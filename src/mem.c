/* memfd backend: GPUOBJ / ion allocations land on plain anonymous memory */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "fkgsl.h"

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif
#ifndef SYS_memfd_create
#define SYS_memfd_create 279
#endif

#define FK_PAGE 4096ULL

struct fk_obj {
  uint64_t size;
  uint64_t mmapsize;
  uint64_t at;
  int inuse;
};

struct fk_region {
  void *host;
  uint64_t len;
  int shadow;
};

static struct fk_obj objs[FK_MAX_OBJ];
static struct fk_region regions[FK_MAX_REGION];
static int nregions;
static uint32_t next_objid = 1;

int fk_memfd(uint64_t len) {
  long fd = syscall(SYS_memfd_create, "libkgsl_fake", MFD_CLOEXEC);
  if (fd < 0) return -1;
  if (ftruncate((int)fd, (off_t)(len ? len : FK_PAGE)) < 0) {
    int saved = errno;
    close((int)fd);
    errno = saved;
    return -1;
  }
  return (int)fd;
}

static uint64_t round_page(uint64_t len) { return (len + FK_PAGE - 1) & ~(FK_PAGE - 1); }

/* One backing store for every GPU object, handed out in page-aligned slices.

   The kernel driver costs a process no descriptor per allocation: an id plus an mmap offset. A memfd
   per object is a lot friendlier to write and it did work until a real model build ran into
   RLIMIT_NOFILE: on Android/bionic the soft limit is 1024 by default, and a model with around a
   thousand live objects exhausts the descriptor table at the memfd call. GPUOBJ_ALLOC then fails and
   the closed runtime surfaces it as nothing more than a null buffer from clCreateBuffer - measured,
   with the peak live object count sitting at 1005. So the arena keeps the descriptor count flat at 1.

   Slices are never recycled: a freed range is abandoned rather than reissued, which keeps two objects
   from ever sharing a page. The cost is that the arena span equals everything allocated so far (204 MB
   for a whole model build, sparse until touched) instead of everything live at once. */
#define FK_ARENA_STEP (8ULL << 20)

static int arena_fd = -1;
static uint64_t arena_used;
static uint64_t arena_span;

static int arena_reserve(uint64_t need) {
  if (arena_fd < 0) {
    arena_fd = fk_memfd(FK_PAGE);
    if (arena_fd < 0) return -1;
  }
  if (need > arena_span) {
    uint64_t grown = (need + FK_ARENA_STEP - 1) & ~(FK_ARENA_STEP - 1);
    if (ftruncate(arena_fd, (off_t)grown) < 0) return -1;
    arena_span = grown;
  }
  return 0;
}

static uint64_t arena_take(uint64_t bytes) {
  uint64_t at = arena_used;
  if (arena_reserve(at + bytes) < 0) return (uint64_t)-1;
  arena_used += bytes;
  return at;
}

int fk_backing(uint64_t len, int *fd, uint64_t *at) {
  uint64_t taken = arena_take(round_page(len));
  if (taken == (uint64_t)-1) return -ENOMEM;
  *fd = arena_fd;
  *at = taken;
  return 0;
}

uint32_t fk_obj_alloc(uint64_t size, uint64_t *mmapsize) {
  uint32_t id = 0;
  for (unsigned scan = 0; scan < FK_MAX_OBJ; scan++) {
    id = next_objid;
    next_objid = next_objid >= FK_MAX_OBJ ? 1 : next_objid + 1;
    if (!objs[id].inuse) break;
  }
  if (!id || objs[id].inuse) return 0;
  uint64_t mapped = round_page(size), at = arena_take(mapped);
  if (at == (uint64_t)-1) return 0;
  objs[id] = (struct fk_obj){ size, mapped, at, 1 };
  *mmapsize = mapped;
  return id;
}

int fk_obj_free(uint32_t id) {
  if (id == 0 || id >= FK_MAX_OBJ || !objs[id].inuse) return -ENOENT;
  objs[id] = (struct fk_obj){ 0, 0, 0, 0 };
  return 0;
}

int fk_obj_size(uint32_t id, uint64_t *size) {
  if (id == 0 || id >= FK_MAX_OBJ || !objs[id].inuse) return -ENOENT;
  *size = objs[id].size;
  return 0;
}

/* A mapping longer than the object itself is answered by growing the arena, not by refusing: the
   object keeps its slice, the neighbour that would have lived next to it simply arrives later. */
int fk_obj_backing(uint32_t id, uint64_t len, int *fd, uint64_t *at) {
  if (id == 0 || id >= FK_MAX_OBJ || !objs[id].inuse) return -ENOENT;
  uint64_t need = objs[id].at + round_page(len ? len : objs[id].mmapsize);
  if (arena_reserve(need) < 0) return -ENOMEM;
  *fd = arena_fd;
  *at = objs[id].at;
  return 0;
}

void fk_region_add(void *host, uint64_t len, int shadow) {
  if (nregions >= FK_MAX_REGION) return;
  regions[nregions++] = (struct fk_region){ host, len, shadow };
}

void fk_shadow_map(void *host, uint64_t len) { fk_region_add(host, len, 1); }

/* A mapping the client has unmapped is not a place to write. Every stamping pass walks this table, so a
   stale entry is a SIGSEGV in the client's process, blamed on this layer. */
void fk_region_drop(void *host, uint64_t len) {
  uintptr_t lo = (uintptr_t)host, hi = lo + len;
  for (int i = 0; i < nregions; i++) {
    uintptr_t rlo = (uintptr_t)regions[i].host, rhi = rlo + regions[i].len;
    if (!regions[i].host || rlo >= hi || rhi <= lo) continue;
    fk_log("region %p len=0x%llx dropped (munmap %p len=0x%llx)\n", regions[i].host,
           (unsigned long long)regions[i].len, host, (unsigned long long)len);
    regions[i] = (struct fk_region){ NULL, 0, 0 };
  }
}

/* advance only the shadow page's timestamp slots: soptimestamp at ts_offset,
   eoptimestamp FK_TS_STRIDE (default 8) bytes after it. Execution is not simulated, so the
   only reason this exists is that a runtime spinning on completion would otherwise hang. */
/* FK_TS_COVERAGE=shadow (default) writes just those slots;
   =all paints every mapped region - blunt, but some runtimes spin somewhere else entirely. */
void fk_shadow_stamp(uint32_t ts) {
  uint64_t map_offset = 0, size = 0;
  uint32_t ts_offset = 0x204;
  const char *stride = getenv("FK_TS_STRIDE");
  unsigned step = stride && *stride ? (unsigned)strtoul(stride, NULL, 0) : 8u;

  const char *coverage = getenv("FK_TS_COVERAGE");
  int coverage_all = coverage && strcmp(coverage, "all") == 0;
  (void)map_offset;
  (void)size;
  fk_prop_shadow(&map_offset, &ts_offset, &size);
  for (int i = 0; i < nregions; i++) {
    if (!regions[i].shadow || !regions[i].host) continue;
    if (ts_offset + step + 4 > regions[i].len) {
      fk_log("shadow too small for ts_offset=0x%llx len=0x%llx\n",
             (unsigned long long)ts_offset, (unsigned long long)regions[i].len);
      continue;
    }
    unsigned char *base = (unsigned char *)regions[i].host;
    memcpy(base + ts_offset, &ts, sizeof ts);
    memcpy(base + ts_offset + step, &ts, sizeof ts);
    if (coverage_all) {
      for (uint64_t k = 0; k + 4 <= regions[i].len; k += 4) memcpy(base + k, &ts, sizeof ts);
    }
  }
}

struct fk_ion {
  int fd;
  uint64_t len;
};
static struct fk_ion ions[FK_MAX_ION];

/* A fence is handed to callers as a descriptor they poll, read or ioctl, so it has to be one.
   Everything this layer submits is already complete, which makes an already-signaled fence the honest
   answer: a real file descriptor that reads as ready and answers every request with success. Handing
   back nothing (the field left unset) instead is what a closed runtime notices as an endless
   "ioctl returned -1 with errno 25" loop -- measured on an Adreno a5xx userland. */
int fk_fence_fd(void) {
  int fd = fk_memfd(8);
  if (fd < 0) return -1;
  const char payload[8] = {0};
  if (write(fd, payload, sizeof payload) != (ssize_t)sizeof payload) {
    int saved = errno;
    close(fd);
    errno = saved;
    return -1;
  }
  lseek(fd, 0, SEEK_SET);
  fk_fd_note(fd, FK_FENCE);
  return fd;
}

int fk_ion_alloc(uint64_t len) {
  uint64_t rounded = round_page(len);
  int fd = fk_memfd(rounded);
  if (fd < 0) return -1;
  for (int i = 0; i < FK_MAX_ION; i++) {
    if (ions[i].fd == 0) {
      ions[i] = (struct fk_ion){ fd, rounded };
      fk_fd_note(fd, FK_ION);
      return fd;
    }
  }
  close(fd);
  return -ENOMEM;
}

int fk_ion_lookup(int fd, uint64_t *len) {
  for (int i = 0; i < FK_MAX_ION; i++) {
    if (ions[i].fd == fd) {
      *len = ions[i].len;
      return 0;
    }
  }
  return -ENOENT;
}

void fk_ion_release(int fd) {
  for (int i = 0; i < FK_MAX_ION; i++) {
    if (ions[i].fd == fd) {
      ions[i] = (struct fk_ion){ 0, 0 };
      return;
    }
  }
}
