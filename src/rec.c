/* LD_PRELOAD recorder: keeps the real GPU and copies the ioctl surface + command batches verbatim.
   Use 1: re-record props (docs/record.md). Use 2: check whether the fake device missed a requirement. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "fkgsl.h"
#include "msm_kgsl.h"
#include "raw.h"

#if defined(__BIONIC__)
/* Android/bionic declares ioctl() with an int request behind an overloadable generic macro, so this
   shim's definition cannot reuse that C name. Compile the body under a private name and let the
   alias at the bottom of the file hand out the link name every caller resolves. */
#define ioctl fk_ioctl
#endif

struct fk_pm4_op {
  unsigned op;
  const char *name;
};
#include "pm4_names.h"

static int (*real_open)(const char *, int, ...);
static int (*real_openat)(int, const char *, int, ...);
static int (*real_ioctl)(int, unsigned long, void *);
static void *(*real_mmap)(void *, size_t, int, int, int, off_t);
static int (*real_close)(int);

static struct fk_sink sink;
static int sink_tried;
static int kgsl_fd = -1;
static char dump_dir[256];

struct rec_obj {
  uint32_t id;
  uint64_t size;
  uint64_t mmapsize;
  void *host;
  int inuse;
};
static struct rec_obj objs[FK_MAX_OBJ];

static struct fk_sink *out(void) {
  if (!sink_tried) {
    const char *path = getenv("FK_REC_OUT");
    sink_tried = 1;
    if (fk_sink_open(&sink, path && *path ? path : FK_DEFAULT_REC) != 0) {
      sink.fp = NULL;
      sink.fd = -1;
    }
    const char *dir = getenv("FK_DUMP_DIR");
    if (dir && *dir) snprintf(dump_dir, sizeof dump_dir, "%s", dir);
  }
  return (sink.fp || sink.fd >= 0) ? &sink : NULL;
}

static void say(const char *fmt, ...) {
  struct fk_sink *f = out();
  if (!f) return;
  va_list ap;
  va_start(ap, fmt);
  fk_sink_vprintf(f, fmt, ap);
  va_end(ap);
}

static const char *pm4_name(unsigned op) {
  for (unsigned i = 0; i < FK_PM4_OP_COUNT; i++)
    if (fk_pm4_ops[i].op == op) return fk_pm4_ops[i].name;
  return "OP_UNKNOWN";
}

static void *host_of(uint32_t id, uint64_t offset) {
  if (id >= FK_MAX_OBJ || !objs[id].inuse || !objs[id].host) return NULL;
  if (offset >= objs[id].size) return NULL;
  return (char *)objs[id].host + offset;
}

/* from a6xx on, pkt0/pkt3 are gone: typ4 (register writes) / typ7 (packets carrying an opcode) */
static void decode_pm4(const unsigned char *dat, size_t len, const char *tag) {
  const unsigned *words = (const unsigned *)dat;
  size_t nw = len / 4;
  struct fk_sink *f = out();
  for (size_t ptr = 0; ptr < nw; ) {
    unsigned cmd = words[ptr];
    if ((cmd >> 24) == 0x70) {
      unsigned opcode = (cmd >> 16) & 0x7F, size = cmd & 0x3FFF;
      say("    %s %zX typ7 %-24s size=%u", tag, ptr, pm4_name(opcode), size);
      for (unsigned i = 0; i < size && ptr + 1 + i < nw; i++) say(" %08x", words[ptr + 1 + i]);
      say("\n");
      const char *opname = pm4_name(opcode);
      if (strstr(opname, "CP_LOAD_STATE6") && size >= 3 && f) {
        uint64_t src = ((uint64_t)words[ptr + 3] << 32) | words[ptr + 2];
        unsigned state_block = (words[ptr + 1] >> 18) & 0xF, state_type = (words[ptr + 1] >> 14) & 3;
        unsigned num_unit = words[ptr + 1] >> 22;
        say("      LOAD_STATE6 %s: block=%u type=%u num_unit=%u src=0x%llx\n", opname, state_block,
            state_type, num_unit, (unsigned long long)src);
        if (dump_dir[0] && (state_block == 13 /* CS_SHADER */)) {
          char path[384];
          snprintf(path, sizeof path, "%s/load_%s_b%u_t%u_%llx.bin", dump_dir,
                   strstr(opname, "FRAG") ? "frag" : "geom", state_block, state_type,
                   (unsigned long long)src);
          size_t bytes = num_unit * (state_type == 0 ? 128u : 4u);
          if (bytes && bytes <= 0x400000) {
            struct fk_sink d;
            if (fk_sink_create(&d, path) == 0) {
              const unsigned char *p = (const unsigned char *)(uintptr_t)src;
              fk_sink_write(&d, (const char *)p, bytes);
              fk_sink_close(&d);
              say("      dumped %zu bytes -> %s\n", bytes, path);
            }
          }
        }
      }
      ptr += 1 + size;
    } else if ((cmd >> 28) == 0x4) {
      unsigned offset = (cmd >> 8) & 0x7FFFF, size = cmd & 0x7F;
      say("    %s %zX typ4 reg=0x%05x size=%u", tag, ptr, offset, size);
      for (unsigned i = 0; i < size && ptr + 1 + i < nw; i++) say(" %08x", words[ptr + 1 + i]);
      say("\n");
      ptr += 1 + size;
    } else {
      say("    %s %zX raw %08x\n", tag, ptr, cmd);
      ptr += 1;
    }
  }
}

static void dump_cmdstream(struct kgsl_gpu_command *cmd) {
  say("IOCTL GPU_COMMAND ts=%u ctx=%u numcmds=%u numobjs=%u\n", cmd->timestamp, cmd->context_id,
      cmd->numcmds, cmd->numobjs);
  for (unsigned i = 0; i < cmd->numcmds; i++) {
    struct kgsl_command_object *o = &((struct kgsl_command_object *)(uintptr_t)cmd->cmdlist)[i];
    void *host = host_of(o->id, o->offset);
    say("  cmd[%u] id=%u offset=0x%llx gpuaddr=0x%llx size=0x%llx flags=0x%x host=%s\n", i, o->id,
        (unsigned long long)o->offset, (unsigned long long)o->gpuaddr,
        (unsigned long long)o->size, o->flags, host ? "MAPPED" : "unmapped");
    if (host) decode_pm4((const unsigned char *)host, o->size < 0x2000 ? o->size : 0x2000, "  |");
  }
  for (unsigned i = 0; i < cmd->numobjs; i++) {
    struct kgsl_command_object *o = &((struct kgsl_command_object *)(uintptr_t)cmd->objlist)[i];
    say("  obj[%u] id=%u offset=0x%llx gpuaddr=0x%llx size=0x%llx flags=0x%x\n", i, o->id,
        (unsigned long long)o->offset, (unsigned long long)o->gpuaddr,
        (unsigned long long)o->size, o->flags);
  }
}

static void note_ret(int fd, unsigned long request, void *arg, long ret) {
  struct kgsl_device_getproperty *prop;
  if (_IOC_NR(request) == 0x02 && arg) {
    prop = arg;
    if (prop->value && prop->sizebytes && prop->sizebytes <= 64 && ret == 0) {
      say("IOCTL PROPSAVE type=0x%02x size=%zu value=", prop->type, prop->sizebytes);
      const unsigned char *b = prop->value;
      for (size_t i = 0; i < prop->sizebytes; i++) say("%02x", b[i]);
      say("\n");
    }
  }
  if (_IOC_NR(request) == 0x4a && arg && ret == 0) dump_cmdstream(arg);
}

static int watch_path(const char *path) {
  return strstr(path, "kgsl") || strstr(path, "/dev/ion") || strstr(path, "pmsg") ? 1 : 0;
}

int open(const char *path, int flags, ...);
int open(const char *path, int flags, ...) {
  mode_t mode = 0;
  va_list ap;
  va_start(ap, flags);
  mode = va_arg(ap, mode_t);
  va_end(ap);
  if (!real_open) real_open = fk_real_sym("open");
  int fd = real_open(path, flags, mode);
  if (watch_path(path)) say("OPEN %s -> fd %d\n", path, fd);
  if (strstr(path, "kgsl-3d0") && fd >= 0) kgsl_fd = fd;
  return fd;
}

int openat(int dirfd, const char *path, int flags, ...);
int openat(int dirfd, const char *path, int flags, ...) {
  mode_t mode = 0;
  va_list ap;
  va_start(ap, flags);
  mode = va_arg(ap, mode_t);
  va_end(ap);
  if (!real_openat) real_openat = fk_real_sym("openat");
  int fd = real_openat(dirfd, path, flags, mode);
  if (watch_path(path)) say("OPENAT %s -> fd %d\n", path, fd);
  if (strstr(path, "kgsl-3d0") && fd >= 0) kgsl_fd = fd;
  return fd;
}

int open64(const char *path, int flags, ...);
int open64(const char *path, int flags, ...) { return open(path, flags); }
int openat64(int dirfd, const char *path, int flags, ...);
int openat64(int dirfd, const char *path, int flags, ...) { return openat(dirfd, path, flags); }

static void *rec_mmap(void *addr, size_t len, int prot, int flags, int fd, uint64_t offset,
                      const char *how) {
  if (!real_mmap) real_mmap = fk_real_sym("mmap");
  void *mapped = real_mmap(addr, len, prot, flags, fd, (off_t)offset);
  if (fd == kgsl_fd) {
    uint32_t id = (uint32_t)(offset >> 12);
    say("%s fd=%d len=0x%zx prot=%d off=0x%llx -> %p\n", how, fd, len, prot,
        (unsigned long long)offset, mapped);
    if (id < FK_MAX_OBJ && objs[id].inuse && mapped != MAP_FAILED) objs[id].host = mapped;
  }
  return mapped;
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t offset);
void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t offset) {
  return rec_mmap(addr, len, prot, flags, fd, (uint64_t)offset, "MMAP");
}
void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off64_t offset);
void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off64_t offset) {
  return rec_mmap(addr, len, prot, flags, fd, (uint64_t)offset, "MMAP64");
}

int ioctl(int fd, unsigned long request, ...);
int ioctl(int fd, unsigned long request, ...) {
  void *arg;
  va_list ap;
  va_start(ap, request);
  arg = va_arg(ap, void *);
  va_end(ap);
  if (!real_ioctl) real_ioctl = fk_real_sym("ioctl");
  if (fd != kgsl_fd) return real_ioctl(fd, request, arg);

  unsigned nr = _IOC_NR(request);
  int pret = -1;
  if (nr == 0x45 && arg) {
    struct kgsl_gpuobj_alloc *a = arg;
    pret = (int)real_ioctl(fd, request, arg);
    if (pret == 0 && a->id < FK_MAX_OBJ)
      objs[a->id] = (struct rec_obj){ a->id, a->size, a->mmapsize, NULL, 1 };
    say("IOCTL GPUOBJ_ALLOC size=0x%llx flags=0x%llx -> id=%u mmapsize=0x%llx ret=%d\n",
        (unsigned long long)a->size, (unsigned long long)a->flags, a->id,
        (unsigned long long)a->mmapsize, pret);
    return pret;
  }
  if (nr == 0x46 && arg) {
    struct kgsl_gpuobj_free *a = arg;
    say("IOCTL GPUOBJ_FREE id=%u\n", a->id);
    if (a->id < FK_MAX_OBJ) objs[a->id] = (struct rec_obj){ 0, 0, 0, NULL, 0 };
  }
  if (nr == 0x02 && arg) {
    struct kgsl_device_getproperty *g = arg;
    say("IOCTL GETPROPERTY type=0x%02x size=%zu", g->type, g->sizebytes);
  } else {
    say("IOCTL nr=0x%02x size=%u", nr, _IOC_SIZE(request));
  }
  long ret = real_ioctl(fd, request, arg);
  if (nr == 0x02) say(" ret=%ld\n", ret);
  else say(" ret=%ld\n", ret);
  note_ret(fd, request, arg, ret);
  return (int)ret;
}

int close(int fd);
int close(int fd) {
  if (!real_close) real_close = fk_real_sym("close");
  if (fd == kgsl_fd) kgsl_fd = -1;
  return real_close(fd);
}

#if !defined(__ANDROID__)
static void __attribute__((constructor)) rec_ctor(void) { out(); }
#else
/* No constructor here. Android's linker holds a non-recursive lock while it runs constructors, and a
   lazy PLT call taken from inside one waits on that same lock: measured on API 23, opening the log file
   in the constructor stops the process in futex_wait_queue_me before main(). out() is called by every
   write anyway, so starting it later costs nothing. */
#endif

#if defined(__BIONIC__)
__asm__(".globl ioctl\n.set ioctl, fk_ioctl");
#endif
