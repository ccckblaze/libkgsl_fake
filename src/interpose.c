/* the LD_PRELOAD surface: take over the device nodes, ioctl and mmap; everything else goes to libc untouched */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
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

#ifndef O_LARGEFILE
#define O_LARGEFILE 0
#endif

static int (*real_open)(const char *, int, ...);
static int (*real_openat)(int, const char *, int, ...);
static int (*real_ioctl)(int, unsigned long, void *);
static void *(*real_mmap)(void *, size_t, int, int, int, off_t);
static int (*real_close)(int);
static int (*real_munmap)(void *, size_t);
static int (*real_fstat)(int, struct stat *);
static int (*real_stat)(const char *, struct stat *);
static int (*real_lstat)(const char *, struct stat *);
static int (*real_access)(const char *, int);
static long (*real_syscall)(long, ...);

static int disabled = -1;
static int fake_kgsl_fd = -1;
static uint64_t shadow_offset;
static uint64_t shadow_size;

static void *sym(const char *name) {
  /* fk_real_sym, not dlsym: on bionic the loader cannot be asked from inside a constructor */
  void *p = fk_real_sym(name);
  if (!p) fk_fatal("libc symbol %s not found, the shim cannot work", name);
  return p;
}

static int shim_disabled(void) {
  if (disabled < 0) {
    const char *v = getenv("FK_DISABLE");
    disabled = (v && *v && strcmp(v, "0") != 0) ? 1 : 0;
    fk_props_init();
    if (fk_prop_shadow(&shadow_offset, NULL, &shadow_size))
      fk_log("shadow page: mmap off=0x%llx size=0x%llx\n", (unsigned long long)shadow_offset,
             (unsigned long long)shadow_size);
    int bypass = fk_bypass_begin();
    real_open = sym("open");
    int probe = real_open("/dev/kgsl-3d0", O_RDONLY | O_CLOEXEC);
    fk_bypass_end(bypass);
    fk_log("shim ready: strict=%d exec=%s real /dev/kgsl-3d0=%s\n", fk_strict(),
           fk_exec_abort() ? "abort" : "stub", probe >= 0 ? "present (hidden by the fake device)" : "absent");
    if (probe >= 0) close(probe);
  }
  return disabled;
}

static int fake_open(enum fk_kind kind, const char *path, const char *how) {
  int fd = fk_memfd(0x1000);
  if (fd < 0) return -1;
  fk_fd_note(fd, kind);
  if (kind == FK_KGSL) fake_kgsl_fd = fd;
  fk_log("FAKE %s %s -> fd=%d\n", how, path, fd);
  return fd;
}

static int fk_mirror(const char *path, char *buf, size_t buflen);

static int interpose_open(int dirfd, const char *path, int flags, mode_t mode, const char *how) {
  enum fk_kind kind;
  char mirror[512];
  if (shim_disabled()) return dirfd < 0 ? real_open(path, flags, mode) : real_openat(dirfd, path, flags, mode);
  kind = (enum fk_kind)fk_kind_of_path(path);
  if (kind != FK_NONE) return fake_open(kind, path, how);
  if (fk_mirror(path, mirror, sizeof mirror) == 0) {
    fk_log("MIRROR %s %s -> %s\n", how, path, mirror);
    return real_open(mirror, flags, mode);
  }
  if (dirfd < 0) return real_open(path, flags, mode);
  return real_openat(dirfd, path, flags, mode);
}

int open(const char *path, int flags, ...);
int open(const char *path, int flags, ...) {
  mode_t mode = 0;
  va_list ap;
  va_start(ap, flags);
  mode = va_arg(ap, mode_t);
  va_end(ap);
  if (!real_open) real_open = sym("open");
  return interpose_open(-1, path, flags, mode, "open");
}

int openat(int dirfd, const char *path, int flags, ...);
int openat(int dirfd, const char *path, int flags, ...) {
  mode_t mode = 0;
  va_list ap;
  va_start(ap, flags);
  mode = va_arg(ap, mode_t);
  va_end(ap);
  if (!real_openat) real_openat = sym("openat");
  return interpose_open(dirfd, path, flags, mode, "openat");
}

/* glibc's 64-bit aliases. Miss one and Python's os.open() hands back the real GPU fd,
   so the experiment "passes" for the wrong reason - a lesson this project learned the hard way. */
int open64(const char *path, int flags, ...);
int open64(const char *path, int flags, ...) {
  mode_t mode = 0;
  va_list ap;
  va_start(ap, flags);
  mode = va_arg(ap, mode_t);
  va_end(ap);
  if (!real_open) real_open = sym("open");
  return interpose_open(-1, path, flags | O_LARGEFILE, mode, "open64");
}

int openat64(int dirfd, const char *path, int flags, ...);
int openat64(int dirfd, const char *path, int flags, ...) {
  mode_t mode = 0;
  va_list ap;
  va_start(ap, flags);
  mode = va_arg(ap, mode_t);
  va_end(ap);
  if (!real_openat) real_openat = sym("openat");
  return interpose_open(dirfd, path, flags | O_LARGEFILE, mode, "openat64");
}

static void fake_stat(struct stat *st) {
  memset(st, 0, sizeof *st);
  st->st_mode = S_IFCHR | 0666;
}

/* A board's identity is not a GPU question: it sits in the /sys/devices/soc0 nodes, and a vendor
   runtime reads the SoC id from there before it will go looking for a GPU at all. A box with no board
   has no such node, so the runtime names the target UNKNOWN and refuses the GPU backend -- and a
   substitute device layer cannot talk it into trying. FK_SYSFS names the directory standing in for
   /sys: with FK_SYSFS=/usr/local/fake-sys, the file /usr/local/fake-sys/devices/soc0/soc_id answers
   for /sys/devices/soc0/soc_id. Nothing is invented here -- the mirror holds what was read off a
   board that works, the same way the device table does. */
static int fk_mirror(const char *path, char *buf, size_t buflen) {
  const char *root = getenv("FK_SYSFS");
  struct stat probe;
  if (!root || !*root || !path || strncmp(path, "/sys/", 5) != 0) return -1;
  if ((size_t)snprintf(buf, buflen, "%s/%s", root, path + 5) >= buflen) return -1;
  if (!real_stat) real_stat = sym("stat");
  return real_stat(buf, &probe) == 0 ? 0 : -1;
}

int stat(const char *path, struct stat *st);
int stat(const char *path, struct stat *st) {
  char mirror[512];
  if (!real_stat) real_stat = sym("stat");
  if (!shim_disabled() && fk_kind_of_path(path) && strcmp(path, "/dev/ion") != 0) {
    fake_stat(st);
    return 0;
  }
  if (!shim_disabled() && fk_mirror(path, mirror, sizeof mirror) == 0) return real_stat(mirror, st);
  return real_stat(path, st);
}

int lstat(const char *path, struct stat *st);
int lstat(const char *path, struct stat *st) {
  char mirror[512];
  if (!real_lstat) real_lstat = sym("lstat");
  if (!shim_disabled() && fk_kind_of_path(path) && strcmp(path, "/dev/ion") != 0) {
    fake_stat(st);
    return 0;
  }
  if (!shim_disabled() && fk_mirror(path, mirror, sizeof mirror) == 0) return real_lstat(mirror, st);
  return real_lstat(path, st);
}

int access(const char *path, int mode);
int access(const char *path, int mode) {
  char mirror[512];
  if (!real_access) real_access = sym("access");
  if (!shim_disabled() && fk_kind_of_path(path)) return 0;
  if (!shim_disabled() && fk_mirror(path, mirror, sizeof mirror) == 0) return 0;
  return real_access(path, mode);
}

int fstat(int fd, struct stat *st);
int fstat(int fd, struct stat *st) {
  if (!real_fstat) real_fstat = sym("fstat");
  if (!shim_disabled() && fk_fd_kind(fd) != FK_NONE) {
    fake_stat(st);
    return 0;
  }
  return real_fstat(fd, st);
}

static int interpose_ioctl(int fd, unsigned long request, void *arg) {
  enum fk_kind kind;
  if (shim_disabled()) return real_ioctl(fd, request, arg);
  kind = fk_fd_kind(fd);
  if (kind == FK_KGSL) return fk_kgsl_ioctl(request, arg);
  if (kind == FK_ION) return fk_ion_ioctl(request, arg);
  if (kind == FK_PMSG) return 0;
  if (kind == FK_FENCE) return 0; /* a fence that is already signaled answers any wait with success */
  /* a real GPU fd opened elsewhere (or via a direct syscall) arrived here */
  if (fk_kgsl_request(request)) {
    if (fk_strict())
      fk_fatal("fd=%d is not a fake fd issued by this shim, yet it carries a KGSL ioctl nr=0x%02x."
               " The program bypassed symbol interception (static linking / raw syscalls / fd passing),"
               " so it is talking to a real GPU and every \"fake device passed\" conclusion is void."
               " Set FK_STRICT=0 to pass it through for a control run.", fd, _IOC_NR(request));
    fk_log("TRIPWIRE real device fd=%d nr=0x%02x passed through (strict=0)\n", fd, _IOC_NR(request));
  }
  return real_ioctl(fd, request, arg);
}

int ioctl(int fd, unsigned long request, ...);
int ioctl(int fd, unsigned long request, ...) {
  void *arg;
  va_list ap;
  va_start(ap, request);
  arg = va_arg(ap, void *);
  va_end(ap);
  if (!real_ioctl) real_ioctl = sym("ioctl");
  return interpose_ioctl(fd, request, arg);
}

/* callers that go straight through syscall() (libgsl does, so does python's os.ioctl wrapper) */
long syscall(long number, ...);
long syscall(long number, ...) {
  va_list ap;
  long a0, a1, a2, a3, a4, a5;
  long result;
  int bypass;
  char mirror[512];
  const char *mpath;
  va_start(ap, number);
  a0 = va_arg(ap, long);
  a1 = va_arg(ap, long);
  a2 = va_arg(ap, long);
  a3 = va_arg(ap, long);
  a4 = va_arg(ap, long);
  a5 = va_arg(ap, long);
  va_end(ap);
  if (!real_syscall) real_syscall = sym("syscall");
  if (shim_disabled()) return real_syscall(number, a0, a1, a2, a3, a4, a5);

  if (number == SYS_ioctl && fk_fd_kind((int)a0) == FK_KGSL)
    return fk_kgsl_ioctl((unsigned long)a1, (void *)a2);
  if (number == SYS_ioctl && fk_fd_kind((int)a0) == FK_ION)
    return fk_ion_ioctl((unsigned long)a1, (void *)a2);
#ifdef SYS_open
  if (number == SYS_open && fk_kind_of_path((const char *)a0))
    return fake_open((enum fk_kind)fk_kind_of_path((const char *)a0), (const char *)a0, "syscall(open)");
#endif
  if (number == SYS_openat && fk_kind_of_path((const char *)a1))
    return fake_open((enum fk_kind)fk_kind_of_path((const char *)a1), (const char *)a1, "syscall(openat)");

  /* the same mirror, for the callers that never touch the open() symbol at all */
  mpath = NULL;
#ifdef SYS_open
  if (number == SYS_open) mpath = (const char *)a0;
#endif
  if (number == SYS_openat) mpath = (const char *)a1;
#ifdef SYS_stat
  if (number == SYS_stat) mpath = (const char *)a0;
#endif
#ifdef SYS_lstat
  if (number == SYS_lstat) mpath = (const char *)a0;
#endif
#ifdef SYS_newfstatat
  if (number == SYS_newfstatat) mpath = (const char *)a1;
#endif
#ifdef SYS_statx
  if (number == SYS_statx) mpath = (const char *)a1;
#endif
  if (mpath && fk_mirror(mpath, mirror, sizeof mirror) == 0) {
    fk_log("MIRROR syscall(%ld) %s -> %s\n", number, mpath, mirror);
    if (number == SYS_openat) return real_syscall(number, a0, (long)mirror, a2, a3, a4, a5);
#ifdef SYS_newfstatat
    if (number == SYS_newfstatat) return real_syscall(number, a0, (long)mirror, a2, a3, a4, a5);
#endif
#ifdef SYS_statx
    if (number == SYS_statx) return real_syscall(number, a0, (long)mirror, a2, a3, a4, a5);
#endif
    return real_syscall(number, (long)mirror, a1, a2, a3, a4, a5);
  }

  bypass = fk_bypass_begin();
  result = real_syscall(number, a0, a1, a2, a3, a4, a5);
  fk_bypass_end(bypass);
  return result;
}

/* mmap convention: offset = obj_id << 12; the shadow page uses the address DEVICE_SHADOW reports as offset */
static void *interpose_mmap(void *addr, size_t len, int prot, int flags, int fd, uint64_t offset,
                            const char *how) {
  enum fk_kind kind;
  uint32_t id;
  int backed;
  uint64_t at;
  void *mapped;
  if (shim_disabled() || fk_fd_kind(fd) != FK_KGSL)
    return real_mmap(addr, len, prot, flags, fd, (off_t)offset);

  id = (uint32_t)(offset >> 12);
  if (fk_obj_backing(id, len, &backed, &at) == 0) {
    mapped = real_mmap(addr, len, prot | PROT_WRITE, flags | MAP_SHARED, backed, (off_t)at);
    fk_log("%s kgsl id=%u off=0x%llx len=0x%zx -> %p\n", how, id, (unsigned long long)offset, len,
           mapped);
    return mapped;
  }

  if (shadow_offset && offset == shadow_offset) {
    if (fk_backing(len < shadow_size ? shadow_size : len, &backed, &at) < 0) return MAP_FAILED;
    mapped = real_mmap(addr, len, prot | PROT_WRITE, flags | MAP_SHARED, backed, (off_t)at);
    if (mapped != MAP_FAILED) {
      fk_shadow_map(mapped, len);
      fk_shadow_stamp(0);
      fk_log("%s kgsl SHADOW off=0x%llx len=0x%zx -> %p\n", how, (unsigned long long)offset, len,
             mapped);
    }
    return mapped;
  }

  kind = fk_fd_kind(fd);
  fk_log("%s kgsl unknown offset=0x%llx (kind=%d) -> anonymous page\n", how, (unsigned long long)offset, kind);
  if (fk_backing(len, &backed, &at) < 0) return MAP_FAILED;
  mapped = real_mmap(addr, len, prot | PROT_WRITE, flags | MAP_SHARED, backed, (off_t)at);
  if (mapped != MAP_FAILED) fk_region_add(mapped, len, 0);
  return mapped;
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t offset);
void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t offset) {
  if (!real_mmap) real_mmap = sym("mmap");
  return interpose_mmap(addr, len, prot, flags, fd, (uint64_t)offset, "mmap");
}

void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off64_t offset);
void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off64_t offset) {
  if (!real_mmap) real_mmap = sym("mmap");
  return interpose_mmap(addr, len, prot, flags, fd, (uint64_t)offset, "mmap64");
}

int munmap(void *addr, size_t len);
int munmap(void *addr, size_t len) {
  if (!real_munmap) real_munmap = sym("munmap");
  if (!shim_disabled()) fk_region_drop(addr, len);
  return real_munmap(addr, len);
}

int close(int fd);
int close(int fd) {
  enum fk_kind kind;
  if (!real_close) real_close = sym("close");
  if (shim_disabled()) return real_close(fd);
  kind = fk_fd_kind(fd);
  if (kind != FK_NONE) {
    if (fd == fake_kgsl_fd) fake_kgsl_fd = -1;
    fk_fd_drop(fd);
    fk_log("CLOSE fake fd=%d kind=%d\n", fd, kind);
  } else {
    fk_ion_release(fd);
  }
  return real_close(fd);
}

#if !defined(__ANDROID__)
static void __attribute__((constructor)) fk_ctor(void) { shim_disabled(); }
#else
/* No constructor here, for the reason written in rec.c: a lazy PLT call made while the linker holds its
   lock for constructors deadlocks on it. shim_disabled() is the first thing every hook does, and it is
   idempotent, so nothing depends on this running at load time. */
#endif

#if defined(__BIONIC__)
/* No dlsym side door here: bionic has no dlvsym to bootstrap a real dlsym from, and the door exists
   for one specific caller shape -- a program that dlopens libc by name and dlsyms mmap/ioctl out of
   that handle (measured: python's ctypes). The native paths this layer serves resolve those symbols
   through the PLT, where plain interposition already applies. */
#else
/* ---------- the ctypes side door: rewriting dlsym ----------
   tinygrad's FileIOInterface.mmap goes through ctypes.CDLL("libc.so.6").mmap, i.e. dlopen a
   specific library and dlsym against that handle: you always get libc's own implementation,
   the PLT is never involved, so LD_PRELOAD interception does not apply at all (measured: the
   fake fd mmap'd a 4KB memfd as if it were large, and the first out-of-range touch SIGBUSed).
   Closed runtimes use this pattern a lot, so the door has to be shut too.
   The rule is conservative: substitute only when the resolved address is exactly libc's implementation. */
static void *(*real_dlsym)(void *, const char *);
static const char *const fk_dlsym_versions[] = { "GLIBC_2.17", "GLIBC_2.2.5", "GLIBC_2.34", "GLIBC_2.0" };

struct fk_sym_slot {
  const char *name;
  void *shim;
  void *libc; /* filled lazily: libc's own implementation, used for the comparison above */
};

static struct fk_sym_slot fk_sym_slots[] = {
  { "mmap", (void *)mmap, NULL },     { "mmap64", (void *)mmap64, NULL },
  { "ioctl", (void *)ioctl, NULL },   { "open", (void *)open, NULL },
  { "open64", (void *)open64, NULL }, { "openat", (void *)openat, NULL },
  { "openat64", (void *)openat64, NULL },
  { "stat", (void *)stat, NULL },     { "lstat", (void *)lstat, NULL },
  { "fstat", (void *)fstat, NULL },   { "access", (void *)access, NULL },
  { "munmap", (void *)munmap, NULL },
};

static int dlsym_bootstrap(void) {
  for (unsigned i = 0; i < sizeof fk_dlsym_versions / sizeof *fk_dlsym_versions && !real_dlsym; i++)
    real_dlsym = dlvsym(RTLD_NEXT, "dlsym", fk_dlsym_versions[i]);
  if (!real_dlsym) fk_log("cannot obtain libc dlsym (tried %s), symbol rewriting is off\n", fk_dlsym_versions[0]);
  return real_dlsym != NULL;
}

static void *libc_impl(struct fk_sym_slot *slot) {
  if (!slot->libc) slot->libc = real_dlsym(RTLD_NEXT, slot->name);
  return slot->libc;
}

void *dlsym(void *handle, const char *name);
void *dlsym(void *handle, const char *name) {
  void *found;
  if (!real_dlsym && !dlsym_bootstrap()) return NULL;
  found = real_dlsym(handle, name);
  if (found == NULL || shim_disabled()) return found;
  /* the shim resolves the real implementations through RTLD_NEXT, so never rewrite those */
  if (handle == RTLD_NEXT || handle == RTLD_DEFAULT) return found;
  for (unsigned i = 0; i < sizeof fk_sym_slots / sizeof *fk_sym_slots; i++) {
    struct fk_sym_slot *slot = &fk_sym_slots[i];
    if (strcmp(name, slot->name) != 0) continue;
    if (found != libc_impl(slot)) return found; /* not libc's own implementation, leave it alone */
    fk_log("dlsym(%p, \"%s\") resolved libc implementation -> rewritten to shim %p\n", handle, name, slot->shim);
    return slot->shim;
  }
  return found;
}
#endif

#if defined(__BIONIC__)
__asm__(".globl ioctl\n.set ioctl, fk_ioctl");
#endif
