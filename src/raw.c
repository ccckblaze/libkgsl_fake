/* The real implementations, obtained without asking the loader for anything.

   Android/bionic guards dlopen and dlsym with a non-recursive lock and runs library constructors while
   holding it. A preload that resolves its `real_open` with dlsym(RTLD_NEXT, ...) therefore deadlocks in
   its own constructor: measured on an API 23 aarch64 userland, every preloaded process stops in
   futex_wait_queue_me and never reaches main(). Only raw system calls can be issued from inside a
   constructor, so on bionic the table below replaces the loader lookup.

   glibc keeps what it always had: dlsym(RTLD_NEXT, ...), versioned, from a loader that tolerates it. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include "raw.h"

#if defined(__ANDROID__) || defined(__BIONIC__)

#if !defined(__aarch64__)
#error "the bionic path of this layer is written for aarch64"
#endif

/* arm64 has no bare open(2) and no mmap2; these are the numbers that syscall table uses */
#ifndef __NR_ioctl
#define __NR_ioctl 29
#endif
#ifndef __NR_faccessat
#define __NR_faccessat 48
#endif
#ifndef __NR_newfstatat
#define __NR_newfstatat 79
#endif
#ifndef __NR_fstat
#define __NR_fstat 80
#endif
#ifndef __NR_openat
#define __NR_openat 56
#endif
#ifndef __NR_close
#define __NR_close 57
#endif
#ifndef __NR_mmap
#define __NR_mmap 222
#endif
#ifndef __NR_munmap
#define __NR_munmap 215
#endif
#ifndef __NR_write
#define __NR_write 64
#endif
#ifndef __NR_read
#define __NR_read 63
#endif

static long svc6(long nr, long a0, long a1, long a2, long a3, long a4, long a5) {
  register long x8 __asm__("x8") = nr;
  register long x0 __asm__("x0") = a0;
  register long x1 __asm__("x1") = a1;
  register long x2 __asm__("x2") = a2;
  register long x3 __asm__("x3") = a3;
  register long x4 __asm__("x4") = a4;
  register long x5 __asm__("x5") = a5;
  __asm__ volatile("svc #0"
                   : "+r"(x0)
                   : "r"(x8), "r"(x1), "r"(x2), "r"(x3), "r"(x4), "r"(x5)
                   : "memory");
  return x0;
}

/* libc convention, not the kernel's: failures come back as -1 with errno set */
static long ret(long r) {
  if (r < 0 && r > -4096) {
    errno = (int)-r;
    return -1;
  }
  return r;
}

static int raw_open(const char *path, int flags, ...) {
  mode_t mode = 0;
  va_list ap;
  va_start(ap, flags);
  mode = (mode_t)va_arg(ap, int);
  va_end(ap);
  return (int)ret(svc6(__NR_openat, AT_FDCWD, (long)path, flags, mode, 0, 0));
}

static int raw_openat(int dirfd, const char *path, int flags, ...) {
  mode_t mode = 0;
  va_list ap;
  va_start(ap, flags);
  mode = (mode_t)va_arg(ap, int);
  va_end(ap);
  return (int)ret(svc6(__NR_openat, dirfd, (long)path, flags, mode, 0, 0));
}

static int raw_ioctl(int fd, unsigned long request, void *argp) {
  return (int)ret(svc6(__NR_ioctl, fd, (long)request, (long)argp, 0, 0, 0));
}

static void *raw_mmap(void *addr, size_t len, int prot, int flags, int fd, off_t offset) {
  long r = ret(svc6(__NR_mmap, (long)addr, (long)len, prot, flags, fd, (long)offset));
  return r == -1 ? MAP_FAILED : (void *)r;
}

static int raw_close(int fd) { return (int)ret(svc6(__NR_close, fd, 0, 0, 0, 0, 0)); }

static int raw_munmap(void *addr, size_t len) { return (int)ret(svc6(__NR_munmap, (long)addr, len, 0, 0, 0, 0)); }

static int raw_fstat(int fd, struct stat *buf) {
  return (int)ret(svc6(__NR_fstat, fd, (long)buf, 0, 0, 0, 0));
}

static int raw_stat(const char *path, struct stat *buf) {
  return (int)ret(svc6(__NR_newfstatat, AT_FDCWD, (long)path, (long)buf, 0, 0, 0));
}

static int raw_lstat(const char *path, struct stat *buf) {
  return (int)ret(svc6(__NR_newfstatat, AT_FDCWD, (long)path, (long)buf, AT_SYMLINK_NOFOLLOW, 0, 0));
}

static int raw_access(const char *path, int mode) {
  return (int)ret(svc6(__NR_faccessat, AT_FDCWD, (long)path, mode, 0, 0, 0));
}

static long raw_syscall(long nr, ...) {
  long a[6] = {0};
  va_list ap;
  va_start(ap, nr);
  for (int i = 0; i < 6; i++) a[i] = va_arg(ap, long);
  va_end(ap);
  return svc6(nr, a[0], a[1], a[2], a[3], a[4], a[5]);
}

static void fk_err_raw(const char *buf, size_t len) {
  size_t done = 0;
  while (done < len) {
    long w = svc6(__NR_write, 2, (long)(buf + done), len - done, 0, 0, 0);
    if (w <= 0) return;
    done += (size_t)w;
  }
}

static const struct {
  const char *name;
  void *impl;
} raw_table[] = {
  { "open", (void *)raw_open },       { "open64", (void *)raw_open },
  { "openat", (void *)raw_openat },   { "openat64", (void *)raw_openat },
  { "ioctl", (void *)raw_ioctl },     { "mmap", (void *)raw_mmap },
  { "mmap64", (void *)raw_mmap },     { "close", (void *)raw_close },
  { "fstat", (void *)raw_fstat },     { "stat", (void *)raw_stat },
  { "lstat", (void *)raw_lstat },     { "access", (void *)raw_access },
  { "munmap", (void *)raw_munmap },
  { "syscall", (void *)raw_syscall },
};

void *fk_real_sym(const char *name) {
  for (unsigned i = 0; i < sizeof raw_table / sizeof *raw_table; i++)
    if (strcmp(name, raw_table[i].name) == 0) return raw_table[i].impl;
  return NULL;
}

#else /* glibc and everyone else: the loader is safe to ask */

void *fk_real_sym(const char *name) { return dlsym(RTLD_NEXT, name); }

#endif

/* ---------- the sink: one line per write, stdio never ----------
   These hooks run in the middle of other people's libc calls, and on Android that includes stdio's own
   start-up: opening the log from inside open() re-enters a locked stdio and the process stops inside
   __sinit forever (measured with debuggerd on API 23: fopen -> __sfp -> __sinit -> pthread_mutex_lock).
   So nothing this layer prints goes through a FILE. glibc keeps stdio here because it survives that
   re-entry and the tests measure its output; bionic gets raw descriptors. Same text either way. */
#if defined(__ANDROID__) || defined(__BIONIC__)

int fk_sink_open(struct fk_sink *s, const char *path) {
  s->fp = NULL;
  s->fd = (int)ret(svc6(__NR_openat, AT_FDCWD, (long)path, O_WRONLY | O_CREAT | O_APPEND, 0644, 0, 0));
  return s->fd >= 0 ? 0 : -1;
}

int fk_sink_create(struct fk_sink *s, const char *path) {
  s->fp = NULL;
  s->fd = (int)ret(svc6(__NR_openat, AT_FDCWD, (long)path, O_WRONLY | O_CREAT | O_TRUNC, 0644, 0, 0));
  return s->fd >= 0 ? 0 : -1;
}

void fk_sink_write(struct fk_sink *s, const char *buf, size_t len) {
  size_t done = 0;
  while (done < len) {
    long w = svc6(__NR_write, s->fd, (long)(buf + done), len - done, 0, 0, 0);
    if (w <= 0) return;
    done += (size_t)w;
  }
}

void fk_sink_close(struct fk_sink *s) {
  if (s->fd >= 0) {
    svc6(__NR_close, s->fd, 0, 0, 0, 0, 0);
    s->fd = -1;
  }
}

void fk_sink_vprintf(struct fk_sink *s, const char *fmt, va_list ap) {
  char stack_buf[2048];
  int need = vsnprintf(stack_buf, sizeof stack_buf, fmt, ap);
  if (need < 0) return;
  if ((size_t)need < sizeof stack_buf) {
    if (s) fk_sink_write(s, stack_buf, (size_t)need);
    else fk_err_raw(stack_buf, (size_t)need);
    return;
  }
  char *big = malloc((size_t)need + 1);
  if (!big) return;
  va_list copy;
  va_copy(copy, ap);
  need = vsnprintf(big, (size_t)need + 1, fmt, copy);
  va_end(copy);
  if (need > 0) {
    if (s) fk_sink_write(s, big, (size_t)need);
    else fk_err_raw(big, (size_t)need);
  }
  free(big);
}

int fk_sink_printf(struct fk_sink *s, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fk_sink_vprintf(s, fmt, ap);
  va_end(ap);
  return 0;
}

void fk_err_printf(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fk_sink_vprintf(NULL, fmt, ap);
  va_end(ap);
}

void fk_err_vprintf(const char *prefix, const char *fmt, va_list ap) {
  size_t n = strlen(prefix);
  fk_err_raw(prefix, n);
  fk_sink_vprintf(NULL, fmt, ap);
  fk_err_raw("\n", 1);
}

char *fk_read_file(const char *path, size_t *len_out) {
  int fd = (int)ret(svc6(__NR_openat, AT_FDCWD, (long)path, O_RDONLY, 0, 0, 0));
  if (fd < 0) return NULL;
  size_t len = 0, cap = 4096;
  char *buf = malloc(cap);
  if (!buf) {
    svc6(__NR_close, fd, 0, 0, 0, 0, 0);
    return NULL;
  }
  for (;;) {
    if (len + 1 >= cap) {
      char *grown = realloc(buf, cap * 2);
      if (!grown) {
        free(buf);
        svc6(__NR_close, fd, 0, 0, 0, 0, 0);
        return NULL;
      }
      buf = grown;
      cap *= 2;
    }
    long got = svc6(__NR_read, fd, (long)(buf + len), (long)(cap - len - 1), 0, 0, 0);
    if (got <= 0) break;
    len += (size_t)got;
  }
  svc6(__NR_close, fd, 0, 0, 0, 0, 0);
  buf[len] = '\0';
  if (len_out) *len_out = len;
  return buf;
}

#else /* glibc: stdio is re-entrant enough here, and this is the shape the tests were built against */

int fk_sink_open(struct fk_sink *s, const char *path) {
  s->fd = -1;
  s->fp = fopen(path, "a");
  if (!s->fp) return -1;
  setvbuf((FILE *)s->fp, NULL, _IOLBF, 0);
  return 0;
}

int fk_sink_create(struct fk_sink *s, const char *path) {
  s->fd = -1;
  s->fp = fopen(path, "wb");
  return s->fp ? 0 : -1;
}

void fk_sink_write(struct fk_sink *s, const char *buf, size_t len) {
  if (!s->fp) return;
  fwrite(buf, 1, len, (FILE *)s->fp);
  fflush((FILE *)s->fp);
}

void fk_sink_vprintf(struct fk_sink *s, const char *fmt, va_list ap) {
  if (!s || !s->fp) return;
  vfprintf((FILE *)s->fp, fmt, ap);
  fflush((FILE *)s->fp);
}

int fk_sink_printf(struct fk_sink *s, const char *fmt, ...) {
  if (!s->fp) return -1;
  va_list ap;
  va_start(ap, fmt);
  fk_sink_vprintf(s, fmt, ap);
  va_end(ap);
  return 0;
}

void fk_sink_close(struct fk_sink *s) {
  if (s->fp) {
    fclose((FILE *)s->fp);
    s->fp = NULL;
  }
}

void fk_err_vprintf(const char *prefix, const char *fmt, va_list ap) {
  fprintf(stderr, "%s", prefix);
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  fflush(stderr);
}

void fk_err_printf(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fk_err_vprintf("", fmt, ap);
  va_end(ap);
}

char *fk_read_file(const char *path, size_t *len_out) {
  FILE *in = fopen(path, "r");
  if (!in) return NULL;
  size_t len = 0, cap = 4096;
  char *buf = malloc(cap);
  if (!buf) {
    fclose(in);
    return NULL;
  }
  while (len + 1 < cap) {
    size_t got = fread(buf + len, 1, cap - len - 1, in);
    if (!got) break;
    len += got;
    if (len + 1 >= cap) {
      char *grown = realloc(buf, cap * 2);
      if (!grown) break;
      buf = grown;
      cap *= 2;
    }
  }
  fclose(in);
  buf[len] = '\0';
  if (len_out) *len_out = len;
  return buf;
}

#endif

