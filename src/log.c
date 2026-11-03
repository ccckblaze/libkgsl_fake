/* env switches, logging, fd ownership table, path classification that catches real-device leaks */
#define _GNU_SOURCE
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "fkgsl.h"

static enum fk_kind fd_kind[FK_MAX_FD];
static struct fk_sink log_sink;
static int log_sink_tried;
static int cfg_verbose = -1;
static int cfg_strict = -1;
static int cfg_exec_abort = -1;
static int cfg_pmsg = -1;
static char kgsl_path[128] = "/dev/kgsl-3d0";
static char ion_path[128] = "/dev/ion";
static char pmsg_path[128] = "/dev/pmsg0";

static int env_int(const char *name, int dflt) {
  const char *v = getenv(name);
  if (!v || !*v) return dflt;
  return strcmp(v, "0") == 0 ? 0 : 1;
}

static void cfg_init(void) {
  const char *v;
  if ((v = getenv("FK_KGSL_PATH")) && *v) snprintf(kgsl_path, sizeof kgsl_path, "%s", v);
  if ((v = getenv("FK_ION_PATH")) && *v) snprintf(ion_path, sizeof ion_path, "%s", v);
  if ((v = getenv("FK_PMSG_PATH")) && *v) snprintf(pmsg_path, sizeof pmsg_path, "%s", v);
  cfg_pmsg = env_int("FK_PMSG", 1);
  cfg_verbose = env_int("FK_VERBOSE", 0);
  cfg_strict = env_int("FK_STRICT", 1);
  v = getenv("FK_EXEC");
  cfg_exec_abort = (v && strcmp(v, "abort") == 0) ? 1 : 0;
}

/* the log is opened once, and a failure is remembered rather than retried per line */
static struct fk_sink *log_out(void) {
  if (!log_sink_tried) {
    const char *path = getenv("FK_LOG");
    log_sink_tried = 1;
    if (fk_sink_open(&log_sink, path && *path ? path : FK_DEFAULT_LOG) != 0) {
      log_sink.fp = NULL;
      log_sink.fd = -1;
    }
  }
  return (log_sink.fp || log_sink.fd >= 0) ? &log_sink : NULL;
}

int fk_verbose(void) {
  if (cfg_verbose < 0) cfg_init();
  return cfg_verbose;
}

int fk_strict(void) {
  if (cfg_strict < 0) cfg_init();
  return cfg_strict;
}

int fk_exec_abort(void) {
  if (cfg_exec_abort < 0) cfg_init();
  return cfg_exec_abort;
}

void fk_log(const char *fmt, ...) {
  if (!fk_verbose()) return;
  struct fk_sink *out = log_out();
  if (!out) return;
  va_list ap;
  va_start(ap, fmt);
  fk_sink_vprintf(out, fmt, ap);
  va_end(ap);
}

void fk_fatal(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fk_err_vprintf("libkgsl_fake: FATAL: ", fmt, ap);
  va_end(ap);
  abort();
}

/* path -> device class. Only the nodes named in the config are taken over; everything else
   (kgsl-2d0 included) passes through, so on a GPU-less box it stays as unopenable as on a real device. */
int fk_kind_of_path(const char *path) {
  if (!path) return FK_NONE;
  if (strstr(path, kgsl_path)) return FK_KGSL;
  if (strstr(path, ion_path)) return FK_ION;
  if (cfg_pmsg < 0) cfg_init();
  if (cfg_pmsg && strstr(path, pmsg_path)) return FK_PMSG;
  return FK_NONE;
}

void fk_fd_note(int fd, enum fk_kind kind) {
  if (fd >= 0 && fd < FK_MAX_FD) fd_kind[fd] = kind;
}

enum fk_kind fk_fd_kind(int fd) {
  if (fd < 0 || fd >= FK_MAX_FD) return FK_NONE;
  return fd_kind[fd];
}

void fk_fd_drop(int fd) {
  if (fd >= 0 && fd < FK_MAX_FD) fd_kind[fd] = FK_NONE;
}

/* calls this shim forwards to libc itself must bypass the shim, otherwise it recurses. */
static int bypass_depth;

int fk_bypass_begin(void) {
  int saved = bypass_depth;
  bypass_depth = 1;
  return saved;
}

void fk_bypass_end(int saved) { bypass_depth = saved; }

int fk_in_bypass(void) { return bypass_depth != 0; }
