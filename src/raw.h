/* how this layer reaches the real libc, and how it writes without touching stdio */
#ifndef FK_RAW_H
#define FK_RAW_H

#include <stdarg.h>
#include <stddef.h>

/* A line-oriented byte sink. Callers hold one of these statically; nothing is allocated. */
struct fk_sink {
  void *fp;
  int fd;
};

int fk_sink_open(struct fk_sink *s, const char *path);
int fk_sink_create(struct fk_sink *s, const char *path);
void fk_sink_write(struct fk_sink *s, const char *buf, size_t len);
int fk_sink_printf(struct fk_sink *s, const char *fmt, ...);
void fk_sink_vprintf(struct fk_sink *s, const char *fmt, va_list ap);
void fk_sink_close(struct fk_sink *s);

/* the same rules for the two channels every caller already expects */
void fk_err_printf(const char *fmt, ...);
void fk_err_vprintf(const char *prefix, const char *fmt, va_list ap);
char *fk_read_file(const char *path, size_t *len_out);

/* the real implementation of a libc entry point, or NULL if this layer has no answer for it */
void *fk_real_sym(const char *name);

#endif
