#ifndef FK_GSL_H
#define FK_GSL_H

#include <stddef.h>
#include <stdint.h>

#include "raw.h"

#define FK_DEFAULT_LOG "/tmp/libkgsl_fake.log"
#define FK_DEFAULT_REC "/tmp/libkgsl_rec.log"

#define FK_MAX_OBJ 8192
#define FK_MAX_FD 8192
#define FK_MAX_ION 512
#define FK_MAX_REGION 128

enum fk_kind { FK_NONE = 0, FK_KGSL, FK_ION, FK_PMSG, FK_FENCE };

struct fk_prop {
  unsigned type;
  unsigned char bytes[64];
  size_t len;
};

/* log.c : env switches, logging, fd ownership table */
int fk_kind_of_path(const char *path);
void fk_fd_note(int fd, enum fk_kind kind);
enum fk_kind fk_fd_kind(int fd);
void fk_fd_drop(int fd);
int fk_strict(void);
int fk_exec_abort(void);
int fk_verbose(void);
int fk_in_bypass(void);
void fk_log(const char *fmt, ...);
void fk_fatal(const char *fmt, ...) __attribute__((noreturn));
int fk_bypass_begin(void);
void fk_bypass_end(int saved);

/* mem.c : memfd backend, faked GPUOBJ / ion */
int fk_memfd(uint64_t len);
uint32_t fk_obj_alloc(uint64_t size, uint64_t *mmapsize);
int fk_obj_free(uint32_t id);
int fk_obj_size(uint32_t id, uint64_t *size);
int fk_obj_backing(uint32_t id, uint64_t len, int *fd, uint64_t *at);
int fk_backing(uint64_t len, int *fd, uint64_t *at);
void fk_region_add(void *host, uint64_t len, int shadow);
void fk_region_drop(void *host, uint64_t len);
int fk_fence_fd(void);
int fk_ion_alloc(uint64_t len);
int fk_ion_lookup(int fd, uint64_t *len);
void fk_ion_release(int fd);

/* props.c : device-property replay */
void fk_props_init(void);
int fk_prop_fill(unsigned type, void *dst, size_t want);
int fk_prop_shadow(uint64_t *map_offset, uint32_t *ts_offset, uint64_t *size);

/* device.c / ion.c : ioctl semantics */
int fk_kgsl_ioctl(unsigned long request, void *arg);
int fk_ion_ioctl(unsigned long request, void *arg);
int fk_kgsl_request(unsigned long request);

/* shadow page: advance only the timestamp fields inside it, never paint the whole region */
void fk_shadow_map(void *host, uint64_t len);
void fk_shadow_stamp(uint32_t ts);

#endif
