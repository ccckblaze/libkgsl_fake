/* libgsl only uses ION to obtain dma-bufs; memfd stands in for them */
#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <sys/ioctl.h>

#include "fkgsl.h"

struct fk_ion_allocation {
  uint64_t len;
  uint64_t alignment;
  unsigned heap_flags;
  unsigned fd;
};
struct fk_ion_handle {
  uint64_t handle;
};
struct fk_ion_fd_data {
  unsigned fd;
};

#define ION_IOC_TYPE 'I'
#define ION_IOC_ALLOC_NR 0
#define ION_IOC_IMPORT_NR 5

int fk_ion_ioctl(unsigned long request, void *arg) {
  if (_IOC_TYPE(request) != ION_IOC_TYPE) {
    /* A dma-buf descriptor travels between libraries, and each one ioctls it its own way: cache and
       sync operations from libgsl, DMA_BUF_SYNC from whoever maps it. Answering ENOTTY -- the kernel's
       own "no such ioctl on this object" -- is what a closed runtime retries until it gives up, and the
       failure it reports is a NULL buffer with this layer named nowhere. The memory behind the fd is
       ours either way, so every foreign request is answered as a success and written down. */
    fk_log("ION foreign ioctl type=0x%02x nr=0x%02x size=%u -> success\n", _IOC_TYPE(request),
           _IOC_NR(request), _IOC_SIZE(request));
    return 0;
  }
  unsigned nr = _IOC_NR(request);

  if (nr == ION_IOC_ALLOC_NR) {
    struct fk_ion_allocation *alloc = arg;
    int exported = fk_ion_alloc(alloc->len);
    if (exported < 0) return -ENOMEM;
    alloc->fd = (unsigned)exported;
    fk_log("ION_ALLOC len=0x%llx align=0x%llx -> fd=%d\n", (unsigned long long)alloc->len,
           (unsigned long long)alloc->alignment, exported);
    return 0;
  }
  if (nr == ION_IOC_IMPORT_NR) {
    struct fk_ion_fd_data *data = arg;
    uint64_t len = 0;
    int original = (int)data->fd;
    int imported = fk_ion_alloc(fk_ion_lookup(original, &len) == 0 ? len : 0x100000);
    if (imported < 0) return -ENOMEM;
    fk_log("ION_IMPORT fd=%d -> fd=%d\n", original, imported);
    data->fd = (unsigned)imported;
    return 0;
  }
  fk_log("ION nr=0x%02x ignored\n", nr);
  return 0;
}
