/* KGSL ioctl semantics: property replay + object allocation + contexts/sync fudged + commands acked only */
#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>

#include "fkgsl.h"
#include "msm_kgsl.h"

static uint32_t next_ctxt = 1;
static uint32_t next_sync = 1;
static uint32_t timestamp;
static int exec_warned;

int fk_kgsl_request(unsigned long request) {
  return _IOC_TYPE(request) == KGSL_IOC_TYPE;
}

static uint64_t env_u64(const char *name, uint64_t dflt) {
  const char *v = getenv(name);
  return v && *v ? strtoull(v, NULL, 0) : dflt;
}

int fk_kgsl_ioctl(unsigned long request, void *arg) {
  unsigned nr = _IOC_NR(request);
  unsigned want = _IOC_SIZE(request);
  struct kgsl_device_getproperty *prop;
  struct kgsl_gpuobj_alloc *alloc;
  struct kgsl_gpuobj_free *freereq;
  struct kgsl_gpuobj_info *info;
  struct kgsl_gpu_command *cmd;
  struct kgsl_drawctxt_create *ctxt;
  struct kgsl_syncsource_create *sync;
  unsigned *out;
  uint64_t size;

  if (!arg) return -EINVAL;

  switch (nr) {
    case 0x02: /* DEVICE_GETPROPERTY */
      prop = arg;
      if (!prop->value || !prop->sizebytes) return -EINVAL;
      if (prop->sizebytes > sizeof(struct kgsl_devinfo) + 64) return -E2BIG;
      if (!fk_prop_fill(prop->type, prop->value, prop->sizebytes)) {
        fk_log("GETPROPERTY type=0x%x not recorded\n", prop->type);
        if (fk_strict()) fk_fatal("strict mode: device properties have no type=0x%02x, re-record with libkgsl_rec", prop->type);
      }
      fk_log("GETPROPERTY type=0x%x size=%zu replayed\n", prop->type, prop->sizebytes);
      return 0;

    case 0x32: /* DEVICE_SETPROPERTY */
      fk_log("SETPROPERTY size=%u ignored\n", want);
      return 0;

    case 0x45: /* GPUOBJ_ALLOC */
      alloc = arg;
      alloc->mmapsize = 0;
      alloc->id = fk_obj_alloc(alloc->size, &alloc->mmapsize);
      if (!alloc->id) {
        fk_log("GPUOBJ_ALLOC size=0x%llx FAILED (no slot or arena out of room)\n",
               (unsigned long long)alloc->size);
        return -ENOMEM;
      }
      fk_log("GPUOBJ_ALLOC size=0x%llx flags=0x%llx -> id=%u mmapsize=0x%llx\n",
             (unsigned long long)alloc->size, (unsigned long long)alloc->flags, alloc->id,
             (unsigned long long)alloc->mmapsize);
      return 0;

    case 0x46: /* GPUOBJ_FREE */
      freereq = arg;
      fk_log("GPUOBJ_FREE id=%u\n", freereq->id);
      return fk_obj_free(freereq->id) == 0 ? 0 : -EINVAL;

    case 0x47: /* GPUOBJ_INFO */
      info = arg;
      if (fk_obj_size(info->id, &size) != 0) return -EINVAL;
      info->size = size;
      info->va_len = size;
      /* on a6xx the CPU and GPU share an address space and the runtime uses the host pointer
         returned by mmap as its gpuaddr, so no separate GPU page-table address is invented here. */
      info->gpuaddr = env_u64("FK_FAKE_GPUADDR", 0);
      fk_log("GPUOBJ_INFO id=%u -> size=0x%llx gpuaddr=0x%llx\n", info->id,
             (unsigned long long)size, (unsigned long long)info->gpuaddr);
      return 0;

    case 0x13: /* DRAWCTXT_CREATE */
      ctxt = arg;
      ctxt->drawctxt_id = next_ctxt++;
      fk_log("DRAWCTXT_CREATE flags=0x%x -> id=%u\n", ctxt->flags, ctxt->drawctxt_id);
      return 0;

    case 0x14: /* DRAWCTXT_DESTROY */
      fk_log("DRAWCTXT_DESTROY id=%u\n", ((struct kgsl_drawctxt_destroy *)arg)->drawctxt_id);
      return 0;

    case 0x40: /* SYNCSOURCE_CREATE */
      sync = arg;
      sync->id = next_sync++;
      fk_log("SYNCSOURCE_CREATE -> id=%u\n", sync->id);
      return 0;

    case 0x41: /* SYNCSOURCE_DESTROY */
    case 0x42: { /* SYNCSOURCE_CREATE_FENCE -- the caller keeps the returned descriptor */
      struct kgsl_syncsource_create_fence *fence = arg;
      int fd = fk_fence_fd();
      if (fd < 0) return -ENOMEM;
      fence->fence_fd = fd;
      fk_log("SYNCSOURCE_CREATE_FENCE id=%u -> fence_fd=%d\n", fence->id, fd);
      return 0;
    }

    case 0x43: /* SYNCSOURCE_SIGNAL_FENCE -- the fence was handed out already signaled */
      fk_log("SYNCSOURCE op=0x%02x\n", nr);
      return 0;

    case 0x49: /* GPUOBJ_SYNC */
      fk_log("GPUOBJ_SYNC count=%u\n", ((struct kgsl_gpuobj_sync *)arg)->count);
      return 0;

    case 0x06: /* DEVICE_WAITTIMESTAMP */
    case 0x07: /* WAITTIMESTAMP_CTXTID */
      fk_log("WAITTIMESTAMP -> reported complete\n");
      return 0;

    case 0x11: /* RINGBUFFER_CMDSTREAM_READTIMESTAMP */
    case 0x16: /* READTIMESTAMP_CTXTID */
      out = arg;
      *out = timestamp;
      fk_log("READTIMESTAMP -> %u\n", timestamp);
      return 0;

    case 0x33: /* TIMESTAMP_EVENT */
      fk_log("TIMESTAMP_EVENT\n");
      return 0;

    case 0x4a: /* GPU_COMMAND -- the one place that genuinely needs the GPU */
      cmd = arg;
      if (fk_exec_abort())
        fk_fatal("GPU_COMMAND received (%u cmds, context=%u) while FK_EXEC=abort forbids execution."
                 " If this came from the compile path, the closed runtime needs the GPU to build -- send the log.",
                 cmd->numcmds, cmd->context_id);
      if (!exec_warned++) {
        fk_log("WARN GPU_COMMAND acked without execution: any reading that expects a GPU write-back is untrustworthy\n");
        fk_err_printf("libkgsl_fake: GPU_COMMAND acked without execution "
                      "(compiling works, computing does not; see README tier 3)");
      }
      timestamp = cmd->timestamp > timestamp ? cmd->timestamp : timestamp + 1;
      fk_shadow_stamp(timestamp);
      fk_log("GPU_COMMAND cmds=%u objs=%u ctx=%u ts=%u -> ack %u\n", cmd->numcmds, cmd->numobjs,
             cmd->context_id, cmd->timestamp, timestamp);
      return 0;

    default:
      fk_log("UNHANDLED nr=0x%02x size=%u dir=%u\n", nr, want, _IOC_DIR(request));
      if (fk_strict()) fk_fatal("strict mode: unrecorded KGSL ioctl nr=0x%02x (size=%u)", nr, want);
      return 0;
  }
}
