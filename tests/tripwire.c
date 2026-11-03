/* Leak-guard field test: obtain the real GPU fd with svc instructions only, so libc symbol
   interception never sees it, then issue a KGSL ioctl through libc ioctl: strict mode must abort
   on the spot - a "pass" bought by silently using real hardware is worthless. Needs a real device:
   without a GPU the node cannot be opened and the program skips itself. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "msm_kgsl.h"

static long raw_syscall(long number, long a0, long a1, long a2) {
#if defined(__aarch64__)
  register long x8 __asm__("x8") = number;
  register long x0 __asm__("x0") = a0;
  register long x1 __asm__("x1") = a1;
  register long x2 __asm__("x2") = a2;
  __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8), "r"(x1), "r"(x2) : "memory", "cc");
  return x0;
#else
  return syscall(number, a0, a1, a2);
#endif
}

int main(void) {
  struct kgsl_device_getproperty prop;
  unsigned char value[64];
  long fd = raw_syscall(SYS_openat, AT_FDCWD, (long)"/dev/kgsl-3d0", O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    printf("TRIPWIRE skip: no usable /dev/kgsl-3d0 on this machine\n");
    return 0;
  }
  printf("TRIPWIRE real device fd=%ld, issuing a KGSL ioctl directly\n", fd);
  memset(value, 0, sizeof value);
  prop.type = KGSL_PROP_DEVICE_BITNESS;
  prop.value = value;
  prop.sizebytes = sizeof value;
  ioctl((int)fd, IOCTL_KGSL_DEVICE_GETPROPERTY, &prop);
  printf("TRIPWIRE BAD: nothing stopped it, bitness=0x%02x -- the fake device was fooled\n", value[0]);
  return 3;
}
