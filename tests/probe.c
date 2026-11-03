/* Staged OpenCL probe.
   usage: probe [compile|full]   compile (default) stops once the ISA comes back; full goes on to submit
   env:   FK_ISA_OUT    where the ISA is written (default /tmp/fkgsl_probe_isa.bin)
          FK_SRC_FILE   an external .cl source; unset means the fixed source below
   A fixed source plus fixed build options is what makes the hash comparable with a real device - do not edit SRC casually. */
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *SRC =
    "__kernel void probe(__global float *o) {\n"
    "  int i = get_global_id(0);\n"
    "  o[i] = i * 2.0f + 1.0f;\n"
    "}\n";

#define CHECK(call)                                                     \
  do {                                                                  \
    err = (call);                                                       \
    if (err != CL_SUCCESS) {                                            \
      fprintf(stderr, "== FAIL %s -> %d\n", #call, (int)err);           \
      return 1;                                                         \
    }                                                                   \
  } while (0)

static void mark(const char *what) {
  fprintf(stderr, "== MARK %s\n", what);
  fflush(stderr);
}

int main(int argc, char **argv) {
  int full = argc > 1 && strcmp(argv[1], "full") == 0;
  const char *isa_path = getenv("FK_ISA_OUT") ? getenv("FK_ISA_OUT") : "/tmp/fkgsl_probe_isa.bin";
  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem buffer = NULL;
  char name[256] = { 0 };
  char kernel_name_buf[64] = { 0 };
  const char *kernel_name = "probe";

  CHECK(clGetPlatformIDs(1, &platform, NULL));
  CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));
  CHECK(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof name, name, NULL));
  fprintf(stderr, "== DEVICE %s\n", name);

  mark("CTX");
  cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
  context = clCreateContext(props, 1, &device, NULL, NULL, &err);
  if (!context || err != CL_SUCCESS) {
    fprintf(stderr, "== FAIL clCreateContext -> %d\n", (int)err);
    return 1;
  }
  queue = clCreateCommandQueue(context, device, 0, &err);
  if (!queue || err != CL_SUCCESS) {
    fprintf(stderr, "== FAIL clCreateCommandQueue -> %d\n", (int)err);
    return 1;
  }

  mark("BUILD");
  char *src = (char *)SRC;
  size_t srclen = strlen(src);
  const char *src_file = getenv("FK_SRC_FILE");
  if (src_file && *src_file) {
    FILE *sf = fopen(src_file, "rb");
    if (!sf) {
      fprintf(stderr, "== FAIL cannot read %s\n", src_file);
      return 1;
    }
    fseek(sf, 0, SEEK_END);
    long bytes = ftell(sf);
    fseek(sf, 0, SEEK_SET);
    src = calloc((size_t)bytes + 1, 1);
    if (fread(src, 1, (size_t)bytes, sf) != (size_t)bytes) {
      fprintf(stderr, "== FAIL short read on %s\n", src_file);
      return 1;
    }
    fclose(sf);
    srclen = (size_t)bytes;
    const char *mark = strstr(src, "__kernel void ");
    if (mark) {
      size_t skip = 14;
      const char *start = mark + skip;
      const char *stop = start;
      while (*stop && *stop != '(' && *stop != ' ' && *stop != '\n' && *stop != '\r' && *stop != ';') stop++;
      memcpy(kernel_name_buf, start, (size_t)(stop - start) < sizeof kernel_name_buf - 1
                                        ? (size_t)(stop - start)
                                        : sizeof kernel_name_buf - 1);
      kernel_name = kernel_name_buf;
    }
    fprintf(stderr, "== SRC %s (%ld bytes, kernel=%s)\n", src_file, bytes, kernel_name);
  }
  const char *sources[1] = { (const char *)src };
  program = clCreateProgramWithSource(context, 1, sources, &srclen, &err);
  if (!program || err != CL_SUCCESS) {
    fprintf(stderr, "== FAIL clCreateProgramWithSource -> %d\n", (int)err);
    return 1;
  }
  CHECK(clBuildProgram(program, 1, &device, "", NULL, NULL));

  mark("BINARY");
  size_t sizes[2] = { 0, 0 };
  cl_uint ndev = 0;
  CHECK(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof sizes, sizes, NULL));
  CHECK(clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof ndev, &ndev, NULL));
  size_t total = 0;
  for (unsigned i = 0; i < ndev && i < 2; i++) total += sizes[i];
  unsigned char *blob = malloc(total ? total : 1);
  unsigned char *ptrs[2] = { blob, ndev > 1 ? blob + sizes[0] : NULL };
  CHECK(clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof ptrs, ptrs, NULL));
  FILE *fp = fopen(isa_path, "wb");
  if (fp) {
    fwrite(blob, 1, total, fp);
    fclose(fp);
  }
  fprintf(stderr, "== ISA %zu bytes (devices=%u) -> %s\n", total, ndev, isa_path);

  mark("LOG");
  size_t loglen = 0;
  if (clGetProgramInfo(program, CL_PROGRAM_BUILD_LOG, 0, NULL, &loglen) == CL_SUCCESS && loglen > 1) {
    char *log = calloc(loglen + 1, 1);
    clGetProgramInfo(program, CL_PROGRAM_BUILD_LOG, loglen, log, NULL);
    fprintf(stderr, "== BUILDLOG %zu: %.160s\n", loglen, log);
    free(log);
  } else {
    fprintf(stderr, "== BUILDLOG none\n");
  }

  if (!full) {
    mark("DONE");
    return 0;
  }

  mark("EXEC");
  kernel = clCreateKernel(program, kernel_name, &err);
  if (!kernel || err != CL_SUCCESS) {
    fprintf(stderr, "== FAIL clCreateKernel -> %d\n", (int)err);
    return 1;
  }
  enum { COUNT = 16 };
  float out[COUNT] = { 0 };
  buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof out, out, &err);
  if (!buffer || err != CL_SUCCESS) {
    fprintf(stderr, "== FAIL clCreateBuffer -> %d\n", (int)err);
    return 1;
  }
  CHECK(clSetKernelArg(kernel, 0, sizeof buffer, &buffer));
  size_t global = COUNT;
  CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL));
  CHECK(clFinish(queue));
  memset(out, 0, sizeof out);
  CHECK(clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof out, out, 0, NULL, NULL));

  mark("VERIFY");
  int bad = 0;
  for (int i = 0; i < COUNT; i++)
    if (out[i] != i * 2.0f + 1.0f) bad++;
  fprintf(stderr, "== VERIFY %s (bad=%d, out[5]=%f)\n", bad ? "MISMATCH" : "OK", bad, (double)out[5]);

  clReleaseMemObject(buffer);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  mark("DONE");
  return 0;
}
