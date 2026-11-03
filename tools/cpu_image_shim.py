"""Give the CPU (Clang) backend OpenCL image2d semantics, so image ASTs can be recomputed verbatim on a
machine without a GPU.

The layout matches the real device (runtime/ops_gpu.py: clCreateImage2D(width=shape[1], height=shape[0],
CL_RGBA half/float)), and the index math follows tinygrad's own image_fixup (codegen/devectorizer.py):
  element i -> pixel(x,y) = ((i//4) % W, (i//4) // W), the low 2 bits of x select one of 4 channels
so read_imagef(b, smp, coord) is the group of 4 elements starting at flat index (y*W+x)*4, and a buffer
holds H*W*4 elements. imageh stores __fp16, imagef stores float, both as float4 in registers.

Three semantic/dialect differences to emulate:
  1) CLK_ADDRESS_CLAMP: out-of-range reads return 0, out-of-range writes are dropped. Convolution windows
     produce negative coordinates at the border and the real hardware absorbs them via CLAMP; without this
     check it is an out-of-bounds access (measured: straight to SIGSEGV).
  2) clang's vector_size types have no .x/.y members, so components are taken through pointers.
  3) ClangRenderer renders coordinates as (int2){a,b}; the comma inside the braces is a macro argument
     separator to the preprocessor, so the pattern wraps them in extra parentheses.
"""
from tinygrad.dtype import dtypes, ImageDType
from tinygrad.uop.ops import Ops, PatternMatcher, UPat

_SHIM = """typedef int sampler_t;
#define CLK_NORMALIZED_COORDS_FALSE 0
#define CLK_ADDRESS_CLAMP 0
#define CLK_FILTER_NEAREST 0
#define CLK_LOCAL_MEM_FENCE 0
#define CLK_GLOBAL_MEM_FENCE 0
static inline float4 cpu_img_read_h(const __fp16 *b, int W, int H, int2 p){ const int *q=(const int*)&p;
  if ((unsigned)q[0] >= (unsigned)W || (unsigned)q[1] >= (unsigned)H) { const float4 z = (float4){0,0,0,0}; return z; }
  const int o=(q[1]*W+q[0])*4;
  return (float4){(float)b[o],(float)b[o+1],(float)b[o+2],(float)b[o+3]}; }
static inline float4 cpu_img_read_f(const float *b, int W, int H, int2 p){ const int *q=(const int*)&p;
  if ((unsigned)q[0] >= (unsigned)W || (unsigned)q[1] >= (unsigned)H) { const float4 z = (float4){0,0,0,0}; return z; }
  const int o=(q[1]*W+q[0])*4;
  return (float4){b[o],b[o+1],b[o+2],b[o+3]}; }
static inline void cpu_img_write_h(__fp16 *b, int W, int H, int2 p, float4 v){ const int *q=(const int*)&p; const float *s=(const float*)&v;
  if ((unsigned)q[0] >= (unsigned)W || (unsigned)q[1] >= (unsigned)H) return;
  const int o=(q[1]*W+q[0])*4;
  b[o]=(__fp16)s[0]; b[o+1]=(__fp16)s[1]; b[o+2]=(__fp16)s[2]; b[o+3]=(__fp16)s[3]; }
static inline void cpu_img_write_f(float *b, int W, int H, int2 p, float4 v){ const int *q=(const int*)&p; const float *s=(const float*)&v;
  if ((unsigned)q[0] >= (unsigned)W || (unsigned)q[1] >= (unsigned)H) return;
  const int o=(q[1]*W+q[0])*4;
  b[o]=s[0]; b[o+1]=s[1]; b[o+2]=s[2]; b[o+3]=s[3]; }
#define read_imagef(b,s,p)  _Generic((b), __fp16*: cpu_img_read_h, default: cpu_img_read_f)(b, NI_##b, NH_##b, p)
#define write_imagef(b,p,v) _Generic((b), __fp16*: cpu_img_write_h, default: cpu_img_write_f)(b, NI_##b, NH_##b, p, v)"""


def make_renderer(cpu_renderer):
  f4 = dtypes.float.vec(4)
  i2 = dtypes.int.vec(2)
  image_pm = PatternMatcher([
    (UPat(Ops.LOAD, dtype=f4, src=(UPat.var("buf").index(UPat.var("idx", i2), UPat.var("gate")), UPat.var("var"))),
     lambda ctx, buf, idx, var, gate: f"({ctx[gate]}?read_imagef({ctx[buf]}, smp, ({ctx[idx]})):{ctx[var]})"),
    (UPat(Ops.LOAD, dtype=f4, src=(UPat.var("buf").index(UPat.var("idx", i2)),)),
     lambda ctx, buf, idx: f"read_imagef({ctx[buf]}, smp, ({ctx[idx]}))"),
    (UPat(Ops.STORE, src=(UPat.var("buf").index(UPat.var("idx", i2)), UPat.var("var", f4)), allow_any_len=True),
     lambda ctx, buf, idx, var: f"write_imagef({ctx[buf]}, ({ctx[idx]}), ({ctx[var]}));"),
  ])

  class CpuImageRenderer(type(cpu_renderer)):
    string_rewrite = image_pm + type(cpu_renderer).string_rewrite

    def render_dtype(self, dt, mutable=True):
      if isinstance(dt, ImageDType): return ("__fp16" if dt.itemsize == 2 else "float") + "*"
      return super().render_dtype(dt, mutable)

    def render_kernel(self, function_name, kernel, bufs, uops, prefix=None):
      images = [(name, dtype) for name, (dtype, _) in bufs if isinstance(dtype, ImageDType)]
      if images:
        # each image buffer carries its own width/height; height is derived from the element count, matching image_fixup
        need = [f"#define NI_{name} {dtype.shape[1]}"
                f"\n#define NH_{name} {dtype.size // (4*dtype.shape[1])}" for name, dtype in images]
        # ClangRenderer only typedefs vector types seen in the uops; add the missing ones (duplicate typedefs error out)
        have = "\n".join(self._render_defines(uops))
        for dt, name in ((dtypes.float.vec(4), "float4"), (dtypes.int.vec(2), "int2")):
          if f"typedef {self.render_dtype(dt.scalar())} {name} " not in have:
            need.append(self.render_vector_prefix(dt))
        prefix = [_SHIM] + need + (prefix or [])
      return super().render_kernel(function_name, kernel, bufs, uops, prefix)

  return CpuImageRenderer()
