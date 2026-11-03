/* a real kernel with loops and memory access, used as a compile-side reference (not a minimal probe like tests/probe.c) */
__kernel void e2x(__global float4 *x, __global float4 *out, const __global float4 *w, int gsz) {
  int idx = get_global_id(1) * gsz + get_global_id(0);
  int i = idx % 596;
  float4 s0 = (float4)(0.f);
  for (int k = 0; k < 8; k++) {
    float4 xk = x[(i * 8 + k) / 4];
    float4 wk = w[(i * 8 + k) / 4];
    s0 += xk * wk;
  }
  float4 r = fmax(s0, (float4)(0.f));
  out[i * 2] = r;
  out[i * 2 + 1] = (float4)(1.0f);
}
