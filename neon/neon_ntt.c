#include <arm_neon.h>
#include "params.h"
#include "ntt.h"
#include "reduce.h"

#define _V (((1U << 26) + KYBER_Q / 2) / KYBER_Q)

/******8888888888888888888888888***************/

// Load int16x8_t c <= ptr*
#define vload(c, ptr) c = vld1q_s16(ptr);

// Store *ptr <= c
#define vstore(ptr, c) vst1q_s16(ptr, c);

// Load int16x8_t c <= ptr*
#define vloadx2(c, ptr) c = vld1q_s16_x2(ptr);

// Store *ptr <= c
#define vstorex2(ptr, c) vst1q_s16_x2(ptr, c);

// Load int16x8_t c <= ptr*
#define vloadx4(c, ptr) c = vld1q_s16_x4(ptr);

// Store *ptr <= c
#define vstorex4(ptr, c) vst1q_s16_x4(ptr, c);

// Load int16x8_t c <= ptr*
#define vload4(c, ptr) c = vld4q_s16(ptr);

// Store *ptr <= c
#define vstore4(ptr, c) vst4q_s16(ptr, c);

// Combine in16x8_t c: low | high
#define vcombine(c, low, high) c = vcombine_s16(low, high);

// c (int16x8) = a + b (int16x8)
#define vadd8(c, a, b) c = vaddq_s16(a, b);

// c (int16x8) = a - b (int16x8)
#define vsub8(c, a, b) c = vsubq_s16(a, b);

// get_low c (int16x4) = low(a) (int16x8)
#define vlo(c, a) c = vget_low_s16(a);

// get_high c (int16x4) = high(a) (int16x8)
#define vhi(c, a) c = vget_high_s16(a);

// c = const
#define vdup(c, const) c = vdup_n_s16(const);

/*************************************************
* Name:        fqmul
*
* Description: Multiplication followed by Montgomery reduction
*
* Arguments:   - int16_t a: first factor
*              - int16_t b: second factor
*
* Returns 16-bit integer congruent to a*b*R^{-1} mod q

// inout: input/output : int16x8_t
// zeta: input : int16x8_t
// a1, a2: temp : int32x4_t
// u1, u2: temp : int32x4_t
// neon_qinv: const   : int32x4_t
// neon_kyberq: const : int32x4_t

rewrite pseudo code:
int16_t fqmul(int16_t b, int16_t c) {
  int32_t t, u, a;

  a = (int32_t) b*c;
  u = a * (QINV << 16);
  u >>= 16;
  t = u * (-KYBER_Q);
  t += a;
  t >>= 16;
  return t;
}
**************************************************/
#define fqmul(out, in, zeta, a1, a2, u1, u2)            \
  a1 = vmull_s16(vget_low_s16(in), vget_low_s16(zeta)); \
  a2 = vmull_high_s16(in, zeta);                        \
  u1 = vmulq_s32(a1, neon_qinv);                        \
  u2 = vmulq_s32(a2, neon_qinv);                        \
  u1 = vshrq_n_s32(u1, 16);                             \
  u2 = vshrq_n_s32(u2, 16);                             \
  u1 = vmulq_s32(u1, neon_kyberq);                      \
  u2 = vmulq_s32(u2, neon_kyberq);                      \
  out = vaddhn_high_s32(vaddhn_s32(u1, a1), u2, a2);

/*
inout: int16x4_t
t32 : int32x4_t
t16: int16x4_t
neon_v: int16x4_t
neon_kyberq16: inout int16x4_t

int16_t barrett_reduce(int16_t a) {
  int16_t t;
  const int16_t v = ((1U << 26) + KYBER_Q / 2) / KYBER_Q;

  t32 = (int32_t)v * a;
  t16 = t32 >> 26;
  t = a + t16 * (-KYBER_Q);
  return t;
}
*/

/*
reduce low and high of 
inout: 
int16x8_t inout, 
t32_1, t32_2: int32x4_t 
t16: int16x8_t 
neon_v, neon_kyber16
*/
#define barret_lh(inout, t32_1, t32_2, t16)                     \
  t32_1 = vmull_s16(vget_low_s16(inout), vget_low_s16(neon_v)); \
  t32_2 = vmull_high_s16(inout, neon_v);                        \
  t32_1 = vshrq_n_s32(t32_1, 26);                               \
  t32_2 = vshrq_n_s32(t32_2, 26);                               \
  t16 = vmovn_high_s32(vmovn_s32(t32_1), t32_2);                \
  inout = vmlaq_s16(inout, t16, neon_kyberq16);

/*
v1, v2: int16x8_t 
out1, out2: int16x8_t
t32_1, t32_2: int32x4_t 
t16: int16x8_t
*/
#define barret_hi(v1, v2, t32_1, t32_2, t16_1, t16_2)                      \
  t32_1 = vmull_high_s16(v1, neon_v);                                      \
  t32_2 = vmull_high_s16(v2, neon_v);                                      \
  t32_1 = vshrq_n_s32(t32_1, 26);                                          \
  t32_2 = vshrq_n_s32(t32_2, 26);                                          \
  t16_1 = vmovn_high_s32(vmovn_s32(t32_1), t32_2);                         \
  t16_2 = (int16x8_t)vzip2q_s64((int64x2_t)v1, (int64x2_t)v2);             \
  t16_2 = vmlaq_s16(t16_2, t16_1, neon_kyberq16);                          \
  v1 = (int16x8_t)vcopyq_laneq_s64((int64x2_t)v1, 1, (int64x2_t)t16_2, 0); \
  v2 = (int16x8_t)vcopyq_laneq_s64((int64x2_t)v2, 1, (int64x2_t)t16_2, 1);

#define barret_lo(v1, v2, t32_1, t32_2, t16_1, t16_2)                      \
  t32_1 = vmull_s16(vget_low_s16(v1), vget_low_s16(neon_v));               \
  t32_2 = vmull_s16(vget_low_s16(v2), vget_low_s16(neon_v));               \
  t32_1 = vshrq_n_s32(t32_1, 26);                                          \
  t32_2 = vshrq_n_s32(t32_2, 26);                                          \
  t16_1 = vmovn_high_s32(vmovn_s32(t32_1), t32_2);                         \
  t16_2 = (int16x8_t)vzip1q_s64((int64x2_t)v1, (int64x2_t)v2);             \
  t16_2 = vmlaq_s16(t16_2, t16_1, neon_kyberq16);                          \
  v1 = (int16x8_t)vcopyq_laneq_s64((int64x2_t)v1, 0, (int64x2_t)t16_2, 0); \
  v2 = (int16x8_t)vcopyq_laneq_s64((int64x2_t)v2, 0, (int64x2_t)t16_2, 1);

/*
Matrix 4x4 transpose: v
Input: int16x8x4_t v, tmp
Output: int16x8x4_t v
*/
#define transpose(v, tmp)                                                         \
  tmp.val[0] = vtrn1q_s16(v.val[0], v.val[1]);                                    \
  tmp.val[1] = vtrn2q_s16(v.val[0], v.val[1]);                                    \
  tmp.val[2] = vtrn1q_s16(v.val[2], v.val[3]);                                    \
  tmp.val[3] = vtrn2q_s16(v.val[2], v.val[3]);                                    \
  v.val[0] = (int16x8_t)vtrn1q_s32((int32x4_t)tmp.val[0], (int32x4_t)tmp.val[2]); \
  v.val[2] = (int16x8_t)vtrn2q_s32((int32x4_t)tmp.val[0], (int32x4_t)tmp.val[2]); \
  v.val[1] = (int16x8_t)vtrn1q_s32((int32x4_t)tmp.val[1], (int32x4_t)tmp.val[3]); \
  v.val[3] = (int16x8_t)vtrn2q_s32((int32x4_t)tmp.val[1], (int32x4_t)tmp.val[3]);

/*
Re-arrange vector
*/
#define arrange(v_out, v_in, i, j, m, n)                                                \
  v_out.val[0] = (int16x8_t)vtrn1q_s64((int64x2_t)v_in.val[i], (int64x2_t)v_in.val[j]); \
  v_out.val[1] = (int16x8_t)vtrn2q_s64((int64x2_t)v_in.val[i], (int64x2_t)v_in.val[j]); \
  v_out.val[2] = (int16x8_t)vtrn1q_s64((int64x2_t)v_in.val[m], (int64x2_t)v_in.val[n]); \
  v_out.val[3] = (int16x8_t)vtrn2q_s64((int64x2_t)v_in.val[m], (int64x2_t)v_in.val[n]);

/*
Butterfly Unit
Input: 
v: int16x8x4_t
i, j, m, n: index 
tmp1, tmp2: int16x8_t
Output: 
v: int16x8x4_t
*/
#define addsub(v, i, j, m, n, t1, t2) \
  t1 = v.val[i];                      \
  t2 = v.val[m];                      \
  vadd8(v.val[i], t1, v.val[j]);      \
  vsub8(v.val[j], t1, v.val[j]);      \
  vadd8(v.val[m], t2, v.val[n]);      \
  vsub8(v.val[n], t2, v.val[n]);

#define addsub_x4(v0, v1, va)             \
  va.val[0] = v0.val[0];                  \
  va.val[1] = v0.val[1];                  \
  va.val[2] = v0.val[2];                  \
  va.val[3] = v0.val[3];                  \
  vadd8(v0.val[0], va.val[0], v1.val[0]); \
  vadd8(v0.val[1], va.val[1], v1.val[1]); \
  vadd8(v0.val[2], va.val[2], v1.val[2]); \
  vadd8(v0.val[3], va.val[3], v1.val[3]); \
  vsub8(v1.val[0], va.val[0], v1.val[0]); \
  vsub8(v1.val[1], va.val[1], v1.val[1]); \
  vsub8(v1.val[2], va.val[2], v1.val[2]); \
  vsub8(v1.val[3], va.val[3], v1.val[3]);

#define addsub_twist(v, v_in, i, j, m, n, t1, t2) \
  t1 = v_in.val[i];                               \
  t2 = v_in.val[m];                               \
  vadd8(v.val[i], t1, v_in.val[j]);               \
  vsub8(v.val[m], t1, v_in.val[j]);               \
  vadd8(v.val[j], t2, v_in.val[n]);               \
  vsub8(v.val[n], t2, v_in.val[n]);

#define subadd_x4(v2, v0, va)             \
  vsub8(v2.val[0], v0.val[0], va.val[0]); \
  vsub8(v2.val[1], v0.val[1], va.val[1]); \
  vsub8(v2.val[2], v0.val[2], va.val[2]); \
  vsub8(v2.val[3], v0.val[3], va.val[3]); \
  vadd8(v0.val[0], v0.val[0], va.val[0]); \
  vadd8(v0.val[1], v0.val[1], va.val[1]); \
  vadd8(v0.val[2], v0.val[2], va.val[2]); \
  vadd8(v0.val[3], v0.val[3], va.val[3]);

#define subadd(v, i, j, m, n, t1, t2) \
  vsub8(v.val[j], v.val[i], t1);      \
  vadd8(v.val[i], v.val[i], t1);      \
  vsub8(v.val[n], v.val[m], t2);      \
  vadd8(v.val[m], v.val[m], t2);

#define subadd_twist(v, v_in, i, j, m, n)    \
  vsub8(v.val[j], v_in.val[i], v_in.val[j]); \
  vadd8(v.val[i], v_in.val[i], v_in.val[j]); \
  vsub8(v.val[n], v_in.val[m], v_in.val[n]); \
  vadd8(v.val[m], v_in.val[m], v_in.val[n]);

/*************************************************
* Name:        invntt_tomont
*
* Description: Inplace inverse number-theoretic transform in Rq and
*              multiplication by Montgomery factor 2^16.
*              Input is in bitreversed order, output is in standard order
*
* Arguments:   - int16_t r[256]: pointer to input/output vector of elements
*                                of Zq
**************************************************/
void neon_invntt(int16_t r[256])
{
  int j, k1, k2, k3, k4, k5, k6;
  // Register: Total 26
  int16x8x4_t vt, v0, v1, v2, v3, z; // 28
  int16x4_t a_lo, a_hi, b_lo, b_hi,
      c_lo, c_hi, d_lo, d_hi;
  int32x4_t t1, t2, t3, t4; // 4
  // Scalar
  k1 = 0;
  k2 = 64;
  k3 = 96;
  k4 = 112;
  k5 = 120;
  k6 = 124;
  // End
  int32x4_t neon_qinv, neon_kyberq;
  int16x8_t neon_v, neon_kyberq16;
  neon_qinv = vdupq_n_s32(QINV << 16);
  neon_kyberq = vdupq_n_s32(-KYBER_Q);
  neon_v = vdupq_n_s16(_V);
  neon_kyberq16 = vdupq_n_s16(-KYBER_Q);

  // *Vectorize* barret_reduction over 96 points rather than 896 points
  // Optimimal Barret reduction for Kyber N=256, B=9 is 78 points, see here:
  // https://eprint.iacr.org/2020/1377.pdf

  // Layer 1, 2, 3, 4, 5, 6
  // Total register: 26
  for (j = 0; j < 256; j += 128)
  {
    // 1st layer : v0.val[0] x v0.val[2] | v0.val[1] x v0.val[3]
    // v0.val[0]: 0, 4, 8,  12 | 16, 20, 24, 28
    // v0.val[1]: 1, 5, 9,  13 | 17, 21, 25, 29
    // v0.val[2]: 2, 6, 10, 14 | 18, 22, 26, 30
    // v0.val[3]: 3, 7, 11, 15 | 19, 23, 27, 31
    vload4(v0, &r[j]);
    vload4(v1, &r[j + 32]);
    vload4(v2, &r[j + 64]);
    vload4(v3, &r[j + 96]);

    addsub(v0, 0, 2, 1, 3, vt.val[0], vt.val[1]);
    addsub(v1, 0, 2, 1, 3, vt.val[2], vt.val[3]);
    addsub(v2, 0, 2, 1, 3, vt.val[0], vt.val[1]);
    addsub(v3, 0, 2, 1, 3, vt.val[2], vt.val[3]);

    vloadx4(z, &zetas_inv[k1]);
    k1 += 32;

    fqmul(v0.val[2], v0.val[2], z.val[0], t1, t2, t3, t4);
    fqmul(v0.val[3], v0.val[3], z.val[0], t1, t2, t3, t4);
    fqmul(v1.val[2], v1.val[2], z.val[1], t1, t2, t3, t4);
    fqmul(v1.val[3], v1.val[3], z.val[1], t1, t2, t3, t4);

    fqmul(v2.val[2], v2.val[2], z.val[2], t1, t2, t3, t4);
    fqmul(v2.val[3], v2.val[3], z.val[2], t1, t2, t3, t4);
    fqmul(v3.val[2], v3.val[2], z.val[3], t1, t2, t3, t4);
    fqmul(v3.val[3], v3.val[3], z.val[3], t1, t2, t3, t4);

    // Layer 2: v0.val[0] x v0.val[1] | v0.val[2] x v0.val[3]
    // transpose 4x4
    transpose(v0, vt);
    transpose(v1, vt);
    transpose(v2, vt);
    transpose(v3, vt);
    // v0.val[0]: 0,  1,  2,  3  | 16,  17,  18,  19
    // v0.val[1]: 4,  5,  6,  7  | 20,  21,  22,  23
    // v0.val[2]: 8,  9,  10, 11 | 24,  25,  26,  27
    // v0.val[3]: 12, 13, 14, 15 | 28,  29,  30,  31

    addsub(v0, 0, 1, 2, 3, vt.val[0], vt.val[1]);
    addsub(v1, 0, 1, 2, 3, vt.val[2], vt.val[3]);
    addsub(v2, 0, 1, 2, 3, vt.val[0], vt.val[1]);
    addsub(v3, 0, 1, 2, 3, vt.val[2], vt.val[3]);

    // TODO: Replace with new zetas_inv table for direct load
    vdup(a_lo, zetas_inv[k2]);
    vdup(b_lo, zetas_inv[k2 + 1]);
    vdup(a_hi, zetas_inv[k2 + 2]);
    vdup(b_hi, zetas_inv[k2 + 3]);
    vdup(c_lo, zetas_inv[k2 + 4]);
    vdup(d_lo, zetas_inv[k2 + 5]);
    vdup(c_hi, zetas_inv[k2 + 6]);
    vdup(d_hi, zetas_inv[k2 + 7]);

    vcombine(z.val[0], a_lo, a_hi);
    vcombine(z.val[1], b_lo, b_hi);
    vcombine(z.val[2], c_lo, c_hi);
    vcombine(z.val[3], d_lo, d_hi);

    fqmul(v0.val[1], v0.val[1], z.val[0], t1, t2, t3, t4);
    fqmul(v0.val[3], v0.val[3], z.val[1], t1, t2, t3, t4);
    fqmul(v1.val[1], v1.val[1], z.val[2], t1, t2, t3, t4);
    fqmul(v1.val[3], v1.val[3], z.val[3], t1, t2, t3, t4);

    vdup(a_lo, zetas_inv[k2 + 8]);
    vdup(b_lo, zetas_inv[k2 + 9]);
    vdup(a_hi, zetas_inv[k2 + 10]);
    vdup(b_hi, zetas_inv[k2 + 11]);
    vdup(c_lo, zetas_inv[k2 + 12]);
    vdup(d_lo, zetas_inv[k2 + 13]);
    vdup(c_hi, zetas_inv[k2 + 14]);
    vdup(d_hi, zetas_inv[k2 + 15]);

    vcombine(z.val[0], a_lo, a_hi);
    vcombine(z.val[1], b_lo, b_hi);
    vcombine(z.val[2], c_lo, c_hi);
    vcombine(z.val[3], d_lo, d_hi);
    k2 += 16;

    fqmul(v2.val[1], v2.val[1], z.val[0], t1, t2, t3, t4);
    fqmul(v2.val[3], v2.val[3], z.val[1], t1, t2, t3, t4);
    fqmul(v3.val[1], v3.val[1], z.val[2], t1, t2, t3, t4);
    fqmul(v3.val[3], v3.val[3], z.val[3], t1, t2, t3, t4);

    // Layer 3 : v0.val[0] x v0.val[2] | v0.val[1] x v0.val[3]
    // v0.val[0]: 0,  1,  2,  3  | 16,  17,  18,  19
    // v0.val[1]: 4,  5,  6,  7  | 20,  21,  22,  23
    // v0.val[2]: 8,  9,  10, 11 | 24,  25,  26,  27
    // v0.val[3]: 12, 13, 14, 15 | 28,  29,  30,  31

    addsub(v0, 0, 2, 1, 3, vt.val[0], vt.val[1]);
    addsub(v1, 0, 2, 1, 3, vt.val[2], vt.val[3]);
    addsub(v2, 0, 2, 1, 3, vt.val[0], vt.val[1]);
    addsub(v3, 0, 2, 1, 3, vt.val[2], vt.val[3]);

    // TODO: Replace with new zetas_inv table for direct load
    vdup(a_lo, zetas_inv[k3]);
    vdup(a_hi, zetas_inv[k3 + 1]);
    vdup(b_lo, zetas_inv[k3 + 2]);
    vdup(b_hi, zetas_inv[k3 + 3]);
    vdup(c_lo, zetas_inv[k3 + 4]);
    vdup(c_hi, zetas_inv[k3 + 5]);
    vdup(d_lo, zetas_inv[k3 + 6]);
    vdup(d_hi, zetas_inv[k3 + 7]);

    vcombine(z.val[0], a_lo, a_hi);
    vcombine(z.val[1], b_lo, b_hi);
    vcombine(z.val[2], c_lo, c_hi);
    vcombine(z.val[3], d_lo, d_hi);
    k3 += 8;

    fqmul(v0.val[2], v0.val[2], z.val[0], t1, t2, t3, t4);
    fqmul(v0.val[3], v0.val[3], z.val[0], t1, t2, t3, t4);
    fqmul(v1.val[2], v1.val[2], z.val[1], t1, t2, t3, t4);
    fqmul(v1.val[3], v1.val[3], z.val[1], t1, t2, t3, t4);

    fqmul(v2.val[2], v2.val[2], z.val[2], t1, t2, t3, t4);
    fqmul(v2.val[3], v2.val[3], z.val[2], t1, t2, t3, t4);
    fqmul(v3.val[2], v3.val[2], z.val[3], t1, t2, t3, t4);
    fqmul(v3.val[3], v3.val[3], z.val[3], t1, t2, t3, t4);

    // 16, 17, 18, 19
    barret_hi(v0.val[0], v1.val[0], t1, t2, vt.val[0], vt.val[1]);
    barret_hi(v2.val[0], v3.val[0], t3, t4, vt.val[2], vt.val[3]);

    // Layer 4: v0.val[0] x v0.val[1] | v0.val[2] x v0.val[3]
    // Re-arrange vector

    // v2.val[0]: 0,  1,  2,  3  | 4,  5,  6,  7
    // v2.val[1]: 16, 17, 18, 19 | 20, 21, 22, 23
    // v2.val[2]: 8,  9,  10, 11 | 12, 13, 14, 15
    // v2.val[3]: 24, 25, 26, 27 | 28, 29, 30, 31

    arrange(vt, v0, 0, 1, 2, 3);
    addsub_twist(v0, vt, 0, 1, 2, 3, z.val[0], z.val[1]);
    arrange(vt, v1, 0, 1, 2, 3);
    addsub_twist(v1, vt, 0, 1, 2, 3, z.val[2], z.val[3]);

    arrange(vt, v2, 0, 1, 2, 3);
    addsub_twist(v2, vt, 0, 1, 2, 3, z.val[0], z.val[1]);
    arrange(vt, v3, 0, 1, 2, 3);
    addsub_twist(v3, vt, 0, 1, 2, 3, z.val[2], z.val[3]);

    z.val[0] = vdupq_n_s16(zetas_inv[k4]);
    z.val[1] = vdupq_n_s16(zetas_inv[k4 + 1]);
    z.val[2] = vdupq_n_s16(zetas_inv[k4 + 2]);
    z.val[3] = vdupq_n_s16(zetas_inv[k4 + 3]);
    k4 += 4;

    fqmul(v0.val[2], v0.val[2], z.val[0], t1, t2, t3, t4);
    fqmul(v0.val[3], v0.val[3], z.val[0], t1, t2, t3, t4);
    fqmul(v1.val[2], v1.val[2], z.val[1], t1, t2, t3, t4);
    fqmul(v1.val[3], v1.val[3], z.val[1], t1, t2, t3, t4);

    fqmul(v2.val[2], v2.val[2], z.val[2], t1, t2, t3, t4);
    fqmul(v2.val[3], v2.val[3], z.val[2], t1, t2, t3, t4);
    fqmul(v3.val[2], v3.val[2], z.val[3], t1, t2, t3, t4);
    fqmul(v3.val[3], v3.val[3], z.val[3], t1, t2, t3, t4);

    // 0, 1, 2, 3
    barret_lo(v0.val[0], v1.val[0], t1, t2, vt.val[0], vt.val[1]);
    barret_lo(v2.val[0], v3.val[0], t3, t4, vt.val[2], vt.val[3]);

    // Layer 5: v0 x v1 | v2 x v3
    // v0: 0  -> 31
    // v1: 32 -> 63
    // v2: 64 -> 95
    // v3: 96 -> 127

    addsub_x4(v0, v1, vt);
    addsub_x4(v2, v3, vt);

    barret_hi(v0.val[0], v2.val[0], t1, t2, vt.val[0], vt.val[1]);

    z.val[0] = vdupq_n_s16(zetas_inv[k5]);
    z.val[1] = vdupq_n_s16(zetas_inv[k5 + 1]);
    k5 += 2;

    fqmul(v1.val[0], v1.val[0], z.val[0], t1, t2, t3, t4);
    fqmul(v1.val[1], v1.val[1], z.val[0], t1, t2, t3, t4);
    fqmul(v1.val[2], v1.val[2], z.val[0], t1, t2, t3, t4);
    fqmul(v1.val[3], v1.val[3], z.val[0], t1, t2, t3, t4);

    fqmul(v3.val[0], v3.val[0], z.val[1], t1, t2, t3, t4);
    fqmul(v3.val[1], v3.val[1], z.val[1], t1, t2, t3, t4);
    fqmul(v3.val[2], v3.val[2], z.val[1], t1, t2, t3, t4);
    fqmul(v3.val[3], v3.val[3], z.val[1], t1, t2, t3, t4);

    // Layer 6: v0 x v2 | v1 x v3
    // v0: 0  -> 31
    // v2: 64 -> 95
    // v1: 32 -> 63
    // v3: 96 -> 127

    addsub_x4(v0, v2, vt);
    addsub_x4(v1, v3, vt);

    barret_lh(v0.val[1], t1, t2, vt.val[0]);
    barret_lh(v1.val[1], t3, t4, vt.val[1]);

    z.val[0] = vdupq_n_s16(zetas_inv[k6++]);

    fqmul(v2.val[0], v2.val[0], z.val[0], t1, t2, t3, t4);
    fqmul(v2.val[1], v2.val[1], z.val[0], t1, t2, t3, t4);
    fqmul(v2.val[2], v2.val[2], z.val[0], t1, t2, t3, t4);
    fqmul(v2.val[3], v2.val[3], z.val[0], t1, t2, t3, t4);

    fqmul(v3.val[0], v3.val[0], z.val[0], t1, t2, t3, t4);
    fqmul(v3.val[1], v3.val[1], z.val[0], t1, t2, t3, t4);
    fqmul(v3.val[2], v3.val[2], z.val[0], t1, t2, t3, t4);
    fqmul(v3.val[3], v3.val[3], z.val[0], t1, t2, t3, t4);

    vstorex4(&r[j], v0);
    vstorex4(&r[j + 32], v1);
    vstorex4(&r[j + 64], v2);
    vstorex4(&r[j + 96], v3);
  }

  // Layer 7, inv_mul
  z.val[0] = vdupq_n_s16(zetas_inv[126]);
  z.val[1] = vdupq_n_s16(zetas_inv[127]);
  for (j = 0; j < 128; j += 64)
  {
    // Layer 7: v0 x v2 | v1 x v3
    // v0: 0   -> 31
    // v1: 32  -> 64
    // v2: 128 -> 159
    // v3: 160 -> 191
    vloadx4(v0, &r[j + 0]);
    vloadx4(v1, &r[j + 32]);
    vloadx4(v2, &r[j + 128]);
    vloadx4(v3, &r[j + 160]);

    addsub_x4(v0, v2, vt);
    addsub_x4(v1, v3, vt);

    // After layer 7, no need for barrett_reduction

    // v2
    fqmul(v2.val[0], v2.val[0], z.val[0], t1, t2, t3, t4);
    fqmul(v2.val[0], v2.val[0], z.val[1], t1, t2, t3, t4);
    fqmul(v2.val[1], v2.val[1], z.val[0], t1, t2, t3, t4);
    fqmul(v2.val[1], v2.val[1], z.val[1], t1, t2, t3, t4);

    fqmul(v2.val[2], v2.val[2], z.val[0], t1, t2, t3, t4);
    fqmul(v2.val[2], v2.val[2], z.val[1], t1, t2, t3, t4);
    fqmul(v2.val[3], v2.val[3], z.val[0], t1, t2, t3, t4);
    fqmul(v2.val[3], v2.val[3], z.val[1], t1, t2, t3, t4);

    // v3
    fqmul(v3.val[0], v3.val[0], z.val[0], t1, t2, t3, t4);
    fqmul(v3.val[0], v3.val[0], z.val[1], t1, t2, t3, t4);
    fqmul(v3.val[1], v3.val[1], z.val[0], t1, t2, t3, t4);
    fqmul(v3.val[1], v3.val[1], z.val[1], t1, t2, t3, t4);

    fqmul(v3.val[2], v3.val[2], z.val[0], t1, t2, t3, t4);
    fqmul(v3.val[2], v3.val[2], z.val[1], t1, t2, t3, t4);
    fqmul(v3.val[3], v3.val[3], z.val[0], t1, t2, t3, t4);
    fqmul(v3.val[3], v3.val[3], z.val[1], t1, t2, t3, t4);

    // v0
    fqmul(v0.val[0], v0.val[0], z.val[1], t1, t2, t3, t4);
    fqmul(v0.val[1], v0.val[1], z.val[1], t1, t2, t3, t4);
    fqmul(v0.val[2], v0.val[2], z.val[1], t1, t2, t3, t4);
    fqmul(v0.val[3], v0.val[3], z.val[1], t1, t2, t3, t4);

    // v1
    fqmul(v1.val[0], v1.val[0], z.val[1], t1, t2, t3, t4);
    fqmul(v1.val[1], v1.val[1], z.val[1], t1, t2, t3, t4);
    fqmul(v1.val[2], v1.val[2], z.val[1], t1, t2, t3, t4);
    fqmul(v1.val[3], v1.val[3], z.val[1], t1, t2, t3, t4);

    vstorex4(&r[j + 0], v0);
    vstorex4(&r[j + 32], v1);
    vstorex4(&r[j + 128], v2);
    vstorex4(&r[j + 160], v3);
  }
}

/*************************************************
* Name:        ntt
*
* Description: Inplace number-theoretic transform (NTT) in Rq
*              input is in standard order, output is in bitreversed order
*
* Arguments:   - int16_t r[256]: pointer to input/output vector of elements
*                                of Zq
**************************************************/
// Merged NTT layer
void neon_ntt(int16_t r[256])
{
  int j, k1, k2, k3, k4, k5, k6;
  // Register: Total 26
  int16x8x4_t vt1, vt2, v0, v1, v2, v3, z; // 28
  int16x4_t a_lo, a_hi, b_lo, b_hi;        // 4
  int32x4_t t1, t2, t3, t4;                // 4
  int32x4_t neon_qinv, neon_kyberq;        // 2
  neon_qinv = vdupq_n_s32(QINV << 16);
  neon_kyberq = vdupq_n_s32(-KYBER_Q);
  // Scalar
  k1 = 64;
  k2 = 32;
  k3 = 16;
  k4 = 8;
  k5 = 4;
  k6 = 2;
  // End

  // Layer 7
  // Total registers: 32
  z.val[0] = vdupq_n_s16(zetas[1]);
  for (j = 0; j < 128; j += 64)
  {
    // Layer 7: v0 x v2 | v1 x v3
    // v0: 0   -> 31
    // v1: 32  -> 64
    // v2: 128 -> 159
    // v3: 160 -> 191
    vloadx4(v0, &r[j + 0]);
    vloadx4(v1, &r[j + 32]);
    vloadx4(v2, &r[j + 128]);
    vloadx4(v3, &r[j + 160]);

    fqmul(vt1.val[0], v2.val[0], z.val[0], t1, t2, t3, t4);
    fqmul(vt1.val[1], v2.val[1], z.val[0], t1, t2, t3, t4);
    fqmul(vt1.val[2], v2.val[2], z.val[0], t1, t2, t3, t4);
    fqmul(vt1.val[3], v2.val[3], z.val[0], t1, t2, t3, t4);

    fqmul(vt2.val[0], v3.val[0], z.val[0], t1, t2, t3, t4);
    fqmul(vt2.val[1], v3.val[1], z.val[0], t1, t2, t3, t4);
    fqmul(vt2.val[2], v3.val[2], z.val[0], t1, t2, t3, t4);
    fqmul(vt2.val[3], v3.val[3], z.val[0], t1, t2, t3, t4);

    // 128: 0 +- 128
    subadd_x4(v2, v0, vt1);
    // 160: 32 +- 160
    subadd_x4(v3, v1, vt2);

    vstorex4(&r[j + 0], v0);
    vstorex4(&r[j + 32], v1);
    vstorex4(&r[j + 128], v2);
    vstorex4(&r[j + 160], v3);
  }

  // Layer 6, 5, 4, 3, 2, 1
  for (j = 0; j < 256; j += 128)
  {
    // Layer 6: v0 x v2 | v1 x v3
    // v0: 0   -> 31
    // v1: 32  -> 63
    // v2: 64  -> 95
    // v3: 96  -> 127
    vloadx4(v0, &r[j + 0]);
    vloadx4(v1, &r[j + 32]);
    vloadx4(v2, &r[j + 64]);
    vloadx4(v3, &r[j + 96]);

    z.val[0] = vdupq_n_s16(zetas[k6++]);

    fqmul(vt1.val[0], v2.val[0], z.val[0], t1, t2, t3, t4);
    fqmul(vt1.val[1], v2.val[1], z.val[0], t1, t2, t3, t4);
    fqmul(vt1.val[2], v2.val[2], z.val[0], t1, t2, t3, t4);
    fqmul(vt1.val[3], v2.val[3], z.val[0], t1, t2, t3, t4);

    fqmul(vt2.val[0], v3.val[0], z.val[0], t1, t2, t3, t4);
    fqmul(vt2.val[1], v3.val[1], z.val[0], t1, t2, t3, t4);
    fqmul(vt2.val[2], v3.val[2], z.val[0], t1, t2, t3, t4);
    fqmul(vt2.val[3], v3.val[3], z.val[0], t1, t2, t3, t4);

    // 64: 0 +- 64
    subadd_x4(v2, v0, vt1);
    // 96: 32 +- 96
    subadd_x4(v3, v1, vt2);

    // Layer 5: v0 x v1 | v2 x v3
    // v0: 0   -> 31
    // v1: 32  -> 63
    // v2: 64  -> 95
    // v3: 96  -> 127

    z.val[0] = vdupq_n_s16(zetas[k5++]);
    z.val[1] = vdupq_n_s16(zetas[k5++]);

    fqmul(vt1.val[0], v1.val[0], z.val[0], t1, t2, t3, t4);
    fqmul(vt1.val[1], v1.val[1], z.val[0], t1, t2, t3, t4);
    fqmul(vt1.val[2], v1.val[2], z.val[0], t1, t2, t3, t4);
    fqmul(vt1.val[3], v1.val[3], z.val[0], t1, t2, t3, t4);

    fqmul(vt2.val[0], v3.val[0], z.val[1], t1, t2, t3, t4);
    fqmul(vt2.val[1], v3.val[1], z.val[1], t1, t2, t3, t4);
    fqmul(vt2.val[2], v3.val[2], z.val[1], t1, t2, t3, t4);
    fqmul(vt2.val[3], v3.val[3], z.val[1], t1, t2, t3, t4);

    // 32: 0 +- 32
    subadd_x4(v1, v0, vt1);
    // 96: 64 +- 96
    subadd_x4(v3, v2, vt2);

    // Layer 4: v0.val[0] x v0.val[2] | v0.val[1] x v0.val[3]
    // val[0]: 0  -> 7
    // val[1]: 8  -> 15
    // val[2]: 16 -> 23
    // val[3]: 24 -> 32
    z.val[0] = vdupq_n_s16(zetas[k4]);
    z.val[1] = vdupq_n_s16(zetas[k4 + 1]);
    z.val[2] = vdupq_n_s16(zetas[k4 + 2]);
    z.val[3] = vdupq_n_s16(zetas[k4 + 3]);
    k4 += 4;

    fqmul(vt1.val[0], v0.val[2], z.val[0], t1, t2, t3, t4);
    fqmul(vt1.val[1], v0.val[3], z.val[0], t1, t2, t3, t4);
    fqmul(vt1.val[2], v1.val[2], z.val[1], t1, t2, t3, t4);
    fqmul(vt1.val[3], v1.val[3], z.val[1], t1, t2, t3, t4);

    fqmul(vt2.val[0], v2.val[2], z.val[2], t1, t2, t3, t4);
    fqmul(vt2.val[1], v2.val[3], z.val[2], t1, t2, t3, t4);
    fqmul(vt2.val[2], v3.val[2], z.val[3], t1, t2, t3, t4);
    fqmul(vt2.val[3], v3.val[3], z.val[3], t1, t2, t3, t4);

    subadd(v0, 0, 2, 1, 3, vt1.val[0], vt1.val[1]);
    subadd(v1, 0, 2, 1, 3, vt1.val[2], vt1.val[3]);
    subadd(v2, 0, 2, 1, 3, vt2.val[0], vt2.val[1]);
    subadd(v3, 0, 2, 1, 3, vt2.val[2], vt2.val[3]);

    // Layer 3: v0.val[0] x v0.val[1] | v0.val[2] x v0.val[3]
    // val[0]: 0  -> 7
    // val[1]: 8  -> 15
    // val[2]: 16 -> 23
    // val[3]: 24 -> 32
    z.val[0] = vdupq_n_s16(zetas[k3]);
    z.val[1] = vdupq_n_s16(zetas[k3 + 1]);
    z.val[2] = vdupq_n_s16(zetas[k3 + 2]);
    z.val[3] = vdupq_n_s16(zetas[k3 + 3]);

    fqmul(vt1.val[0], v0.val[1], z.val[0], t1, t2, t3, t4);
    fqmul(vt1.val[1], v0.val[3], z.val[1], t1, t2, t3, t4);
    fqmul(vt1.val[2], v1.val[1], z.val[2], t1, t2, t3, t4);
    fqmul(vt1.val[3], v1.val[3], z.val[3], t1, t2, t3, t4);

    subadd(v0, 0, 1, 2, 3, vt1.val[0], vt1.val[1]);
    subadd(v1, 0, 1, 2, 3, vt1.val[2], vt1.val[3]);

    z.val[0] = vdupq_n_s16(zetas[k3 + 4]);
    z.val[1] = vdupq_n_s16(zetas[k3 + 5]);
    z.val[2] = vdupq_n_s16(zetas[k3 + 6]);
    z.val[3] = vdupq_n_s16(zetas[k3 + 7]);
    k3 += 8;

    fqmul(vt2.val[0], v2.val[1], z.val[0], t1, t2, t3, t4);
    fqmul(vt2.val[1], v2.val[3], z.val[1], t1, t2, t3, t4);
    fqmul(vt2.val[2], v3.val[1], z.val[2], t1, t2, t3, t4);
    fqmul(vt2.val[3], v3.val[3], z.val[3], t1, t2, t3, t4);

    subadd(v2, 0, 1, 2, 3, vt2.val[0], vt2.val[1]);
    subadd(v3, 0, 1, 2, 3, vt2.val[2], vt2.val[3]);

    // Layer 2: l0 x l1   | l2 x l3
    // Input:
    // 0,  1,  2,  3  | 4,  5,  6,  7
    // 8,  9,  10, 11 | 12, 13, 14, 15
    // 16, 17, 18, 19 | 20, 21, 22, 23
    // 24, 25, 26, 27 | 28, 29, 30, 31

    // Swap (v0.val[0], v0.val[2]) and (v0.val[1], v0.val[3])
    // Output:
    // 0,  1,  2,  3  | 16, 17, 18, 19
    // 4,  5,  6,  7  | 20, 21, 22, 23
    // 8,  9,  10, 11 | 24, 25, 26, 27
    // 12, 13, 14, 15 | 28, 29, 30, 31
    arrange(vt1, v0, 0, 2, 1, 3);
    arrange(vt2, v1, 0, 2, 1, 3);

    vdup(a_lo, zetas[k2]);
    vdup(b_lo, zetas[k2 + 1]);
    vdup(a_hi, zetas[k2 + 2]);
    vdup(b_hi, zetas[k2 + 3]);
    vcombine(z.val[0], a_lo, a_hi);
    vcombine(z.val[1], b_lo, b_hi);

    vdup(a_lo, zetas[k2 + 4]);
    vdup(b_lo, zetas[k2 + 5]);
    vdup(a_hi, zetas[k2 + 6]);
    vdup(b_hi, zetas[k2 + 7]);
    vcombine(z.val[2], a_lo, a_hi);
    vcombine(z.val[3], b_lo, b_hi);

    fqmul(vt1.val[1], vt1.val[1], z.val[0], t1, t2, t3, t4);
    fqmul(vt1.val[3], vt1.val[3], z.val[1], t1, t2, t3, t4);

    fqmul(vt2.val[1], vt2.val[1], z.val[2], t1, t2, t3, t4);
    fqmul(vt2.val[3], vt2.val[3], z.val[3], t1, t2, t3, t4);

    subadd_twist(v0, vt1, 0, 1, 2, 3);
    subadd_twist(v1, vt2, 0, 1, 2, 3);

    //
    arrange(vt1, v2, 0, 2, 1, 3);
    arrange(vt2, v3, 0, 2, 1, 3);

    vdup(a_lo, zetas[k2 + 8]);
    vdup(b_lo, zetas[k2 + 9]);
    vdup(a_hi, zetas[k2 + 10]);
    vdup(b_hi, zetas[k2 + 11]);
    vcombine(z.val[0], a_lo, a_hi);
    vcombine(z.val[1], b_lo, b_hi);

    vdup(a_lo, zetas[k2 + 12]);
    vdup(b_lo, zetas[k2 + 13]);
    vdup(a_hi, zetas[k2 + 14]);
    vdup(b_hi, zetas[k2 + 15]);
    vcombine(z.val[2], a_lo, a_hi);
    vcombine(z.val[3], b_lo, b_hi);

    k2 += 16;

    fqmul(vt1.val[1], vt1.val[1], z.val[0], t1, t2, t3, t4);
    fqmul(vt1.val[3], vt1.val[3], z.val[1], t1, t2, t3, t4);
    fqmul(vt2.val[1], vt2.val[1], z.val[2], t1, t2, t3, t4);
    fqmul(vt2.val[3], vt2.val[3], z.val[3], t1, t2, t3, t4);

    subadd_twist(v2, vt1, 0, 1, 2, 3);
    subadd_twist(v3, vt2, 0, 1, 2, 3);

    // Layer 1: v0.val[0] x v0.val[2] | v0.val[1] x v0.val[3]
    // v0.val[0]: 0,  1,  2,  3  | 16, 17, 18, 19
    // v0.val[1]: 4,  5,  6,  7  | 20, 21, 22, 23
    // v0.val[2]: 8,  9,  10, 11 | 24, 25, 26, 27
    // v0.val[3]: 12, 13, 14, 15 | 28, 29, 30, 31
    // transpose 4x4
    transpose(v0, vt1);
    transpose(v1, vt2);
    transpose(v2, vt1);
    transpose(v3, vt2);
    // v0.val[0]: 0, 4, 8,  12 | 16, 20, 24, 28
    // v0.val[1]: 1, 5, 9,  13 | 17, 21, 25, 29
    // v0.val[2]: 2, 6, 10, 14 | 18, 22, 26, 30
    // v0.val[3]: 3, 7, 11, 15 | 19, 23, 27, 31

    vloadx4(z, &zetas[k1]);
    k1 += 32;

    fqmul(vt1.val[0], v0.val[2], z.val[0], t1, t2, t3, t4);
    fqmul(vt1.val[1], v0.val[3], z.val[0], t1, t2, t3, t4);
    fqmul(vt1.val[2], v1.val[2], z.val[1], t1, t2, t3, t4);
    fqmul(vt1.val[3], v1.val[3], z.val[1], t1, t2, t3, t4);

    fqmul(vt2.val[0], v2.val[2], z.val[2], t1, t2, t3, t4);
    fqmul(vt2.val[1], v2.val[3], z.val[2], t1, t2, t3, t4);
    fqmul(vt2.val[2], v3.val[2], z.val[3], t1, t2, t3, t4);
    fqmul(vt2.val[3], v3.val[3], z.val[3], t1, t2, t3, t4);

    subadd(v0, 0, 2, 1, 3, vt1.val[0], vt1.val[1]);
    subadd(v1, 0, 2, 1, 3, vt1.val[2], vt1.val[3]);
    subadd(v2, 0, 2, 1, 3, vt2.val[0], vt2.val[1]);
    subadd(v3, 0, 2, 1, 3, vt2.val[2], vt2.val[3]);

    vstore4(&r[j], v0);
    vstore4(&r[j + 32], v1);
    vstore4(&r[j + 64], v2);
    vstore4(&r[j + 96], v3);
  }
}
