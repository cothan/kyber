#include <arm_neon.h>
#include "params.h"
#include "ntt.h"
#include "reduce.h"

#define _V (((1U << 26) + KYBER_Q / 2) / KYBER_Q)

/******8888888888888888888888888***************/

// Load int16x8_t c <= ptr*
#define vload(c, ptr) c = vld1q_s16(ptr);

// Load int16x8_t c <= ptr*
#define vloadx4(c, ptr) c = vld1q_s16_x4(ptr);

// Store *ptr <= c
#define vstorex4(ptr, c) vst1q_s16_x4(ptr, c);

// Load int16x8_t c <= ptr*
#define vload4(c, ptr) c = vld4q_s16(ptr);

// Store *ptr <= c
#define vstore4(ptr, c) vst4q_s16(ptr, c);

// c (int16x8) = a + b (int16x8)
#define vadd8(c, a, b) c = vaddq_s16(a, b);

// c (int16x8) = a - b (int16x8)
#define vsub8(c, a, b) c = vsubq_s16(a, b);

// c = a
#define vcopy(c, a) c = vorrq_s16(a, a);

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
#define fqmul(out, in, zeta, t)                               \
  t.val[0] = vmull_s16(vget_low_s16(in), vget_low_s16(zeta)); \
  t.val[1] = vmull_high_s16(in, zeta);                        \
  t.val[2] = vmulq_s32(t.val[0], neon_qinv);                  \
  t.val[3] = vmulq_s32(t.val[1], neon_qinv);                  \
  t.val[2] = vshrq_n_s32(t.val[2], 16);                       \
  t.val[3] = vshrq_n_s32(t.val[3], 16);                       \
  t.val[2] = vmulq_s32(t.val[2], neon_kyberq);                \
  t.val[3] = vmulq_s32(t.val[3], neon_kyberq);                \
  out = vaddhn_high_s32(vaddhn_s32(t.val[2], t.val[0]), t.val[3], t.val[1]);

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
  vcopy(t1, v.val[i]);                      \
  vcopy(t2, v.val[m]);                      \
  vadd8(v.val[i], t1, v.val[j]);      \
  vsub8(v.val[j], t1, v.val[j]);      \
  vadd8(v.val[m], t2, v.val[n]);      \
  vsub8(v.val[n], t2, v.val[n]);

#define addsub_x4(v0, v1, va)             \
  vcopy(va.val[0] , v0.val[0]);                  \
  vcopy(va.val[1] , v0.val[1]);                  \
  vcopy(va.val[2] , v0.val[2]);                  \
  vcopy(va.val[3] , v0.val[3]);                  \
  vadd8(v0.val[0], va.val[0], v1.val[0]); \
  vadd8(v0.val[1], va.val[1], v1.val[1]); \
  vadd8(v0.val[2], va.val[2], v1.val[2]); \
  vadd8(v0.val[3], va.val[3], v1.val[3]); \
  vsub8(v1.val[0], va.val[0], v1.val[0]); \
  vsub8(v1.val[1], va.val[1], v1.val[1]); \
  vsub8(v1.val[2], va.val[2], v1.val[2]); \
  vsub8(v1.val[3], va.val[3], v1.val[3]);

#define addsub_twist(v, v_in, i, j, m, n, t1, t2) \
  vcopy(t1 , v_in.val[i]);                               \
  vcopy(t2 , v_in.val[m]);                               \
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

// void print_vector(int16x8x4_t a, int bound, const char* string)
// {
//   for (int i = 0; i < bound; i++)
//   {
//     printf("%s: ", string);
//     for (int j = 0; j < 8; j++){
//       printf("%d,", a.val[i][j]);
//     }
//     printf("\n");
//   }
// }

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
  int j, k = 0;
  // Register: Total 32 + 4(const) = 36
  int16x8x4_t vt, v0, v1, v2, v3, z; // 28
  int32x4x4_t t;                     // 4
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

    vloadx4(z, &neon_zetas_inv[k]);

    fqmul(v0.val[2], v0.val[2], z.val[0], t);
    fqmul(v0.val[3], v0.val[3], z.val[0], t);
    fqmul(v1.val[2], v1.val[2], z.val[1], t);
    fqmul(v1.val[3], v1.val[3], z.val[1], t);

    fqmul(v2.val[2], v2.val[2], z.val[2], t);
    fqmul(v2.val[3], v2.val[3], z.val[2], t);
    fqmul(v3.val[2], v3.val[2], z.val[3], t);
    fqmul(v3.val[3], v3.val[3], z.val[3], t);

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

    vloadx4(z, &neon_zetas_inv[k + 32]);

    fqmul(v0.val[1], v0.val[1], z.val[0], t);
    fqmul(v0.val[3], v0.val[3], z.val[1], t);
    fqmul(v1.val[1], v1.val[1], z.val[2], t);
    fqmul(v1.val[3], v1.val[3], z.val[3], t);

    vloadx4(z, &neon_zetas_inv[k + 64]);

    fqmul(v2.val[1], v2.val[1], z.val[0], t);
    fqmul(v2.val[3], v2.val[3], z.val[1], t);
    fqmul(v3.val[1], v3.val[1], z.val[2], t);
    fqmul(v3.val[3], v3.val[3], z.val[3], t);

    // Layer 3 : v0.val[0] x v0.val[2] | v0.val[1] x v0.val[3]
    // v0.val[0]: 0,  1,  2,  3  | 16,  17,  18,  19
    // v0.val[1]: 4,  5,  6,  7  | 20,  21,  22,  23
    // v0.val[2]: 8,  9,  10, 11 | 24,  25,  26,  27
    // v0.val[3]: 12, 13, 14, 15 | 28,  29,  30,  31

    addsub(v0, 0, 2, 1, 3, vt.val[0], vt.val[1]);
    addsub(v1, 0, 2, 1, 3, vt.val[2], vt.val[3]);
    addsub(v2, 0, 2, 1, 3, vt.val[0], vt.val[1]);
    addsub(v3, 0, 2, 1, 3, vt.val[2], vt.val[3]);

    vloadx4(z, &neon_zetas_inv[k + 96]);

    fqmul(v0.val[2], v0.val[2], z.val[0], t);
    fqmul(v0.val[3], v0.val[3], z.val[0], t);
    fqmul(v1.val[2], v1.val[2], z.val[1], t);
    fqmul(v1.val[3], v1.val[3], z.val[1], t);

    fqmul(v2.val[2], v2.val[2], z.val[2], t);
    fqmul(v2.val[3], v2.val[3], z.val[2], t);
    fqmul(v3.val[2], v3.val[2], z.val[3], t);
    fqmul(v3.val[3], v3.val[3], z.val[3], t);

    // 16, 17, 18, 19
    barret_hi(v0.val[0], v1.val[0], t.val[0], t.val[1], vt.val[0], vt.val[1]);
    barret_hi(v2.val[0], v3.val[0], t.val[2], t.val[3], vt.val[2], vt.val[3]);

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

    vloadx4(z, &neon_zetas_inv[k + 128]);

    fqmul(v0.val[2], v0.val[2], z.val[0], t);
    fqmul(v0.val[3], v0.val[3], z.val[0], t);
    fqmul(v1.val[2], v1.val[2], z.val[1], t);
    fqmul(v1.val[3], v1.val[3], z.val[1], t);

    fqmul(v2.val[2], v2.val[2], z.val[2], t);
    fqmul(v2.val[3], v2.val[3], z.val[2], t);
    fqmul(v3.val[2], v3.val[2], z.val[3], t);
    fqmul(v3.val[3], v3.val[3], z.val[3], t);

    // 0, 1, 2, 3
    barret_lo(v0.val[0], v1.val[0], t.val[0], t.val[1], vt.val[0], vt.val[1]);
    barret_lo(v2.val[0], v3.val[0], t.val[2], t.val[3], vt.val[2], vt.val[3]);

    // Layer 5: v0 x v1 | v2 x v3
    // v0: 0  -> 31
    // v1: 32 -> 63
    // v2: 64 -> 95
    // v3: 96 -> 127

    addsub_x4(v0, v1, vt);
    addsub_x4(v2, v3, vt);

    barret_hi(v0.val[0], v2.val[0], t.val[0], t.val[1], vt.val[0], vt.val[1]);

    vload(z.val[0], &neon_zetas_inv[k + 160]);
    vload(z.val[1], &neon_zetas_inv[k + 168]);

    fqmul(v1.val[0], v1.val[0], z.val[0], t);
    fqmul(v1.val[1], v1.val[1], z.val[0], t);
    fqmul(v1.val[2], v1.val[2], z.val[0], t);
    fqmul(v1.val[3], v1.val[3], z.val[0], t);

    fqmul(v3.val[0], v3.val[0], z.val[1], t);
    fqmul(v3.val[1], v3.val[1], z.val[1], t);
    fqmul(v3.val[2], v3.val[2], z.val[1], t);
    fqmul(v3.val[3], v3.val[3], z.val[1], t);

    // Layer 6: v0 x v2 | v1 x v3
    // v0: 0  -> 31
    // v2: 64 -> 95
    // v1: 32 -> 63
    // v3: 96 -> 127

    addsub_x4(v0, v2, vt);
    addsub_x4(v1, v3, vt);

    barret_lh(v0.val[1], t.val[0], t.val[1], vt.val[0]);
    barret_lh(v1.val[1], t.val[2], t.val[3], vt.val[1]);

    vload(z.val[0], &neon_zetas_inv[k + 176]);

    fqmul(v2.val[0], v2.val[0], z.val[0], t);
    fqmul(v2.val[1], v2.val[1], z.val[0], t);
    fqmul(v2.val[2], v2.val[2], z.val[0], t);
    fqmul(v2.val[3], v2.val[3], z.val[0], t);

    fqmul(v3.val[0], v3.val[0], z.val[0], t);
    fqmul(v3.val[1], v3.val[1], z.val[0], t);
    fqmul(v3.val[2], v3.val[2], z.val[0], t);
    fqmul(v3.val[3], v3.val[3], z.val[0], t);

    vstorex4(&r[j], v0);
    vstorex4(&r[j + 32], v1);
    vstorex4(&r[j + 64], v2);
    vstorex4(&r[j + 96], v3);
    k += 184;
  }

  // Layer 7, inv_mul
  vload(z.val[0], &neon_zetas_inv[368]);
  vload(z.val[1], &neon_zetas_inv[376]);

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
    fqmul(v2.val[0], v2.val[0], z.val[0], t);
    fqmul(v2.val[0], v2.val[0], z.val[1], t);
    fqmul(v2.val[1], v2.val[1], z.val[0], t);
    fqmul(v2.val[1], v2.val[1], z.val[1], t);

    fqmul(v2.val[2], v2.val[2], z.val[0], t);
    fqmul(v2.val[2], v2.val[2], z.val[1], t);
    fqmul(v2.val[3], v2.val[3], z.val[0], t);
    fqmul(v2.val[3], v2.val[3], z.val[1], t);

    // v3
    fqmul(v3.val[0], v3.val[0], z.val[0], t);
    fqmul(v3.val[0], v3.val[0], z.val[1], t);
    fqmul(v3.val[1], v3.val[1], z.val[0], t);
    fqmul(v3.val[1], v3.val[1], z.val[1], t);

    fqmul(v3.val[2], v3.val[2], z.val[0], t);
    fqmul(v3.val[2], v3.val[2], z.val[1], t);
    fqmul(v3.val[3], v3.val[3], z.val[0], t);
    fqmul(v3.val[3], v3.val[3], z.val[1], t);

    // v0
    fqmul(v0.val[0], v0.val[0], z.val[1], t);
    fqmul(v0.val[1], v0.val[1], z.val[1], t);
    fqmul(v0.val[2], v0.val[2], z.val[1], t);
    fqmul(v0.val[3], v0.val[3], z.val[1], t);

    // v1
    fqmul(v1.val[0], v1.val[0], z.val[1], t);
    fqmul(v1.val[1], v1.val[1], z.val[1], t);
    fqmul(v1.val[2], v1.val[2], z.val[1], t);
    fqmul(v1.val[3], v1.val[3], z.val[1], t);

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
  int j, k = 8;
  // Register: Total 28 + 4 + 2 (const) = 34
  int16x8x4_t vt1, vt2, v0, v1, v2, v3, z; // 28
  int32x4x4_t t;                           // 4
  int32x4_t neon_qinv, neon_kyberq;        // 2
  neon_qinv = vdupq_n_s32(QINV << 16);
  neon_kyberq = vdupq_n_s32(-KYBER_Q);
  // End

  // Layer 7
  // Total registers: 32
  vload(z.val[0], &neon_zetas[0]);
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

    fqmul(vt1.val[0], v2.val[0], z.val[0], t);
    fqmul(vt1.val[1], v2.val[1], z.val[0], t);
    fqmul(vt1.val[2], v2.val[2], z.val[0], t);
    fqmul(vt1.val[3], v2.val[3], z.val[0], t);

    fqmul(vt2.val[0], v3.val[0], z.val[0], t);
    fqmul(vt2.val[1], v3.val[1], z.val[0], t);
    fqmul(vt2.val[2], v3.val[2], z.val[0], t);
    fqmul(vt2.val[3], v3.val[3], z.val[0], t);

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

    vload(z.val[0], &neon_zetas[k]);

    fqmul(vt1.val[0], v2.val[0], z.val[0], t);
    fqmul(vt1.val[1], v2.val[1], z.val[0], t);
    fqmul(vt1.val[2], v2.val[2], z.val[0], t);
    fqmul(vt1.val[3], v2.val[3], z.val[0], t);

    fqmul(vt2.val[0], v3.val[0], z.val[0], t);
    fqmul(vt2.val[1], v3.val[1], z.val[0], t);
    fqmul(vt2.val[2], v3.val[2], z.val[0], t);
    fqmul(vt2.val[3], v3.val[3], z.val[0], t);

    // 64: 0 +- 64
    subadd_x4(v2, v0, vt1);
    // 96: 32 +- 96
    subadd_x4(v3, v1, vt2);

    // Layer 5: v0 x v1 | v2 x v3
    // v0: 0   -> 31
    // v1: 32  -> 63
    // v2: 64  -> 95
    // v3: 96  -> 127

    vload(z.val[0], &neon_zetas[k + 8]);
    vload(z.val[1], &neon_zetas[k + 16]);

    fqmul(vt1.val[0], v1.val[0], z.val[0], t);
    fqmul(vt1.val[1], v1.val[1], z.val[0], t);
    fqmul(vt1.val[2], v1.val[2], z.val[0], t);
    fqmul(vt1.val[3], v1.val[3], z.val[0], t);

    fqmul(vt2.val[0], v3.val[0], z.val[1], t);
    fqmul(vt2.val[1], v3.val[1], z.val[1], t);
    fqmul(vt2.val[2], v3.val[2], z.val[1], t);
    fqmul(vt2.val[3], v3.val[3], z.val[1], t);

    // 32: 0 +- 32
    subadd_x4(v1, v0, vt1);
    // 96: 64 +- 96
    subadd_x4(v3, v2, vt2);

    // Layer 4: v0.val[0] x v0.val[2] | v0.val[1] x v0.val[3]
    // val[0]: 0  -> 7
    // val[1]: 8  -> 15
    // val[2]: 16 -> 23
    // val[3]: 24 -> 32
    vloadx4(z, &neon_zetas[k + 24]);

    fqmul(vt1.val[0], v0.val[2], z.val[0], t);
    fqmul(vt1.val[1], v0.val[3], z.val[0], t);
    fqmul(vt1.val[2], v1.val[2], z.val[1], t);
    fqmul(vt1.val[3], v1.val[3], z.val[1], t);

    fqmul(vt2.val[0], v2.val[2], z.val[2], t);
    fqmul(vt2.val[1], v2.val[3], z.val[2], t);
    fqmul(vt2.val[2], v3.val[2], z.val[3], t);
    fqmul(vt2.val[3], v3.val[3], z.val[3], t);

    subadd(v0, 0, 2, 1, 3, vt1.val[0], vt1.val[1]);
    subadd(v1, 0, 2, 1, 3, vt1.val[2], vt1.val[3]);
    subadd(v2, 0, 2, 1, 3, vt2.val[0], vt2.val[1]);
    subadd(v3, 0, 2, 1, 3, vt2.val[2], vt2.val[3]);

    // Layer 3: v0.val[0] x v0.val[1] | v0.val[2] x v0.val[3]
    // val[0]: 0  -> 7
    // val[1]: 8  -> 15
    // val[2]: 16 -> 23
    // val[3]: 24 -> 32
    vloadx4(z, &neon_zetas[k + 56]);

    fqmul(vt1.val[0], v0.val[1], z.val[0], t);
    fqmul(vt1.val[1], v0.val[3], z.val[1], t);
    fqmul(vt1.val[2], v1.val[1], z.val[2], t);
    fqmul(vt1.val[3], v1.val[3], z.val[3], t);

    subadd(v0, 0, 1, 2, 3, vt1.val[0], vt1.val[1]);
    subadd(v1, 0, 1, 2, 3, vt1.val[2], vt1.val[3]);

    vloadx4(z, &neon_zetas[k + 88]);

    fqmul(vt2.val[0], v2.val[1], z.val[0], t);
    fqmul(vt2.val[1], v2.val[3], z.val[1], t);
    fqmul(vt2.val[2], v3.val[1], z.val[2], t);
    fqmul(vt2.val[3], v3.val[3], z.val[3], t);

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

    vloadx4(z, &neon_zetas[k + 120]);

    fqmul(vt1.val[1], vt1.val[1], z.val[0], t);
    fqmul(vt1.val[3], vt1.val[3], z.val[1], t);

    fqmul(vt2.val[1], vt2.val[1], z.val[2], t);
    fqmul(vt2.val[3], vt2.val[3], z.val[3], t);

    subadd_twist(v0, vt1, 0, 1, 2, 3);
    subadd_twist(v1, vt2, 0, 1, 2, 3);

    arrange(vt1, v2, 0, 2, 1, 3);
    arrange(vt2, v3, 0, 2, 1, 3);

    vloadx4(z, &neon_zetas[k + 152]);

    fqmul(vt1.val[1], vt1.val[1], z.val[0], t);
    fqmul(vt1.val[3], vt1.val[3], z.val[1], t);
    fqmul(vt2.val[1], vt2.val[1], z.val[2], t);
    fqmul(vt2.val[3], vt2.val[3], z.val[3], t);

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

    vloadx4(z, &neon_zetas[k + 184]);

    fqmul(vt1.val[0], v0.val[2], z.val[0], t);
    fqmul(vt1.val[1], v0.val[3], z.val[0], t);
    fqmul(vt1.val[2], v1.val[2], z.val[1], t);
    fqmul(vt1.val[3], v1.val[3], z.val[1], t);

    fqmul(vt2.val[0], v2.val[2], z.val[2], t);
    fqmul(vt2.val[1], v2.val[3], z.val[2], t);
    fqmul(vt2.val[2], v3.val[2], z.val[3], t);
    fqmul(vt2.val[3], v3.val[3], z.val[3], t);

    subadd(v0, 0, 2, 1, 3, vt1.val[0], vt1.val[1]);
    subadd(v1, 0, 2, 1, 3, vt1.val[2], vt1.val[3]);
    subadd(v2, 0, 2, 1, 3, vt2.val[0], vt2.val[1]);
    subadd(v3, 0, 2, 1, 3, vt2.val[2], vt2.val[3]);

    vstore4(&r[j], v0);
    vstore4(&r[j + 32], v1);
    vstore4(&r[j + 64], v2);
    vstore4(&r[j + 96], v3);

    k += 216;
  }
}
