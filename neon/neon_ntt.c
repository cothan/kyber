#include <arm_neon.h>
#include "params.h"
#include "ntt.h"
#include "reduce.h"

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

// c (int16x4) = a - b (int16x4)
#define vsub4(c, a, b) c = vsub_s16(a, b);

// c (int16x4) = a + b (int16x4)
#define vadd4(c, a, b) c = vadd_s16(a, b);

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
#define fqmul(out, in, zeta, a1, a2, u1, u2, neon_qinv, neon_kyberq) \
  a1 = vmull_s16(vget_low_s16(in), vget_low_s16(zeta));                  \
  a2 = vmull_high_s16(in, zeta);                                         \
  u1 = vmulq_s32(a1, neon_qinv);                                         \
  u2 = vmulq_s32(a2, neon_qinv);                                         \
  u1 = vshrq_n_s32(u1, 16);                                              \
  u2 = vshrq_n_s32(u2, 16);                                              \
  u1 = vmulq_s32(neon_kyberq, u1);                                       \
  u2 = vmulq_s32(neon_kyberq, u2);                                       \
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
#define barrett(inout, t32, t16, neon_v, neon_kyberq16) \
  t32 = vmull_s16(inout, neon_v);                       \
  t32 = vshrq_n_s32(t32, 26);                           \
  t16 = vmovn_s32(t32);                                 \
  inout = vmla_s16(inout, t16, neon_kyberq16);

/*
Matrix 4x4 tranpose: v
Input: int16x8x4_t v, tmp
Output: int16x8x4_t v
*/
#define tranpose(v, tmp)                                                          \
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
#define arrange(v_out, v_in)                                                            \
  v_out.val[0] = (int16x8_t)vtrn1q_s64((int64x2_t)v_in.val[0], (int64x2_t)v_in.val[1]); \
  v_out.val[1] = (int16x8_t)vtrn2q_s64((int64x2_t)v_in.val[0], (int64x2_t)v_in.val[1]); \
  v_out.val[2] = (int16x8_t)vtrn1q_s64((int64x2_t)v_in.val[2], (int64x2_t)v_in.val[3]); \
  v_out.val[3] = (int16x8_t)vtrn2q_s64((int64x2_t)v_in.val[2], (int64x2_t)v_in.val[3]);

/*
Butterfly Unit
Input: v, i, j, tmp
Output: v
*/
#define addsub(v, v_in, i, j, tmp)   \
  tmp = v_in.val[i];                 \
  vadd8(v.val[i], tmp, v_in.val[j]); \
  vsub8(v.val[j], tmp, v_in.val[j]);

#define addsub_twist(v, v_in, i, j, m, n, tmp1, tmp2)   \
  tmp1 = v_in.val[i];                 \
  vadd8(v.val[i], tmp1, v_in.val[j]); \
  vsub8(v.val[m], tmp1, v_in.val[j]); \
  tmp2 = v_in.val[m];                 \
  vadd8(v.val[j], tmp2, v_in.val[n]); \
  vsub8(v.val[n], tmp2, v_in.val[n]);


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
  int16x8x4_t va, vb, v0, v1, v2, v3; // 20
  int16x8_t r0, r1, r2, r3, l0, l1, l2, l3,
            o_tmp, e_tmp;                     // 10
  int16x4_t a_lo, a_hi, b_lo, b_hi, zlo, zhi; // 6
  int32x4_t t1, t2, t3, t4;                   // 6
  int16x8_t z0, z1, z2, z3;
  // Constant: Total 4
  int32x4_t neon_qinv, neon_kyberq;
  int16x4_t neon_v, neon_kyberq16;
  neon_qinv = vdupq_n_s32(QINV << 16);
  neon_kyberq = vdupq_n_s32(-KYBER_Q);
  neon_kyberq16 = vdup_n_s16(-KYBER_Q);
  neon_v = vdup_n_s16(((1U << 26) + KYBER_Q / 2) / KYBER_Q);
  // Scalar
  k1 = 0;
  k2 = 64;
  k3 = 96;
  k4 = 112;
  k5 = 120;
  k6 = 124;
  // End

  // *Vectorize* barret_reduction over 96 points rather than 896 points
  // Optimimal Barret reduction for Kyber N=256, B=9 is 78 points, see here: 
  // https://eprint.iacr.org/2020/1377.pdf

  // Layer 1, 2, 3, 4, 5
  // Total register: 26
  for (j = 0; j < 256; j += 64)
  {
    // 1st layer : v0.val[0] x v0.val[2] | v0.val[1] x v0.val[3]
    // v0.val[0]: 0, 4, 8,  12 | 16, 20, 24, 28
    // v0.val[1]: 1, 5, 9,  13 | 17, 21, 25, 29
    // v0.val[2]: 2, 6, 10, 14 | 18, 22, 26, 30
    // v0.val[3]: 3, 7, 11, 15 | 19, 23, 27, 31
    vload4(v0, &r[j]);
    vload4(v1, &r[j+32]);

    addsub(v0, v0, 0, 2, e_tmp);
    addsub(v0, v0, 1, 3, o_tmp);

    addsub(v1, v1, 0, 2, e_tmp);
    addsub(v1, v1, 1, 3, o_tmp);

    vload(z0, &zetas_inv[k1]);
    vload(z1, &zetas_inv[k1+8]);
    k1 += 16;
    fqmul(v0.val[2], v0.val[2], z0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v0.val[3], v0.val[3], z0, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    fqmul(v1.val[2], v1.val[2], z1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v1.val[3], v1.val[3], z1, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    // Layer 2: v0.val[0] x v0.val[1] | v0.val[2] x v0.val[3]
    // Tranpose 4x4
    tranpose(v0, v2);
    tranpose(v1, v3);

    // v0.val[0]: 0,  1,  2,  3  | 16,  17,  18,  19
    // v0.val[1]: 4,  5,  6,  7  | 20,  21,  22,  23
    // v0.val[2]: 8,  9,  10, 11 | 24,  25,  26,  27
    // v0.val[3]: 12, 13, 14, 15 | 28,  29,  30,  31

    addsub(v0, v0, 0, 1, e_tmp);
    addsub(v0, v0, 2, 3, o_tmp);

    addsub(v1, v1, 0, 1, e_tmp);
    addsub(v1, v1, 2, 3, o_tmp);

    vdup(a_lo, zetas_inv[k2]);
    vdup(b_lo, zetas_inv[k2 + 1]);
    vdup(a_hi, zetas_inv[k2 + 2]);
    vdup(b_hi, zetas_inv[k2 + 3]);

    vcombine(z0, a_lo, a_hi);
    vcombine(z1, b_lo, b_hi);
    
    vdup(a_lo, zetas_inv[k2 + 4]);
    vdup(b_lo, zetas_inv[k2 + 5]);
    vdup(a_hi, zetas_inv[k2 + 6]);
    vdup(b_hi, zetas_inv[k2 + 7]);

    vcombine(z2, a_lo, a_hi);
    vcombine(z3, b_lo, b_hi);
    k2 += 8;

    fqmul(v0.val[1], v0.val[1], z0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v0.val[3], v0.val[3], z1, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    fqmul(v1.val[1], v1.val[1], z2, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v1.val[3], v1.val[3], z3, t1, t2, t3, t4, neon_qinv, neon_kyberq);


    // Layer 3 : v0.val[0] x v0.val[2] | v0.val[1] x v0.val[3]
    // v0.val[0]: 0,  1,  2,  3  | 16,  17,  18,  19
    // v0.val[1]: 4,  5,  6,  7  | 20,  21,  22,  23
    // v0.val[2]: 8,  9,  10, 11 | 24,  25,  26,  27
    // v0.val[3]: 12, 13, 14, 15 | 28,  29,  30,  31

    addsub(v0, v0, 0, 2, e_tmp);
    addsub(v0, v0, 1, 3, o_tmp);

    addsub(v1, v1, 0, 2, e_tmp);
    addsub(v1, v1, 1, 3, o_tmp);

    vdup(a_lo, zetas_inv[k3]);
    vdup(a_hi, zetas_inv[k3 + 1]);
    vdup(b_lo, zetas_inv[k3 + 2]);
    vdup(b_hi, zetas_inv[k3 + 3]);
    k3 += 4;
    vcombine(z0, a_lo, a_hi);
    vcombine(z1, b_lo, b_hi);

    fqmul(v0.val[2], v0.val[2], z0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v0.val[3], v0.val[3], z0, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    fqmul(v1.val[2], v1.val[2], z1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v1.val[3], v1.val[3], z1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    
    // 16, 17, 18, 19
    vlo(a_lo, v0.val[0]);
    vhi(a_hi, v0.val[0]);
    vlo(b_lo, v1.val[0]);
    vhi(b_hi, v1.val[0]);

    barrett(a_hi, t1, zhi, neon_v, neon_kyberq16);
    barrett(b_hi, t2, zhi, neon_v, neon_kyberq16);

    vcombine(v0.val[0], a_lo, a_hi);
    vcombine(v1.val[0], b_lo, b_hi);


    // Layer 4: v0.val[0] x v0.val[1] | v0.val[2] x v0.val[3]
    // Re-arrange vector
    arrange(v2, v0);
    arrange(v3, v1);

    // v2.val[0]: 0,  1,  2,  3  | 4,  5,  6,  7
    // v2.val[1]: 16, 17, 18, 19 | 20, 21, 22, 23
    // v2.val[2]: 8,  9,  10, 11 | 12, 13, 14, 15
    // v2.val[3]: 24, 25, 26, 27 | 28, 29, 30, 31

    addsub_twist(v0, v2, 0, 1, 2, 3, e_tmp, o_tmp);
    addsub_twist(v1, v3, 0, 1, 2, 3, e_tmp, o_tmp);

    z0 = vdupq_n_s16(zetas_inv[k4]);
    z1 = vdupq_n_s16(zetas_inv[k4+1]);
    k4 += 2;
    fqmul(v0.val[2], v0.val[2], z0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v0.val[3], v0.val[3], z0, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    fqmul(v1.val[2], v1.val[2], z1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v1.val[3], v1.val[3], z1, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    // 0, 1, 2, 3
    vlo(a_lo, v0.val[0]);
    vhi(a_hi, v0.val[0]);
    vlo(b_lo, v1.val[0]);
    vhi(b_hi, v1.val[0]);

    barrett(a_lo, t1, zlo, neon_v, neon_kyberq16);
    barrett(b_lo, t1, zlo, neon_v, neon_kyberq16);

    vcombine(v0.val[0], a_lo, a_hi);
    vcombine(v1.val[0], b_lo, b_hi);

    // Layer 5: 
    // v0.val[0] x v1.val[0]
    // v0.val[1] x v1.val[1]
    // v0.val[2] x v1.val[2]
    // v0.val[3] x v1.val[3]
    
    // // 0 + 32
    // vadd8(v0.val[0], va.val[0], v1.val[0]);
    // vadd8(v0.val[1], va.val[1], v1.val[1]);
    // vadd8(v0.val[2], va.val[2], v1.val[2]);
    // vadd8(v0.val[3], va.val[3], v1.val[3]);
    // // 0 - 32
    // vsub8(v1.val[0], va.val[0], v1.val[0]);
    // vsub8(v1.val[1], va.val[1], v1.val[1]);
    // vsub8(v1.val[2], va.val[2], v1.val[2]);
    // vsub8(v1.val[3], va.val[3], v1.val[3]);

    // vlo(a_lo, v0.val[0]);
    // vhi(a_hi, v0.val[0]);

    // barrett(a_hi, t1, zhi, neon_v, neon_kyberq16);
    // vcombine(v0.val[0], a_lo, a_hi);

    // l0 = vdupq_n_s16(zetas_inv[k5++]);
    // fqmul(v1.val[0], v1.val[0], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    // fqmul(v1.val[1], v1.val[1], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    // fqmul(v1.val[2], v1.val[2], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    // fqmul(v1.val[3], v1.val[3], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    vstorex4(&r[j], v0);
    vstorex4(&r[j+32], v1);
  }

  // Layer 5, 6
  // Total register: 5x4 + 4(const) + 2(zlo, zhi) + 4(ab_hi, lo) + 2(t1-t6) = 32
  for (j = 0; j < 256; j += 128)
  {
    // Layer 5: v0 x v1 | v2 x v3
    // v0: 0  -> 31
    // v1: 32 -> 63
    // v2: 64 -> 95
    // v3: 96 -> 127
    vloadx4(v0, &r[j + 0]);
    vloadx4(v1, &r[j + 32]);
    vloadx4(v2, &r[j + 64]);
    vloadx4(v3, &r[j + 96]);

    va.val[0] = v0.val[0];
    va.val[1] = v0.val[1];
    va.val[2] = v0.val[2];
    va.val[3] = v0.val[3];

    // 0 + 32
    vadd8(v0.val[0], va.val[0], v1.val[0]);
    vadd8(v0.val[1], va.val[1], v1.val[1]);
    vadd8(v0.val[2], va.val[2], v1.val[2]);
    vadd8(v0.val[3], va.val[3], v1.val[3]);
    // 0 - 32
    vsub8(v1.val[0], va.val[0], v1.val[0]);
    vsub8(v1.val[1], va.val[1], v1.val[1]);
    vsub8(v1.val[2], va.val[2], v1.val[2]);
    vsub8(v1.val[3], va.val[3], v1.val[3]);

    vb.val[0] = v2.val[0];
    vb.val[1] = v2.val[1];
    vb.val[2] = v2.val[2];
    vb.val[3] = v2.val[3];

    // // 64 + 96
    vadd8(v2.val[0], vb.val[0], v3.val[0]);
    vadd8(v2.val[1], vb.val[1], v3.val[1]);
    vadd8(v2.val[2], vb.val[2], v3.val[2]);
    vadd8(v2.val[3], vb.val[3], v3.val[3]);
    // 64 - 96
    vsub8(v3.val[0], vb.val[0], v3.val[0]);
    vsub8(v3.val[1], vb.val[1], v3.val[1]);
    vsub8(v3.val[2], vb.val[2], v3.val[2]);
    vsub8(v3.val[3], vb.val[3], v3.val[3]);

    vlo(a_lo, v0.val[0]);
    vhi(a_hi, v0.val[0]);
    vlo(b_lo, v2.val[0]);
    vhi(b_hi, v2.val[0]);
    
    barrett(a_hi, t1, zhi, neon_v, neon_kyberq16);
    barrett(b_hi, t2, zhi, neon_v, neon_kyberq16);
    
    vcombine(v0.val[0], a_lo, a_hi);
    vcombine(v2.val[0], b_lo, b_hi);

    //
    // vdup(zlo, zetas_inv[k5++]);
    l0 = vdupq_n_s16(zetas_inv[k5++]);
    l1 = vdupq_n_s16(zetas_inv[k5++]);

    fqmul(v1.val[0], v1.val[0], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v1.val[1], v1.val[1], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v1.val[2], v1.val[2], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v1.val[3], v1.val[3], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    fqmul(v3.val[0], v3.val[0], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v3.val[1], v3.val[1], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v3.val[2], v3.val[2], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v3.val[3], v3.val[3], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    // Layer 6: v0 x v2 | v1 x v3
    // v0: 0  -> 31
    // v2: 64 -> 95
    // v1: 32 -> 63
    // v3: 96 -> 127

    va.val[0] = v0.val[0];
    va.val[1] = v0.val[1];
    va.val[2] = v0.val[2];
    va.val[3] = v0.val[3];

    // 0 + 64
    vadd8(v0.val[0], va.val[0], v2.val[0]);
    vadd8(v0.val[1], va.val[1], v2.val[1]);
    vadd8(v0.val[2], va.val[2], v2.val[2]);
    vadd8(v0.val[3], va.val[3], v2.val[3]);
    // 0 - 64
    vsub8(v2.val[0], va.val[0], v2.val[0]);
    vsub8(v2.val[1], va.val[1], v2.val[1]);
    vsub8(v2.val[2], va.val[2], v2.val[2]);
    vsub8(v2.val[3], va.val[3], v2.val[3]);

    vb.val[0] = v1.val[0];
    vb.val[1] = v1.val[1];
    vb.val[2] = v1.val[2];
    vb.val[3] = v1.val[3];

    // 32 + 96
    vadd8(v1.val[0], vb.val[0], v3.val[0]);
    vadd8(v1.val[1], vb.val[1], v3.val[1]);
    vadd8(v1.val[2], vb.val[2], v3.val[2]);
    vadd8(v1.val[3], vb.val[3], v3.val[3]);
    // 32 - 96
    vsub8(v3.val[0], vb.val[0], v3.val[0]);
    vsub8(v3.val[1], vb.val[1], v3.val[1]);
    vsub8(v3.val[2], vb.val[2], v3.val[2]);
    vsub8(v3.val[3], vb.val[3], v3.val[3]);

    vlo(a_lo, v0.val[1]);
    vhi(a_hi, v0.val[1]);
    vlo(b_lo, v1.val[1]);
    vhi(b_hi, v1.val[1]);
    
    barrett(a_lo, t1, zlo, neon_v, neon_kyberq16);
    barrett(a_hi, t2, zhi, neon_v, neon_kyberq16);
    barrett(b_lo, t3, zlo, neon_v, neon_kyberq16);
    barrett(b_hi, t1, zhi, neon_v, neon_kyberq16);
    
    vcombine(v0.val[1], a_lo, a_hi);
    vcombine(v1.val[1], b_lo, b_hi);
    
    l0 = vdupq_n_s16(zetas_inv[k6++]);

    fqmul(v2.val[0], v2.val[0], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v2.val[1], v2.val[1], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v2.val[2], v2.val[2], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v2.val[3], v2.val[3], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    fqmul(v3.val[0], v3.val[0], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v3.val[1], v3.val[1], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v3.val[2], v3.val[2], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v3.val[3], v3.val[3], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    vstorex4(&r[j + 0], v0);
    vstorex4(&r[j + 32], v1);
    vstorex4(&r[j + 64], v2);
    vstorex4(&r[j + 96], v3);
  }

  // Layer 7, inv_mul
  l0 = vdupq_n_s16(zetas_inv[126]);
  l1 = vdupq_n_s16(zetas_inv[127]);
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

    va.val[0] = v0.val[0];
    va.val[1] = v0.val[1];
    va.val[2] = v0.val[2];
    va.val[3] = v0.val[3];

    // 0 + 128
    vadd8(v0.val[0], va.val[0], v2.val[0]);
    vadd8(v0.val[1], va.val[1], v2.val[1]);
    vadd8(v0.val[2], va.val[2], v2.val[2]);
    vadd8(v0.val[3], va.val[3], v2.val[3]);
    // 0 - 128
    vsub8(v2.val[0], va.val[0], v2.val[0]);
    vsub8(v2.val[1], va.val[1], v2.val[1]);
    vsub8(v2.val[2], va.val[2], v2.val[2]);
    vsub8(v2.val[3], va.val[3], v2.val[3]);

    vb.val[0] = v1.val[0];
    vb.val[1] = v1.val[1];
    vb.val[2] = v1.val[2];
    vb.val[3] = v1.val[3];

    // 32 + 160
    vadd8(v1.val[0], vb.val[0], v3.val[0]);
    vadd8(v1.val[1], vb.val[1], v3.val[1]);
    vadd8(v1.val[2], vb.val[2], v3.val[2]);
    vadd8(v1.val[3], vb.val[3], v3.val[3]);
    // 32 - 160
    vsub8(v3.val[0], vb.val[0], v3.val[0]);
    vsub8(v3.val[1], vb.val[1], v3.val[1]);
    vsub8(v3.val[2], vb.val[2], v3.val[2]);
    vsub8(v3.val[3], vb.val[3], v3.val[3]);

    // After layer 7, no need for barrett_reduction

    fqmul(v2.val[0], v2.val[0], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v2.val[0], v2.val[0], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v2.val[1], v2.val[1], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v2.val[1], v2.val[1], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    fqmul(v2.val[2], v2.val[2], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v2.val[2], v2.val[2], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v2.val[3], v2.val[3], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v2.val[3], v2.val[3], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
  
    fqmul(v3.val[0], v3.val[0], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v3.val[0], v3.val[0], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v3.val[1], v3.val[1], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v3.val[1], v3.val[1], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    fqmul(v3.val[2], v3.val[2], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v3.val[2], v3.val[2], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v3.val[3], v3.val[3], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v3.val[3], v3.val[3], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    
    // v0
    fqmul(v0.val[0], v0.val[0], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v0.val[1], v0.val[1], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v0.val[2], v0.val[2], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v0.val[3], v0.val[3], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    fqmul(v1.val[0], v1.val[0], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v1.val[1], v1.val[1], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v1.val[2], v1.val[2], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(v1.val[3], v1.val[3], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);

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
  int16x8x4_t va, vb, v0, v1, v2, v3; // 20
  int16x8_t r0, r1, r2, r3,
      l0, l1, l2, l3;               // 8
  int16x4_t a_lo, a_hi, b_lo, b_hi; // 6
  int32x4_t t1, t2, t3, t4;         // 6
  // Constant: Total 2
  int32x4_t neon_qinv, neon_kyberq;
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
  l0 = vdupq_n_s16(zetas[1]);
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

    fqmul(va.val[0], v2.val[0], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(va.val[1], v2.val[1], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(va.val[2], v2.val[2], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(va.val[3], v2.val[3], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    fqmul(vb.val[0], v3.val[0], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(vb.val[1], v3.val[1], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(vb.val[2], v3.val[2], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(vb.val[3], v3.val[3], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);


    // 128: 0 - 128
    vsub8(v2.val[0], v0.val[0], va.val[0]);
    vsub8(v2.val[1], v0.val[1], va.val[1]);
    vsub8(v2.val[2], v0.val[2], va.val[2]);
    vsub8(v2.val[3], v0.val[3], va.val[3]);

    //   0: 0 + 128
    vadd8(v0.val[0], v0.val[0], va.val[0]);
    vadd8(v0.val[1], v0.val[1], va.val[1]);
    vadd8(v0.val[2], v0.val[2], va.val[2]);
    vadd8(v0.val[3], v0.val[3], va.val[3]);

    // 160: 32 - 160
    vsub8(v3.val[0], v1.val[0], vb.val[0]);
    vsub8(v3.val[1], v1.val[1], vb.val[1]);
    vsub8(v3.val[2], v1.val[2], vb.val[2]);
    vsub8(v3.val[3], v1.val[3], vb.val[3]);

    //  32:  32 + 160
    vadd8(v1.val[0], v1.val[0], vb.val[0]);
    vadd8(v1.val[1], v1.val[1], vb.val[1]);
    vadd8(v1.val[2], v1.val[2], vb.val[2]);
    vadd8(v1.val[3], v1.val[3], vb.val[3]);

    vstorex4(&r[j + 0], v0);
    vstorex4(&r[j + 32], v1);
    vstorex4(&r[j + 128], v2);
    vstorex4(&r[j + 160], v3);
  }

  // Layer 6, 5
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

    l0 = vdupq_n_s16(zetas[k6++]);

    fqmul(va.val[0], v2.val[0], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(va.val[1], v2.val[1], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(va.val[2], v2.val[2], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(va.val[3], v2.val[3], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    fqmul(vb.val[0], v3.val[0], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(vb.val[1], v3.val[1], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(vb.val[2], v3.val[2], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(vb.val[3], v3.val[3], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);


    // 64: 0 - 64
    vsub8(v2.val[0], v0.val[0], va.val[0]);
    vsub8(v2.val[1], v0.val[1], va.val[1]);
    vsub8(v2.val[2], v0.val[2], va.val[2]);
    vsub8(v2.val[3], v0.val[3], va.val[3]);

    //  0: 0 + 64
    vadd8(v0.val[0], v0.val[0], va.val[0]);
    vadd8(v0.val[1], v0.val[1], va.val[1]);
    vadd8(v0.val[2], v0.val[2], va.val[2]);
    vadd8(v0.val[3], v0.val[3], va.val[3]);

    // 96: 32 - 96
    vsub8(v3.val[0], v1.val[0], vb.val[0]);
    vsub8(v3.val[1], v1.val[1], vb.val[1]);
    vsub8(v3.val[2], v1.val[2], vb.val[2]);
    vsub8(v3.val[3], v1.val[3], vb.val[3]);

    // 32:  32 + 96
    vadd8(v1.val[0], v1.val[0], vb.val[0]);
    vadd8(v1.val[1], v1.val[1], vb.val[1]);
    vadd8(v1.val[2], v1.val[2], vb.val[2]);
    vadd8(v1.val[3], v1.val[3], vb.val[3]);

    // Layer 5: v0 x v1 | v2 x v3
    // v0: 0   -> 31
    // v1: 32  -> 63
    // v2: 64  -> 95
    // v3: 96  -> 127

    l0 = vdupq_n_s16(zetas[k5++]);
    l1 = vdupq_n_s16(zetas[k5++]);

    fqmul(va.val[0], v1.val[0], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(va.val[1], v1.val[1], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(va.val[2], v1.val[2], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(va.val[3], v1.val[3], l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    fqmul(vb.val[0], v3.val[0], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(vb.val[1], v3.val[1], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(vb.val[2], v3.val[2], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(vb.val[3], v3.val[3], l1, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    // 32: 0 - 32
    vsub8(v1.val[0], v0.val[0], va.val[0]);
    vsub8(v1.val[1], v0.val[1], va.val[1]);
    vsub8(v1.val[2], v0.val[2], va.val[2]);
    vsub8(v1.val[3], v0.val[3], va.val[3]);

    //  0: 0 + 32
    vadd8(v0.val[0], v0.val[0], va.val[0]);
    vadd8(v0.val[1], v0.val[1], va.val[1]);
    vadd8(v0.val[2], v0.val[2], va.val[2]);
    vadd8(v0.val[3], v0.val[3], va.val[3]);

    // 96: 64 - 96
    vsub8(v3.val[0], v2.val[0], vb.val[0]);
    vsub8(v3.val[1], v2.val[1], vb.val[1]);
    vsub8(v3.val[2], v2.val[2], vb.val[2]);
    vsub8(v3.val[3], v2.val[3], vb.val[3]);

    //  64: 64 + 96
    vadd8(v2.val[0], v2.val[0], vb.val[0]);
    vadd8(v2.val[1], v2.val[1], vb.val[1]);
    vadd8(v2.val[2], v2.val[2], vb.val[2]);
    vadd8(v2.val[3], v2.val[3], vb.val[3]);

    vstorex4(&r[j + 0], v0);
    vstorex4(&r[j + 32], v1);
    vstorex4(&r[j + 64], v2);
    vstorex4(&r[j + 96], v3);
  }

  // Layer 4, 3, 2, 1
  for (j = 0; j < 256; j += 32)
  {
    vloadx4(v0, &r[j]);

    r0 = v0.val[0];
    r1 = v0.val[1];
    r2 = v0.val[2];
    r3 = v0.val[3];

    // Layer 4: r0 x r2 | r1 x r3
    // r0: 0  -> 7
    // r1: 8  -> 15
    // r2: 16 -> 23
    // r3: 24 -> 32
    l0 = vdupq_n_s16(zetas[k4++]);

    fqmul(l2, r2, l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(l3, r3, l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    vsub8(r2, r0, l2);
    vadd8(r0, r0, l2);

    vsub8(r3, r1, l3);
    vadd8(r1, r1, l3);

    // Layer 3: r0 x r1 | r2 x r3
    // r0: 0  -> 7
    // r1: 8  -> 15
    // r2: 16 -> 23
    // r3: 24 -> 32
    l0 = vdupq_n_s16(zetas[k3++]);
    l2 = vdupq_n_s16(zetas[k3++]);
  
    fqmul(l1, r1, l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(l3, r3, l2, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    vsub8(r1, r0, l1);
    vadd8(r0, r0, l1);

    vsub8(r3, r2, l3);
    vadd8(r2, r2, l3);

    // Layer 2: l0 x l1   | l2 x l3
    // r0: 0,  1,  2,  3  | 4,  5,  6,  7
    // r1: 8,  9,  10, 11 | 12, 13, 14, 15
    // r2: 16, 17, 18, 19 | 20, 21, 22, 23
    // r3: 24, 25, 26, 27 | 28, 29, 30, 31
    // Swap (r0, r2) and (r1, r3)
    l0 = (int16x8_t)vtrn1q_s64((int64x2_t)r0, (int64x2_t)r2);
    l1 = (int16x8_t)vtrn2q_s64((int64x2_t)r0, (int64x2_t)r2);
    l2 = (int16x8_t)vtrn1q_s64((int64x2_t)r1, (int64x2_t)r3);
    l3 = (int16x8_t)vtrn2q_s64((int64x2_t)r1, (int64x2_t)r3);
    // l0: 0,  1,  2,  3  | 16, 17, 18, 19
    // l1: 4,  5,  6,  7  | 20, 21, 22, 23
    // l2: 8,  9,  10, 11 | 24, 25, 26, 27
    // l3: 12, 13, 14, 15 | 28, 29, 30, 31

    vdup(a_lo, zetas[k2]);
    vdup(b_lo, zetas[k2+1]);
    vdup(a_hi, zetas[k2+2]);
    vdup(b_hi, zetas[k2+3]);
    vcombine(r0, a_lo, a_hi);
    vcombine(r2, b_lo, b_hi);

    fqmul(l1, l1, r0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(l3, l3, r2, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    k2 += 4;

    vsub8(r1, l0, l1);
    vadd8(r0, l0, l1);

    vsub8(r3, l2, l3);
    vadd8(r2, l2, l3);

    // Layer 1: r0 x r2 | r1 x r3
    // r0: 0,  1,  2,  3  | 16, 17, 18, 19
    // r1: 4,  5,  6,  7  | 20, 21, 22, 23
    // r2: 8,  9,  10, 11 | 24, 25, 26, 27
    // r3: 12, 13, 14, 15 | 28, 29, 30, 31
    // Tranpose 4x4
    l0 = vtrn1q_s16(r0, r1);
    l1 = vtrn2q_s16(r0, r1);
    l2 = vtrn1q_s16(r2, r3);
    l3 = vtrn2q_s16(r2, r3);
    r0 = (int16x8_t)vtrn1q_s32((int32x4_t)l0, (int32x4_t)l2);
    r2 = (int16x8_t)vtrn2q_s32((int32x4_t)l0, (int32x4_t)l2);
    r1 = (int16x8_t)vtrn1q_s32((int32x4_t)l1, (int32x4_t)l3);
    r3 = (int16x8_t)vtrn2q_s32((int32x4_t)l1, (int32x4_t)l3);
    // r0: 0, 4, 8,  12 | 16, 20, 24, 28
    // r1: 1, 5, 9,  13 | 17, 21, 25, 29
    // r2: 2, 6, 10, 14 | 18, 22, 26, 30
    // r3: 3, 7, 11, 15 | 19, 23, 27, 31

    vlo(a_lo, r2);
    vhi(a_hi, r2);

    vlo(b_lo, r3);
    vhi(b_hi, r3);

    vload(l0, &zetas[k1]);
    k1 += 8;

    fqmul(l2, r2, l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);
    fqmul(l3, r3, l0, t1, t2, t3, t4, neon_qinv, neon_kyberq);

    vsub8(r2, r0, l2);
    vadd8(r0, r0, l2);

    vsub8(r3, r1, l3);
    vadd8(r1, r1, l3);

    v0.val[0] = r0;
    v0.val[1] = r1;
    v0.val[2] = r2;
    v0.val[3] = r3;

    vstore4(&r[j], v0);
  }
}
