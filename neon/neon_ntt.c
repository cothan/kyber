#include <arm_neon.h>
#include "params.h"
#include "ntt.h"
#include "reduce.h"

/******8888888888888888888888888***************/

// Load int16x8_t c <= ptr*
#define vload8(c, ptr) c = vld1q_s16(ptr);

// Store *ptr <= c
#define vstore8(ptr, c) vst1q_s16(ptr, c);

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

inout: input/output : int16x4_t
zeta: input : int16x4_t
t: temp : int32x4_t
a: temp : int32x4_t
u: temp : int32x4_t
neon_qinv: const   : int32x4_t
neon_kyberq: const : int32x4_t

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
#define fqmul(inout, zeta, t, a, u, neon_qinv, neon_kyberq)                    \
  a = vmull_s16(inout, zeta);                                                  \
  u = vmulq_s32(a, neon_qinv);                                                 \
  u = vshrq_n_s32(u, 16);                                                      \
  t = vmlaq_s32(a, neon_kyberq, u);                                            \
  t = vshrq_n_s32(t, 16);                                                      \
  inout = vmovn_s32(t);

/*
v: int16x8_t
t : int16x8_t
t_lo: int16x4_t
t_hi: int16x4_t
t1: int32x4_t
t2: int32x4_t
temp: int16x8_t
a: inout int16x8_t

rewrite pseudo code:
int16_t barrett_reduce(int16_t a) {
  int16_t t;
  const int16_t v = ((1U << 26) + KYBER_Q / 2) / KYBER_Q;

  t = (int32_t)v * a >> 26;
  t = a + t * (-KYBER_Q);
  return t;
}
*/
#define barrett(inout, t, t_lo, t_hi, t1, t2, neon_v, neon_kyberq16)           \
  vlo(t_lo, inout);                                                            \
  vhi(t_hi, inout);                                                            \
  t1 = vmull_s16(t_lo, neon_v);                                                \
  t2 = vmull_s16(t_hi, neon_v);                                                \
  t1 = vshrq_n_s32(t1, 26);                                                    \
  t2 = vshrq_n_s32(t2, 26);                                                    \
  t_lo = vmovn_s32(t1);                                                        \
  t_hi = vmovn_s32(t2);                                                        \
  vcombine(t, t_lo, t_hi);                                                     \
  inout = vmlaq_s16(inout, t, neon_kyberq16);

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
void neon_invntt(int16_t r[256]) {
  // NEON Registers
  int16x8_t a, b, c, d, at, bt, ct, dt, neon_zetas, neon_kyberq16;  // 9
  int16x4_t a_lo, a_hi, b_lo, b_hi, c_lo, c_hi, d_lo, d_hi;         // 8
  int16x4_t neon_zeta1, neon_zeta2, neon_zeta3, neon_zeta4, neon_v; // 4
  int32x4_t t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc;         // 12
  int32x4_t neon_qinv, neon_kyberq;                                 // 2
  int16x8x4_t ab;                                                   // 4
  // End
  neon_qinv = vdupq_n_s32(QINV << 16);
  neon_kyberq = vdupq_n_s32(-KYBER_Q);
  neon_kyberq16 = vdupq_n_s16(-KYBER_Q);
  neon_v = vdup_n_s16(((1U << 26) + KYBER_Q / 2) / KYBER_Q);
  // Scalar variable
  uint16_t start, len, j, k;
  // End

  k = 0;

  //   Layer 1
  for (j = 0; j < 256; j += 32) {
    // ab.val[0] = 0, 4, 8, 12, 16, 20, 24, 28
    // ab.val[1] = 1, 5, 9, 13, 17, 21, 25, 29
    // ab.val[2] = 2, 6, 10, 14, 18, 22, 26, 30
    // al.val[3] = 3, 7, 11, 15, 19, 23, 27, 31

    vload8(neon_zetas, &zetas_inv[k]);
    vlo(neon_zeta1, neon_zetas);
    vhi(neon_zeta2, neon_zetas);
    //
    ab = vld4q_s16(&r[j]);

    at = ab.val[0];

    // 0: 0 + 2
    vadd8(ab.val[0], at, ab.val[2]);
    // 2: 0 - 2
    vsub8(ab.val[2], at, ab.val[2]);

    // a_lo
    vlo(a_lo, ab.val[2]);
    vhi(a_hi, ab.val[2]);

    fqmul(a_lo, neon_zeta1, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(a_hi, neon_zeta2, t6, t7, t8, neon_qinv, neon_kyberq);

    barrett(ab.val[0], ct, c_lo, c_hi, t1, t2, neon_v, neon_kyberq16);

    vcombine(ab.val[2], a_lo, a_hi);
    //

    bt = ab.val[1];
    // 1: 1 + 3
    vadd8(ab.val[1], bt, ab.val[3]);
    // 3: 1 - 3
    vsub8(ab.val[3], bt, ab.val[3]);

    vlo(b_lo, ab.val[3]);
    vhi(b_hi, ab.val[3]);

    fqmul(b_lo, neon_zeta1, t4, t5, t6, neon_qinv, neon_kyberq);
    fqmul(b_hi, neon_zeta2, t9, ta, tb, neon_qinv, neon_kyberq);

    barrett(ab.val[1], dt, d_lo, d_hi, t1, t2, neon_v, neon_kyberq16);

    vcombine(ab.val[3], b_lo, b_hi);

    vst4q_s16(&r[j], ab);
    //
    k += 8;
  }

  //   Layer 2
  for (j = 0; j < 256; j += 32) {
    vdup(neon_zeta1, zetas_inv[k++]);
    vdup(neon_zeta2, zetas_inv[k++]);
    //
    //   a_lo - a_hi,
    //   b_lo - b_hi,
    //   c_lo - c_hi,
    //   d_lo - d_hi

    // a: 0, 1, 2, 3, | 4, 5, 6, 7
    // b: 8, 9, 10, 11 | 12, 13, 14, 15
    // c: 16, 17, 18, 19 | 20, 21, 22, 23
    // d: 24, 25, 26, 27 | 28, 29, 30, 31
    vload8(a, &r[j]);
    vload8(b, &r[j + 8]);

    vlo(a_lo, a);
    vhi(a_hi, a);
    //
    vlo(b_lo, b);
    vhi(b_hi, b);

    c_lo = a_lo;
    d_lo = b_lo;
    // 0 + 4
    vadd4(a_lo, c_lo, a_hi);
    // 0 - 4
    vsub4(a_hi, c_lo, a_hi);
    // 8 + 12
    vadd4(b_lo, d_lo, b_hi);
    // 8 - 12
    vsub4(b_hi, d_lo, b_hi);

    fqmul(a_hi, neon_zeta1, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_hi, neon_zeta2, t4, t5, t6, neon_qinv, neon_kyberq);

    vcombine(at, a_lo, b_lo);
    barrett(at, bt, c_hi, d_hi, ta, tb, neon_v, neon_kyberq16);
    vlo(a_lo, at);
    vhi(b_lo, at);

    vcombine(a, a_lo, a_hi);
    vcombine(b, b_lo, b_hi);

    vstore8(&r[j], a);
    vstore8(&r[j + 8], b);

    vdup(neon_zeta3, zetas_inv[k++]);
    vdup(neon_zeta4, zetas_inv[k++]);

    vload8(c, &r[j + 16]);
    vload8(d, &r[j + 24]);
    //
    vlo(c_lo, c);
    vhi(c_hi, c);
    //
    vlo(d_lo, d);
    vhi(d_hi, d);

    a_lo = c_lo;
    b_lo = d_lo;
    // 16 + 20
    vadd4(c_lo, a_lo, c_hi);
    // 16 - 20
    vsub4(c_hi, a_lo, c_hi);
    // 24 + 28
    vadd4(d_lo, b_lo, d_hi);
    // 24 - 28
    vsub4(d_hi, b_lo, d_hi);

    fqmul(c_hi, neon_zeta3, t6, t7, t8, neon_qinv, neon_kyberq);
    fqmul(d_hi, neon_zeta4, t9, ta, tb, neon_qinv, neon_kyberq);

    vcombine(ct, c_lo, d_lo);
    barrett(ct, dt, a_hi, b_hi, t1, t2, neon_v, neon_kyberq16);
    vlo(c_lo, ct);
    vhi(d_lo, ct);

    vcombine(c, c_lo, c_hi);
    vcombine(d, d_lo, d_hi);

    vstore8(&r[j + 16], c);
    vstore8(&r[j + 24], d);
  }

  //   Layer 3
  for (j = 0; j < 256; j += 32) {
    //   a - b, c - d
    vdup(neon_zeta1, zetas_inv[k++]);
    //
    vload8(a, &r[j]);
    vload8(b, &r[j + 8]);

    c = a;
    //  a : c(a) + b
    vadd8(a, c, b);
    //  b : c(a) - b
    vsub8(b, c, b);

    //
    vlo(b_lo, b);
    vhi(b_hi, b);
    //
    fqmul(b_lo, neon_zeta1, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_hi, neon_zeta1, t4, t5, t6, neon_qinv, neon_kyberq);

    barrett(a, ct, c_lo, c_hi, t1, t2, neon_v, neon_kyberq16);

    vcombine(b, b_lo, b_hi);

    vstore8(&r[j], a);
    vstore8(&r[j + 8], b);
    //
    vdup(neon_zeta2, zetas_inv[k++]);
    //
    vload8(c, &r[j + 16]);
    vload8(d, &r[j + 24]);
    //
    a = c;
    // c: a(c) + d
    vadd8(c, a, d);
    // d: a(c) - d
    vsub8(d, a, d);
    //
    vlo(d_lo, d);
    vhi(d_hi, d);

    fqmul(d_lo, neon_zeta2, t7, t8, t9, neon_qinv, neon_kyberq);
    fqmul(d_hi, neon_zeta2, ta, tb, tc, neon_qinv, neon_kyberq);

    barrett(c, at, a_lo, a_hi, ta, tb, neon_v, neon_kyberq16);

    vcombine(d, d_lo, d_hi);

    vstore8(&r[j + 16], c);
    vstore8(&r[j + 24], d);
  }

  // Layer 4, 5, 6, 7
  for (len = 16; len <= 128; len <<= 1) {
    for (start = 0; start < 256; start = j + len) {
      vdup(neon_zeta1, zetas_inv[k++]);
      for (j = start; j < start + len; j += 16) {
        //   a - c, b - d
        vload8(a, &r[j]);
        vload8(b, &r[j + 8]);
        vload8(c, &r[j + len]);
        vload8(d, &r[j + len + 8]);

        at = a;
        // a = at(a) + c
        vadd8(a, at, c);
        // c = at(a) - c
        vsub8(c, at, c);
        //
        vlo(c_lo, c);
        vhi(c_hi, c);

        fqmul(c_lo, neon_zeta1, t1, t2, t3, neon_qinv, neon_kyberq);
        fqmul(c_hi, neon_zeta1, t4, t5, t6, neon_qinv, neon_kyberq);

        barrett(a, bt, b_lo, b_hi, t1, t2, neon_v, neon_kyberq16);

        vcombine(c, c_lo, c_hi);

        bt = b;
        // b = bt(b) + d
        vadd8(b, bt, d);
        // d = bt(b) - d
        vsub8(d, bt, d);

        //
        vlo(d_lo, d);
        vhi(d_hi, d);

        //
        fqmul(d_lo, neon_zeta1, t7, t8, t9, neon_qinv, neon_kyberq);
        fqmul(d_hi, neon_zeta1, ta, tb, tc, neon_qinv, neon_kyberq);

        barrett(b, at, a_lo, a_hi, ta, tb, neon_v, neon_kyberq16);

        vcombine(d, d_lo, d_hi);

        vstore8(&r[j], a);
        vstore8(&r[j + 8], b);
        vstore8(&r[j + len], c);
        vstore8(&r[j + len + 8], d);
      }
    }
  }

  vdup(neon_zeta1, zetas_inv[127]);
  for (j = 0; j < 256; j += 32) {
    ab = vld1q_s16_x4(&r[j]);

    vlo(a_lo, ab.val[0]);
    vhi(a_hi, ab.val[0]);
    vlo(b_lo, ab.val[1]);
    vhi(b_hi, ab.val[1]);
    vlo(c_lo, ab.val[2]);
    vhi(c_hi, ab.val[2]);
    vlo(d_lo, ab.val[3]);
    vhi(d_hi, ab.val[3]);

    fqmul(a_lo, neon_zeta1, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_lo, neon_zeta1, t4, t5, t6, neon_qinv, neon_kyberq);
    fqmul(c_lo, neon_zeta1, t7, t8, t9, neon_qinv, neon_kyberq);
    fqmul(d_lo, neon_zeta1, ta, tb, tc, neon_qinv, neon_kyberq);
    fqmul(a_hi, neon_zeta1, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_hi, neon_zeta1, t4, t5, t6, neon_qinv, neon_kyberq);
    fqmul(c_hi, neon_zeta1, t7, t8, t9, neon_qinv, neon_kyberq);
    fqmul(d_hi, neon_zeta1, ta, tb, tc, neon_qinv, neon_kyberq);

    vcombine(ab.val[0], a_lo, a_hi);
    vcombine(ab.val[1], b_lo, b_hi);
    vcombine(ab.val[2], c_lo, c_hi);
    vcombine(ab.val[3], d_lo, d_hi);

    vst1q_s16_x4(&r[j], ab);
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
void neon_ntt(int16_t r[256]) {
  // NEON Registers
  int16x8_t a, b, c, d, at, bt, ct, dt, neon_zetas;         // 9
  int16x4_t a_lo, a_hi, b_lo, b_hi, c_lo, c_hi, d_lo, d_hi; // 8
  int16x4_t neon_zeta1, neon_zeta2, neon_zeta3, neon_zeta4; // 4
  int32x4_t t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc; // 12
  int32x4_t neon_qinv, neon_kyberq;                         // 2
  int16x8x4_t ab;                                           // 4
  // End
  neon_qinv = vdupq_n_s32(QINV << 16);
  neon_kyberq = vdupq_n_s32(-KYBER_Q);

  // Scalar variable
  uint16_t start, len, j, k;
  // End

  // Layer 7, 6, 5, 4
  k = 1;
  for (len = 128; len >= 16; len >>= 1) {
    for (start = 0; start < 256; start = j + len) {
      vdup(neon_zeta1, zetas[k++]);
      for (j = start; j < start + len; j += 16) {
        //   a - c, b - d
        vload8(a, &r[j]);
        vload8(b, &r[j + 8]);

        vload8(c, &r[j + len]);
        vload8(d, &r[j + len + 8]);

        vlo(c_lo, c);
        vhi(c_hi, c);
        //
        vlo(d_lo, d);
        vhi(d_hi, d);

        fqmul(c_lo, neon_zeta1, t1, t2, t3, neon_qinv, neon_kyberq);
        fqmul(c_hi, neon_zeta1, t4, t5, t6, neon_qinv, neon_kyberq);
        //
        fqmul(d_lo, neon_zeta1, t7, t8, t9, neon_qinv, neon_kyberq);
        fqmul(d_hi, neon_zeta1, ta, tb, tc, neon_qinv, neon_kyberq);

        vcombine(ct, c_lo, c_hi);
        vcombine(dt, d_lo, d_hi);

        vsub8(c, a, ct);
        vadd8(a, a, ct);
        //
        vsub8(d, b, dt);
        vadd8(b, b, dt);

        vstore8(&r[j], a);
        vstore8(&r[j + 8], b);

        vstore8(&r[j + len], c);
        vstore8(&r[j + len + 8], d);
      }
    }
  }

  //   Layer 3
  for (j = 0; j < 256; j += 32) {
    //   a - b, c - d
    vload8(a, &r[j]);
    vload8(b, &r[j + 8]);
    //
    vload8(c, &r[j + 16]);
    vload8(d, &r[j + 24]);

    //
    vlo(b_lo, b);
    vhi(b_hi, b);
    //
    vlo(d_lo, d);
    vhi(d_hi, d);

    vdup(neon_zeta1, zetas[k++]);
    vdup(neon_zeta2, zetas[k++]);

    fqmul(b_lo, neon_zeta1, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_hi, neon_zeta1, t4, t5, t6, neon_qinv, neon_kyberq);
    //
    fqmul(d_lo, neon_zeta2, t7, t8, t9, neon_qinv, neon_kyberq);
    fqmul(d_hi, neon_zeta2, ta, tb, tc, neon_qinv, neon_kyberq);

    vcombine(bt, b_lo, b_hi);
    vcombine(dt, d_lo, d_hi);

    vsub8(b, a, bt);
    vadd8(a, a, bt);
    //
    vsub8(d, c, dt);
    vadd8(c, c, dt);

    vstore8(&r[j], a);
    vstore8(&r[j + 8], b);
    vstore8(&r[j + 16], c);
    vstore8(&r[j + 24], d);
  }

  // Layer 2
  for (j = 0; j < 256; j += 32) {
    //   a_lo - a_hi,
    //   b_lo - b_hi,
    //   c_lo - c_hi,
    //   d_lo - d_hi

    // a: 0, 1, 2, 3, | 4, 5, 6, 7
    // b: 8, 9, 10, 11 | 12, 13, 14, 15
    // c: 16, 17, 18, 19 | 20, 21, 22, 23
    // d: 24, 25, 26, 27 | 28, 29, 30, 31
    vload8(a, &r[j]);
    vload8(b, &r[j + 8]);
    vload8(c, &r[j + 16]);
    vload8(d, &r[j + 24]);

    vlo(a_lo, a);
    vhi(a_hi, a);
    //
    vlo(b_lo, b);
    vhi(b_hi, b);

    vdup(neon_zeta1, zetas[k++]);
    vdup(neon_zeta2, zetas[k++]);

    fqmul(a_hi, neon_zeta1, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_hi, neon_zeta2, t4, t5, t6, neon_qinv, neon_kyberq);

    vsub4(c_hi, a_lo, a_hi);
    vadd4(a_lo, a_lo, a_hi);
    vcombine(a, a_lo, c_hi);

    vsub4(d_hi, b_lo, b_hi);
    vadd4(b_lo, b_lo, b_hi);
    vcombine(b, b_lo, d_hi);

    //
    vlo(c_lo, c);
    vhi(c_hi, c);
    //
    vlo(d_lo, d);
    vhi(d_hi, d);

    vdup(neon_zeta3, zetas[k++]);
    vdup(neon_zeta4, zetas[k++]);

    fqmul(c_hi, neon_zeta3, t6, t7, t8, neon_qinv, neon_kyberq);
    fqmul(d_hi, neon_zeta4, t9, ta, tb, neon_qinv, neon_kyberq);

    vsub4(a_hi, c_lo, c_hi);
    vadd4(c_lo, c_lo, c_hi);
    vcombine(c, c_lo, a_hi);

    vsub4(b_hi, d_lo, d_hi);
    vadd4(d_lo, d_lo, d_hi);
    vcombine(d, d_lo, b_hi);

    vstore8(&r[j], a);
    vstore8(&r[j + 8], b);
    vstore8(&r[j + 16], c);
    vstore8(&r[j + 24], d);
  }

  //   Layer 1
  for (j = 0; j < 256; j += 32) {
    // ab.val[0] = 0, 4, 8, 12, 16, 20, 24, 28
    // ab.val[1] = 1, 5, 9, 13, 17, 21, 25, 29
    // ab.val[2] = 2, 6, 10, 14, 18, 22, 26, 30
    // al.val[3] = 3, 7, 11, 15, 19, 23, 27, 31

    ab = vld4q_s16(&r[j]);

    // a_lo
    vlo(a_lo, ab.val[2]);
    vhi(a_hi, ab.val[2]);
    vlo(b_lo, ab.val[3]);
    vhi(b_hi, ab.val[3]);

    vload8(neon_zetas, &zetas[k]);

    vlo(neon_zeta1, neon_zetas);
    vhi(neon_zeta2, neon_zetas);

    fqmul(a_lo, neon_zeta1, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_lo, neon_zeta1, t4, t5, t6, neon_qinv, neon_kyberq);

    fqmul(a_hi, neon_zeta2, t6, t7, t8, neon_qinv, neon_kyberq);
    fqmul(b_hi, neon_zeta2, t9, ta, tb, neon_qinv, neon_kyberq);

    vcombine(at, a_lo, a_hi);
    vcombine(bt, b_lo, b_hi);

    // 2: 0 - 2
    vsub8(ab.val[2], ab.val[0], at);
    // 0: 0 + 2
    vadd8(ab.val[0], ab.val[0], at);

    // 3: 1 - 3
    vsub8(ab.val[3], ab.val[1], bt);
    // 1: 1 + 3
    vadd8(ab.val[1], ab.val[1], bt);

    k += 8;
    //
    vst4q_s16(&r[j], ab);
  }
}