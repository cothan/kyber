#include <arm_neon.h>
#include "params.h"
#include "ntt.h"
#include "reduce.h"
#include <stdio.h>


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
#define fqmul(inout, zeta, a, u, t, neon_qinv, neon_kyberq) \
  a = vmull_s16(inout, zeta);                               \
  u = vmulq_s32(a, neon_qinv);                              \
  u = vshrq_n_s32(u, 16);                                   \
  t = vmlaq_s32(a, neon_kyberq, u);                         \
  t = vshrq_n_s32(t, 16);                                   \
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
  int16x8x2_t aa, bb;                                               // 2
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
    vload(neon_zetas, &zetas_inv[k]);
    vlo(neon_zeta1, neon_zetas);
    vhi(neon_zeta2, neon_zetas);
    //
    vload4(ab, &r[j]);

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

    vstore4(&r[j], ab);

    k += 8;
  }
  printf("1 = [");
  for (j = 0; j < 32; j++)
  {
    printf("%d, ", r[j]);
  }
  printf("]\n");

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
    vloadx2(aa, &r[j]);
    a = aa.val[0];
    b = aa.val[1];

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

    aa.val[0] = a;
    aa.val[1] = b;
    vstorex2(&r[j], aa);

    vdup(neon_zeta3, zetas_inv[k++]);
    vdup(neon_zeta4, zetas_inv[k++]);

    vloadx2(bb, &r[j+16]);
    c = bb.val[0];
    d = bb.val[1];
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

    bb.val[0] = c;
    bb.val[1] = d;
    vstorex2(&r[j+16], bb);
  }

  printf("2 = [");
  for (j = 0; j < 32; j++)
  {
    printf("%d, ", r[j]);
  }
  printf("]\n");

  //   Layer 3
  for (j = 0; j < 256; j += 32) {
    //   a - b, c - d
    vdup(neon_zeta1, zetas_inv[k++]);
    //
    vloadx2(aa, &r[j]);
    a = aa.val[0];
    b = aa.val[1];

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

    aa.val[0] = a;
    aa.val[1] = b;
    vstorex2(&r[j], aa);
    //
    vdup(neon_zeta2, zetas_inv[k++]);
    //
    vloadx2(bb, &r[j + 16]);
    c = bb.val[0];
    d = bb.val[1];
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

    bb.val[0] = c;
    bb.val[1] = d;
    vstorex2(&r[j + 16], bb);
  }

  printf("3 = [");
  for (j = 0; j < 32; j++)
  {
    printf("%d, ", r[j]);
  }
  printf("]\n");

  // Layer 4, 5, 6, 7
  for (len = 16; len <= 64; len <<= 1) {
    printf("len = %d\n", len);
    for (start = 0; start < 256; start = j + len) {
      printf("zetas_inv: %d :%d\n", k, zetas_inv[k]);
      vdup(neon_zeta1, zetas_inv[k++]);
      for (j = start; j < start + len; j += 16) {
        //   a - c, b - d
        vloadx2(aa, &r[j]);
        a = aa.val[0];
        b = aa.val[1];
        vloadx2(bb, &r[j + len]);
        c = bb.val[0];
        d = bb.val[1];

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

        aa.val[0] = a;
        aa.val[1] = b;
        vstorex2(&r[j], aa);
        bb.val[0] = c;
        bb.val[1] = d;
        vstorex2(&r[j + len], bb);
      }
    }
    printf("%d = [", len);
    for (j = 0; j < 128; j++)
    {
      printf("%d, ", r[j]);
    }
    printf("]\n");
  }

  // printf("4 = [");
  // for (j = 0; j < 32; j++)
  // {
  //   printf("%d, ", r[j]);
  // }
  // printf("]\n");

  // vdup(neon_zeta1, zetas_inv[127]);
  // for (j = 0; j < 256; j += 32) {
  //   vloadx4(ab, &r[j]);

  //   vlo(a_lo, ab.val[0]);
  //   vhi(a_hi, ab.val[0]);
  //   vlo(b_lo, ab.val[1]);
  //   vhi(b_hi, ab.val[1]);
  //   vlo(c_lo, ab.val[2]);
  //   vhi(c_hi, ab.val[2]);
  //   vlo(d_lo, ab.val[3]);
  //   vhi(d_hi, ab.val[3]);

  //   fqmul(a_lo, neon_zeta1, t1, t2, t3, neon_qinv, neon_kyberq);
  //   fqmul(b_lo, neon_zeta1, t4, t5, t6, neon_qinv, neon_kyberq);
  //   fqmul(c_lo, neon_zeta1, t7, t8, t9, neon_qinv, neon_kyberq);
  //   fqmul(d_lo, neon_zeta1, ta, tb, tc, neon_qinv, neon_kyberq);
  //   fqmul(a_hi, neon_zeta1, t1, t2, t3, neon_qinv, neon_kyberq);
  //   fqmul(b_hi, neon_zeta1, t4, t5, t6, neon_qinv, neon_kyberq);
  //   fqmul(c_hi, neon_zeta1, t7, t8, t9, neon_qinv, neon_kyberq);
  //   fqmul(d_hi, neon_zeta1, ta, tb, tc, neon_qinv, neon_kyberq);

  //   vcombine(ab.val[0], a_lo, a_hi);
  //   vcombine(ab.val[1], b_lo, b_hi);
  //   vcombine(ab.val[2], c_lo, c_hi);
  //   vcombine(ab.val[3], d_lo, d_hi);

  //   vstorex4(&r[j], ab);
  // }
}


void combined_neon_invntt(int16_t r[256])
{
  int j, k1, k2, k3, k4, k5, k6, k7; 
  // Register: Total 26
  int16x8x4_t v; // 4
  int16x8_t r0, r1, r2, r3,
      l0, l1, l2, l3,
      o_tmp, e_tmp;                           // 10
  int16x4_t a_lo, a_hi, b_lo, b_hi, zlo, zhi; // 6
  int32x4_t t1, t2, t3, t4, t5, t6;           // 6
  // Constant: Total 4 
  int32x4_t neon_qinv, neon_kyberq;
  int16x8_t neon_kyberq16;
  int16x4_t neon_v;
  neon_qinv = vdupq_n_s32(QINV << 16);
  neon_kyberq = vdupq_n_s32(-KYBER_Q);
  neon_kyberq16 = vdupq_n_s16(-KYBER_Q);
  neon_v = vdup_n_s16(((1U << 26) + KYBER_Q / 2) / KYBER_Q);
  // Scalar 
  k1 = 0;
  k2 = 8*(256/32); // 64
  k3 = 96;
  k4 = 112;
  k5 = 120;
  k6 = 124;
  k7 = 126;
  // End 

  // Layer 1, 2, 3, 4
  for (j = 0; j < 256; j+=32)
  {
    // 1st layer : r0 x r2 | r1 x r3
    // r0: 0, 4, 8,  12 | 16, 20, 24, 28
    // r1: 1, 5, 9,  13 | 17, 21, 25, 29
    // r2: 2, 6, 10, 14 | 18, 22, 26, 30
    // r3: 3, 7, 11, 15 | 19, 23, 27, 31
    vload4(v, &r[j]);

    r0 = v.val[0];
    r1 = v.val[1];
    r2 = v.val[2];
    r3 = v.val[3];

    e_tmp = r0;
    // 0 + 2 
    vadd8(r0, e_tmp, r2);
    // 0 - 2
    vsub8(r2, e_tmp, r2);

    o_tmp = r1;
    // 1 + 3
    vadd8(r1, o_tmp, r3);
    // 1 - 3
    vsub8(r3, o_tmp, r3);

    vlo(a_lo, r2);
    vhi(a_hi, r2);

    vlo(b_lo, r3);
    vhi(b_hi, r3);

    vload(l0, &zetas_inv[k1]);
    k1 += 8;
    vlo(zlo, l0);
    vhi(zhi, l0);

    fqmul(a_lo, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(a_hi, zhi, t4, t5, t6, neon_qinv, neon_kyberq);
    fqmul(b_lo, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_hi, zhi, t4, t5, t6, neon_qinv, neon_kyberq);

    barrett(r0, e_tmp, zlo, zhi, t1, t2, neon_v, neon_kyberq16);
    barrett(r1, o_tmp, zlo, zhi, t3, t4, neon_v, neon_kyberq16);

    vcombine(r2, a_lo, a_hi);
    vcombine(r3, b_lo, b_hi);

    // Layer 2: r0 x r1 | r2 x r3
    // Tranpose 4x4 
    l0 = vtrn1q_s16(r0, r1);
    l1 = vtrn2q_s16(r0, r1);
    l2 = vtrn1q_s16(r2, r3);
    l3 = vtrn2q_s16(r2, r3);
    r0 = (int16x8_t) vtrn1q_s32( (int32x4_t) l0, (int32x4_t) l2 );
    r2 = (int16x8_t) vtrn2q_s32( (int32x4_t) l0, (int32x4_t) l2 );
    r1 = (int16x8_t) vtrn1q_s32( (int32x4_t) l1, (int32x4_t) l3 );
    r3 = (int16x8_t) vtrn2q_s32( (int32x4_t) l1, (int32x4_t) l3 );
    
    // r0: 0,  1,  2,  3  | 16,  17,  18,  19
    // r1: 4,  5,  6,  7  | 20,  21,  22,  23
    // r2: 8,  9,  10, 11 | 24,  25,  26,  27
    // r3: 12, 13, 14, 15 | 28,  29,  30,  31

    e_tmp = r0;
    // 0 + 4
    vadd8(r0, e_tmp, r1);
    // 0 - 4
    vsub8(r1, e_tmp, r1);

    o_tmp = r2; 
    // 8 + 12 
    vadd8(r2, o_tmp, r3);
    // 8 - 12 
    vsub8(r3, o_tmp, r3);

    vlo(a_lo, r1);
    vlo(b_lo, r3);

    vhi(a_hi, r1);
    vhi(b_hi, r3);

    vdup(zlo, zetas_inv[k2++]);
    fqmul(a_lo, zlo, t1, t2, t3, neon_qinv, neon_kyberq);

    vdup(zlo, zetas_inv[k2++]);
    fqmul(b_lo, zlo, t4, t5, t6, neon_qinv, neon_kyberq);
    
    vdup(zhi, zetas_inv[k2++]);
    fqmul(a_hi, zhi, t1, t2, t3, neon_qinv, neon_kyberq);
    
    vdup(zhi, zetas_inv[k2++]);
    fqmul(b_hi, zhi, t4, t5, t6, neon_qinv, neon_kyberq);

    barrett(r0, e_tmp, zlo, zhi, t1, t2, neon_v, neon_kyberq16);
    barrett(r2, o_tmp, zlo, zhi, t3, t4, neon_v, neon_kyberq16);

    vcombine(r1, a_lo, a_hi);
    vcombine(r3, b_lo, b_hi);

    // Layer 3 : r0 x r2 | r1 x r3
    // r0: 0,  1,  2,  3  | 16,  17,  18,  19
    // r1: 4,  5,  6,  7  | 20,  21,  22,  23
    // r2: 8,  9,  10, 11 | 24,  25,  26,  27
    // r3: 12, 13, 14, 15 | 28,  29,  30,  31

    e_tmp = r0;
    // 0 + 8  
    vadd8(r0, e_tmp, r2);
    // 0 - 8
    vsub8(r2, e_tmp, r2);

    o_tmp = r1;
    // 4 + 12
    vadd8(r1, o_tmp, r3);
    // 4 - 12
    vsub8(r3, o_tmp, r3);

    vlo(a_lo, r2);
    vhi(a_hi, r2);

    vlo(b_lo, r3);
    vhi(b_hi, r3);

    vdup(zlo, zetas_inv[k3++]);
    vdup(zhi, zetas_inv[k3++]);

    fqmul(a_lo, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(a_hi, zhi, t4, t5, t6, neon_qinv, neon_kyberq);
    fqmul(b_lo, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_hi, zhi, t4, t5, t6, neon_qinv, neon_kyberq);

    barrett(r0, e_tmp, zlo, zhi, t1, t2, neon_v, neon_kyberq16);
    barrett(r1, o_tmp, zlo, zhi, t3, t4, neon_v, neon_kyberq16);

    vcombine(r2, a_lo, a_hi);
    vcombine(r3, b_lo, b_hi);

    // Layer 4: r0 x r1 | r2 x r3
    // Re-arrange vector 
    l0 = (int16x8_t) vtrn1q_s64( (int64x2_t) r0, (int64x2_t) r1);
    l1 = (int16x8_t) vtrn2q_s64( (int64x2_t) r0, (int64x2_t) r1);
    l2 = (int16x8_t) vtrn1q_s64( (int64x2_t) r2, (int64x2_t) r3);
    l3 = (int16x8_t) vtrn2q_s64( (int64x2_t) r2, (int64x2_t) r3);

    // l0: 0,  1,  2,  3  | 4,  5,  6,  7  
    // l1: 16, 17, 18, 19 | 20, 21, 22, 23
    // l2: 8,  9,  10, 11 | 12, 13, 14, 15 
    // l3: 24, 25, 26, 27 | 28, 29, 30, 31

    e_tmp = l0;
    // 0 + 16
    vadd8(r0, e_tmp, l1);
    // 0 - 16
    vsub8(r2, e_tmp, l1);

    o_tmp = l2; 
    // 8 + 24
    vadd8(r1, o_tmp, l3);
    // 8 - 24 
    vsub8(r3, o_tmp, l3);

    vlo(a_lo, r2);
    vhi(a_hi, r2);

    vlo(b_lo, r3);
    vhi(b_hi, r3);

    vdup(zhi, zetas_inv[k4++]);
    // zlo = zhi;

    fqmul(a_lo, zhi, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(a_hi, zhi, t4, t5, t6, neon_qinv, neon_kyberq);
    fqmul(b_lo, zhi, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_hi, zhi, t4, t5, t6, neon_qinv, neon_kyberq);

    barrett(r0, e_tmp, zlo, zhi, t1, t2, neon_v, neon_kyberq16);
    barrett(r1, o_tmp, zlo, zhi, t3, t4, neon_v, neon_kyberq16);

    vcombine(r2, a_lo, a_hi);
    vcombine(r3, b_lo, b_hi);

    v.val[0] = r0;
    v.val[1] = r1;
    v.val[2] = r2;
    v.val[3] = r3;

    vstorex4(&r[j], v);
  }

  int16x8x4_t v0, v1, v2, v3;

  // Layer 5, 6
  // Total register: 5x4 + 4(const) + 2(zlo, zhi) + 4(ab_hi, lo) + 2(t1-t6) = 32
  for (j = 0; j < 256; j+=128)
  {
    // Layer 5: v0 x v1 | v2 x v3 
    // v0: 0  -> 31
    // v1: 32 -> 63
    // v2: 64 -> 95
    // v3: 96 -> 127
    vloadx4(v0, &r[j+ 0]);
    vloadx4(v1, &r[j+32]);
    vloadx4(v2, &r[j+64]);
    vloadx4(v3, &r[j+96]);

    v.val[0] = v0.val[0];
    v.val[1] = v0.val[1];
    v.val[2] = v0.val[2];
    v.val[3] = v0.val[3];

    // 0 + 32 
    vadd8(v0.val[0], v.val[0], v1.val[0]);
    vadd8(v0.val[1], v.val[1], v1.val[1]);
    vadd8(v0.val[2], v.val[2], v1.val[2]);
    vadd8(v0.val[3], v.val[3], v1.val[3]);
    // 0 - 32 
    vsub8(v1.val[0], v.val[0], v1.val[0]);
    vsub8(v1.val[1], v.val[1], v1.val[1]);
    vsub8(v1.val[2], v.val[2], v1.val[2]);
    vsub8(v1.val[3], v.val[3], v1.val[3]);

    v.val[0] = v2.val[0];
    v.val[1] = v2.val[1];
    v.val[2] = v2.val[2];
    v.val[3] = v2.val[3];

    // // 64 + 96
    vadd8(v2.val[0], v.val[0], v3.val[0]);
    vadd8(v2.val[1], v.val[1], v3.val[1]);
    vadd8(v2.val[2], v.val[2], v3.val[2]);
    vadd8(v2.val[3], v.val[3], v3.val[3]);
    // 64 - 96
    vsub8(v3.val[0], v.val[0], v3.val[0]);
    vsub8(v3.val[1], v.val[1], v3.val[1]);
    vsub8(v3.val[2], v.val[2], v3.val[2]);
    vsub8(v3.val[3], v.val[3], v3.val[3]);

    barrett(v0.val[0], v.val[0], zlo, zhi, t1, t2, neon_v, neon_kyberq16);
    barrett(v0.val[1], v.val[1], zlo, zhi, t3, t1, neon_v, neon_kyberq16);
    barrett(v0.val[2], v.val[2], zlo, zhi, t2, t3, neon_v, neon_kyberq16);
    barrett(v0.val[3], v.val[3], zlo, zhi, t1, t2, neon_v, neon_kyberq16);

    barrett(v2.val[0], v.val[0], zlo, zhi, t3, t1, neon_v, neon_kyberq16);
    barrett(v2.val[1], v.val[1], zlo, zhi, t2, t3, neon_v, neon_kyberq16);
    barrett(v2.val[2], v.val[2], zlo, zhi, t1, t2, neon_v, neon_kyberq16);
    barrett(v2.val[3], v.val[3], zlo, zhi, t3, t1, neon_v, neon_kyberq16);

    // 
    vdup(zlo, zetas_inv[k5++]);

    vlo(a_lo, v1.val[0]);
    vhi(a_hi, v1.val[0]);
    vlo(b_lo, v1.val[1]);
    vhi(b_hi, v1.val[1]);

    fqmul(a_lo, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(a_hi, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_lo, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_hi, zlo, t1, t2, t3, neon_qinv, neon_kyberq);

    vcombine(v1.val[0], a_lo, a_hi);
    vcombine(v1.val[1], b_lo, b_hi);

    vlo(a_lo, v1.val[2]);
    vhi(a_hi, v1.val[2]);
    vlo(b_lo, v1.val[3]);
    vhi(b_hi, v1.val[3]);

    fqmul(a_lo, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(a_hi, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_lo, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_hi, zlo, t1, t2, t3, neon_qinv, neon_kyberq);

    vcombine(v1.val[2], a_lo, a_hi);
    vcombine(v1.val[3], b_lo, b_hi);

    // 
    vdup(zhi, zetas_inv[k5++]);

    vlo(a_lo, v3.val[0]);
    vhi(a_hi, v3.val[0]);
    vlo(b_lo, v3.val[1]);
    vhi(b_hi, v3.val[1]);

    fqmul(a_lo, zhi, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(a_hi, zhi, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_lo, zhi, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_hi, zhi, t1, t2, t3, neon_qinv, neon_kyberq);

    vcombine(v3.val[0], a_lo, a_hi);
    vcombine(v3.val[1], b_lo, b_hi);

    vlo(a_lo, v3.val[2]);
    vhi(a_hi, v3.val[2]);
    vlo(b_lo, v3.val[3]);
    vhi(b_hi, v3.val[3]);

    fqmul(a_lo, zhi, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(a_hi, zhi, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_lo, zhi, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_hi, zhi, t1, t2, t3, neon_qinv, neon_kyberq);

    vcombine(v3.val[2], a_lo, a_hi);
    vcombine(v3.val[3], b_lo, b_hi);

    // Layer 6: v0 x v2 | v1 x v3 
    // v0: 0  -> 31
    // v2: 64 -> 95
    // v1: 32 -> 63
    // v3: 96 -> 127

    v.val[0] = v0.val[0];
    v.val[1] = v0.val[1];
    v.val[2] = v0.val[2];
    v.val[3] = v0.val[3];

    // 0 + 64 
    vadd8(v0.val[0], v.val[0], v2.val[0]);
    vadd8(v0.val[1], v.val[1], v2.val[1]);
    vadd8(v0.val[2], v.val[2], v2.val[2]);
    vadd8(v0.val[3], v.val[3], v2.val[3]);
    // 0 - 64
    vsub8(v2.val[0], v.val[0], v2.val[0]);
    vsub8(v2.val[1], v.val[1], v2.val[1]);
    vsub8(v2.val[2], v.val[2], v2.val[2]);
    vsub8(v2.val[3], v.val[3], v2.val[3]);

    v.val[0] = v1.val[0];
    v.val[1] = v1.val[1];
    v.val[2] = v1.val[2];
    v.val[3] = v1.val[3];

    // 32 + 96
    vadd8(v1.val[0], v.val[0], v3.val[0]);
    vadd8(v1.val[1], v.val[1], v3.val[1]);
    vadd8(v1.val[2], v.val[2], v3.val[2]);
    vadd8(v1.val[3], v.val[3], v3.val[3]);
    // 32 - 96
    vsub8(v3.val[0], v.val[0], v3.val[0]);
    vsub8(v3.val[1], v.val[1], v3.val[1]);
    vsub8(v3.val[2], v.val[2], v3.val[2]);
    vsub8(v3.val[3], v.val[3], v3.val[3]);

    barrett(v0.val[0], v.val[0], zlo, zhi, t1, t2, neon_v, neon_kyberq16);
    barrett(v0.val[1], v.val[1], zlo, zhi, t1, t2, neon_v, neon_kyberq16);
    barrett(v0.val[2], v.val[2], zlo, zhi, t1, t2, neon_v, neon_kyberq16);
    barrett(v0.val[3], v.val[3], zlo, zhi, t1, t2, neon_v, neon_kyberq16);

    vstorex4(&r[j+ 0], v0);

    barrett(v1.val[0], v.val[0], zlo, zhi, t1, t2, neon_v, neon_kyberq16);
    barrett(v1.val[1], v.val[1], zlo, zhi, t1, t2, neon_v, neon_kyberq16);
    barrett(v1.val[2], v.val[2], zlo, zhi, t1, t2, neon_v, neon_kyberq16);
    barrett(v1.val[3], v.val[3], zlo, zhi, t1, t2, neon_v, neon_kyberq16);

    vstorex4(&r[j+32], v1);

    // Reduce register usage afterwards

    vdup(zlo, zetas_inv[k6++]);

    vlo(a_lo, v2.val[0]);
    vhi(a_hi, v2.val[0]);
    vlo(b_lo, v2.val[1]);
    vhi(b_hi, v2.val[1]);

    fqmul(a_lo, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(a_hi, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_lo, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_hi, zlo, t1, t2, t3, neon_qinv, neon_kyberq);

    vcombine(v2.val[0], a_lo, a_hi);
    vcombine(v2.val[1], b_lo, b_hi);

    vlo(a_lo, v2.val[2]);
    vhi(a_hi, v2.val[2]);
    vlo(b_lo, v2.val[3]);
    vhi(b_hi, v2.val[3]);

    fqmul(a_lo, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(a_hi, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_lo, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_hi, zlo, t1, t2, t3, neon_qinv, neon_kyberq);

    vcombine(v2.val[2], a_lo, a_hi);
    vcombine(v2.val[3], b_lo, b_hi);

    vstorex4(&r[j+64], v2);

    //  
    vlo(a_lo, v3.val[0]);
    vhi(a_hi, v3.val[0]);
    vlo(b_lo, v3.val[1]);
    vhi(b_hi, v3.val[1]);

    fqmul(a_lo, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(a_hi, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_lo, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_hi, zlo, t1, t2, t3, neon_qinv, neon_kyberq);

    vcombine(v3.val[0], a_lo, a_hi);
    vcombine(v3.val[1], b_lo, b_hi);

    vlo(a_lo, v3.val[2]);
    vhi(a_hi, v3.val[2]);
    vlo(b_lo, v3.val[3]);
    vhi(b_hi, v3.val[3]);

    fqmul(a_lo, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(a_hi, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_lo, zlo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(b_hi, zlo, t1, t2, t3, neon_qinv, neon_kyberq);

    vcombine(v3.val[2], a_lo, a_hi);
    vcombine(v3.val[3], b_lo, b_hi);
    
    vstorex4(&r[j+96], v3);
  }

  // Layer 7, inv_mul 
  // Layer 7: 
  // v0: 0   -> 31 
  // v1: 32  -> 64
  // v2: 128 -> 159
  // v3: 160 -> 191


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
  int16x8x2_t aa, bb;                                       // 2
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
        vloadx2(aa, &r[j]);
        a = aa.val[0];
        b = aa.val[1];
        vloadx2(bb, &r[j + len]);
        c = bb.val[0];
        d = bb.val[1];

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

        aa.val[0] = a;
        aa.val[1] = b;
        vstorex2(&r[j], aa);
        bb.val[0] = c;
        bb.val[1] = d;
        vstorex2(&r[j + len], bb);
      }
    }
  }

  //   Layer 3
  for (j = 0; j < 256; j += 32) {
    //   a - b, c - d
    vloadx4(ab, &r[j]);
    a = ab.val[0];
    b = ab.val[1];
    //
    c = ab.val[2];
    d = ab.val[3];

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

    ab.val[0] = a;
    ab.val[1] = b;
    ab.val[2] = c;
    ab.val[3] = d;
    vstorex4(&r[j], ab);
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
    vloadx4(ab, &r[j]);
    a = ab.val[0];
    b = ab.val[1];
    c = ab.val[2];
    d = ab.val[3];

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

    ab.val[0] = a;
    ab.val[1] = b;
    ab.val[2] = c;
    ab.val[3] = d;
    vstorex4(&r[j], ab);
  }

  //   Layer 1
  for (j = 0; j < 256; j += 32) {
    // ab.val[0] = 0, 4, 8, 12, 16, 20, 24, 28
    // ab.val[1] = 1, 5, 9, 13, 17, 21, 25, 29
    // ab.val[2] = 2, 6, 10, 14, 18, 22, 26, 30
    // al.val[3] = 3, 7, 11, 15, 19, 23, 27, 31
    vload4(ab, &r[j]);

    // a_lo
    vlo(a_lo, ab.val[2]);
    vhi(a_hi, ab.val[2]);
    vlo(b_lo, ab.val[3]);
    vhi(b_hi, ab.val[3]);

    vload(neon_zetas, &zetas[k]);

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
    vstore4(&r[j], ab);
  }
}


int compare(int16_t *a, int16_t *b, int length)
{
  int i, j, count = 0;
  for (i = 0; i < length; i+=8)
  {
    for (j = i; j < i + 8; j++)
    {
      if (a[j] != b[j])
      {
        printf("%d: %d != %d\n", j, a[j], b[j]);
        count++;
      }
      if (count > 8) 
        return 1;
    }
  }
  printf("Correct!!\n");
  return 0;
}


int main(void)
{
  int16_t r_gold[256], r[256];
  int16_t i, t; 
  for (i  = 0; i < 256; i++)
  {
    t = i; 
    r[i] = t;
    r_gold[i] = t;
  }

  neon_invntt(r_gold);
  printf("\n\n\n");
  combined_neon_invntt(r);

  if (compare(r_gold, r, 256))
    return 1;

  return 0; 
}