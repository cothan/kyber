#include <papi.h>
#include <stdio.h>
#include <arm_neon.h>
#include "params.h"
#include "ntt.h"
#include "reduce.h"

// clang ntt.c reduce.c neon_ntt.c speed_ntt.c -o neon_ntt -O3 -g3 -Wall -Werror -Wextra -Wpedantic -lpapi
// gcc   ntt.c reduce.c neon_ntt.c speed_ntt.c -o neon_ntt -O3 -g3 -Wall -Werror -Wextra -Wpedantic -lpapi

static
const int16_t ref_zetas[128] = {
  2285, 2571, 2970, 1812, 1493, 1422, 287, 202, 
  3158, 622, 1577, 182, 962, 2127, 1855, 1468, 
  573, 2004, 264, 383, 2500, 1458, 1727, 3199, 
  2648, 1017, 732, 608, 1787, 411, 3124, 1758, 
  1223, 652, 2777, 1015, 2036, 1491, 3047, 1785, 
  516, 3321, 3009, 2663, 1711, 2167, 126, 1469, 
  2476, 3239, 3058, 830, 107, 1908, 3082, 2378, 
  2931, 961, 1821, 2604, 448, 2264, 677, 2054, 
  2226, 430, 555, 843, 2078, 871, 1550, 105, 
  422, 587, 177, 3094, 3038, 2869, 1574, 1653, 
  3083, 778, 1159, 3182, 2552, 1483, 2727, 1119, 
  1739, 644, 2457, 349, 418, 329, 3173, 3254, 
  817, 1097, 603, 610, 1322, 2044, 1864, 384, 
  2114, 3193, 1218, 1994, 2455, 220, 2142, 1670, 
  2144, 1799, 2051, 794, 1819, 2475, 2459, 478, 
  3221, 3021, 996, 991, 958, 1869, 1522, 1628
};

static 
const int16_t ref_zetas_inv[128] = {
  1701, 1807, 1460, 2371, 2338, 2333, 308, 108, 2851, 870, 854, 1510, 2535,
  1278, 1530, 1185, 1659, 1187, 3109, 874, 1335, 2111, 136, 1215, 2945, 1465,
  1285, 2007, 2719, 2726, 2232, 2512, 75, 156, 3000, 2911, 2980, 872, 2685,
  1590, 2210, 602, 1846, 777, 147, 2170, 2551, 246, 1676, 1755, 460, 291, 235,
  3152, 2742, 2907, 3224, 1779, 2458, 1251, 2486, 2774, 2899, 1103, 1275, 2652,
  1065, 2881, 725, 1508, 2368, 398, 951, 247, 1421, 3222, 2499, 271, 90, 853,
  1860, 3203, 1162, 1618, 666, 320, 8, 2813, 1544, 282, 1838, 1293, 2314, 552,
  2677, 2106, 1571, 205, 2918, 1542, 2721, 2597, 2312, 681, 130, 1602, 1871,
  829, 2946, 3065, 1325, 2756, 1861, 1474, 1202, 2367, 3147, 1752, 2707, 171,
  3127, 3042, 1907, 1836, 1517, 359, 758, 1441
};

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

static 
void unroll_neon_invntt(int16_t r[256])
{
  // NEON Registers
  int16x8_t a, b, c, d, at, bt, ct, my_zetas;  // 9
  int16x4_t a_lo, a_hi, b_lo, b_hi, c_lo, c_hi, d_lo, d_hi, zlo, zhi;// 8
  int16x4_t neon_zeta1, neon_zeta2, neon_zeta3, neon_zeta4, neon_v, neon_kyberq16; // 4
  int32x4_t t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc;         // 12
  int32x4_t neon_qinv, neon_kyberq;                                 // 2
  int16x8x4_t ab;                                                   // 4
  int16x8x2_t aa, bb;                                               // 2
  // End
  neon_qinv = vdupq_n_s32(QINV << 16);
  neon_kyberq = vdupq_n_s32(-KYBER_Q);
  neon_kyberq16 = vdup_n_s16(-KYBER_Q);
  neon_v = vdup_n_s16(((1U << 26) + KYBER_Q / 2) / KYBER_Q);
  // Scalar variable
  uint16_t start, len, j, k;
  // End

  k = 0;

  // *Vectorize* barret_reduction over 96 points rather than 896 points
  // Optimimal Barret reduction for Kyber N=256, B=9 is 78 points, see here: 
  // https://eprint.iacr.org/2020/1377.pdf

  //   Layer 1
  for (j = 0; j < 256; j += 32)
  {
    // ab.val[0] = 0, 4, 8, 12, 16, 20, 24, 28
    // ab.val[1] = 1, 5, 9, 13, 17, 21, 25, 29
    // ab.val[2] = 2, 6, 10, 14, 18, 22, 26, 30
    // al.val[3] = 3, 7, 11, 15, 19, 23, 27, 31
    vload(my_zetas, &ref_zetas_inv[k]);
    vlo(neon_zeta1, my_zetas);
    vhi(neon_zeta2, my_zetas);
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

    vcombine(ab.val[3], b_lo, b_hi);

    vstore4(&r[j], ab);

    k += 8;
  }

  //   Layer 2
  for (j = 0; j < 256; j += 32)
  {
    vdup(neon_zeta1, ref_zetas_inv[k++]);
    vdup(neon_zeta2, ref_zetas_inv[k++]);
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
    vlo(a_lo, at);
    vhi(b_lo, at);

    vcombine(a, a_lo, a_hi);
    vcombine(b, b_lo, b_hi);

    aa.val[0] = a;
    aa.val[1] = b;
    vstorex2(&r[j], aa);

    vdup(neon_zeta3, ref_zetas_inv[k++]);
    vdup(neon_zeta4, ref_zetas_inv[k++]);

    vloadx2(bb, &r[j + 16]);
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
    vlo(c_lo, ct);
    vhi(d_lo, ct);

    vcombine(c, c_lo, c_hi);
    vcombine(d, d_lo, d_hi);

    bb.val[0] = c;
    bb.val[1] = d;
    vstorex2(&r[j + 16], bb);
  }

  //   Layer 3
  for (j = 0; j < 256; j += 32)
  {
    //   a - b, c - d
    vdup(neon_zeta1, ref_zetas_inv[k++]);
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
    vcombine(b, b_lo, b_hi);  

    aa.val[0] = a;
    aa.val[1] = b;
    vstorex2(&r[j], aa);
    //
    vdup(neon_zeta2, ref_zetas_inv[k++]);
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
    vcombine(d, d_lo, d_hi);

    vlo(c_lo, c);
    vhi(c_hi, c);
    barrett(c_lo, t1, zlo, neon_v, neon_kyberq16);
    vcombine(c, c_lo, c_hi);

    bb.val[0] = c;
    bb.val[1] = d;
    vstorex2(&r[j + 16], bb);
  }

  // Layer 4, 5, 6, 7
  for (len = 16; len <= 128; len <<= 1)
  {
    for (start = 0; start < 256; start = j + len)
    {
      vdup(neon_zeta1, ref_zetas_inv[k++]);
      for (j = start; j < start + len; j += 16)
      {
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

        vcombine(c, c_lo, c_hi);
        // barrett(a, bt, b_lo, b_hi, t1, t2, neon_v, neon_kyberq16);
        vlo(a_lo, a);
        vhi(a_hi, a);
        barrett(a_lo, t1, zlo, neon_v, neon_kyberq16);
        barrett(a_hi, t2, zlo, neon_v, neon_kyberq16);
        vcombine(a, a_lo, a_hi);


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

        // barrett(b, at, a_lo, a_hi, ta, tb, neon_v, neon_kyberq16);
        vlo(b_lo, b);
        vhi(b_hi, b);
        barrett(b_lo, t3, zhi, neon_v, neon_kyberq16);
        barrett(b_hi, t4, zhi, neon_v, neon_kyberq16);
        vcombine(b, b_lo, b_hi);
        

        vcombine(d, d_lo, d_hi);

        aa.val[0] = a;
        aa.val[1] = b;
        vstorex2(&r[j], aa);
        bb.val[0] = c;
        bb.val[1] = d;
        vstorex2(&r[j + len], bb);
      }
    }
  }

  vdup(neon_zeta1, ref_zetas_inv[127]);
  for (j = 0; j < 256; j += 32)
  {
    vloadx4(ab, &r[j]);

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

    vstorex4(&r[j], ab);
  }
}

static
void unroll_neon_ntt(int16_t r[256])
{
  // NEON Registers
  int16x8_t a, b, c, d, at, bt, ct, dt, my_zetas;         // 9
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
  for (len = 128; len >= 16; len >>= 1)
  {
    for (start = 0; start < 256; start = j + len)
    {
      vdup(neon_zeta1, ref_zetas[k++]);
      for (j = start; j < start + len; j += 16)
      {
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
  for (j = 0; j < 256; j += 32)
  {
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

    vdup(neon_zeta1, ref_zetas[k++]);
    vdup(neon_zeta2, ref_zetas[k++]);

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
  for (j = 0; j < 256; j += 32)
  {
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

    vdup(neon_zeta1, ref_zetas[k++]);
    vdup(neon_zeta2, ref_zetas[k++]);

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

    vdup(neon_zeta3, ref_zetas[k++]);
    vdup(neon_zeta4, ref_zetas[k++]);

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
  for (j = 0; j < 256; j += 32)
  {
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

    vload(my_zetas, &ref_zetas[k]);

    vlo(neon_zeta1, my_zetas);
    vhi(neon_zeta2, my_zetas);

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


static 
int compare(int16_t *a, int16_t *b, int length, const char *string)
{
  int i, j, count = 0;
  int16_t aa, bb;
  for (i = 0; i < length; i += 8)
  {
    for (j = i; j < i + 8; j++)
    {
      aa = a[j]; // % KYBER_Q;
      bb = b[j]; // % KYBER_Q;
      if (aa != bb)
      {
        printf("%d: %d != %d: %d != %d ", j, a[j], b[j], aa, bb);
        if ( (aa + KYBER_Q == bb) || (aa - KYBER_Q == bb) )
        {
          printf(": OK\n");
        }
        else{
          count++;
        }
      }
      if (count > 8)
      {
        printf("%s Incorrect!!\n", string);
        return 1;
      }
    }
  }
  if (count) {
    printf("%s Incorrect!!\n", string);
    return 1;
  }
  
  return 0;
}


static int16_t fqmul1(int16_t a, int16_t b) {
  return montgomery_reduce((int32_t)a*b);
}

static
void ntt(int16_t r[256]) {
  unsigned int len, start, j, k;
  int16_t t, zeta;

  k = 1;
  for(len = 128; len >= 2; len >>= 1) {
    for(start = 0; start < 256; start = j + len) {
      zeta = ref_zetas[k++];
      for(j = start; j < start + len; ++j) {
        t = fqmul1(zeta, r[j + len]);
        r[j + len] = r[j] - t;
        r[j] = r[j] + t;
      }
    }
  }
}

static
void invntt(int16_t r[256]) {
  unsigned int start, len, j, k;
  int16_t t, zeta;

  k = 0;
  for(len = 2; len <= 128; len <<= 1) {
    for(start = 0; start < 256; start = j + len) {
      zeta = ref_zetas_inv[k++];
      for(j = start; j < start + len; ++j) {
        t = r[j];
        r[j] = barrett_reduce(t + r[j + len]);
        r[j + len] = t - r[j + len];
        r[j + len] = fqmul1(zeta, r[j + len]);
      }
    }
  }

  for(j = 0; j < 256; ++j)
    r[j] = fqmul1(r[j], ref_zetas_inv[127]);
}

#include <sys/random.h>
#include <string.h>

#define TESTS 1000000

int main(void)
{
  int16_t r_gold[256], r1[256], r2[256];
  int retval;

  getrandom(r_gold, sizeof(r_gold), 0);
  for (int i = 0; i < 256; i++){
    r_gold[i] %= KYBER_Q;
  }
  memcpy(r1, r_gold, sizeof(r_gold));
  memcpy(r2, r_gold, sizeof(r_gold));

  // Test INTT
  retval = PAPI_hl_region_begin("c_invntt");
  for (int j = 0; j < TESTS; j++)
  {
    invntt(r_gold);
  }
  retval = PAPI_hl_region_end("c_invntt");

  retval = PAPI_hl_region_begin("merged_neon_invntt");
  for (int j = 0; j < TESTS; j++)
  {
    neon_invntt(r1);
  }
  retval = PAPI_hl_region_end("merged_neon_invntt");
  
  retval = PAPI_hl_region_begin("unroll_neon_invntt");
  for (int j = 0; j < TESTS; j++)
  {
    unroll_neon_invntt(r2);
  }
  retval = PAPI_hl_region_end("unroll_neon_invntt");


  // Test NTT
  retval = PAPI_hl_region_begin("c_ntt");
  for (int j = 0; j < TESTS; j++)
  {
    ntt(r_gold);
  }
  retval = PAPI_hl_region_end("c_ntt");
  
  retval = PAPI_hl_region_begin("merged_neon_ntt");  
  for (int j = 0; j < TESTS; j++)
  {
    neon_ntt(r1);
  }
  retval = PAPI_hl_region_end("merged_neon_ntt");
  
  retval = PAPI_hl_region_begin("unroll_neon_ntt");
  for (int j = 0; j < TESTS; j++)
  {
    unroll_neon_ntt(r2);
  }
  retval = PAPI_hl_region_end("unroll_neon_ntt");
  /* Do some computation here */
  int comp = 0;
  comp = compare(r_gold, r1, 256, "r_gold vs neon_invntt");
  comp = comp && compare(r_gold, r2, 256,  "r_gold vs unroll_neon_invntt");
  if (comp)
    return 1;

  return 0;
}
