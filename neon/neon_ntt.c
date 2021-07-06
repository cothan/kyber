#include <arm_neon.h>
#include "params.h"
#include "neon_ntt.h"
#include "reduce.h"

/* Code to generate neon_zetas and neon_zetas_inv used in the number-theoretic transform: `gen_ntt.c` */

const int16_t neon_zetas [224] = {
  -359, 1493, 1422, -171, 622, 1577, 182, 573, 
  -1325, 264, 383, -829, 1458, -1602, -130, -758, 
  1223, 1223, 1223, 1223, -552, -552, -552, -552, 
  652, 652, 652, 652, 1015, 1015, 1015, 1015, 
  -1293, -1293, -1293, -1293, -282, -282, -282, -282, 
  1491, 1491, 1491, 1491, -1544, -1544, -1544, -1544, 
  516, 516, 516, 516, -320, -320, -320, -320, 
  -8, -8, -8, -8, -666, -666, -666, -666, 
  -1618, -1618, -1618, -1618, 126, 126, 126, 126, 
  -1162, -1162, -1162, -1162, 1469, 1469, 1469, 1469, 
  -1103, 430, 555, 843, -1251, 871, 1550, 105, 
  422, 587, 177, -235, -291, -460, 1574, 1653, 
  -246, 778, 1159, -147, -777, 1483, -602, 1119, 
  -1590, 644, -872, 349, 418, 329, -156, -75, 
  -1517, 287, 202, 962, -1202, -1474, 1468, -681, 
  1017, 732, 608, -1542, 411, -205, -1571, -853, 
  -853, -853, -853, -853, -271, -271, -271, -271, 
  -90, -90, -90, -90, 830, 830, 830, 830, 
  107, 107, 107, 107, -247, -247, -247, -247, 
  -1421, -1421, -1421, -1421, -951, -951, -951, -951, 
  -398, -398, -398, -398, -1508, -1508, -1508, -1508, 
  961, 961, 961, 961, -725, -725, -725, -725, 
  448, 448, 448, 448, 677, 677, 677, 677, 
  -1065, -1065, -1065, -1065, -1275, -1275, -1275, -1275, 
  817, 1097, 603, 610, 1322, -1285, -1465, 384, 
  -1215, -136, 1218, -1335, -874, 220, -1187, -1659, 
  -1185, -1530, -1278, 794, -1510, -854, -870, 478, 
  -108, -308, 996, 991, 958, -1460, 1522, 1628, 
};

const int16_t neon_zetas_qinv [224] = {
14745, 13525, -12402, -20907, 27758, -3799, -15690, -5827, 
17363, -26360, -29057, 5571, -1102, 21438, -26242, 31498, 
-5689, -5689, -5689, -5689, 1496, 1496, 1496, 1496, 
-6516, -6516, -6516, -6516, 30967, 30967, 30967, 30967, 
-23565, -23565, -23565, -23565, 20710, 20710, 20710, 20710, 
20179, 20179, 20179, 20179, 25080, 25080, 25080, 25080, 
-12796, -12796, -12796, -12796, 16064, 16064, 16064, 16064, 
26616, 26616, 26616, 26616, -12442, -12442, -12442, -12442, 
9134, 9134, 9134, 9134, -25986, -25986, -25986, -25986, 
-650, -650, -650, -650, 27837, 27837, 27837, 27837, 
-335, 11182, -11477, 13387, -32227, -14233, 20494, -21655, 
-27738, 13131, 945, -4587, -14883, 23092, 6182, 5493, 
32010, -32502, 10631, 30317, 29175, -18741, -28762, 12639, 
-18486, 20100, 17560, 18525, -14430, 19529, -5276, -12619, 

787, 28191, -16694, 10690, 1358, -11202, 31164, -28073, 
24313, -10532, 8800, 18426, 8859, 26675, -16163, 19883, 
19883, 19883, 19883, 19883, -15887, -15887, -15887, -15887, 
-28250, -28250, -28250, -28250, -8898, -8898, -8898, -8898, 
-28309, -28309, -28309, -28309, -30199, -30199, -30199, -30199, 
9075, 9075, 9075, 9075, 18249, 18249, 18249, 18249, 
13426, 13426, 13426, 13426, -29156, -29156, -29156, -29156, 
14017, 14017, 14017, 14017, -12757, -12757, -12757, -12757, 
16832, 16832, 16832, 16832, -24155, -24155, -24155, -24155, 
4311, 4311, 4311, 4311, -17915, -17915, -17915, -17915, 
-31183, 20297, 25435, 2146, -7382, 15355, 24391, -32384, 
-20927, -6280, 10946, -14903, 24214, -11044, 16989, 14469, 
10335, -21498, -7934, -20198, -22502, 23210, 10906, -17442, 
31636, -23860, 28644, -20257, 23998, 7756, -17422, 23132, 
};

const int16_t neon_zetas_inv [272] = {
  1628, 1522, -1460, 958, 991, 996, -308, -108, 
  478, -870, -854, -1510, 794, -1278, -1530, -1185, 
  -1659, -1187, 220, -874, -1335, 1218, -136, -1215, 
  384, -1465, -1285, 1322, 610, 603, 1097, 817, 
  -1275, -1275, -1275, -1275, -1065, -1065, -1065, -1065, 
  677, 677, 677, 677, 448, 448, 448, 448, 
  -725, -725, -725, -725, 961, 961, 961, 961, 
  -1508, -1508, -1508, -1508, -398, -398, -398, -398, 
  -951, -951, -951, -951, -1421, -1421, -1421, -1421, 
  -247, -247, -247, -247, 107, 107, 107, 107, 
  830, 830, 830, 830, -90, -90, -90, -90, 
  -271, -271, -271, -271, -853, -853, -853, -853, 
  -1571, -1571, -1571, -1571, -205, -205, -205, -205, 
  411, 411, 411, 411, -1542, -1542, -1542, -1542, 
  608, 608, 608, 608, 732, 732, 732, 732, 
  1017, 1017, 1017, 1017, -681, -681, -681, -681, 
  1468, -1474, -1202, 962, 202, 287, -1517, -1517, 
  -75, -156, 329, 418, 349, -872, 644, -1590, 
  1119, -602, 1483, -777, -147, 1159, 778, -246, 
  1653, 1574, -460, -291, -235, 177, 587, 422, 
  105, 1550, 871, -1251, 843, 555, 430, -1103, 
  1469, 1469, 1469, 1469, -1162, -1162, -1162, -1162, 
  126, 126, 126, 126, -1618, -1618, -1618, -1618, 
  -666, -666, -666, -666, -8, -8, -8, -8, 
  -320, -320, -320, -320, 516, 516, 516, 516, 
  -1544, -1544, -1544, -1544, 1491, 1491, 1491, 1491, 
  -282, -282, -282, -282, -1293, -1293, -1293, -1293, 
  1015, 1015, 1015, 1015, 652, 652, 652, 652, 
  -552, -552, -552, -552, 1223, 1223, 1223, 1223, 
  -130, -130, -130, -130, -1602, -1602, -1602, -1602, 
  1458, 1458, 1458, 1458, -829, -829, -829, -829, 
  383, 383, 383, 383, 264, 264, 264, 264, 
  -1325, -1325, -1325, -1325, 573, 573, 573, 573, 
  182, 1577, 622, -171, 1422, 1493, -359, 1397, 
};

const int16_t neon_zetas_inv_qinv [272] = {
23132, -17422, 7756, 23998, -20257, 28644, -23860, 31636, 
-17442, 10906, 23210, -22502, -20198, -7934, -21498, 10335, 
14469, 16989, -11044, 24214, -14903, 10946, -6280, -20927, 
-32384, 24391, 15355, -7382, 2146, 25435, 20297, -31183, 
-17915, -17915, -17915, -17915, 4311, 4311, 4311, 4311, 
-24155, -24155, -24155, -24155, 16832, 16832, 16832, 16832, 
-12757, -12757, -12757, -12757, 14017, 14017, 14017, 14017, 
-29156, -29156, -29156, -29156, 13426, 13426, 13426, 13426, 
18249, 18249, 18249, 18249, 9075, 9075, 9075, 9075, 
-30199, -30199, -30199, -30199, -28309, -28309, -28309, -28309, 
-8898, -8898, -8898, -8898, -28250, -28250, -28250, -28250, 
-15887, -15887, -15887, -15887, 19883, 19883, 19883, 19883, 
-16163, -16163, -16163, -16163, 26675, 26675, 26675, 26675, 
8859, 8859, 8859, 8859, 18426, 18426, 18426, 18426, 
8800, 8800, 8800, 8800, -10532, -10532, -10532, -10532, 
24313, 24313, 24313, 24313, -28073, -28073, -28073, -28073, 
31164, -11202, 1358, 10690, -16694, 28191, 787, 787, 

-12619, -5276, 19529, -14430, 18525, 17560, 20100, -18486, 
12639, -28762, -18741, 29175, 30317, 10631, -32502, 32010, 
5493, 6182, 23092, -14883, -4587, 945, 13131, -27738, 
-21655, 20494, -14233, -32227, 13387, -11477, 11182, -335, 
27837, 27837, 27837, 27837, -650, -650, -650, -650, 
-25986, -25986, -25986, -25986, 9134, 9134, 9134, 9134, 
-12442, -12442, -12442, -12442, 26616, 26616, 26616, 26616, 
16064, 16064, 16064, 16064, -12796, -12796, -12796, -12796, 
25080, 25080, 25080, 25080, 20179, 20179, 20179, 20179, 
20710, 20710, 20710, 20710, -23565, -23565, -23565, -23565, 
30967, 30967, 30967, 30967, -6516, -6516, -6516, -6516, 
1496, 1496, 1496, 1496, -5689, -5689, -5689, -5689, 
-26242, -26242, -26242, -26242, 21438, 21438, 21438, 21438, 
-1102, -1102, -1102, -1102, 5571, 5571, 5571, 5571, 
-29057, -29057, -29057, -29057, -26360, -26360, -26360, -26360, 
17363, 17363, 17363, 17363, -5827, -5827, -5827, -5827, 
-15690, -3799, 27758, -20907, -12402, 13525, 14745, 5237, 
};

/*************************************************/
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

out, in: int16x8_t
zeta: input : int16x8_t
t : int16x8x4_t
neon_qinv: const   : int16x8_t
neon_kyberq: const : int16x8_t
rewrite pseudo code:
int16_t fqmul(int16_t b, int16_t c) {
  int32_t t, u, a;

  // fqmul: 4 MUL
  t0 = a_H = (2*b*c)_H; 
  t1 = a_L = ((b * QINV)_L * b)_L; 
  t2 = (2*a_L * Q)_H; 
  out = (t0 - t2)/2 = (b*c)_H - (a_L * Q)_H;

  // fqmul_qinv: 3 MUL + precomputed
  t0 = a_H = (2*b*c)_H; 
  t1 = a_L = (b_qinv * c)_L; 
  t2 = (t1 * Q)_H; 
  out = (t0 - t2)/2 = (b*c)_H - (a_L * Q)_H;

*************************************************/
#define fqmul(out, in, zeta, zeta_qinv, t)                          \
  t.val[0] = vqdmulhq_s16(in, zeta);              /* a_H */         \
  t.val[1] = vmulq_s16(zeta_qinv, in);            /* a_L */         \
  t.val[2] = vqdmulhq_s16(t.val[1], neon_kyberq); /* (2*a_L*Q)_H */ \
  out = vhsubq_s16(t.val[0], t.val[2]);           /* (t0 - t2)/2 */

#define fqmul_lane(out, in, zeta, zeta_qinv, t, lane)                \
  t.val[0] = vqdmulhq_laneq_s16(in, zeta, lane);   /* a_H */         \
  t.val[1] = vmulq_laneq_s16(in, zeta_qinv, lane); /* a_L */         \
  t.val[2] = vqdmulhq_s16(t.val[1], neon_kyberq);  /* (2*a_L*Q)_H */ \
  out = vhsubq_s16(t.val[0], t.val[2]);            /* (t0 - t2)/2 */

/*
inout: int16x4_t
t32 : int32x4_t
t16: int16x4_t
neon_v: int16x4_t
neon_kyberq16: inout int16x4_t

int16_t barrett_reduce(int16_t a) {
  int16_t t;
  const int16_t v = ((1U << 26) + KYBER_Q / 2) / KYBER_Q;

  t = (int32_t)v * a;
  (t_L, t_H) = t; 
  t_h = t_H + (1 << 9);
  t_H = t_H >> 10;
  t_H = a - t_H * KYBER_Q;
  return t_H;
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
#define barrett(inout, t, i)                                        \
  t.val[i] = vqdmulhq_s16(inout, neon_v);        /* (2*a*v)_H */    \
  t.val[i + 1] = vhaddq_s16(t.val[i], neon_one); /* (2a*v + 2)/2 */ \
  t.val[i + 1] = vshrq_n_s16(t.val[i + 1], 10);                     \
  inout = vmlsq_s16(inout, t.val[i + 1], neon_kyberq);

/*
v1, v2: int16x8_t 
out1, out2: int16x8_t
t32_1, t32_2: int32x4_t 
t16: int16x8_t
*/
#define barrett_hi(v1, v2, t, i)                                                  \
  t.val[i] = (int16x8_t)vmull_high_s16(v1, neon_v);                               \
  t.val[i + 1] = (int16x8_t)vmull_high_s16(v2, neon_v);                           \
  t.val[i] = vuzp2q_s16(t.val[i], t.val[i + 1]);                                  \
  t.val[i] = vaddq_s16(t.val[i], neon_one);                                       \
  t.val[i] = vshrq_n_s16(t.val[i], 10);                                           \
  t.val[i + 1] = (int16x8_t)vzip2q_s64((int64x2_t)v1, (int64x2_t)v2);             \
  t.val[i + 1] = vmlsq_s16(t.val[i + 1], t.val[i], neon_kyberq);                  \
  v1 = (int16x8_t)vcopyq_laneq_s64((int64x2_t)v1, 1, (int64x2_t)t.val[i + 1], 0); \
  v2 = (int16x8_t)vcopyq_laneq_s64((int64x2_t)v2, 1, (int64x2_t)t.val[i + 1], 1);

#define barrett_lo(v1, v2, t, i)                                                   \
  t.val[i] = (int16x8_t)vmull_s16(vget_low_s16(v1), vget_low_s16(neon_v));        \
  t.val[i + 1] = (int16x8_t)vmull_s16(vget_low_s16(v2), vget_low_s16(neon_v));    \
  t.val[i] = vuzp2q_s16(t.val[i], t.val[i + 1]);                                  \
  t.val[i] = vaddq_s16(t.val[i], neon_one);                                       \
  t.val[i] = vshrq_n_s16(t.val[i], 10);                                           \
  t.val[i + 1] = (int16x8_t)vzip1q_s64((int64x2_t)v1, (int64x2_t)v2);             \
  t.val[i + 1] = vmlsq_s16(t.val[i + 1], t.val[i], neon_kyberq);                  \
  v1 = (int16x8_t)vcopyq_laneq_s64((int64x2_t)v1, 0, (int64x2_t)t.val[i + 1], 0); \
  v2 = (int16x8_t)vcopyq_laneq_s64((int64x2_t)v2, 0, (int64x2_t)t.val[i + 1], 1);

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
#define addsub(v, i, j, m, n, t, k)        \
  vcopy(t.val[k], v.val[i]);               \
  vcopy(t.val[k + 1], v.val[m]);           \
  vadd8(v.val[i], v.val[j], t.val[k]);     \
  vsub8(v.val[j], v.val[j], t.val[k]);     \
  vadd8(v.val[m], v.val[n], t.val[k + 1]); \
  vsub8(v.val[n], v.val[n], t.val[k + 1]);

#define addsub_x4(v0, v1, va)             \
  vcopy(va.val[0], v0.val[0]);            \
  vcopy(va.val[1], v0.val[1]);            \
  vcopy(va.val[2], v0.val[2]);            \
  vcopy(va.val[3], v0.val[3]);            \
  vadd8(v0.val[0], v1.val[0], va.val[0]); \
  vadd8(v0.val[1], v1.val[1], va.val[1]); \
  vadd8(v0.val[2], v1.val[2], va.val[2]); \
  vadd8(v0.val[3], v1.val[3], va.val[3]); \
  vsub8(v1.val[0], v1.val[0], va.val[0]); \
  vsub8(v1.val[1], v1.val[1], va.val[1]); \
  vsub8(v1.val[2], v1.val[2], va.val[2]); \
  vsub8(v1.val[3], v1.val[3], va.val[3]);

#define addsub_twist(v, v_in, i, j, m, n, t, k) \
  vcopy(t.val[k], v_in.val[i]);                 \
  vcopy(t.val[k + 1], v_in.val[m]);             \
  vadd8(v.val[i], v_in.val[j], t.val[k]);       \
  vsub8(v.val[m], v_in.val[j], t.val[k]);       \
  vadd8(v.val[j], v_in.val[n], t.val[k + 1]);   \
  vsub8(v.val[n], v_in.val[n], t.val[k + 1]);

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
* Name:        ntt
*
* Description: Inplace number-theoretic transform (NTT) in Rq.
*              input is in standard order, output is in bitreversed order
*
* Arguments:   - int16_t r[256]: pointer to input/output vector of elements of Zq
**************************************************/
void neon_ntt(int16_t r[256])
{
  int j, k = 0;
  // Register: Total 32 + 1 (const) = 33
  // vt1, vt2 and t are temporary registers
  int16x8x4_t vt1, vt2, v0, v1, v2, v3, z, z_qinv; // 32
  int16x8x2_t z2, z2_qinv;                         // 4
  int16x8x3_t t;                                   // 3
  int16x8_t neon_kyberq;                           // 1
  neon_kyberq = vdupq_n_s16(KYBER_Q);
  // End

  // Layer 7
  // Total registers: 32
  z2 = vld1q_s16_x2(&neon_zetas[k]);
  z2_qinv = vld1q_s16_x2(&neon_zetas_qinv[k]);
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

    fqmul_lane(vt1.val[0], v2.val[0], z2.val[1], z2_qinv.val[1], t, 7);
    fqmul_lane(vt1.val[1], v2.val[1], z2.val[1], z2_qinv.val[1], t, 7);
    fqmul_lane(vt1.val[2], v2.val[2], z2.val[1], z2_qinv.val[1], t, 7);
    fqmul_lane(vt1.val[3], v2.val[3], z2.val[1], z2_qinv.val[1], t, 7);

    fqmul_lane(vt2.val[0], v3.val[0], z2.val[1], z2_qinv.val[1], t, 7);
    fqmul_lane(vt2.val[1], v3.val[1], z2.val[1], z2_qinv.val[1], t, 7);
    fqmul_lane(vt2.val[2], v3.val[2], z2.val[1], z2_qinv.val[1], t, 7);
    fqmul_lane(vt2.val[3], v3.val[3], z2.val[1], z2_qinv.val[1], t, 7);

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

    fqmul_lane(vt1.val[0], v2.val[0], z2.val[0], z2_qinv.val[0], t, 0);
    fqmul_lane(vt1.val[1], v2.val[1], z2.val[0], z2_qinv.val[0], t, 0);
    fqmul_lane(vt1.val[2], v2.val[2], z2.val[0], z2_qinv.val[0], t, 0);
    fqmul_lane(vt1.val[3], v2.val[3], z2.val[0], z2_qinv.val[0], t, 0);

    fqmul_lane(vt2.val[0], v3.val[0], z2.val[0], z2_qinv.val[0], t, 0);
    fqmul_lane(vt2.val[1], v3.val[1], z2.val[0], z2_qinv.val[0], t, 0);
    fqmul_lane(vt2.val[2], v3.val[2], z2.val[0], z2_qinv.val[0], t, 0);
    fqmul_lane(vt2.val[3], v3.val[3], z2.val[0], z2_qinv.val[0], t, 0);

    // 64: 0 +- 64
    subadd_x4(v2, v0, vt1);
    // 96: 32 +- 96
    subadd_x4(v3, v1, vt2);

    // Layer 5: v0 x v1 | v2 x v3
    // v0: 0   -> 31
    // v1: 32  -> 63
    // v2: 64  -> 95
    // v3: 96  -> 127

    fqmul_lane(vt1.val[0], v1.val[0], z2.val[0], z2_qinv.val[0], t, 1);
    fqmul_lane(vt1.val[1], v1.val[1], z2.val[0], z2_qinv.val[0], t, 1);
    fqmul_lane(vt1.val[2], v1.val[2], z2.val[0], z2_qinv.val[0], t, 1);
    fqmul_lane(vt1.val[3], v1.val[3], z2.val[0], z2_qinv.val[0], t, 1);

    fqmul_lane(vt2.val[0], v3.val[0], z2.val[0], z2_qinv.val[0], t, 2);
    fqmul_lane(vt2.val[1], v3.val[1], z2.val[0], z2_qinv.val[0], t, 2);
    fqmul_lane(vt2.val[2], v3.val[2], z2.val[0], z2_qinv.val[0], t, 2);
    fqmul_lane(vt2.val[3], v3.val[3], z2.val[0], z2_qinv.val[0], t, 2);

    // 32: 0 +- 32
    subadd_x4(v1, v0, vt1);
    // 96: 64 +- 96
    subadd_x4(v3, v2, vt2);

    // Layer 4: v0.val[0] x v0.val[2] | v0.val[1] x v0.val[3]
    // val[0]: 0  -> 7
    // val[1]: 8  -> 15
    // val[2]: 16 -> 23
    // val[3]: 24 -> 32

    fqmul_lane(vt1.val[0], v0.val[2], z2.val[0], z2_qinv.val[0], t, 3);
    fqmul_lane(vt1.val[1], v0.val[3], z2.val[0], z2_qinv.val[0], t, 3);
    fqmul_lane(vt1.val[2], v1.val[2], z2.val[0], z2_qinv.val[0], t, 4);
    fqmul_lane(vt1.val[3], v1.val[3], z2.val[0], z2_qinv.val[0], t, 4);

    fqmul_lane(vt2.val[0], v2.val[2], z2.val[0], z2_qinv.val[0], t, 5);
    fqmul_lane(vt2.val[1], v2.val[3], z2.val[0], z2_qinv.val[0], t, 5);
    fqmul_lane(vt2.val[2], v3.val[2], z2.val[0], z2_qinv.val[0], t, 6);
    fqmul_lane(vt2.val[3], v3.val[3], z2.val[0], z2_qinv.val[0], t, 6);

    subadd(v0, 0, 2, 1, 3, vt1.val[0], vt1.val[1]);
    subadd(v1, 0, 2, 1, 3, vt1.val[2], vt1.val[3]);
    subadd(v2, 0, 2, 1, 3, vt2.val[0], vt2.val[1]);
    subadd(v3, 0, 2, 1, 3, vt2.val[2], vt2.val[3]);

    // Layer 3: v0.val[0] x v0.val[1] | v0.val[2] x v0.val[3]
    // val[0]: 0  -> 7
    // val[1]: 8  -> 15
    // val[2]: 16 -> 23
    // val[3]: 24 -> 32

    fqmul_lane(vt1.val[0], v0.val[1], z2.val[0], z2_qinv.val[0], t, 7);
    fqmul_lane(vt1.val[1], v0.val[3], z2.val[1], z2_qinv.val[1], t, 0);
    fqmul_lane(vt1.val[2], v1.val[1], z2.val[1], z2_qinv.val[1], t, 1);
    fqmul_lane(vt1.val[3], v1.val[3], z2.val[1], z2_qinv.val[1], t, 2);

    fqmul_lane(vt2.val[0], v2.val[1], z2.val[1], z2_qinv.val[1], t, 3);
    fqmul_lane(vt2.val[1], v2.val[3], z2.val[1], z2_qinv.val[1], t, 4);
    fqmul_lane(vt2.val[2], v3.val[1], z2.val[1], z2_qinv.val[1], t, 5);
    fqmul_lane(vt2.val[3], v3.val[3], z2.val[1], z2_qinv.val[1], t, 6);

    subadd(v0, 0, 1, 2, 3, vt1.val[0], vt1.val[1]);
    subadd(v1, 0, 1, 2, 3, vt1.val[2], vt1.val[3]);
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

    vloadx4(z, &neon_zetas[k + 16]);
    vloadx4(z_qinv, &neon_zetas_qinv[k + 16]);

    fqmul(vt1.val[1], vt1.val[1], z.val[0], z_qinv.val[0], t);
    fqmul(vt1.val[3], vt1.val[3], z.val[1], z_qinv.val[1], t);
    fqmul(vt2.val[1], vt2.val[1], z.val[2], z_qinv.val[2], t);
    fqmul(vt2.val[3], vt2.val[3], z.val[3], z_qinv.val[3], t);

    subadd_twist(v0, vt1, 0, 1, 2, 3);
    subadd_twist(v1, vt2, 0, 1, 2, 3);

    arrange(vt1, v2, 0, 2, 1, 3);
    arrange(vt2, v3, 0, 2, 1, 3);

    vloadx4(z, &neon_zetas[k + 48]);
    vloadx4(z_qinv, &neon_zetas_qinv[k + 48]);

    fqmul(vt1.val[1], vt1.val[1], z.val[0], z_qinv.val[0], t);
    fqmul(vt1.val[3], vt1.val[3], z.val[1], z_qinv.val[1], t);
    fqmul(vt2.val[1], vt2.val[1], z.val[2], z_qinv.val[2], t);
    fqmul(vt2.val[3], vt2.val[3], z.val[3], z_qinv.val[3], t);

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

    vloadx4(z, &neon_zetas[k + 80]);
    vloadx4(z_qinv, &neon_zetas_qinv[k + 80]);

    fqmul(vt1.val[0], v0.val[2], z.val[0], z_qinv.val[0], t);
    fqmul(vt1.val[1], v0.val[3], z.val[0], z_qinv.val[0], t);
    fqmul(vt1.val[2], v1.val[2], z.val[1], z_qinv.val[1], t);
    fqmul(vt1.val[3], v1.val[3], z.val[1], z_qinv.val[1], t);

    fqmul(vt2.val[0], v2.val[2], z.val[2], z_qinv.val[2], t);
    fqmul(vt2.val[1], v2.val[3], z.val[2], z_qinv.val[2], t);
    fqmul(vt2.val[2], v3.val[2], z.val[3], z_qinv.val[3], t);
    fqmul(vt2.val[3], v3.val[3], z.val[3], z_qinv.val[3], t);

    subadd(v0, 0, 2, 1, 3, vt1.val[0], vt1.val[1]);
    subadd(v1, 0, 2, 1, 3, vt1.val[2], vt1.val[3]);
    subadd(v2, 0, 2, 1, 3, vt2.val[0], vt2.val[1]);
    subadd(v3, 0, 2, 1, 3, vt2.val[2], vt2.val[3]);

    vstore4(&r[j], v0);
    vstore4(&r[j + 32], v1);
    vstore4(&r[j + 64], v2);
    vstore4(&r[j + 96], v3);

    k = 112;

    z2 = vld1q_s16_x2(&neon_zetas[k]);
    z2_qinv = vld1q_s16_x2(&neon_zetas_qinv[k]);
  }
}

/*************************************************
* Name:        invntt_tomont
*
* Description: Inplace inverse number-theoretic transform in Rq and
*              multiplication by Montgomery factor 2^16.
*              Input is in bitreversed order, output is in standard order
*
* Arguments:   - int16_t r[256] in {-(q-1)/2,...,(q-1)/2} 
*              pointer to input/output vector of elements of Zq
**************************************************/
void neon_invntt(int16_t r[256])
{
  int j, k = 0;
  // Register: Total 24 + 4(const) = 28
  int16x8x4_t t, v0, v1, v2, v3, z, z_qinv; // 24
  int16x8_t z2, z2_qinv;                    // 2
  // End
  int16x8_t neon_v, neon_kyberq, neon_one;
  neon_kyberq = vdupq_n_s16(KYBER_Q);
  neon_v = vdupq_n_s16(((1U << 26) + KYBER_Q / 2) / KYBER_Q);
  neon_one = vdupq_n_s16(1 << 9);

  const int16_t f = 1441;        // mont^2/128
  const int16_t f_qinv = -10079; // f*qinv mod 2^16

  // *Vectorize* barret_reduction over *64* points rather than 896=128*7 points
  // Optimimal Barret reduction for Kyber N=256, B=9 is 72 points, see here:
  // https://eprint.iacr.org/2020/1377.pdf

  // Layer 1, 2, 3, 4, 5, 6
  // Total register: 27
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

    addsub(v0, 0, 2, 1, 3, t, 0);
    addsub(v1, 0, 2, 1, 3, t, 2);
    addsub(v2, 0, 2, 1, 3, t, 0);
    addsub(v3, 0, 2, 1, 3, t, 2);

    vloadx4(z, &neon_zetas_inv[k]);
    vloadx4(z_qinv, &neon_zetas_inv_qinv[k]);

    fqmul(v0.val[2], v0.val[2], z.val[0], z_qinv.val[0], t);
    fqmul(v0.val[3], v0.val[3], z.val[0], z_qinv.val[0], t);
    fqmul(v1.val[2], v1.val[2], z.val[1], z_qinv.val[1], t);
    fqmul(v1.val[3], v1.val[3], z.val[1], z_qinv.val[1], t);

    fqmul(v2.val[2], v2.val[2], z.val[2], z_qinv.val[2], t);
    fqmul(v2.val[3], v2.val[3], z.val[2], z_qinv.val[2], t);
    fqmul(v3.val[2], v3.val[2], z.val[3], z_qinv.val[3], t);
    fqmul(v3.val[3], v3.val[3], z.val[3], z_qinv.val[3], t);

    // Layer 2: v0.val[0] x v0.val[1] | v0.val[2] x v0.val[3]
    // transpose 4x4
    transpose(v0, t);
    transpose(v1, t);
    transpose(v2, t);
    transpose(v3, t);
    // v0.val[0]: 0,  1,  2,  3  | 16,  17,  18,  19
    // v0.val[1]: 4,  5,  6,  7  | 20,  21,  22,  23
    // v0.val[2]: 8,  9,  10, 11 | 24,  25,  26,  27
    // v0.val[3]: 12, 13, 14, 15 | 28,  29,  30,  31

    addsub(v0, 0, 1, 2, 3, t, 0);
    addsub(v1, 0, 1, 2, 3, t, 2);
    addsub(v2, 0, 1, 2, 3, t, 0);
    addsub(v3, 0, 1, 2, 3, t, 2);

    vloadx4(z, &neon_zetas_inv[k + 32]);
    vloadx4(z_qinv, &neon_zetas_inv_qinv[k + 32]);

    fqmul(v0.val[1], v0.val[1], z.val[0], z_qinv.val[0], t);
    fqmul(v0.val[3], v0.val[3], z.val[1], z_qinv.val[1], t);
    fqmul(v1.val[1], v1.val[1], z.val[2], z_qinv.val[2], t);
    fqmul(v1.val[3], v1.val[3], z.val[3], z_qinv.val[3], t);

    vloadx4(z, &neon_zetas_inv[k + 64]);
    vloadx4(z_qinv, &neon_zetas_inv_qinv[k + 64]);

    fqmul(v2.val[1], v2.val[1], z.val[0], z_qinv.val[0], t);
    fqmul(v2.val[3], v2.val[3], z.val[1], z_qinv.val[1], t);
    fqmul(v3.val[1], v3.val[1], z.val[2], z_qinv.val[2], t);
    fqmul(v3.val[3], v3.val[3], z.val[3], z_qinv.val[3], t);

    // Layer 3 : v0.val[0] x v0.val[2] | v0.val[1] x v0.val[3]
    // v0.val[0]: 0,  1,  2,  3  | 16,  17,  18,  19
    // v0.val[1]: 4,  5,  6,  7  | 20,  21,  22,  23
    // v0.val[2]: 8,  9,  10, 11 | 24,  25,  26,  27
    // v0.val[3]: 12, 13, 14, 15 | 28,  29,  30,  31

    addsub(v0, 0, 2, 1, 3, t, 0);
    addsub(v1, 0, 2, 1, 3, t, 2);
    addsub(v2, 0, 2, 1, 3, t, 0);
    addsub(v3, 0, 2, 1, 3, t, 2);

    vloadx4(z, &neon_zetas_inv[k + 96]);
    vloadx4(z_qinv, &neon_zetas_inv_qinv[k + 96]);

    fqmul(v0.val[2], v0.val[2], z.val[0], z_qinv.val[0], t);
    fqmul(v0.val[3], v0.val[3], z.val[0], z_qinv.val[0], t);
    fqmul(v1.val[2], v1.val[2], z.val[1], z_qinv.val[1], t);
    fqmul(v1.val[3], v1.val[3], z.val[1], z_qinv.val[1], t);

    fqmul(v2.val[2], v2.val[2], z.val[2], z_qinv.val[2], t);
    fqmul(v2.val[3], v2.val[3], z.val[2], z_qinv.val[2], t);
    fqmul(v3.val[2], v3.val[2], z.val[3], z_qinv.val[3], t);
    fqmul(v3.val[3], v3.val[3], z.val[3], z_qinv.val[3], t);

    // Layer 4: v0.val[0] x v0.val[1] | v0.val[2] x v0.val[3]
    // Re-arrange vector

    // v2.val[0]: 0,  1,  2,  3  | 4,  5,  6,  7
    // v2.val[1]: 16, 17, 18, 19 | 20, 21, 22, 23
    // v2.val[2]: 8,  9,  10, 11 | 12, 13, 14, 15
    // v2.val[3]: 24, 25, 26, 27 | 28, 29, 30, 31

    arrange(t, v0, 0, 1, 2, 3);
    addsub_twist(v0, t, 0, 1, 2, 3, z, 0);
    arrange(t, v1, 0, 1, 2, 3);
    addsub_twist(v1, t, 0, 1, 2, 3, z, 2);

    arrange(t, v2, 0, 1, 2, 3);
    addsub_twist(v2, t, 0, 1, 2, 3, z, 0);
    arrange(t, v3, 0, 1, 2, 3);
    addsub_twist(v3, t, 0, 1, 2, 3, z, 2);

    // 0, 1, 2, 3: 8 points
    barrett_lo(v1.val[0], v3.val[0], t, 0);

    z2 = vld1q_s16(&neon_zetas_inv[k + 128]);
    z2_qinv = vld1q_s16(&neon_zetas_inv_qinv[k + 128]);

    fqmul_lane(v0.val[2], v0.val[2], z2, z2_qinv, t, 0);
    fqmul_lane(v0.val[3], v0.val[3], z2, z2_qinv, t, 0);
    fqmul_lane(v1.val[2], v1.val[2], z2, z2_qinv, t, 1);
    fqmul_lane(v1.val[3], v1.val[3], z2, z2_qinv, t, 1);

    fqmul_lane(v2.val[2], v2.val[2], z2, z2_qinv, t, 2);
    fqmul_lane(v2.val[3], v2.val[3], z2, z2_qinv, t, 2);
    fqmul_lane(v3.val[2], v3.val[2], z2, z2_qinv, t, 3);
    fqmul_lane(v3.val[3], v3.val[3], z2, z2_qinv, t, 3);

    // Layer 5: v0 x v1 | v2 x v3
    // v0: 0  -> 31
    // v1: 32 -> 63
    // v2: 64 -> 95
    // v3: 96 -> 127

    addsub_x4(v0, v1, t);
    addsub_x4(v2, v3, t);

    // 0...7: 16 points
    barrett(v0.val[0], t, 0);
    barrett(v2.val[0], t, 2);

    fqmul_lane(v1.val[0], v1.val[0], z2, z2_qinv, t, 4);
    fqmul_lane(v1.val[1], v1.val[1], z2, z2_qinv, t, 4);
    fqmul_lane(v1.val[2], v1.val[2], z2, z2_qinv, t, 4);
    fqmul_lane(v1.val[3], v1.val[3], z2, z2_qinv, t, 4);

    fqmul_lane(v3.val[0], v3.val[0], z2, z2_qinv, t, 5);
    fqmul_lane(v3.val[1], v3.val[1], z2, z2_qinv, t, 5);
    fqmul_lane(v3.val[2], v3.val[2], z2, z2_qinv, t, 5);
    fqmul_lane(v3.val[3], v3.val[3], z2, z2_qinv, t, 5);

    // Layer 6: v0 x v2 | v1 x v3
    // v0: 0  -> 31
    // v2: 64 -> 95
    // v1: 32 -> 63
    // v3: 96 -> 127

    addsub_x4(v0, v2, t);
    addsub_x4(v1, v3, t);

    // 8, 9, ... 15: 8 points
    barrett(v0.val[1], t, 0);

    fqmul_lane(v2.val[0], v2.val[0], z2, z2_qinv, t, 6);
    fqmul_lane(v2.val[1], v2.val[1], z2, z2_qinv, t, 6);
    fqmul_lane(v2.val[2], v2.val[2], z2, z2_qinv, t, 6);
    fqmul_lane(v2.val[3], v2.val[3], z2, z2_qinv, t, 6);

    fqmul_lane(v3.val[0], v3.val[0], z2, z2_qinv, t, 6);
    fqmul_lane(v3.val[1], v3.val[1], z2, z2_qinv, t, 6);
    fqmul_lane(v3.val[2], v3.val[2], z2, z2_qinv, t, 6);
    fqmul_lane(v3.val[3], v3.val[3], z2, z2_qinv, t, 6);

    vstorex4(&r[j], v0);
    vstorex4(&r[j + 32], v1);
    vstorex4(&r[j + 64], v2);
    vstorex4(&r[j + 96], v3);

    k += 136;
  }

  // Layer 7, inv_mul
  z.val[1] = vdupq_n_s16(f);
  z_qinv.val[1] = vdupq_n_s16(f_qinv);

  // After layer 7, no need for barrett_reduction
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

    addsub_x4(v0, v2, t);
    addsub_x4(v1, v3, t);

    // v2
    fqmul_lane(v2.val[0], v2.val[0], z2, z2_qinv, t, 7);
    fqmul_lane(v2.val[1], v2.val[1], z2, z2_qinv, t, 7);
    fqmul_lane(v2.val[2], v2.val[2], z2, z2_qinv, t, 7);
    fqmul_lane(v2.val[3], v2.val[3], z2, z2_qinv, t, 7);

    // v3
    fqmul_lane(v3.val[0], v3.val[0], z2, z2_qinv, t, 7);
    fqmul_lane(v3.val[1], v3.val[1], z2, z2_qinv, t, 7);
    fqmul_lane(v3.val[2], v3.val[2], z2, z2_qinv, t, 7);
    fqmul_lane(v3.val[3], v3.val[3], z2, z2_qinv, t, 7);

    // v0
    fqmul(v0.val[0], v0.val[0], z.val[1], z_qinv.val[1], t);
    fqmul(v0.val[1], v0.val[1], z.val[1], z_qinv.val[1], t);
    fqmul(v0.val[2], v0.val[2], z.val[1], z_qinv.val[1], t);
    fqmul(v0.val[3], v0.val[3], z.val[1], z_qinv.val[1], t);

    // v1
    fqmul(v1.val[0], v1.val[0], z.val[1], z_qinv.val[1], t);
    fqmul(v1.val[1], v1.val[1], z.val[1], z_qinv.val[1], t);
    fqmul(v1.val[2], v1.val[2], z.val[1], z_qinv.val[1], t);
    fqmul(v1.val[3], v1.val[3], z.val[1], z_qinv.val[1], t);

    vstorex4(&r[j + 0], v0);
    vstorex4(&r[j + 32], v1);
    vstorex4(&r[j + 128], v2);
    vstorex4(&r[j + 160], v3);
  }
}
