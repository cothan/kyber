#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include "params.h"
#include "neon_ntt.h"
#include "reduce.h"

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

  a = (int32_t) b*c;
  (a_L, a_H) = a
  a_L = a_L * QINV;
  t = a_L * Q;
  (t_L, t_H) = t;
  return t_H - a_H;

  a = (int32_t) b*c;
  (a_L, a_H) = a
  a_L = a_L * QINV;
  t = a_L * Q;
  (t_L, t_H) = t;
  return t_H - a_H;
}
*************************************************/
// static void print_vector(int16x8x4_t a, int bound, const char *string)
// {
//   for (int i = 0; i < bound; i++)
//   {
//     for (int j = 0; j < 8; j++)
//     {
//       printf("%d, ", (int16_t)(a.val[i][j] & 0xffff));
//     }
//     printf("\\\\ %s", string);
//     printf("\n");
//   }
// }

// static void print_vector(int16x8_t a, const char *string)
// {
//   for (int j = 0; j < 8; j++)
//   {
//     printf("%d, ", (int16_t)(a[j] & 0xffff));
//   }
//   printf("\\\\ %s\n", string);
// }
  

  // print_vector(in, "in fqmul");                                     
  // print_vector(t.val[1], "zeta_qinv fqmul");                        
  // print_vector(t.val[2], "output fqmul");                           
#define fqmul1(out, in, zeta, t)                                    \
  t.val[0] = vqdmulhq_s16(in, zeta);     /* (2*a)_H */              \
  t.val[1] = vmulq_s16(neon_qinv, zeta); /* a_L */                  \
  t.val[2] = vmulq_s16(t.val[1], in);    /* a_L = a_L * QINV */     \
  t.val[3] = vqdmulhq_s16(t.val[2], neon_kyberq); /* (2*a_L*Q)_H */ \
  out = vhsubq_s16(t.val[0], t.val[3]);           /* ((2*a)_H - (2*a_L*Q)_H)/2 */

  // print_vector(in, "in fqmul");                                  
  // print_vector(zeta_qinv, "zeta_qinv fqmul");                    
  // print_vector(t.val[1], "output fqmul");                        
#define fqmul2(out, in, zeta, zeta_qinv, t)                         \
  t.val[0] = vqdmulhq_s16(in, zeta);   /* (2*a)_H */                \
  t.val[1] = vmulq_s16(zeta_qinv, in); /* a_L */                    \
  t.val[2] = vqdmulhq_s16(t.val[1], neon_kyberq); /* (2*a_L*Q)_H */ \
  out = vhsubq_s16(t.val[0], t.val[2]);           /* ((2*a)_H - (2*a_L*Q)_H)/2 */

#define fqmul2_lane(out, in, zeta, zeta_qinv, t, lane)                         \
  t.val[0] = vqdmulhq_laneq_s16(in, zeta, lane);   /* (2*a)_H */                \
  t.val[1] = vmulq_laneq_s16(in, zeta_qinv, lane); /* a_L */                    \
  t.val[2] = vqdmulhq_s16(t.val[1], neon_kyberq); /* (2*a_L*Q)_H */ \
  out = vhsubq_s16(t.val[0], t.val[2]);           /* ((2*a)_H - (2*a_L*Q)_H)/2 */


// #define fqmul2(out, in, zeta, zeta_qinv, t)                                    \
//   t.val[0] = (int16x8_t)vqdmulhq_s16(in, zeta); /* (2*a)_H */                  \
//   t.val[0] = vshrq_n_s16(t.val[0], 1);                                         \
//   t.val[1] = vmulq_s16(in, zeta_qinv);                       /* a_L */         \
//   t.val[2] = (int16x8_t)vqdmulhq_s16(t.val[1], neon_kyberq); /* (2*a_L*Q)_H */ \
//   t.val[2] = vshrq_n_s16(t.val[2], 1);                                         \
//   out = vsubq_s16(t.val[0], t.val[2]);

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

void test_neon_ntt(int16_t r[256])
{
  int j, k = 0;
  // Register: Total 32 + 2 (const) = 34
  int16x8x4_t t, vt1, vt2, v0, v1, v2, v3, z; // 32
  int16x8_t neon_qinv, neon_kyberq;           // 2
  neon_qinv = vdupq_n_s16(QINV);
  neon_kyberq = vdupq_n_s16(KYBER_Q);
  // End

  // Layer 7
  // Total registers: 32
  z.val[0] = vdupq_n_s16(neon_zetas[15]);
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

    fqmul1(vt1.val[0], v2.val[0], z.val[0], t);
    fqmul1(vt1.val[1], v2.val[1], z.val[0], t);
    fqmul1(vt1.val[2], v2.val[2], z.val[0], t);
    fqmul1(vt1.val[3], v2.val[3], z.val[0], t);

    fqmul1(vt2.val[0], v3.val[0], z.val[0], t);
    fqmul1(vt2.val[1], v3.val[1], z.val[0], t);
    fqmul1(vt2.val[2], v3.val[2], z.val[0], t);
    fqmul1(vt2.val[3], v3.val[3], z.val[0], t);

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

    z.val[0] = vdupq_n_s16(neon_zetas[k]);

    fqmul1(vt1.val[0], v2.val[0], z.val[0], t);
    fqmul1(vt1.val[1], v2.val[1], z.val[0], t);
    fqmul1(vt1.val[2], v2.val[2], z.val[0], t);
    fqmul1(vt1.val[3], v2.val[3], z.val[0], t);

    fqmul1(vt2.val[0], v3.val[0], z.val[0], t);
    fqmul1(vt2.val[1], v3.val[1], z.val[0], t);
    fqmul1(vt2.val[2], v3.val[2], z.val[0], t);
    fqmul1(vt2.val[3], v3.val[3], z.val[0], t);

    // 64: 0 +- 64
    subadd_x4(v2, v0, vt1);
    // 96: 32 +- 96
    subadd_x4(v3, v1, vt2);

    // Layer 5: v0 x v1 | v2 x v3
    // v0: 0   -> 31
    // v1: 32  -> 63
    // v2: 64  -> 95
    // v3: 96  -> 127

    z.val[0] = vdupq_n_s16(neon_zetas[k + 1]);
    z.val[1] = vdupq_n_s16(neon_zetas[k + 2]);

    fqmul1(vt1.val[0], v1.val[0], z.val[0], t);
    fqmul1(vt1.val[1], v1.val[1], z.val[0], t);
    fqmul1(vt1.val[2], v1.val[2], z.val[0], t);
    fqmul1(vt1.val[3], v1.val[3], z.val[0], t);

    fqmul1(vt2.val[0], v3.val[0], z.val[1], t);
    fqmul1(vt2.val[1], v3.val[1], z.val[1], t);
    fqmul1(vt2.val[2], v3.val[2], z.val[1], t);
    fqmul1(vt2.val[3], v3.val[3], z.val[1], t);

    // 32: 0 +- 32
    subadd_x4(v1, v0, vt1);
    // 96: 64 +- 96
    subadd_x4(v3, v2, vt2);

    // Layer 4: v0.val[0] x v0.val[2] | v0.val[1] x v0.val[3]
    // val[0]: 0  -> 7
    // val[1]: 8  -> 15
    // val[2]: 16 -> 23
    // val[3]: 24 -> 32
    z.val[0] = vdupq_n_s16(neon_zetas[k + 3]);
    z.val[1] = vdupq_n_s16(neon_zetas[k + 4]);
    z.val[2] = vdupq_n_s16(neon_zetas[k + 5]);
    z.val[3] = vdupq_n_s16(neon_zetas[k + 6]);

    fqmul1(vt1.val[0], v0.val[2], z.val[0], t);
    fqmul1(vt1.val[1], v0.val[3], z.val[0], t);
    fqmul1(vt1.val[2], v1.val[2], z.val[1], t);
    fqmul1(vt1.val[3], v1.val[3], z.val[1], t);

    fqmul1(vt2.val[0], v2.val[2], z.val[2], t);
    fqmul1(vt2.val[1], v2.val[3], z.val[2], t);
    fqmul1(vt2.val[2], v3.val[2], z.val[3], t);
    fqmul1(vt2.val[3], v3.val[3], z.val[3], t);

    subadd(v0, 0, 2, 1, 3, vt1.val[0], vt1.val[1]);
    subadd(v1, 0, 2, 1, 3, vt1.val[2], vt1.val[3]);
    subadd(v2, 0, 2, 1, 3, vt2.val[0], vt2.val[1]);
    subadd(v3, 0, 2, 1, 3, vt2.val[2], vt2.val[3]);

    // Layer 3: v0.val[0] x v0.val[1] | v0.val[2] x v0.val[3]
    // val[0]: 0  -> 7
    // val[1]: 8  -> 15
    // val[2]: 16 -> 23
    // val[3]: 24 -> 32
    z.val[0] = vdupq_n_s16(neon_zetas[k + 7]);
    z.val[1] = vdupq_n_s16(neon_zetas[k + 8]);
    z.val[2] = vdupq_n_s16(neon_zetas[k + 9]);
    z.val[3] = vdupq_n_s16(neon_zetas[k + 10]);

    fqmul1(vt1.val[0], v0.val[1], z.val[0], t);
    fqmul1(vt1.val[1], v0.val[3], z.val[1], t);
    fqmul1(vt1.val[2], v1.val[1], z.val[2], t);
    fqmul1(vt1.val[3], v1.val[3], z.val[3], t);

    subadd(v0, 0, 1, 2, 3, vt1.val[0], vt1.val[1]);
    subadd(v1, 0, 1, 2, 3, vt1.val[2], vt1.val[3]);

    z.val[0] = vdupq_n_s16(neon_zetas[k + 11]);
    z.val[1] = vdupq_n_s16(neon_zetas[k + 12]);
    z.val[2] = vdupq_n_s16(neon_zetas[k + 13]);
    z.val[3] = vdupq_n_s16(neon_zetas[k + 14]);

    fqmul1(vt2.val[0], v2.val[1], z.val[0], t);
    fqmul1(vt2.val[1], v2.val[3], z.val[1], t);
    fqmul1(vt2.val[2], v3.val[1], z.val[2], t);
    fqmul1(vt2.val[3], v3.val[3], z.val[3], t);

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

    fqmul1(vt1.val[1], vt1.val[1], z.val[0], t);
    fqmul1(vt1.val[3], vt1.val[3], z.val[1], t);

    fqmul1(vt2.val[1], vt2.val[1], z.val[2], t);
    fqmul1(vt2.val[3], vt2.val[3], z.val[3], t);

    subadd_twist(v0, vt1, 0, 1, 2, 3);
    subadd_twist(v1, vt2, 0, 1, 2, 3);

    arrange(vt1, v2, 0, 2, 1, 3);
    arrange(vt2, v3, 0, 2, 1, 3);

    vloadx4(z, &neon_zetas[k + 48]);

    fqmul1(vt1.val[1], vt1.val[1], z.val[0], t);
    fqmul1(vt1.val[3], vt1.val[3], z.val[1], t);
    fqmul1(vt2.val[1], vt2.val[1], z.val[2], t);
    fqmul1(vt2.val[3], vt2.val[3], z.val[3], t);

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

    fqmul1(vt1.val[0], v0.val[2], z.val[0], t);
    fqmul1(vt1.val[1], v0.val[3], z.val[0], t);
    fqmul1(vt1.val[2], v1.val[2], z.val[1], t);
    fqmul1(vt1.val[3], v1.val[3], z.val[1], t);

    fqmul1(vt2.val[0], v2.val[2], z.val[2], t);
    fqmul1(vt2.val[1], v2.val[3], z.val[2], t);
    fqmul1(vt2.val[2], v3.val[2], z.val[3], t);
    fqmul1(vt2.val[3], v3.val[3], z.val[3], t);

    subadd(v0, 0, 2, 1, 3, vt1.val[0], vt1.val[1]);
    subadd(v1, 0, 2, 1, 3, vt1.val[2], vt1.val[3]);
    subadd(v2, 0, 2, 1, 3, vt2.val[0], vt2.val[1]);
    subadd(v3, 0, 2, 1, 3, vt2.val[2], vt2.val[3]);

    vstore4(&r[j], v0);
    vstore4(&r[j + 32], v1);
    vstore4(&r[j + 64], v2);
    vstore4(&r[j + 96], v3);

    k += 112;
  }
}

void test_neon_ntt_qinv(int16_t r[256])
{
  int j, k = 0;
  // Register: Total 32 + 2 (const) = 34
  int16x8x4_t t, vt1, vt2, v0, v1, v2, v3, z, z_qinv; // 32
  int16x8_t neon_kyberq;                   // 2
  // neon_qinv = vdupq_n_s16(QINV);
  neon_kyberq = vdupq_n_s16(KYBER_Q);
  // End

  // Layer 7
  // Total registers: 32
  z.val[0] = vdupq_n_s16(neon_zetas[15]);
  z_qinv.val[0] = vdupq_n_s16(neon_zetas_qinv[15]);
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

    fqmul2(vt1.val[0], v2.val[0], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[1], v2.val[1], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[2], v2.val[2], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[3], v2.val[3], z.val[0], z_qinv.val[0], t);

    fqmul2(vt2.val[0], v3.val[0], z.val[0], z_qinv.val[0], t);
    fqmul2(vt2.val[1], v3.val[1], z.val[0], z_qinv.val[0], t);
    fqmul2(vt2.val[2], v3.val[2], z.val[0], z_qinv.val[0], t);
    fqmul2(vt2.val[3], v3.val[3], z.val[0], z_qinv.val[0], t);

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

    z.val[0] = vdupq_n_s16(neon_zetas[k]);
    z_qinv.val[0] = vdupq_n_s16(neon_zetas_qinv[k]);

    fqmul2(vt1.val[0], v2.val[0], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[1], v2.val[1], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[2], v2.val[2], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[3], v2.val[3], z.val[0], z_qinv.val[0], t);

    fqmul2(vt2.val[0], v3.val[0], z.val[0], z_qinv.val[0], t);
    fqmul2(vt2.val[1], v3.val[1], z.val[0], z_qinv.val[0], t);
    fqmul2(vt2.val[2], v3.val[2], z.val[0], z_qinv.val[0], t);
    fqmul2(vt2.val[3], v3.val[3], z.val[0], z_qinv.val[0], t);

    // 64: 0 +- 64
    subadd_x4(v2, v0, vt1);
    // 96: 32 +- 96
    subadd_x4(v3, v1, vt2);

    // Layer 5: v0 x v1 | v2 x v3
    // v0: 0   -> 31
    // v1: 32  -> 63
    // v2: 64  -> 95
    // v3: 96  -> 127

    z.val[0] = vdupq_n_s16(neon_zetas[k + 1]);
    z.val[1] = vdupq_n_s16(neon_zetas[k + 2]);
    z_qinv.val[0] = vdupq_n_s16(neon_zetas_qinv[k + 1]);
    z_qinv.val[1] = vdupq_n_s16(neon_zetas_qinv[k + 2]);

    fqmul2(vt1.val[0], v1.val[0], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[1], v1.val[1], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[2], v1.val[2], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[3], v1.val[3], z.val[0], z_qinv.val[0], t);

    fqmul2(vt2.val[0], v3.val[0], z.val[1], z_qinv.val[1], t);
    fqmul2(vt2.val[1], v3.val[1], z.val[1], z_qinv.val[1], t);
    fqmul2(vt2.val[2], v3.val[2], z.val[1], z_qinv.val[1], t);
    fqmul2(vt2.val[3], v3.val[3], z.val[1], z_qinv.val[1], t);

    // 32: 0 +- 32
    subadd_x4(v1, v0, vt1);
    // 96: 64 +- 96
    subadd_x4(v3, v2, vt2);

    // Layer 4: v0.val[0] x v0.val[2] | v0.val[1] x v0.val[3]
    // val[0]: 0  -> 7
    // val[1]: 8  -> 15
    // val[2]: 16 -> 23
    // val[3]: 24 -> 32
    z.val[0] = vdupq_n_s16(neon_zetas[k + 3]);
    z.val[1] = vdupq_n_s16(neon_zetas[k + 4]);
    z.val[2] = vdupq_n_s16(neon_zetas[k + 5]);
    z.val[3] = vdupq_n_s16(neon_zetas[k + 6]);
    z_qinv.val[0] = vdupq_n_s16(neon_zetas_qinv[k + 3]);
    z_qinv.val[1] = vdupq_n_s16(neon_zetas_qinv[k + 4]);
    z_qinv.val[2] = vdupq_n_s16(neon_zetas_qinv[k + 5]);
    z_qinv.val[3] = vdupq_n_s16(neon_zetas_qinv[k + 6]);

    fqmul2(vt1.val[0], v0.val[2], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[1], v0.val[3], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[2], v1.val[2], z.val[1], z_qinv.val[1], t);
    fqmul2(vt1.val[3], v1.val[3], z.val[1], z_qinv.val[1], t);

    fqmul2(vt2.val[0], v2.val[2], z.val[2], z_qinv.val[2], t);
    fqmul2(vt2.val[1], v2.val[3], z.val[2], z_qinv.val[2], t);
    fqmul2(vt2.val[2], v3.val[2], z.val[3], z_qinv.val[3], t);
    fqmul2(vt2.val[3], v3.val[3], z.val[3], z_qinv.val[3], t);

    subadd(v0, 0, 2, 1, 3, vt1.val[0], vt1.val[1]);
    subadd(v1, 0, 2, 1, 3, vt1.val[2], vt1.val[3]);
    subadd(v2, 0, 2, 1, 3, vt2.val[0], vt2.val[1]);
    subadd(v3, 0, 2, 1, 3, vt2.val[2], vt2.val[3]);

    // Layer 3: v0.val[0] x v0.val[1] | v0.val[2] x v0.val[3]
    // val[0]: 0  -> 7
    // val[1]: 8  -> 15
    // val[2]: 16 -> 23
    // val[3]: 24 -> 32
    z.val[0] = vdupq_n_s16(neon_zetas[k + 7]);
    z.val[1] = vdupq_n_s16(neon_zetas[k + 8]);
    z.val[2] = vdupq_n_s16(neon_zetas[k + 9]);
    z.val[3] = vdupq_n_s16(neon_zetas[k + 10]);
    z_qinv.val[0] = vdupq_n_s16(neon_zetas_qinv[k + 7]);
    z_qinv.val[1] = vdupq_n_s16(neon_zetas_qinv[k + 8]);
    z_qinv.val[2] = vdupq_n_s16(neon_zetas_qinv[k + 9]);
    z_qinv.val[3] = vdupq_n_s16(neon_zetas_qinv[k + 10]);

    fqmul2(vt1.val[0], v0.val[1], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[1], v0.val[3], z.val[1], z_qinv.val[1], t);
    fqmul2(vt1.val[2], v1.val[1], z.val[2], z_qinv.val[2], t);
    fqmul2(vt1.val[3], v1.val[3], z.val[3], z_qinv.val[3], t);

    subadd(v0, 0, 1, 2, 3, vt1.val[0], vt1.val[1]);
    subadd(v1, 0, 1, 2, 3, vt1.val[2], vt1.val[3]);

    z.val[0] = vdupq_n_s16(neon_zetas[k + 11]);
    z.val[1] = vdupq_n_s16(neon_zetas[k + 12]);
    z.val[2] = vdupq_n_s16(neon_zetas[k + 13]);
    z.val[3] = vdupq_n_s16(neon_zetas[k + 14]);
    z_qinv.val[0] = vdupq_n_s16(neon_zetas_qinv[k + 11]);
    z_qinv.val[1] = vdupq_n_s16(neon_zetas_qinv[k + 12]);
    z_qinv.val[2] = vdupq_n_s16(neon_zetas_qinv[k + 13]);
    z_qinv.val[3] = vdupq_n_s16(neon_zetas_qinv[k + 14]);

    fqmul2(vt2.val[0], v2.val[1], z.val[0], z_qinv.val[0], t);
    fqmul2(vt2.val[1], v2.val[3], z.val[1], z_qinv.val[1], t);
    fqmul2(vt2.val[2], v3.val[1], z.val[2], z_qinv.val[2], t);
    fqmul2(vt2.val[3], v3.val[3], z.val[3], z_qinv.val[3], t);

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

    fqmul2(vt1.val[1], vt1.val[1], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[3], vt1.val[3], z.val[1], z_qinv.val[1], t);

    fqmul2(vt2.val[1], vt2.val[1], z.val[2], z_qinv.val[2], t);
    fqmul2(vt2.val[3], vt2.val[3], z.val[3], z_qinv.val[3], t);

    subadd_twist(v0, vt1, 0, 1, 2, 3);
    subadd_twist(v1, vt2, 0, 1, 2, 3);

    arrange(vt1, v2, 0, 2, 1, 3);
    arrange(vt2, v3, 0, 2, 1, 3);

    vloadx4(z, &neon_zetas[k + 48]);
    vloadx4(z_qinv, &neon_zetas_qinv[k + 48]);

    fqmul2(vt1.val[1], vt1.val[1], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[3], vt1.val[3], z.val[1], z_qinv.val[1], t);
    fqmul2(vt2.val[1], vt2.val[1], z.val[2], z_qinv.val[2], t);
    fqmul2(vt2.val[3], vt2.val[3], z.val[3], z_qinv.val[3], t);

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

    fqmul2(vt1.val[0], v0.val[2], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[1], v0.val[3], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[2], v1.val[2], z.val[1], z_qinv.val[1], t);
    fqmul2(vt1.val[3], v1.val[3], z.val[1], z_qinv.val[1], t);

    fqmul2(vt2.val[0], v2.val[2], z.val[2], z_qinv.val[2], t);
    fqmul2(vt2.val[1], v2.val[3], z.val[2], z_qinv.val[2], t);
    fqmul2(vt2.val[2], v3.val[2], z.val[3], z_qinv.val[3], t);
    fqmul2(vt2.val[3], v3.val[3], z.val[3], z_qinv.val[3], t);

    subadd(v0, 0, 2, 1, 3, vt1.val[0], vt1.val[1]);
    subadd(v1, 0, 2, 1, 3, vt1.val[2], vt1.val[3]);
    subadd(v2, 0, 2, 1, 3, vt2.val[0], vt2.val[1]);
    subadd(v3, 0, 2, 1, 3, vt2.val[2], vt2.val[3]);

    vstore4(&r[j], v0);
    vstore4(&r[j + 32], v1);
    vstore4(&r[j + 64], v2);
    vstore4(&r[j + 96], v3);

    k += 112;
  }
}


void test_neon_ntt_qinv_lane(int16_t r[256])
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

    fqmul2_lane(vt1.val[0], v2.val[0], z2.val[1], z2_qinv.val[1], t, 7);
    fqmul2_lane(vt1.val[1], v2.val[1], z2.val[1], z2_qinv.val[1], t, 7);
    fqmul2_lane(vt1.val[2], v2.val[2], z2.val[1], z2_qinv.val[1], t, 7);
    fqmul2_lane(vt1.val[3], v2.val[3], z2.val[1], z2_qinv.val[1], t, 7);

    fqmul2_lane(vt2.val[0], v3.val[0], z2.val[1], z2_qinv.val[1], t, 7);
    fqmul2_lane(vt2.val[1], v3.val[1], z2.val[1], z2_qinv.val[1], t, 7);
    fqmul2_lane(vt2.val[2], v3.val[2], z2.val[1], z2_qinv.val[1], t, 7);
    fqmul2_lane(vt2.val[3], v3.val[3], z2.val[1], z2_qinv.val[1], t, 7);

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

    fqmul2_lane(vt1.val[0], v2.val[0], z2.val[0], z2_qinv.val[0], t, 0);
    fqmul2_lane(vt1.val[1], v2.val[1], z2.val[0], z2_qinv.val[0], t, 0);
    fqmul2_lane(vt1.val[2], v2.val[2], z2.val[0], z2_qinv.val[0], t, 0);
    fqmul2_lane(vt1.val[3], v2.val[3], z2.val[0], z2_qinv.val[0], t, 0);

    fqmul2_lane(vt2.val[0], v3.val[0], z2.val[0], z2_qinv.val[0], t, 0);
    fqmul2_lane(vt2.val[1], v3.val[1], z2.val[0], z2_qinv.val[0], t, 0);
    fqmul2_lane(vt2.val[2], v3.val[2], z2.val[0], z2_qinv.val[0], t, 0);
    fqmul2_lane(vt2.val[3], v3.val[3], z2.val[0], z2_qinv.val[0], t, 0);

    // 64: 0 +- 64
    subadd_x4(v2, v0, vt1);
    // 96: 32 +- 96
    subadd_x4(v3, v1, vt2);

    // Layer 5: v0 x v1 | v2 x v3
    // v0: 0   -> 31
    // v1: 32  -> 63
    // v2: 64  -> 95
    // v3: 96  -> 127

    fqmul2_lane(vt1.val[0], v1.val[0], z2.val[0], z2_qinv.val[0], t, 1);
    fqmul2_lane(vt1.val[1], v1.val[1], z2.val[0], z2_qinv.val[0], t, 1);
    fqmul2_lane(vt1.val[2], v1.val[2], z2.val[0], z2_qinv.val[0], t, 1);
    fqmul2_lane(vt1.val[3], v1.val[3], z2.val[0], z2_qinv.val[0], t, 1);

    fqmul2_lane(vt2.val[0], v3.val[0], z2.val[0], z2_qinv.val[0], t, 2);
    fqmul2_lane(vt2.val[1], v3.val[1], z2.val[0], z2_qinv.val[0], t, 2);
    fqmul2_lane(vt2.val[2], v3.val[2], z2.val[0], z2_qinv.val[0], t, 2);
    fqmul2_lane(vt2.val[3], v3.val[3], z2.val[0], z2_qinv.val[0], t, 2);

    // 32: 0 +- 32
    subadd_x4(v1, v0, vt1);
    // 96: 64 +- 96
    subadd_x4(v3, v2, vt2);

    // Layer 4: v0.val[0] x v0.val[2] | v0.val[1] x v0.val[3]
    // val[0]: 0  -> 7
    // val[1]: 8  -> 15
    // val[2]: 16 -> 23
    // val[3]: 24 -> 32

    fqmul2_lane(vt1.val[0], v0.val[2], z2.val[0], z2_qinv.val[0], t, 3);
    fqmul2_lane(vt1.val[1], v0.val[3], z2.val[0], z2_qinv.val[0], t, 3);
    fqmul2_lane(vt1.val[2], v1.val[2], z2.val[0], z2_qinv.val[0], t, 4);
    fqmul2_lane(vt1.val[3], v1.val[3], z2.val[0], z2_qinv.val[0], t, 4);

    fqmul2_lane(vt2.val[0], v2.val[2], z2.val[0], z2_qinv.val[0], t, 5);
    fqmul2_lane(vt2.val[1], v2.val[3], z2.val[0], z2_qinv.val[0], t, 5);
    fqmul2_lane(vt2.val[2], v3.val[2], z2.val[0], z2_qinv.val[0], t, 6);
    fqmul2_lane(vt2.val[3], v3.val[3], z2.val[0], z2_qinv.val[0], t, 6);

    subadd(v0, 0, 2, 1, 3, vt1.val[0], vt1.val[1]);
    subadd(v1, 0, 2, 1, 3, vt1.val[2], vt1.val[3]);
    subadd(v2, 0, 2, 1, 3, vt2.val[0], vt2.val[1]);
    subadd(v3, 0, 2, 1, 3, vt2.val[2], vt2.val[3]);

    // Layer 3: v0.val[0] x v0.val[1] | v0.val[2] x v0.val[3]
    // val[0]: 0  -> 7
    // val[1]: 8  -> 15
    // val[2]: 16 -> 23
    // val[3]: 24 -> 32

    fqmul2_lane(vt1.val[0], v0.val[1], z2.val[0], z2_qinv.val[0], t, 7);
    fqmul2_lane(vt1.val[1], v0.val[3], z2.val[1], z2_qinv.val[1], t, 0);
    fqmul2_lane(vt1.val[2], v1.val[1], z2.val[1], z2_qinv.val[1], t, 1);
    fqmul2_lane(vt1.val[3], v1.val[3], z2.val[1], z2_qinv.val[1], t, 2);

    fqmul2_lane(vt2.val[0], v2.val[1], z2.val[1], z2_qinv.val[1], t, 3);
    fqmul2_lane(vt2.val[1], v2.val[3], z2.val[1], z2_qinv.val[1], t, 4);
    fqmul2_lane(vt2.val[2], v3.val[1], z2.val[1], z2_qinv.val[1], t, 5);
    fqmul2_lane(vt2.val[3], v3.val[3], z2.val[1], z2_qinv.val[1], t, 6);

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

    fqmul2(vt1.val[1], vt1.val[1], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[3], vt1.val[3], z.val[1], z_qinv.val[1], t);
    fqmul2(vt2.val[1], vt2.val[1], z.val[2], z_qinv.val[2], t);
    fqmul2(vt2.val[3], vt2.val[3], z.val[3], z_qinv.val[3], t);

    subadd_twist(v0, vt1, 0, 1, 2, 3);
    subadd_twist(v1, vt2, 0, 1, 2, 3);

    arrange(vt1, v2, 0, 2, 1, 3);
    arrange(vt2, v3, 0, 2, 1, 3);

    vloadx4(z, &neon_zetas[k + 48]);
    vloadx4(z_qinv, &neon_zetas_qinv[k + 48]);

    fqmul2(vt1.val[1], vt1.val[1], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[3], vt1.val[3], z.val[1], z_qinv.val[1], t);
    fqmul2(vt2.val[1], vt2.val[1], z.val[2], z_qinv.val[2], t);
    fqmul2(vt2.val[3], vt2.val[3], z.val[3], z_qinv.val[3], t);

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

    fqmul2(vt1.val[0], v0.val[2], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[1], v0.val[3], z.val[0], z_qinv.val[0], t);
    fqmul2(vt1.val[2], v1.val[2], z.val[1], z_qinv.val[1], t);
    fqmul2(vt1.val[3], v1.val[3], z.val[1], z_qinv.val[1], t);

    fqmul2(vt2.val[0], v2.val[2], z.val[2], z_qinv.val[2], t);
    fqmul2(vt2.val[1], v2.val[3], z.val[2], z_qinv.val[2], t);
    fqmul2(vt2.val[2], v3.val[2], z.val[3], z_qinv.val[3], t);
    fqmul2(vt2.val[3], v3.val[3], z.val[3], z_qinv.val[3], t);

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
