#include <arm_neon.h>
#include "params.h"
#include "reduce.h"
#include "ntt.h"
#include "polyvec.h"

void neon_polyvec_ntt(polyvec *r)
{
  unsigned int i;
  for(i=0;i<KYBER_K;i++)
  {
    neon_ntt(r->vec[i].coeffs);
    neon_poly_reduce(&r->vec[i]);
  }
}

void neon_polyvec_invntt_to_mont(polyvec *r)
{
  unsigned int i;
  for(i=0;i<KYBER_K;i++)
    neon_invntt(r->vec[i].coeffs);
}


void neon_polyvec_add_reduce(polyvec *c, const polyvec *a) {
  unsigned int i;
  for (i = 0; i < KYBER_K; i++) {
    // c = c + a;
    // c = reduce(c);
    neon_poly_add_reduce(&c->vec[i], &a->vec[i]);
  }
}

/******8888888888888888888888888***************/
// Load int16x8x2_t c <= ptr*
#define vload16(c, ptr) c = vld1q_s16_x2(ptr);

// Load int16x8x2_t c <= ptr*
#define vstore16(ptr, c) vst1q_s16_x2(ptr, c);

// Load int16x4_t c <= ptr*
#define vload4(c, ptr) c = vld1_s16(ptr);

// Combine in16x8_t c: low | high
#define vcombine(c, low, high) c = vcombine_s16(low, high);

// c (int16x4) = a + b (int16x4)
#define vadd4(c, a, b) c = vadd_s16(a, b);

// c (int16x8) = a + b (int16x8)
#define vadd8(c, a, b) c = vaddq_s16(a, b);

// get_low c (int16x4) = low(a) (int16x8)
#define vlo(c, a) c = vget_low_s16(a);

// get_high c (int16x4) = high(a) (int16x8)
#define vhi(c, a) c = vget_high_s16(a);

// c = ~a
#define vnot4(c, a) c = vmvn_s16(a);

/*
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
*/
#define fqmul(inout, zeta, t, a, u, neon_qinv, neon_kyberq)                    \
  a = vmull_s16(inout, zeta);                                                  \
  u = vmulq_s32(a, neon_qinv);                                                 \
  u = vshrq_n_s32(u, 16);                                                      \
  t = vmlaq_s32(a, neon_kyberq, u);                                            \
  t = vshrq_n_s32(t, 16);                                                      \
  inout = vmovn_s32(t);

/*
inout: int16x8_t
t : int16x8_t
t_lo: int16x4_t
t_hi: int16x4_t
t1: int32x4_t
t2: int32x4_t
neon_v: int16x4_t
neon_kyberq16: int16x8_t

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

/*
permute Coefficients
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, a, b, c, d, e, f
Input:
reg: int16x8x2_t: vld2q_s16(ptr)
--> vld2q_s16
in1: 0, 2, 4, 6, 8, a, c, e
in2: 1, 3, 5, 7, 9, b, d, f
------->
Output:
out1 : int16x8_t
out2 : int16x8_t
--> permutation inside 01 register
out1: 0, 4, 8, c | 2, 6, a, e
out2: 1, 5, 9, d | 3, 7, b, f
*/
#define transpose(out1, out2, in1, in2)                                        \
  out1 = vtrn1q_s16(in1, in2);                                                 \
  out2 = vtrn2q_s16(in1, in2);

#define permute(out1, out2, in1, in2)                                          \
  out1 = vuzp1q_s16(in1, in2);                                                 \
  out2 = vuzp2q_s16(in1, in2);

#define depermute(out1, out2, in1, in2)                                        \
  out1 = vzip1q_s16(in1, in2);                                                 \
  out2 = vzip2q_s16(in1, in2);


void neon_polyvec_acc_montgomery(poly *c, const polyvec *a, const polyvec *b, const int to_mont) {

  int16x8x2_t aa, bb, sum, cc;                                           //8
  int16x8_t ta1, ta2, ta3, ta4, 
            tb1, tb2, tb3, tb4;                                         // 4
  int16x4_t ta3_lo, ta3_hi, 
            tb3_lo, tb3_hi, 
            ta4_lo, ta4_hi,                                             // 8
            tb4_lo, tb4_hi; 
  int32x4_t t1, t2, t3, t4, t5, t6;                                     // 6
  int16x4_t neon_zeta, neon_one, neon_v, 
            neon_zeta_positive, neon_zeta_negative;                     // 5
  int16x8_t neon_kyberq16;                                              // 1
  int32x4_t neon_qinv, neon_kyberq;                                     // 2

  // Declare constant
  neon_qinv = vdupq_n_s32(QINV << 16);
  neon_kyberq = vdupq_n_s32(-KYBER_Q);
  neon_kyberq16 = vdupq_n_s16(-KYBER_Q);
  neon_v = vdup_n_s16(((1U << 26) + KYBER_Q / 2) / KYBER_Q);
  neon_one = vdup_n_s16(1);

  // Scalar variable
  unsigned int k = 64;
  unsigned int i, j;
  // End

  // Total possible register: Max 34 = 8+4+8+6+5+1+2, Min = 30
  // 1st Iteration
  for (j = 0; j < KYBER_N; j+=16){
    // Load Zeta
    // 64, 65, 66, 67
    vload4(neon_zeta_positive, &zetas[k]);
    // Convert zeta to negative sign
    // -64, -64, -66, -67
    vnot4(neon_zeta_negative, neon_zeta_positive);
    vadd4(neon_zeta_negative, neon_zeta_negative, neon_one);

    // Use max 8 registers
    // 0, 2, 4, 6, 8, a, c, e
    // 1, 3, 5, 7, 9, b, d, f
    aa = vld2q_s16(&a->vec[0].coeffs[j]);
    bb = vld2q_s16(&b->vec[0].coeffs[j]);


    // Tranpose before multiply
    transpose(ta1, ta2, aa.val[0], aa.val[1]);
    transpose(tb1, tb2, bb.val[0], bb.val[1]);

    permute(ta3, ta4, ta1, ta2);
    permute(tb3, tb4, tb1, tb2);

    // Do BaseMul
    // Input: ta3, ta4, tb3, tb4
    // t3: 0, 4, 8, 12, 2, 6, 10, 14
    // t4: 1, 5, 9, 13, 3, 7, 11, 15

    // tc3_lo = ta3[0:4] x tb3[0:4] + ta4[0:4] x tb4[0:4] x zeta_positive
    // tc3_hi = ta3[4:8] x tb3[4:8] + ta4[4:8] x tb4[4:8] x zeta_negative

    // t4_lo = ta3[0:4] x tb4[0:4] + ta4[0:4] x tb3[0:4]
    // t4_hi = ta3[4:8] x tb4[4:8] + ta4[4:8] x tb3[4:8]

    // Split fqmul
    vlo(ta3_lo, ta3);
    vhi(ta3_hi, ta3);
    vlo(tb3_lo, tb3);
    vhi(tb3_hi, tb3);

    vlo(ta4_lo, ta4);
    vhi(ta4_hi, ta4);
    vlo(tb4_lo, tb4);
    vhi(tb4_hi, tb4);

    // ta3[0:4] x tb3[0:4]
    // ta3[4:8] x tb3[4:8]
    fqmul(ta3_lo, tb3_lo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta3_hi, tb3_hi, t4, t5, t6, neon_qinv, neon_kyberq);

    // ta4[0:4] x tb4[0:4] x zeta_positive
    // ta4[4:8] x tb4[4:8] x zeta_negative
    fqmul(ta4_lo, tb4_lo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta4_hi, tb4_hi, t4, t5, t6, neon_qinv, neon_kyberq);
    fqmul(ta4_lo, neon_zeta_positive, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta4_hi, neon_zeta_negative, t4, t5, t6, neon_qinv, neon_kyberq);
    
    vcombine(ta1, ta3_lo, ta3_hi);
    vcombine(ta2, ta4_lo, ta4_hi);

    // tb1 = ta1 + ta2
    vadd8(sum.val[0], ta1, ta2);

    // Split fqmul
    vlo(ta3_lo, ta3);
    vhi(ta3_hi, ta3);
    vlo(tb3_lo, tb3);
    vhi(tb3_hi, tb3);

    vlo(ta4_lo, ta4);
    vhi(ta4_hi, ta4);
    vlo(tb4_lo, tb4);
    vhi(tb4_hi, tb4);

    // ta3[0:4] x tb4[0:4]
    // ta3[4:8] x tb4[4:8]
    fqmul(ta3_lo, tb4_lo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta3_hi, tb4_hi, t4, t5, t6, neon_qinv, neon_kyberq);

    // ta4[0:4] x tb3[0:4]
    // ta4[4:8] x tb3[4:8]
    fqmul(ta4_lo, tb3_lo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta4_hi, tb3_hi, t4, t5, t6, neon_qinv, neon_kyberq);
    
    vcombine(ta1, ta3_lo, ta3_hi);
    vcombine(ta2, ta4_lo, ta4_hi);
    
    // tb2 = ta1 + ta2
    vadd8(sum.val[1], ta1, ta2);
    
    /***************************/

    // 2nd iterator
    aa = vld2q_s16(&a->vec[1].coeffs[j]);
    bb = vld2q_s16(&b->vec[1].coeffs[j]);

    transpose(ta1, ta2, aa.val[0], aa.val[1]);
    transpose(tb1, tb2, bb.val[0], bb.val[1]);

    permute(ta3, ta4, ta1, ta2);
    permute(tb3, tb4, tb1, tb2);

    vlo(ta3_lo, ta3);
    vhi(ta3_hi, ta3);
    vlo(tb3_lo, tb3);
    vhi(tb3_hi, tb3);

    vlo(ta4_lo, ta4);
    vhi(ta4_hi, ta4);
    vlo(tb4_lo, tb4);
    vhi(tb4_hi, tb4);

    fqmul(ta3_lo, tb3_lo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta3_hi, tb3_hi, t4, t5, t6, neon_qinv, neon_kyberq);

    fqmul(ta4_lo, tb4_lo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta4_hi, tb4_hi, t4, t5, t6, neon_qinv, neon_kyberq);
    fqmul(ta4_lo, neon_zeta_positive, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta4_hi, neon_zeta_negative, t4, t5, t6, neon_qinv, neon_kyberq);
    
    vcombine(ta1, ta3_lo, ta3_hi);
    vcombine(ta2, ta4_lo, ta4_hi);

    vadd8(cc.val[0], ta1, ta2);
    vadd8(sum.val[0], sum.val[0], cc.val[0]);

    vlo(ta3_lo, ta3);
    vhi(ta3_hi, ta3);
    vlo(tb3_lo, tb3);
    vhi(tb3_hi, tb3);

    vlo(ta4_lo, ta4);
    vhi(ta4_hi, ta4);
    vlo(tb4_lo, tb4);
    vhi(tb4_hi, tb4);

    fqmul(ta3_lo, tb4_lo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta3_hi, tb4_hi, t4, t5, t6, neon_qinv, neon_kyberq);

    fqmul(ta4_lo, tb3_lo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta4_hi, tb3_hi, t4, t5, t6, neon_qinv, neon_kyberq);
    
    vcombine(ta1, ta3_lo, ta3_hi);
    vcombine(ta2, ta4_lo, ta4_hi);
    
    vadd8(cc.val[1], ta1, ta2);
    vadd8(sum.val[1], sum.val[1], cc.val[1]);

    /***************************/

#if KYBER_K >= 3 
    // 3rd iterator
    aa = vld2q_s16(&a->vec[2].coeffs[j]);
    bb = vld2q_s16(&b->vec[2].coeffs[j]);

    transpose(ta1, ta2, aa.val[0], aa.val[1]);
    transpose(tb1, tb2, bb.val[0], bb.val[1]);

    permute(ta3, ta4, ta1, ta2);
    permute(tb3, tb4, tb1, tb2);

    vlo(ta3_lo, ta3);
    vhi(ta3_hi, ta3);
    vlo(tb3_lo, tb3);
    vhi(tb3_hi, tb3);

    vlo(ta4_lo, ta4);
    vhi(ta4_hi, ta4);
    vlo(tb4_lo, tb4);
    vhi(tb4_hi, tb4);

    fqmul(ta3_lo, tb3_lo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta3_hi, tb3_hi, t4, t5, t6, neon_qinv, neon_kyberq);

    fqmul(ta4_lo, tb4_lo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta4_hi, tb4_hi, t4, t5, t6, neon_qinv, neon_kyberq);
    fqmul(ta4_lo, neon_zeta_positive, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta4_hi, neon_zeta_negative, t4, t5, t6, neon_qinv, neon_kyberq);
    
    vcombine(ta1, ta3_lo, ta3_hi);
    vcombine(ta2, ta4_lo, ta4_hi);

    vadd8(cc.val[0], ta1, ta2);
    vadd8(sum.val[0], sum.val[0], cc.val[0]);

    vlo(ta3_lo, ta3);
    vhi(ta3_hi, ta3);
    vlo(tb3_lo, tb3);
    vhi(tb3_hi, tb3);

    vlo(ta4_lo, ta4);
    vhi(ta4_hi, ta4);
    vlo(tb4_lo, tb4);
    vhi(tb4_hi, tb4);

    fqmul(ta3_lo, tb4_lo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta3_hi, tb4_hi, t4, t5, t6, neon_qinv, neon_kyberq);

    fqmul(ta4_lo, tb3_lo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta4_hi, tb3_hi, t4, t5, t6, neon_qinv, neon_kyberq);
    
    vcombine(ta1, ta3_lo, ta3_hi);
    vcombine(ta2, ta4_lo, ta4_hi);
    
    vadd8(cc.val[1], ta1, ta2);
    vadd8(sum.val[1], sum.val[1], cc.val[1]);
#endif
#if KYBER_K == 4
    // 3rd iterator
    aa = vld2q_s16(&a->vec[3].coeffs[j]);
    bb = vld2q_s16(&b->vec[3].coeffs[j]);

    transpose(ta1, ta2, aa.val[0], aa.val[1]);
    transpose(tb1, tb2, bb.val[0], bb.val[1]);

    permute(ta3, ta4, ta1, ta2);
    permute(tb3, tb4, tb1, tb2);

    vlo(ta3_lo, ta3);
    vhi(ta3_hi, ta3);
    vlo(tb3_lo, tb3);
    vhi(tb3_hi, tb3);

    vlo(ta4_lo, ta4);
    vhi(ta4_hi, ta4);
    vlo(tb4_lo, tb4);
    vhi(tb4_hi, tb4);

    fqmul(ta3_lo, tb3_lo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta3_hi, tb3_hi, t4, t5, t6, neon_qinv, neon_kyberq);

    fqmul(ta4_lo, tb4_lo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta4_hi, tb4_hi, t4, t5, t6, neon_qinv, neon_kyberq);
    fqmul(ta4_lo, neon_zeta_positive, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta4_hi, neon_zeta_negative, t4, t5, t6, neon_qinv, neon_kyberq);
    
    vcombine(ta1, ta3_lo, ta3_hi);
    vcombine(ta2, ta4_lo, ta4_hi);

    vadd8(cc.val[0], ta1, ta2);
    vadd8(sum.val[0], sum.val[0], cc.val[0]);

    vlo(ta3_lo, ta3);
    vhi(ta3_hi, ta3);
    vlo(tb3_lo, tb3);
    vhi(tb3_hi, tb3);

    vlo(ta4_lo, ta4);
    vhi(ta4_hi, ta4);
    vlo(tb4_lo, tb4);
    vhi(tb4_hi, tb4);

    fqmul(ta3_lo, tb4_lo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta3_hi, tb4_hi, t4, t5, t6, neon_qinv, neon_kyberq);

    fqmul(ta4_lo, tb3_lo, t1, t2, t3, neon_qinv, neon_kyberq);
    fqmul(ta4_hi, tb3_hi, t4, t5, t6, neon_qinv, neon_kyberq);
    
    vcombine(ta1, ta3_lo, ta3_hi);
    vcombine(ta2, ta4_lo, ta4_hi);
    
    vadd8(cc.val[1], ta1, ta2);
    vadd8(sum.val[1], sum.val[1], cc.val[1]);
#endif

   // Do poly_reduce:   poly_reduce(r);
    barrett(sum.val[0], cc.val[0], ta3_lo, ta3_hi, t1, t2, neon_v, neon_kyberq16);
    barrett(sum.val[1], cc.val[1], tb3_lo, tb3_hi, t3, t4, neon_v, neon_kyberq16);

    if (to_mont)
    {
      neon_zeta = vdup_n_s16(((1ULL << 32) % KYBER_Q));

      // Split fqmul
      vlo(ta3_lo, sum.val[0]);
      vhi(ta3_hi, sum.val[0]);
      vlo(ta4_lo, sum.val[1]);
      vhi(ta4_hi, sum.val[1]);

      fqmul(ta3_lo, neon_zeta, t1, t2, t3, neon_qinv, neon_kyberq);
      fqmul(ta3_hi, neon_zeta, t4, t5, t6, neon_qinv, neon_kyberq);
      fqmul(ta4_lo, neon_zeta, t1, t2, t3, neon_qinv, neon_kyberq);
      fqmul(ta4_hi, neon_zeta, t4, t5, t6, neon_qinv, neon_kyberq);
      
      vcombine(sum.val[0], ta3_lo, ta3_hi);
      vcombine(sum.val[1], ta4_lo, ta4_hi);
    }

    // Tranpose before store back to memory
    // tb1: 0, 4, 8, 12, 2, 6, 10, 14
    // tb2: 1, 5, 9, 13, 3, 7, 11, 15
    depermute(tb1, tb2, sum.val[0], sum.val[1]);

    // tb3: 0, 1, 4, 5, 8, 9, 12, 13
    // tb4: 2, 3, 6, 7, 10, 11, 14, 15
    transpose(sum.val[0], sum.val[1], tb1, tb2);

    // 0, 2, 4, 6, 8, a, c, e
    // 1, 3, 5, 7, 9, b, d, f
    vst2q_s16(&c->coeffs[j], sum);
    k+=4;
  }
}
