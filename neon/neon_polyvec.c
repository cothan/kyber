#include <arm_neon.h>
#include "params.h"
#include "reduce.h"
#include "ntt.h"
#include "polyvec.h"

/*************************************************
* Name:        polyvec_ntt
*
* Description: Apply forward NTT to all elements of a vector of polynomials
*
* Arguments:   - polyvec *r: pointer to in/output vector of polynomials
**************************************************/
void neon_polyvec_ntt(polyvec *r)
{
  unsigned int i;
  for (i = 0; i < KYBER_K; i++)
  {
    neon_ntt(r->vec[i].coeffs);
    neon_poly_reduce(&r->vec[i]);
  }
}

/*************************************************
* Name:        polyvec_invntt_tomont
*
* Description: Apply inverse NTT to all elements of a vector of polynomials
*              and multiply by Montgomery factor 2^16
*
* Arguments:   - polyvec *r: pointer to in/output vector of polynomials
**************************************************/
void neon_polyvec_invntt_to_mont(polyvec *r)
{
  unsigned int i;
  for (i = 0; i < KYBER_K; i++)
    neon_invntt(r->vec[i].coeffs);
}

/*************************************************
* Name:        polyvec_add
*
* Description: Add vectors of polynomials
*
* Arguments: - polyvec *r:       pointer to output vector of polynomials
*            - const polyvec *a: pointer to first input vector of polynomials
*            - const polyvec *b: pointer to second input vector of polynomials
**************************************************/
/*************************************************
* Name:        polyvec_reduce
*
* Description: Applies Barrett reduction to each coefficient
*              of each element of a vector of polynomials
*              for details of the Barrett reduction see comments in reduce.c
*
* Arguments:   - poly *r: pointer to input/output polynomial
**************************************************/
void neon_polyvec_add_reduce_csubq(polyvec *c, const polyvec *a)
{
  unsigned int i;
  for (i = 0; i < KYBER_K; i++)
  {
    // c = c + a;
    // c = reduce(c);
    neon_poly_add_reduce_csubq(&c->vec[i], &a->vec[i]);
  }
}

/**********************************/
// Load interleave
#define vload2(c, ptr) c = vld2q_s16(ptr);

// Store interleave
#define vstore2(ptr, c) vst2q_s16(ptr, c);

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
inout: input/output : int16x8_t
zeta: input : int16x8_t
a1, a2: temp : int32x4_t
u1, u2: temp : int32x4_t
neon_qinv: const   : int32x4_t
neon_kyberq: const : int32x4_t
*/
#define fqmul(out, in, zeta, t, neon_qinv, neon_kyberq)       \
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
reduce low and high of 
inout: 
int16x8_t inout, 
t32_1, t32_2: int32x4_t 
t16: int16x8_t 
neon_v, neon_kyber16
*/
#define barrett(inout, t32_1, t32_2, t16, neon_v, neon_kyberq16) \
  t32_1 = vmull_s16(vget_low_s16(inout), vget_low_s16(neon_v));  \
  t32_2 = vmull_high_s16(inout, neon_v);                         \
  t32_1 = vshrq_n_s32(t32_1, 26);                                \
  t32_2 = vshrq_n_s32(t32_2, 26);                                \
  t16 = vmovn_high_s32(vmovn_s32(t32_1), t32_2);                 \
  inout = vmlaq_s16(inout, t16, neon_kyberq16);
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
#define transpose(out1, out2, in1, in2) \
  out1 = vtrn1q_s16(in1, in2);          \
  out2 = vtrn2q_s16(in1, in2);

#define permute(out1, out2, in1, in2) \
  out1 = vuzp1q_s16(in1, in2);        \
  out2 = vuzp2q_s16(in1, in2);

#define depermute(out1, out2, in1, in2) \
  out1 = vzip1q_s16(in1, in2);          \
  out2 = vzip2q_s16(in1, in2);

/*************************************************
* Name:        polyvec_pointwise_acc_montgomery
*
* Description: Pointwise multiply elements of a and b, accumulate into r,
*              and multiply by 2^-16.
*
* Arguments: - poly *r:          pointer to output polynomial
*            - const polyvec *a: pointer to first input vector of polynomials
*            - const polyvec *b: pointer to second input vector of polynomials
**************************************************/
void neon_polyvec_acc_montgomery(poly *c, const polyvec *a, const polyvec *b, const int to_mont)
{
  int16x8x4_t ta, tb;                                         // 8
  int16x8x2_t aa, bb, sum, cc;                                // 8
  int16x8_t neon_kyberq16, neon_zeta, neon_v;                 // 3
  int32x4x4_t t;                                              // 4
  int32x4_t neon_qinv, neon_kyberq;                           // 2
  int16x4_t neon_one, neon_zeta_positive, neon_zeta_negative; // 5

  // Declare constant
  neon_qinv = vdupq_n_s32(QINV << 16);
  neon_kyberq = vdupq_n_s32(-KYBER_Q);
  neon_kyberq16 = vdupq_n_s16(-KYBER_Q);
  neon_v = vdupq_n_s16(((1U << 26) + KYBER_Q / 2) / KYBER_Q);
  neon_one = vdup_n_s16(1);

  // Scalar variable
  unsigned int k = 64;
  unsigned int j;
  // End

  // Total possible register: Max 30;
  // 1st Iteration
  for (j = 0; j < KYBER_N; j += 16)
  {
    // Load Zeta
    // 64, 65, 66, 67
    neon_zeta_positive = vld1_s16(&zetas[k]);
    // Convert zeta to negative sign
    // -64, -64, -66, -67
    vnot4(neon_zeta_negative, neon_zeta_positive);
    vadd4(neon_zeta_negative, neon_zeta_negative, neon_one);
    vcombine(neon_zeta, neon_zeta_positive, neon_zeta_negative);

    // Use max 8 registers
    // 0, 2, 4, 6, 8, a, c, e
    // 1, 3, 5, 7, 9, b, d, f
    vload2(aa, &a->vec[0].coeffs[j]);
    vload2(bb, &b->vec[0].coeffs[j]);

    // Tranpose before multiply
    transpose(ta.val[0], ta.val[1], aa.val[0], aa.val[1]);
    transpose(tb.val[0], tb.val[1], bb.val[0], bb.val[1]);

    permute(ta.val[2], ta.val[3], ta.val[0], ta.val[1]);
    permute(tb.val[2], tb.val[3], tb.val[0], tb.val[1]);

    // Do BaseMul
    // Input: ta3, ta4, tb3, tb4
    // t3: 0, 4, 8, 12, 2, 6, 10, 14
    // t4: 1, 5, 9, 13, 3, 7, 11, 15

    // tc3_lo = ta3[0:4] x tb3[0:4] + ta4[0:4] x tb4[0:4] x zeta_positive
    // tc3_hi = ta3[4:8] x tb3[4:8] + ta4[4:8] x tb4[4:8] x zeta_negative

    // t4_lo = ta3[0:4] x tb4[0:4] + ta4[0:4] x tb3[0:4]
    // t4_hi = ta3[4:8] x tb4[4:8] + ta4[4:8] x tb3[4:8]

    // Split fqmul
    // ta3[0:4] x tb3[0:4]
    // ta3[4:8] x tb3[4:8]
    fqmul(ta.val[0], ta.val[2], tb.val[2], t, neon_qinv, neon_kyberq);

    // ta4[0:4] x tb4[0:4] x zeta_positive
    // ta4[4:8] x tb4[4:8] x zeta_negative
    fqmul(ta.val[1], ta.val[3], tb.val[3], t, neon_qinv, neon_kyberq);
    fqmul(ta.val[1], ta.val[1], neon_zeta, t, neon_qinv, neon_kyberq);

    // tb1 = ta1 + ta2
    vadd8(sum.val[0], ta.val[0], ta.val[1]);

    // Split fqmul

    // ta3[0:4] x tb4[0:4]
    // ta3[4:8] x tb4[4:8]
    fqmul(ta.val[0], ta.val[2], tb.val[3], t, neon_qinv, neon_kyberq);

    // ta4[0:4] x tb3[0:4]
    // ta4[4:8] x tb3[4:8]
    fqmul(ta.val[1], ta.val[3], tb.val[2], t, neon_qinv, neon_kyberq);

    // tb2 = ta1 + ta2
    vadd8(sum.val[1], ta.val[0], ta.val[1]);

    /***************************/

    // 2nd iterator
    vload2(aa, &a->vec[1].coeffs[j]);
    vload2(bb, &b->vec[1].coeffs[j]);

    transpose(ta.val[0], ta.val[1], aa.val[0], aa.val[1]);
    transpose(tb.val[0], tb.val[1], bb.val[0], bb.val[1]);

    permute(ta.val[2], ta.val[3], ta.val[0], ta.val[1]);
    permute(tb.val[2], tb.val[3], tb.val[0], tb.val[1]);

    fqmul(ta.val[0], ta.val[2], tb.val[2], t, neon_qinv, neon_kyberq);

    fqmul(ta.val[1], ta.val[3], tb.val[3], t, neon_qinv, neon_kyberq);
    fqmul(ta.val[1], ta.val[1], neon_zeta, t, neon_qinv, neon_kyberq);

    vadd8(cc.val[0], ta.val[0], ta.val[1]);
    vadd8(sum.val[0], sum.val[0], cc.val[0]);

    fqmul(ta.val[0], ta.val[2], tb.val[3], t, neon_qinv, neon_kyberq);
    fqmul(ta.val[1], ta.val[3], tb.val[2], t, neon_qinv, neon_kyberq);

    vadd8(cc.val[1], ta.val[0], ta.val[1]);
    vadd8(sum.val[1], sum.val[1], cc.val[1]);

    /***************************/

#if KYBER_K >= 3
    // 3rd iterator
    vload2(aa, &a->vec[2].coeffs[j]);
    vload2(bb, &b->vec[2].coeffs[j]);

    transpose(ta.val[0], ta.val[1], aa.val[0], aa.val[1]);
    transpose(tb.val[0], tb.val[1], bb.val[0], bb.val[1]);

    permute(ta.val[2], ta.val[3], ta.val[0], ta.val[1]);
    permute(tb.val[2], tb.val[3], tb.val[0], tb.val[1]);

    fqmul(ta.val[0], ta.val[2], tb.val[2], t, neon_qinv, neon_kyberq);

    fqmul(ta.val[1], ta.val[3], tb.val[3], t, neon_qinv, neon_kyberq);
    fqmul(ta.val[1], ta.val[1], neon_zeta, t, neon_qinv, neon_kyberq);

    vadd8(cc.val[0], ta.val[0], ta.val[1]);
    vadd8(sum.val[0], sum.val[0], cc.val[0]);

    fqmul(ta.val[0], ta.val[2], tb.val[3], t, neon_qinv, neon_kyberq);
    fqmul(ta.val[1], ta.val[3], tb.val[2], t, neon_qinv, neon_kyberq);

    vadd8(cc.val[1], ta.val[0], ta.val[1]);
    vadd8(sum.val[1], sum.val[1], cc.val[1]);
#endif
#if KYBER_K == 4
    // 3rd iterator
    vload2(aa, &a->vec[3].coeffs[j]);
    vload2(bb, &b->vec[3].coeffs[j]);

    transpose(ta.val[0], ta.val[1], aa.val[0], aa.val[1]);
    transpose(tb.val[0], tb.val[1], bb.val[0], bb.val[1]);

    permute(ta.val[2], ta.val[3], ta.val[0], ta.val[1]);
    permute(tb.val[2], tb.val[3], tb.val[0], tb.val[1]);

    fqmul(ta.val[0], ta.val[2], tb.val[2], t, neon_qinv, neon_kyberq);

    fqmul(ta.val[1], ta.val[3], tb.val[3], t, neon_qinv, neon_kyberq);
    fqmul(ta.val[1], ta.val[1], neon_zeta, t, neon_qinv, neon_kyberq);

    vadd8(cc.val[0], ta.val[0], ta.val[1]);
    vadd8(sum.val[0], sum.val[0], cc.val[0]);

    fqmul(ta.val[0], ta.val[2], tb.val[3], t, neon_qinv, neon_kyberq);
    fqmul(ta.val[1], ta.val[3], tb.val[2], t, neon_qinv, neon_kyberq);

    vadd8(cc.val[1], ta.val[0], ta.val[1]);
    vadd8(sum.val[1], sum.val[1], cc.val[1]);
#endif

    // Do poly_reduce:   poly_reduce(r);
    barrett(sum.val[0], t.val[0], t.val[1], cc.val[0], neon_v, neon_kyberq16);
    barrett(sum.val[1], t.val[2], t.val[3], cc.val[1], neon_v, neon_kyberq16);

    if (to_mont)
    {
      neon_zeta = vdupq_n_s16(((1ULL << 32) % KYBER_Q));

      // Split fqmul
      fqmul(sum.val[0], sum.val[0], neon_zeta, t, neon_qinv, neon_kyberq);
      fqmul(sum.val[1], sum.val[1], neon_zeta, t, neon_qinv, neon_kyberq);
    }

    // Tranpose before store back to memory
    // tb1: 0, 4, 8, 12, 2, 6, 10, 14
    // tb2: 1, 5, 9, 13, 3, 7, 11, 15
    depermute(tb.val[0], tb.val[1], sum.val[0], sum.val[1]);

    // tb3: 0, 1, 4, 5, 8, 9, 12, 13
    // tb4: 2, 3, 6, 7, 10, 11, 14, 15
    transpose(sum.val[0], sum.val[1], tb.val[0], tb.val[1]);

    // 0, 2, 4, 6, 8, a, c, e
    // 1, 3, 5, 7, 9, b, d, f
    vstore2(&c->coeffs[j], sum);
    k += 4;
  }
}
