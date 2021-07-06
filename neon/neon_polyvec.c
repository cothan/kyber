#include <arm_neon.h>
#include "params.h"
#include "reduce.h"
#include "neon_ntt.h"
#include "poly.h"
#include "polyvec.h"

/*************************************************
* Name:        neon_polyvec_ntt
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
    neon_poly_ntt(&r->vec[i]);
  }
}

/*************************************************
* Name:        neon_polyvec_invntt_to_mont
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
    neon_poly_invntt_tomont(&r->vec[i]);
}

/*************************************************
* Name:        neon_polyvec_add_reduce
*
* Description: Applies Barrett reduction to each coefficient
*              of each element of a vector of polynomials
*              for details of the Barrett reduction see comments in reduce.c
*
* Arguments: - polyvec *r:       pointer to output vector of polynomials
*            - const polyvec *a: pointer to first input vector of polynomials
*            - const polyvec *b: pointer to second input vector of polynomials
**************************************************/
void neon_polyvec_add_reduce(polyvec *c, const polyvec *a)
{
  unsigned int i;
  for (i = 0; i < KYBER_K; i++)
  {
    // c = c + a;
    // c = reduce(c);
    neon_poly_add_reduce(&c->vec[i], &a->vec[i]);
  }
}

/**********************************/
// Load int16x8_t c <= ptr*
#define vload4(c, ptr) c = vld4q_s16(ptr);

// Store *ptr <= c
#define vstore4(ptr, c) vst4q_s16(ptr, c);

// c (int16x8) = a + b (int16x8)
#define vadd8(c, a, b) c = vaddq_s16(a, b);

// c (int16x8) = a - b (int16x8)
#define vsub8(c, a, b) c = vsubq_s16(a, b);

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
  t1 = a_L = (b * c)_L; 
  t2 = (a_L * QINV)_L; 
  t3 = (t2 * Q)_H;
  out = (t0 - t3)/2

  // fqmul_qinv: 3 MUL + precomputed
  t0 = a_H = (2*b*c)_H; 
  t1 = a_L = (b_qinv * c)_L; 
  t2 = (t1 * Q)_H; 
  out = (t0 - t2)/2 = (b*c)_H - (a_L * Q)_H;

*************************************************/
#define fqmul_qinv(out, in, zeta, zeta_qinv, t)                     \
  t.val[0] = vqdmulhq_s16(in, zeta);              /* a_H */         \
  t.val[1] = vmulq_s16(zeta_qinv, in);            /* a_L */         \
  t.val[2] = vqdmulhq_s16(t.val[1], neon_kyberq); /* (2*a_L*Q)_H */ \
  out = vhsubq_s16(t.val[0], t.val[2]);           /* (t0 - t2)/2 */

#define fqmul(out, in, zeta, t)                                      \
  t.val[0] = vqdmulhq_s16(in, zeta);              /* a_H */          \
  t.val[1] = vmulq_s16(in, zeta);                 /* a_L */          \
  t.val[2] = vmulq_s16(t.val[1], neon_qinv);      /* (a_L*QINV)_L */ \
  t.val[3] = vqdmulhq_s16(t.val[2], neon_kyberq); /* (2*t2*Q)_H */   \
  out = vhsubq_s16(t.val[0], t.val[3]);           /* (t0 - t3)/2 */

/*
inout: 
int16x8_t inout, 
int16x8x2_t t, 
int16x8_t neon_v, neon_one, neon_kyberq;
*/
#define barrett(inout, t, i)                                        \
  t.val[i] = vqdmulhq_s16(inout, neon_v);        /* (2*a*v)_H */    \
  t.val[i + 1] = vhaddq_s16(t.val[i], neon_one); /* (2a*v + 2)/2 */ \
  t.val[i + 1] = vshrq_n_s16(t.val[i + 1], 10);                     \
  inout = vmlsq_s16(inout, t.val[i + 1], neon_kyberq);

/*************************************************
* Name:        neon_polyvec_acc_montgomery
*
* Description: Multiply elements of a and b in NTT domain, accumulate into r,
*              and multiply by 2^-16.
*
* Arguments: - poly *r: pointer to output polynomial
*            - const polyvec *a: pointer to first input vector of polynomials
*            - const polyvec *b: pointer to second input vector of polynomials
**************************************************/
void neon_polyvec_acc_montgomery(poly *c, const polyvec *a, const polyvec *b, const int to_mont)
{
  int16x8x4_t aa, bb, r, ta, tb, t; // 24
  int16x8_t neon_v, neon_qinv, neon_kyberq,
            neon_zeta, neon_one, neon_zeta_qinv; // 6

  // Declare constant
  neon_qinv = vdupq_n_s16(QINV);
  neon_kyberq = vdupq_n_s16(KYBER_Q);
  neon_v = vdupq_n_s16((((1U << 26) + KYBER_Q / 2) / KYBER_Q));
  neon_one = vdupq_n_s16(1 << 9);

  // Scalar variable
  unsigned int k = 80;
  unsigned int j, i;
  // End

  // Total possible register: Max 30;
  // 1st Iteration
  for (j = 0; j < KYBER_N; j += 32)
  {
    // Load Zeta
    // 64, 65, 66, 67 =-= 68, 69, 70, 71
    neon_zeta = vld1q_s16(&neon_zetas[k]);
    neon_zeta_qinv = vld1q_s16(&neon_zetas_qinv[k]);

    // Use max 8 registers
    // 0: 0, 4,  8, 12, =-=  16, 20, 24, 28
    // 1: 1, 5,  9, 13, =-=  17, 21, 25, 29
    // 2: 2, 6, 10, 14, =-=  18, 22, 26, 30
    // 3: 3, 7, 11, 15, =-=  19, 23, 27, 31
    vload4(aa, &a->vec[0].coeffs[j]);
    vload4(bb, &b->vec[0].coeffs[j]);

    // => r.val[0] = a.val[1]*b.val[1]*zeta_pos + a.val[0] * b.val[0]
    // => r.val[1] = a.val[0]*b.val[1] + a.val[1] * b.val[0]
    // => r.val[2] = a.val[3]*b.val[3]*zetas_neg + a.val[2]*b.val[2]
    // => r.val[3] = a.val[2]*b.val[3] + a.val[3] * b.val[2]

    fqmul(ta.val[0], aa.val[1], bb.val[1], t);
    fqmul_qinv(ta.val[0], ta.val[0], neon_zeta, neon_zeta_qinv, t);
    fqmul(ta.val[1], aa.val[0], bb.val[1], t);
    fqmul(ta.val[2], aa.val[3], bb.val[3], t);
    fqmul_qinv(ta.val[2], ta.val[2], neon_zeta, neon_zeta_qinv, t);
    fqmul(ta.val[3], aa.val[2], bb.val[3], t);

    fqmul(tb.val[0], aa.val[0], bb.val[0], t);
    fqmul(tb.val[1], aa.val[1], bb.val[0], t);
    fqmul(tb.val[2], aa.val[2], bb.val[2], t);
    fqmul(tb.val[3], aa.val[3], bb.val[2], t);

    vadd8(r.val[0], ta.val[0], tb.val[0]);
    vadd8(r.val[1], ta.val[1], tb.val[1]);
    vsub8(r.val[2], tb.val[2], ta.val[2]);
    vadd8(r.val[3], ta.val[3], tb.val[3]);

    /***************************/

    // 2nd iterator
    vload4(aa, &a->vec[1].coeffs[j]);
    vload4(bb, &b->vec[1].coeffs[j]);

    fqmul(ta.val[0], aa.val[1], bb.val[1], t);
    fqmul_qinv(ta.val[0], ta.val[0], neon_zeta, neon_zeta_qinv, t);
    fqmul(ta.val[1], aa.val[0], bb.val[1], t);
    fqmul(ta.val[2], aa.val[3], bb.val[3], t);
    fqmul_qinv(ta.val[2], ta.val[2], neon_zeta, neon_zeta_qinv, t);
    fqmul(ta.val[3], aa.val[2], bb.val[3], t);

    vadd8(r.val[0], r.val[0], ta.val[0]);
    vadd8(r.val[1], r.val[1], ta.val[1]);
    vsub8(r.val[2], r.val[2], ta.val[2]);
    vadd8(r.val[3], r.val[3], ta.val[3]);

    fqmul(tb.val[0], aa.val[0], bb.val[0], t);
    fqmul(tb.val[1], aa.val[1], bb.val[0], t);
    fqmul(tb.val[2], aa.val[2], bb.val[2], t);
    fqmul(tb.val[3], aa.val[3], bb.val[2], t);

    vadd8(r.val[0], r.val[0], tb.val[0]);
    vadd8(r.val[1], r.val[1], tb.val[1]);
    vadd8(r.val[2], r.val[2], tb.val[2]);
    vadd8(r.val[3], r.val[3], tb.val[3]);

    /***************************/

#if KYBER_K >= 3
    // 3rd iterator
    vload4(aa, &a->vec[2].coeffs[j]);
    vload4(bb, &b->vec[2].coeffs[j]);

    fqmul(ta.val[0], aa.val[1], bb.val[1], t);
    fqmul_qinv(ta.val[0], ta.val[0], neon_zeta, neon_zeta_qinv, t);
    fqmul(ta.val[1], aa.val[0], bb.val[1], t);
    fqmul(ta.val[2], aa.val[3], bb.val[3], t);
    fqmul_qinv(ta.val[2], ta.val[2], neon_zeta, neon_zeta_qinv, t);
    fqmul(ta.val[3], aa.val[2], bb.val[3], t);

    vadd8(r.val[0], r.val[0], ta.val[0]);
    vadd8(r.val[1], r.val[1], ta.val[1]);
    vsub8(r.val[2], r.val[2], ta.val[2]);
    vadd8(r.val[3], r.val[3], ta.val[3]);

    fqmul(tb.val[0], aa.val[0], bb.val[0], t);
    fqmul(tb.val[1], aa.val[1], bb.val[0], t);
    fqmul(tb.val[2], aa.val[2], bb.val[2], t);
    fqmul(tb.val[3], aa.val[3], bb.val[2], t);

    vadd8(r.val[0], r.val[0], tb.val[0]);
    vadd8(r.val[1], r.val[1], tb.val[1]);
    vadd8(r.val[2], r.val[2], tb.val[2]);
    vadd8(r.val[3], r.val[3], tb.val[3]);
#endif
#if KYBER_K == 4
    // 3rd iterator
    vload4(aa, &a->vec[3].coeffs[j]);
    vload4(bb, &b->vec[3].coeffs[j]);

    fqmul(ta.val[0], aa.val[1], bb.val[1], t);
    fqmul_qinv(ta.val[0], ta.val[0], neon_zeta, neon_zeta_qinv, t);
    fqmul(ta.val[1], aa.val[0], bb.val[1], t);
    fqmul(ta.val[2], aa.val[3], bb.val[3], t);
    fqmul_qinv(ta.val[2], ta.val[2], neon_zeta, neon_zeta_qinv, t);
    fqmul(ta.val[3], aa.val[2], bb.val[3], t);

    vadd8(r.val[0], r.val[0], ta.val[0]);
    vadd8(r.val[1], r.val[1], ta.val[1]);
    vsub8(r.val[2], r.val[2], ta.val[2]);
    vadd8(r.val[3], r.val[3], ta.val[3]);

    fqmul(tb.val[0], aa.val[0], bb.val[0], t);
    fqmul(tb.val[1], aa.val[1], bb.val[0], t);
    fqmul(tb.val[2], aa.val[2], bb.val[2], t);
    fqmul(tb.val[3], aa.val[3], bb.val[2], t);

    vadd8(r.val[0], r.val[0], tb.val[0]);
    vadd8(r.val[1], r.val[1], tb.val[1]);
    vadd8(r.val[2], r.val[2], tb.val[2]);
    vadd8(r.val[3], r.val[3], tb.val[3]);
#endif

    // Do poly_reduce:   poly_reduce(r);
    barrett(r.val[0], t, 0);
    barrett(r.val[1], t, 2);
    barrett(r.val[2], t, 0);
    barrett(r.val[3], t, 2);

    if (to_mont)
    {
      neon_zeta = vdupq_n_s16(((1ULL << 32) % KYBER_Q));
      neon_zeta_qinv = vdupq_n_s16( (int16_t) (((1ULL << 32) % KYBER_Q) * QINV));

      // Split fqmul
      fqmul_qinv(r.val[0], r.val[0], neon_zeta, neon_zeta_qinv, t);
      fqmul_qinv(r.val[1], r.val[1], neon_zeta, neon_zeta_qinv, t);
      fqmul_qinv(r.val[2], r.val[2], neon_zeta, neon_zeta_qinv, t);
      fqmul_qinv(r.val[3], r.val[3], neon_zeta, neon_zeta_qinv, t);
    }

    vstore4(&c->coeffs[j], r);
    i = (j != 96) ? 0 : 80;
    k += 8 + i;
  }
}
