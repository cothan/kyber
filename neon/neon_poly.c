#include <arm_neon.h>
#include "params.h"
#include "poly.h"
#include "symmetric.h"
#include "cbd.h"


// Load int16x8x4_t c <= ptr*
#define vload(c, ptr) c = vld1q_s16_x4(ptr);

// Load int16x8x4_t c <= ptr*
#define vstore(ptr, c) vst1q_s16_x4(ptr, c);

// Combine in16x8_t c: low | high
#define vcombine(c, low, high) c = vcombine_s16(low, high);

// c (int16x8) = a + b (int16x8)
#define vadd8(c, a, b) c = vaddq_s16(a, b);

// c (int16x8) = a - b (int16x8)
#define vsub8(c, a, b) c = vsubq_s16(a, b);

// get_low c (int16x4) = low(a) (int16x8)
#define vlo(c, a) c = vget_low_s16(a);

// get_high c (int16x4) = high(a) (int16x8)
#define vhi(c, a) c = vget_high_s16(a);

/*
inout: int16x8_t
t : int16x8_t
t_lo: int16x4_t
t_hi: int16x4_t
t1: int32x4_t
t2: int32x4_t
neon_v: int16x8_t
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


void neon_poly_reduce(poly *c) {
  int16x8x4_t cc, aa;                                                   // 4
  int16x8_t neon_kyberq16;                                  // 2
  int16x4_t neon_v;
  int16x4_t t0_lo, t0_hi, t1_lo, t1_hi, t2_lo, t2_hi, t3_lo, t3_hi; // 8
  int32x4_t t01, t02, t11, t12, t21, t22, t31, t32;                 // 8

  neon_kyberq16 = vdupq_n_s16(-KYBER_Q);
  neon_v = vdup_n_s16(((1U << 26) + KYBER_Q / 2) / KYBER_Q);

  // Total register: 22 registers.
  unsigned int i;
  for (i = 0; i < KYBER_N; i += 32) {
    vload(cc, &c->coeffs[i]);

    // c = reduce(c)
    barrett(cc.val[0], aa.val[0], t0_lo, t0_hi, t01, t02, neon_v, neon_kyberq16);
    barrett(cc.val[1], aa.val[1], t1_lo, t1_hi, t11, t12, neon_v, neon_kyberq16);
    barrett(cc.val[2], aa.val[2], t2_lo, t2_hi, t21, t22, neon_v, neon_kyberq16);
    barrett(cc.val[3], aa.val[3], t3_lo, t3_hi, t31, t32, neon_v, neon_kyberq16);

    // c = t;
    vstore(&c->coeffs[i], cc);
  }
}


/*
c = c + a
c = reduce(c)
*/
void neon_poly_add_reduce(poly *c, const poly *a) {
  int16x8x4_t cc, aa;                                               // 8
  int16x8_t neon_kyberq16;                                  // 2
  int16x4_t neon_v;
  int16x4_t t0_lo, t0_hi, t1_lo, t1_hi, t2_lo, t2_hi, t3_lo, t3_hi; // 8
  int32x4_t t01, t02, t11, t12, t21, t22, t31, t32;                 // 8

  neon_kyberq16 = vdupq_n_s16(-KYBER_Q);
  neon_v = vdup_n_s16(((1U << 26) + KYBER_Q / 2) / KYBER_Q);

  // Total register: 26 registers.
  unsigned int i;
  for (i = 0; i < KYBER_N; i += 32) {
    vload(aa, &a->coeffs[i]);
    vload(cc, &c->coeffs[i]);

    // c = c - a;
    vadd8(cc.val[0], cc.val[0], aa.val[0]);
    vadd8(cc.val[1], cc.val[1], aa.val[1]);
    vadd8(cc.val[2], cc.val[2], aa.val[2]);
    vadd8(cc.val[3], cc.val[3], aa.val[3]);

    // c = reduce(c)
    barrett(cc.val[0], aa.val[0], t0_lo, t0_hi, t01, t02, neon_v, neon_kyberq16);
    barrett(cc.val[1], aa.val[1], t1_lo, t1_hi, t11, t12, neon_v, neon_kyberq16);
    barrett(cc.val[2], aa.val[2], t2_lo, t2_hi, t21, t22, neon_v, neon_kyberq16);
    barrett(cc.val[3], aa.val[3], t3_lo, t3_hi, t31, t32, neon_v, neon_kyberq16);

    // c = t;
    vstore(&c->coeffs[i], cc);
  }
}


/*
c = c - a
c = reduce(c)
*/
void neon_poly_sub_reduce(poly *c, const poly *a) {
  int16x8x4_t cc, aa;                                               // 8
  int16x8_t neon_kyberq16;
  int16x4_t neon_v;                                  // 2
  int16x4_t t0_lo, t0_hi, t1_lo, t1_hi, t2_lo, t2_hi, t3_lo, t3_hi; // 8
  int32x4_t t01, t02, t11, t12, t21, t22, t31, t32;                 // 8

  neon_kyberq16 = vdupq_n_s16(-KYBER_Q);
  neon_v = vdup_n_s16(((1U << 26) + KYBER_Q / 2) / KYBER_Q);

  // Total register: 26 registers.
  unsigned int i;
  for (i = 0; i < KYBER_N; i += 32) {
    vload(aa, &a->coeffs[i]);
    vload(cc, &c->coeffs[i]);

    // c = c - a;
    vsub8(cc.val[0], cc.val[0], aa.val[0]);
    vsub8(cc.val[1], cc.val[1], aa.val[1]);
    vsub8(cc.val[2], cc.val[2], aa.val[2]);
    vsub8(cc.val[3], cc.val[3], aa.val[3]);

    // c = reduce(c)
    barrett(cc.val[0], aa.val[0], t0_lo, t0_hi, t01, t02, neon_v, neon_kyberq16);
    barrett(cc.val[1], aa.val[1], t1_lo, t1_hi, t11, t12, neon_v, neon_kyberq16);
    barrett(cc.val[2], aa.val[2], t2_lo, t2_hi, t21, t22, neon_v, neon_kyberq16);
    barrett(cc.val[3], aa.val[3], t3_lo, t3_hi, t31, t32, neon_v, neon_kyberq16);

    // c = t;
    vstore(&c->coeffs[i], cc);
  }
}


/*
c = c + a + b
c = reduce(c)
*/
void neon_poly_add_add_reduce(poly *c, const poly *a, const poly *b) {
  int16x8x4_t cc, aa, bb;                                           // 12
  int16x8_t neon_kyberq16;
  int16x4_t neon_v;                                  // 2
  int16x4_t t0_lo, t0_hi, t1_lo, t1_hi, t2_lo, t2_hi, t3_lo, t3_hi; // 8
  int32x4_t t01, t02, t11, t12, t21, t22, t31, t32;                 // 8

  neon_kyberq16 = vdupq_n_s16(-KYBER_Q);
  neon_v = vdup_n_s16(((1U << 26) + KYBER_Q / 2) / KYBER_Q);

  // Total register: 30 registers.
  unsigned int i;
  for (i = 0; i < KYBER_N; i += 32) {
    vload(aa, &a->coeffs[i]);
    vload(bb, &b->coeffs[i]);
    vload(cc, &c->coeffs[i]);

    // a' = a + b;
    vadd8(aa.val[0], aa.val[0], bb.val[0]);
    vadd8(aa.val[1], aa.val[1], bb.val[1]);
    vadd8(aa.val[2], aa.val[2], bb.val[2]);
    vadd8(aa.val[3], aa.val[3], bb.val[3]);

    // c = c + a' = c + a + b;
    vadd8(cc.val[0], cc.val[0], aa.val[0]);
    vadd8(cc.val[1], cc.val[1], aa.val[1]);
    vadd8(cc.val[2], cc.val[2], aa.val[2]);
    vadd8(cc.val[3], cc.val[3], aa.val[3]);

    // c = reduce(c)
    barrett(cc.val[0], bb.val[0], t0_lo, t0_hi, t01, t02, neon_v, neon_kyberq16);
    barrett(cc.val[1], bb.val[1], t1_lo, t1_hi, t11, t12, neon_v, neon_kyberq16);
    barrett(cc.val[2], bb.val[2], t2_lo, t2_hi, t21, t22, neon_v, neon_kyberq16);
    barrett(cc.val[3], bb.val[3], t3_lo, t3_hi, t31, t32, neon_v, neon_kyberq16);

    // c = t;
    vstore(&c->coeffs[i], cc);
  }
}

void neon_poly_getnoise_eta1_2x(poly *vec1, poly *vec2,
                                const uint8_t seed[KYBER_SYMBYTES],
                                uint8_t nonce1, uint8_t nonce2)
{
    uint8_t buf1[KYBER_ETA1 * KYBER_N / 4],
            buf2[KYBER_ETA1 * KYBER_N / 4];
    neon_prf(buf1, buf2, sizeof(buf1), seed, nonce1, nonce2);
    neon_cbd_eta1(vec1, buf1);
    neon_cbd_eta1(vec2, buf2);
}

void neon_poly_getnoise_eta2_2x(poly *vec1, poly *vec2,
                                const uint8_t seed[KYBER_SYMBYTES],
                                uint8_t nonce1, uint8_t nonce2)
{
    uint8_t buf1[KYBER_ETA2 * KYBER_N / 4],
            buf2[KYBER_ETA2 * KYBER_N / 4];
    neon_prf(buf1, buf2, sizeof(buf1), seed, nonce1, nonce2);
    neon_cbd_eta2(vec1, buf1);
    neon_cbd_eta2(vec2, buf2);
}

void neon_poly_getnoise_eta2(poly *r,
                             const uint8_t seed[KYBER_SYMBYTES],
                             uint8_t nonce)
{
    uint8_t buf[KYBER_ETA2 * KYBER_N / 4];
    prf(buf, sizeof(buf), seed, nonce);
    neon_cbd_eta2(r, buf);
}