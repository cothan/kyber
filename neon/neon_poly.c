#include <arm_neon.h>
#include "params.h"
#include "poly.h"
#include "symmetric.h"
#include "cbd.h"

// Load int16x8x4_t c <= ptr*
#define vloadx4(c, ptr) c = vld1q_s16_x4(ptr);

// Load int16x8x4_t c <= ptr*
#define vstorex4(ptr, c) vst1q_s16_x4(ptr, c);

// c (int16x8) = a + b (int16x8)
#define vadd(c, a, b) c = vaddq_s16(a, b);

// c (int16x8) = a + b (int16x8)
#define vand(c, a, b) c = vandq_s16((int16x8_t)a, b);

// c (int16x8) = a - b (int16x8)
#define vsub(c, a, b) c = vsubq_s16(a, b);

// compare greater than or equal
#define vcompare(out, a, const_n) out = vcgeq_s16(a, const_n);

/*
reduce low and high of 
inout: 
int16x8_t inout, 
t32_1, t32_2: int32x4_t 
t16: int16x8_t 
neon_v, neon_kyber16
*/
#define barrett(inout, t, i)                                                  \
  t.val[i] = (int16x8_t)vmull_s16(vget_low_s16(inout), vget_low_s16(neon_v)); \
  t.val[i + 1] = (int16x8_t)vmull_high_s16(inout, neon_v);                    \
  t.val[i] = vuzp2q_s16(t.val[i], t.val[i + 1]);                              \
  t.val[i + 1] = vshrq_n_s16(t.val[i], 10);                                   \
  inout = vmlsq_s16(inout, t.val[i + 1], neon_kyberq);

void neon_poly_reduce(poly *c)
{
  int16x8x4_t cc, t;             // 8
  int16x8_t neon_v, neon_kyberq; // 2

  neon_kyberq = vdupq_n_s16(KYBER_Q);
  neon_v = vdupq_n_s16(((1U << 26) + KYBER_Q / 2) / KYBER_Q);

  // Total register: 18 registers.
  for (int i = 0; i < KYBER_N; i += 32)
  {
    vloadx4(cc, &c->coeffs[i]);

    // c = reduce(c)
    barrett(cc.val[0], t, 0);
    barrett(cc.val[1], t, 2);
    barrett(cc.val[2], t, 0);
    barrett(cc.val[3], t, 2);

    // c = t;
    vstorex4(&c->coeffs[i], cc);
  }
}

/*
c = c + a
c = reduce(c)
*/
void neon_poly_add_reduce_csubq(poly *c, const poly *a)
{
  int16x8x4_t cc, aa, t;         // 8
  uint16x8x4_t res;              // 4
  int16x8_t neon_v, neon_kyberq; // 2

  neon_kyberq = vdupq_n_s16(KYBER_Q);
  neon_v = vdupq_n_s16(((1U << 26) + KYBER_Q / 2) / KYBER_Q);

  // Total register: 26 registers.
  unsigned int i;
  for (i = 0; i < KYBER_N; i += 32)
  {
    vloadx4(aa, &a->coeffs[i]);
    vloadx4(cc, &c->coeffs[i]);

    // c = c - a;
    vadd(cc.val[0], cc.val[0], aa.val[0]);
    vadd(cc.val[1], cc.val[1], aa.val[1]);
    vadd(cc.val[2], cc.val[2], aa.val[2]);
    vadd(cc.val[3], cc.val[3], aa.val[3]);

    // c = reduce(c)
    barrett(cc.val[0], t, 0);
    barrett(cc.val[1], t, 2);
    barrett(cc.val[2], t, 0);
    barrett(cc.val[3], t, 2);

    // c = csubq(c)
    // res = 0xffff if cc >= kyber_q else 0
    vcompare(res.val[0], cc.val[0], neon_kyberq);
    vcompare(res.val[1], cc.val[1], neon_kyberq);
    vcompare(res.val[2], cc.val[2], neon_kyberq);
    vcompare(res.val[3], cc.val[3], neon_kyberq);

    // res = res & kyber_q
    vand(aa.val[0], res.val[0], neon_kyberq);
    vand(aa.val[1], res.val[1], neon_kyberq);
    vand(aa.val[2], res.val[2], neon_kyberq);
    vand(aa.val[3], res.val[3], neon_kyberq);

    // c = c - a;
    vsub(cc.val[0], cc.val[0], aa.val[0]);
    vsub(cc.val[1], cc.val[1], aa.val[1]);
    vsub(cc.val[2], cc.val[2], aa.val[2]);
    vsub(cc.val[3], cc.val[3], aa.val[3]);

    // c = t;
    vstorex4(&c->coeffs[i], cc);
  }
}

/*
c = c - a
c = reduce(c)
*/
void neon_poly_sub_reduce_csubq(poly *c, const poly *a)
{
  int16x8x4_t cc, aa, t;         // 8
  uint16x8x4_t res;              // 4
  int16x8_t neon_v, neon_kyberq; // 2

  neon_kyberq = vdupq_n_s16(KYBER_Q);
  neon_v = vdupq_n_s16(((1U << 26) + KYBER_Q / 2) / KYBER_Q);

  // Total register: 26 registers.
  unsigned int i;
  for (i = 0; i < KYBER_N; i += 32)
  {
    vloadx4(aa, &a->coeffs[i]);
    vloadx4(cc, &c->coeffs[i]);

    // c = c - a;
    vsub(cc.val[0], cc.val[0], aa.val[0]);
    vsub(cc.val[1], cc.val[1], aa.val[1]);
    vsub(cc.val[2], cc.val[2], aa.val[2]);
    vsub(cc.val[3], cc.val[3], aa.val[3]);

    // c = reduce(c)
    barrett(cc.val[0], t, 0);
    barrett(cc.val[1], t, 2);
    barrett(cc.val[2], t, 0);
    barrett(cc.val[3], t, 2);

    // c = csubq(c)
    // res = 0xffff if cc >= kyber_q else 0
    vcompare(res.val[0], cc.val[0], neon_kyberq);
    vcompare(res.val[1], cc.val[1], neon_kyberq);
    vcompare(res.val[2], cc.val[2], neon_kyberq);
    vcompare(res.val[3], cc.val[3], neon_kyberq);

    // res = res & kyber_q
    vand(aa.val[0], res.val[0], neon_kyberq);
    vand(aa.val[1], res.val[1], neon_kyberq);
    vand(aa.val[2], res.val[2], neon_kyberq);
    vand(aa.val[3], res.val[3], neon_kyberq);

    // c = c - a;
    vsub(cc.val[0], cc.val[0], aa.val[0]);
    vsub(cc.val[1], cc.val[1], aa.val[1]);
    vsub(cc.val[2], cc.val[2], aa.val[2]);
    vsub(cc.val[3], cc.val[3], aa.val[3]);

    // c = t;
    vstorex4(&c->coeffs[i], cc);
  }
}

/*
c = c + a + b
c = reduce(c)
*/
void neon_poly_add_add_reduce_csubq(poly *c, const poly *a, const poly *b)
{
  int16x8x4_t cc, aa, bb, t;        // 16
  uint16x8x4_t res;              // 4
  int16x8_t neon_v, neon_kyberq; // 3

  neon_kyberq = vdupq_n_s16(KYBER_Q);
  neon_v = vdupq_n_s16(((1U << 26) + KYBER_Q / 2) / KYBER_Q);

  // Total register: 30 registers.
  unsigned int i;
  for (i = 0; i < KYBER_N; i += 32)
  {
    vloadx4(aa, &a->coeffs[i]);
    vloadx4(bb, &b->coeffs[i]);
    vloadx4(cc, &c->coeffs[i]);

    // a' = a + b;
    vadd(aa.val[0], aa.val[0], bb.val[0]);
    vadd(aa.val[1], aa.val[1], bb.val[1]);
    vadd(aa.val[2], aa.val[2], bb.val[2]);
    vadd(aa.val[3], aa.val[3], bb.val[3]);

    // c = c + a' = c + a + b;
    vadd(cc.val[0], cc.val[0], aa.val[0]);
    vadd(cc.val[1], cc.val[1], aa.val[1]);
    vadd(cc.val[2], cc.val[2], aa.val[2]);
    vadd(cc.val[3], cc.val[3], aa.val[3]);

    // c = reduce(c)
    barrett(cc.val[0], t, 0);
    barrett(cc.val[1], t, 2);
    barrett(cc.val[2], t, 0);
    barrett(cc.val[3], t, 2);

    // c = csubq(c)
    // res = 0xffff if cc >= kyber_q else 0
    vcompare(res.val[0], cc.val[0], neon_kyberq);
    vcompare(res.val[1], cc.val[1], neon_kyberq);
    vcompare(res.val[2], cc.val[2], neon_kyberq);
    vcompare(res.val[3], cc.val[3], neon_kyberq);

    // res = res & kyber_q
    vand(aa.val[0], res.val[0], neon_kyberq);
    vand(aa.val[1], res.val[1], neon_kyberq);
    vand(aa.val[2], res.val[2], neon_kyberq);
    vand(aa.val[3], res.val[3], neon_kyberq);

    // c = c - a;
    vsub(cc.val[0], cc.val[0], aa.val[0]);
    vsub(cc.val[1], cc.val[1], aa.val[1]);
    vsub(cc.val[2], cc.val[2], aa.val[2]);
    vsub(cc.val[3], cc.val[3], aa.val[3]);

    // c = t;
    vstorex4(&c->coeffs[i], cc);
  }
}

void neon_poly_getnoise_eta1_2x(poly *vec1, poly *vec2,
                                const uint8_t seed[KYBER_SYMBYTES],
                                uint8_t nonce1, uint8_t nonce2)
{
  uint8_t buf1[KYBER_ETA1 * KYBER_N / 4],
      buf2[KYBER_ETA1 * KYBER_N / 4];
  neon_prf(buf1, buf2, sizeof(buf1), seed, nonce1, nonce2);
  cbd_eta1(vec1, buf1);
  cbd_eta1(vec2, buf2);
}

void neon_poly_getnoise_eta2_2x(poly *vec1, poly *vec2,
                                const uint8_t seed[KYBER_SYMBYTES],
                                uint8_t nonce1, uint8_t nonce2)
{
  uint8_t buf1[KYBER_ETA2 * KYBER_N / 4],
      buf2[KYBER_ETA2 * KYBER_N / 4];
  neon_prf(buf1, buf2, sizeof(buf1), seed, nonce1, nonce2);
  cbd_eta2(vec1, buf1);
  cbd_eta2(vec2, buf2);
}

void neon_poly_getnoise_eta2(poly *r,
                             const uint8_t seed[KYBER_SYMBYTES],
                             uint8_t nonce)
{
  uint8_t buf[KYBER_ETA2 * KYBER_N / 4];
  prf(buf, sizeof(buf), seed, nonce);
  cbd_eta2(r, buf);
}

/*************************************************
* Name:        poly_csubq
*
* Description: Applies conditional subtraction of q to each coefficient
*              of a polynomial. For details of conditional subtraction
*              of q see comments in reduce.c
*
* Arguments:   - poly *r: pointer to input/output polynomial
**************************************************/
void poly_csubq(poly *r)
{
  int16x8_t const_kyberq;
  uint16x8x4_t res;
  int16x8x4_t cc, aa;

  const_kyberq = vdupq_n_s16(KYBER_Q);

  for (unsigned int i = 0; i < KYBER_N; i += 32)
  {
    vloadx4(cc, &r->coeffs[i]);
    // res = 0xffff if cc >= kyber_q else 0
    vcompare(res.val[0], cc.val[0], const_kyberq);
    vcompare(res.val[1], cc.val[1], const_kyberq);
    vcompare(res.val[2], cc.val[2], const_kyberq);
    vcompare(res.val[3], cc.val[3], const_kyberq);

    // res = res & kyber_q
    vand(aa.val[0], res.val[0], const_kyberq);
    vand(aa.val[1], res.val[1], const_kyberq);
    vand(aa.val[2], res.val[2], const_kyberq);
    vand(aa.val[3], res.val[3], const_kyberq);

    // c = c - a;
    vsub(cc.val[0], cc.val[0], aa.val[0]);
    vsub(cc.val[1], cc.val[1], aa.val[1]);
    vsub(cc.val[2], cc.val[2], aa.val[2]);
    vsub(cc.val[3], cc.val[3], aa.val[3]);

    vstorex4(&r->coeffs[i], cc);
  }
}

/*************************************************
* Name:        poly_frombytes
*
* Description: De-serialization of a polynomial;
*              inverse of poly_tobytes
*
* Arguments:   - poly *r:          pointer to output polynomial
*              - const uint8_t *a: pointer to input byte array
*                                  (of KYBER_POLYBYTES bytes)
**************************************************/
void poly_frombytes(poly *r, const uint8_t a[KYBER_POLYBYTES])
{
  uint8x16x3_t neon_buf;
  uint16x8x4_t tmp;
  int16x8x4_t value;
  uint16x8_t const_0xfff;
  const_0xfff = vdupq_n_u16(0xfff);

  unsigned int i, j = 0;
  for (i = 0; i < KYBER_POLYBYTES; i += 48)
  {
    neon_buf = vld3q_u8(&a[i]);

    // Val0: 0-1 | 3-4 | 6-7| 9-10
    tmp.val[0] = (uint16x8_t)vzip1q_u8(neon_buf.val[0], neon_buf.val[1]);
    tmp.val[1] = (uint16x8_t)vzip2q_u8(neon_buf.val[0], neon_buf.val[1]);

    tmp.val[0] = vandq_u16(tmp.val[0], const_0xfff);
    tmp.val[1] = vandq_u16(tmp.val[1], const_0xfff);

    // Val1: 1-2 | 4-5 | 7-8 | 10-11
    tmp.val[2] = (uint16x8_t)vzip1q_u8(neon_buf.val[1], neon_buf.val[2]);
    tmp.val[3] = (uint16x8_t)vzip2q_u8(neon_buf.val[1], neon_buf.val[2]);

    tmp.val[2] = vshrq_n_u16(tmp.val[2], 4);
    tmp.val[3] = vshrq_n_u16(tmp.val[3], 4);

    // Final value
    value.val[0] = (int16x8_t)vzip1q_u16(tmp.val[0], tmp.val[2]);
    value.val[1] = (int16x8_t)vzip2q_u16(tmp.val[0], tmp.val[2]);
    value.val[2] = (int16x8_t)vzip1q_u16(tmp.val[1], tmp.val[3]);
    value.val[3] = (int16x8_t)vzip2q_u16(tmp.val[1], tmp.val[3]);

    vst1q_s16_x4(&r->coeffs[j], value);
    j += 32;
  }
}
