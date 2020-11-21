#ifndef POLY_H
#define POLY_H

#include <stdint.h>
#include "params.h"

/*
 * Elements of R_q = Z_q[X]/(X^n + 1). Represents polynomial
 * coeffs[0] + X*coeffs[1] + X^2*xoeffs[2] + ... + X^{n-1}*coeffs[n-1]
 */
typedef struct{
  int16_t coeffs[KYBER_N];
} poly;

#define poly_compress KYBER_NAMESPACE(_ref_poly_compress)
void poly_compress(uint8_t r[KYBER_POLYCOMPRESSEDBYTES], poly *a);
#define poly_decompress KYBER_NAMESPACE(_ref_poly_decompress)
void poly_decompress(poly *r, const uint8_t a[KYBER_POLYCOMPRESSEDBYTES]);

#define poly_tobytes KYBER_NAMESPACE(_ref_poly_tobytes)
void poly_tobytes(uint8_t r[KYBER_POLYBYTES], poly *a, 
                  const uint8_t reduce);

#define poly_frommsg KYBER_NAMESPACE(_ref_poly_frommsg)
void poly_frommsg(poly *r, const uint8_t msg[KYBER_INDCPA_MSGBYTES]);
#define poly_tomsg KYBER_NAMESPACE(_ref_poly_tomsg)
void poly_tomsg(uint8_t msg[KYBER_INDCPA_MSGBYTES], poly *r);

// NEON

#define neon_poly_reduce KYBER_NAMESPACE(_neon_poly_reduce)
void neon_poly_reduce(poly *c);
#define neon_poly_add_reduce_csubq KYBER_NAMESPACE(_neon_poly_add_reduce_csubq)
void neon_poly_add_reduce_csubq(poly *c, const poly *a);

#define neon_poly_sub_reduce_csubq KYBER_NAMESPACE(_neon_poly_sub_reduce_csubq)
void neon_poly_sub_reduce_csubq(poly *c, const poly *a);

#define neon_poly_add_add_reduce_csubq KYBER_NAMESPACE(_neon_poly_add_add_reduce_csubq)
void neon_poly_add_add_reduce_csubq(poly *c, const poly *a, const poly *b);

#define neon_poly_getnoise_eta1_2x KYBER_NAMESPACE(_neon_poly_getnoise_eta1_2x)
void neon_poly_getnoise_eta1_2x(poly *vec1, poly *vec2,
                                const uint8_t seed[KYBER_SYMBYTES],
                                uint8_t nonce1, uint8_t nonce2);

#define neon_poly_getnoise_eta2_2x KYBER_NAMESPACE(_neon_poly_getnoise_eta2_2x)
void neon_poly_getnoise_eta2_2x(poly *vec1, poly *vec2,
                                const uint8_t seed[KYBER_SYMBYTES],
                                uint8_t nonce1, uint8_t nonce2);

#define neon_poly_getnoise_eta2 KYBER_NAMESPACE(_neon_poly_getnoise_eta2)
void neon_poly_getnoise_eta2(poly *r,
                             const uint8_t seed[KYBER_SYMBYTES],
                             uint8_t nonce);

#define poly_csubq KYBER_NAMESPACE(_neon_poly_csubq)
void poly_csubq(poly *r);

#define poly_frombytes KYBER_NAMESPACE(_neon_poly_frombytes)
void poly_frombytes(poly *r, const uint8_t a[KYBER_POLYBYTES]);

#endif


  
