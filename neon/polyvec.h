#ifndef POLYVEC_H
#define POLYVEC_H

#include <stdint.h>
#include "params.h"
#include "poly.h"

typedef struct{
  poly vec[KYBER_K];
} polyvec;

#define polyvec_compress KYBER_NAMESPACE(_ref_polyvec_compress)
void polyvec_compress(uint8_t r[KYBER_POLYVECCOMPRESSEDBYTES], polyvec *a);
#define polyvec_decompress KYBER_NAMESPACE(_ref_polyvec_decompress)
void polyvec_decompress(polyvec *r,
                        const uint8_t a[KYBER_POLYVECCOMPRESSEDBYTES]);

#define polyvec_tobytes KYBER_NAMESPACE(_ref_polyvec_tobytes)
void polyvec_tobytes(uint8_t r[KYBER_POLYVECBYTES], polyvec *a, 
                     const uint8_t reduce);

#define polyvec_frombytes KYBER_NAMESPACE(_ref_polyvec_frombytes)
void polyvec_frombytes(polyvec *r, const uint8_t a[KYBER_POLYVECBYTES]);

// Add NEON

#define neon_polyvec_ntt KYBER_NAMESPACE(_neon_polyvec_ntt)
void neon_polyvec_ntt(polyvec *r);

#define neon_polyvec_invntt_to_mont KYBER_NAMESPACE(_neon_polyvec_invntt_to_mont)
void neon_polyvec_invntt_to_mont(polyvec *r);

#define neon_polyvec_add_reduce_csubq KYBER_NAMESPACE(_neon_polyvec_add_reduce_csubq)
void neon_polyvec_add_reduce_csubq(polyvec *c, const polyvec *a);

#define neon_polyvec_acc_montgomery KYBER_NAMESPACE(_neon_polyvec_acc_montgomery)
void neon_polyvec_acc_montgomery(poly *c, const polyvec *a, const polyvec *b, const int to_mont);

#define polyvec_csubq KYBER_NAMESPACE(_neon_polyvec_csubq)
void polyvec_csubq(polyvec *r);

#endif
