#ifndef NTT_H
#define NTT_H

#include <stdint.h>
#include "params.h"

#define neon_zetas_inv KYBER_NAMESPACE(neon_zetas_inv)
extern const int16_t neon_zetas_inv[272];

#define neon_zetas KYBER_NAMESPACE(neon_zetas)
extern const int16_t neon_zetas[224];

#define neon_ntt KYBER_NAMESPACE(neon_ntt)
void neon_ntt(int16_t poly[256]);

#define neon_invntt KYBER_NAMESPACE(neon_invntt)
void neon_invntt(int16_t poly[256]);

// #define neon_ntt KYBER_NAMESPACE(test_neon_ntt)
void test_neon_ntt(int16_t poly[256]);
void test_neon_ntt_qinv(int16_t r[256]);
void test_neon_ntt_qinv_lane(int16_t r[256]);

// #define neon_invntt KYBER_NAMESPACE(test_neon_invntt)
void test_neon_invntt(int16_t poly[256]);
void test_neon_invntt_qinv(int16_t poly[256]);
void test_neon_invntt_qinv_lane(int16_t poly[256]);

#endif
