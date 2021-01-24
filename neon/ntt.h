#ifndef NTT_H
#define NTT_H

#include <stdint.h>
#include "params.h"

#define zetas KYBER_NAMESPACE(_ref_zetas)
extern const int16_t zetas[128];

#define zetas_inv KYBER_NAMESPACE(_ref_zetas_inv)
extern const int16_t zetas_inv[128];

#define neon_zetas_inv KYBER_NAMESPACE(_neon_zetas_inv)
extern const int16_t neon_zetas_inv[384];

#define neon_zetas KYBER_NAMESPACE(_neon_zetas)
extern const int16_t neon_zetas[440];

#define neon_ntt KYBER_NAMESPACE(_neon_ntt)
void neon_ntt(int16_t poly[256]);

#define neon_invntt KYBER_NAMESPACE(_neon_invntt)
void neon_invntt(int16_t poly[256]);

#endif
