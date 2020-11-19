#ifndef REJSAMPLE_H
#define REJSAMPLE_H

#include <stdint.h>

#define neon_rej_uniform KYBER_NAMESPACE(_neon_rej_uniform)
unsigned int neon_rej_uniform(int16_t *r, const uint8_t *buf);

#define rej_uniform KYBER_NAMESPACE(_rej_uniform)
unsigned int rej_uniform(int16_t *r,
                         unsigned int len,
                         const uint8_t *buf,
                         unsigned int buflen);
#endif