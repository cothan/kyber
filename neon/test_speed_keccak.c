#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "api.h"
#include "kex.h"
#include "params.h"
#include "indcpa.h"
#include "poly.h"
#include "polyvec.h"
#include "cpucycles.h"
#include "speed_print.h"
#include "ntt.h"
#include "symmetric.h"
#include "papi.h"

#define NTESTS 1000000

// uint64_t t[NTESTS];
uint8_t seed[KYBER_SYMBYTES] = {0};

int main()
{
    unsigned int i;
    unsigned char pk[CRYPTO_PUBLICKEYBYTES] = {0};
    unsigned char sk[CRYPTO_SECRETKEYBYTES] = {0};
    unsigned char ct[CRYPTO_CIPHERTEXTBYTES] = {0};
    unsigned char key[CRYPTO_BYTES] = {0};
    unsigned char kexsenda[KEX_AKE_SENDABYTES] = {0};
    unsigned char kexsendb[KEX_AKE_SENDBBYTES] = {0};
    unsigned char kexkey[KEX_SSBYTES] = {0};
    unsigned char msg[KYBER_INDCPA_MSGBYTES] = {0};
    uint8_t buf1eta2[KYBER_ETA2 * KYBER_N / 4],
            buf2eta2[KYBER_ETA2 * KYBER_N / 4];
    uint8_t buf1eta1[KYBER_ETA1 * KYBER_N / 4],
            buf2eta1[KYBER_ETA1 * KYBER_N / 4];
    polyvec matrix[KYBER_K];
    poly ap, bp;
    int retval; 
    printf("NTESTS: %d\n", NTESTS);
    long_long start, end;

    start = cpucycles();
    for (i = 0; i < NTESTS; i++)
    {
        // t[i] = cpucycles();
        gen_matrix(matrix, seed, 0);
    }
    end = cpucycles() - start;
    printf("gen_a: %lf\n", (double) end/NTESTS);

    start = cpucycles();
    for (i = 0; i < NTESTS; i++)
    {
        // t[i] = cpucycles();
        neon_poly_getnoise_eta1_2x(&ap, &bp, seed, 0, 1);
    }
    end = cpucycles() - start;
    printf("poly_getnoise_eta1: %lf\n", (double) end/NTESTS);

    start = cpucycles();
    for (i = 0; i < NTESTS; i++)
    {
        // t[i] = cpucycles();
        neon_poly_getnoise_eta2_2x(&ap, &bp, seed, 0, 1);
    }
    end = cpucycles() - start;
    printf("poly_getnoise_eta2: %lf\n", (double) end/NTESTS);

    start = cpucycles();
    for (i = 0; i < NTESTS; i++)
    {
        // t[i] = cpucycles();
        neon_prf(buf1eta1, buf2eta1, sizeof(buf2eta1), seed, 0, 1);
    }
    end = cpucycles() - start;
    printf("prf: %d->%d: %lf\n", 32, sizeof(buf1eta1), (double) end/NTESTS);

    start = cpucycles();
    for (i = 0; i < NTESTS; i++)
    {
        // t[i] = cpucycles();
        neon_prf(buf1eta2, buf2eta2, sizeof(buf2eta2), seed, 0, 1);
    }
    end = cpucycles() - start;
    printf("prf: %d->%d: %lf\n", 32, sizeof(buf1eta2), (double) end/NTESTS);




    return 0;
}
