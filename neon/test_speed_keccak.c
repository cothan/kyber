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
#include "papi.h"

#define NTESTS 100000

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
    printf("gen_a: %lld\n", end/NTESTS);

    start = cpucycles();
    for (i = 0; i < NTESTS; i++)
    {
        // t[i] = cpucycles();
        neon_poly_getnoise_eta1_2x(&ap, &bp, seed, 0, 1);
    }
    end = cpucycles() - start;
    printf("neon_poly_getnoise_eta1_2x: %lld\n", end/NTESTS);

    start = cpucycles();
    for (i = 0; i < NTESTS; i++)
    {
        // t[i] = cpucycles();
        neon_poly_getnoise_eta2_2x(&ap, &bp, seed, 0, 1);
    }
    end = cpucycles() - start;
    printf("neon_poly_getnoise_eta2_2x: %lld\n", end/NTESTS);

    // PAPI_hl_region_begin("SHAKE128");
    // for (i = 0; i < NTESTS; i++)
    // {
    //     // t[i] = cpucycles();
    //     (&ap, seed, 0);
    // }
    // PAPI_hl_region_end("SHAKE128");




    return 0;
}
