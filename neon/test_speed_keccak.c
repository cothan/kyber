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
    // unsigned char pk[CRYPTO_PUBLICKEYBYTES] = {0};
    // unsigned char sk[CRYPTO_SECRETKEYBYTES] = {0};
    // unsigned char ct[CRYPTO_CIPHERTEXTBYTES] = {0};
    // unsigned char key[CRYPTO_BYTES] = {0};
    // unsigned char kexsenda[KEX_AKE_SENDABYTES] = {0};
    // unsigned char kexsendb[KEX_AKE_SENDBBYTES] = {0};
    // unsigned char kexkey[KEX_SSBYTES] = {0};
    // unsigned char msg[KYBER_INDCPA_MSGBYTES] = {0};
    uint8_t buf1eta2[KYBER_ETA2 * KYBER_N / 4],
            buf2eta2[KYBER_ETA2 * KYBER_N / 4];
    uint8_t buf1eta1[KYBER_ETA1 * KYBER_N / 4],
            buf2eta1[KYBER_ETA1 * KYBER_N / 4];
    polyvec matrix[KYBER_K];
    poly ap, bp;
    int retval; 
    neon_xof_state state;
    
    #define GEN_MATRIX_NBLOCKS ((12*KYBER_N/8*(1 << 12)/KYBER_Q \
                             + XOF_BLOCKBYTES)/XOF_BLOCKBYTES)

    uint8_t buf0[GEN_MATRIX_NBLOCKS * XOF_BLOCKBYTES + 2],
            buf1[GEN_MATRIX_NBLOCKS * XOF_BLOCKBYTES + 2];

    printf("NTESTS: %d\n", NTESTS);
    long_long start, end;

    //start = cpucycles();
    for (i = 0; i < NTESTS; i++)
    {
        // t[i] = cpucycles();
        PAPI_hl_region_begin("gen_a");  
        gen_matrix(matrix, seed, 0);
        PAPI_hl_region_end("gen_a");
    }
    //end = cpucycles() - start;
    //printf("gen_a: %lf\n", (double) end/NTESTS);

    //start = cpucycles();
    for (i = 0; i < NTESTS; i++)
    {
        // t[i] = cpucycles();
        PAPI_hl_region_begin("noise1_2x");  
        neon_poly_getnoise_eta1_2x(&ap, &bp, seed, 0, 1);
        PAPI_hl_region_end("noise1_2x");
    }
    //end = cpucycles() - start;
    //printf("poly_getnoise_eta1: %lf\n", (double) end/NTESTS);

    //start = cpucycles();
    for (i = 0; i < NTESTS; i++)
    {
        // t[i] = cpucycles();
        PAPI_hl_region_begin("noise2_2x");  
        neon_poly_getnoise_eta2_2x(&ap, &bp, seed, 0, 1);
        PAPI_hl_region_end("noise2_2x");
    }
    //end = cpucycles() - start;
    //printf("poly_getnoise_eta2: %lf\n", (double) end/NTESTS);

    //start = cpucycles();
    for (i = 0; i < NTESTS; i++)
    {
        // t[i] = cpucycles();
        PAPI_hl_region_begin("prf");  
        neon_prf(buf1eta1, buf2eta1, sizeof(buf2eta1), seed, 0, 1);
        PAPI_hl_region_end("prf");
    }
    //end = cpucycles() - start;
    printf("prf: %d->%d \n", 32, sizeof(buf1eta1) );

    //art = cpucycles();
    for (i = 0; i < NTESTS; i++)
    {
        // t[i] = cpucycles();
        PAPI_hl_region_begin("prf2");
        neon_prf(buf1eta2, buf2eta2, sizeof(buf2eta2), seed, 0, 1);
        PAPI_hl_region_end("prf2");
    }
    //end = cpucycles() - start;
    printf("prf: %d->%d\n", 32, sizeof(buf1eta2) );


    //start = cpucycles();
    for (i = 0; i < NTESTS; i++)
    {
        // t[i] = cpucycles();
        PAPI_hl_region_begin("abs");  
        neon_kyber_shake128_absorb(&state, seed, 0, 1, 1, 0);
        neon_xof_squeezeblocks(buf0, buf1, GEN_MATRIX_NBLOCKS, &state);
        PAPI_hl_region_end("abs");
    }
    //end = cpucycles() - start;
    printf("SHAKE128 ABS SQZ: %d->%d \n", 32, sizeof(buf1) );


    return 0;
}
