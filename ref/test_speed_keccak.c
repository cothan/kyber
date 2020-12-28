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
        poly_getnoise_eta1(&ap, seed, 1);
        poly_getnoise_eta1(&bp, seed, 0);
    }
    end = cpucycles() - start;
    printf("poly_getnoise_eta1: %lf\n", (double) end/NTESTS);

    start = cpucycles();
    for (i = 0; i < NTESTS; i++)
    {
        // t[i] = cpucycles();
        poly_getnoise_eta2(&ap, seed, 0);
        poly_getnoise_eta2(&bp, seed, 1);
    }
    end = cpucycles() - start;
    printf("poly_getnoise_eta2: %lf\n", (double) end/NTESTS);



    return 0;
}
