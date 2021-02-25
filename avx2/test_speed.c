#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "kem.h"
#include "kex.h"
#include "params.h"
#include "indcpa.h"
#include "polyvec.h"
#include "poly.h"
#include "randombytes.h"
#include "cpucycles.h"
#include "speed_print.h"
#include <time.h>
#include "print.h"

#define NTESTS 1000000000

uint8_t seed[KYBER_SYMBYTES] = {0};

/* Dummy randombytes for speed tests that simulates a fast randombytes implementation
 * as in SUPERCOP so that we get comparable cycle counts */
void randombytes(__attribute__((unused)) uint8_t *r, __attribute__((unused)) size_t len) {
  return;
}

int main()
{
  unsigned int i;
  uint8_t pk[CRYPTO_PUBLICKEYBYTES];
  uint8_t sk[CRYPTO_SECRETKEYBYTES];
  uint8_t ct[CRYPTO_CIPHERTEXTBYTES];
  uint8_t key[CRYPTO_BYTES];
  uint8_t kexsenda[KEX_AKE_SENDABYTES];
  uint8_t kexsendb[KEX_AKE_SENDBBYTES];
  uint8_t kexkey[KEX_SSBYTES];
  polyvec matrix[KYBER_K];
  poly ap;
  clock_t start, end;
#ifndef KYBER_90S
  poly bp, cp, dp;
#endif

  start = clock();
  for(i=0;i<NTESTS;i++) {
    gen_matrix(matrix, seed, 0);
  }
  end = clock() - start;
  print("gen_matrix:", ((double) end)/CLOCKS_PER_SEC);


  start = clock();
  for(i=0;i<NTESTS;i++) {
    poly_getnoise_eta1(&ap, seed, 0);
  }
  end = clock() - start;
  print("poly_getnoise_eta1:", ((double) end)/CLOCKS_PER_SEC);

  start = clock();
  for(i=0;i<NTESTS;i++) {
    poly_getnoise_eta2(&ap, seed, 0);
  }
  end = clock() - start;
  print("poly_getnoise_eta2:", ((double) end)/CLOCKS_PER_SEC);

#ifndef KYBER_90S
  start = clock();
  for(i=0;i<NTESTS;i++) {
    poly_getnoise_eta1_4x(&ap, &bp, &cp, &dp, seed, 0, 1, 2, 3);
  }
  end = clock() - start;
  print("poly_getnoise_eta1_4x:", ((double) end)/CLOCKS_PER_SEC);
#endif

  start = clock();
  for(i=0;i<NTESTS;i++) {
    poly_ntt(&ap);
  }
  end = clock() - start;
  print("poly_ntt:", ((double) end)/CLOCKS_PER_SEC);

  start = clock();
  for(i=0;i<NTESTS;i++) {
    poly_invntt_tomont(&ap);
  }
  end = clock() - start;
  print("poly_invntt_tomont:", ((double) end)/CLOCKS_PER_SEC);

  start = clock();
  for(i=0;i<NTESTS;i++) {
    polyvec_basemul_acc_montgomery(&ap, &matrix[0], &matrix[1]);
  }
  end = clock() - start;
  print("polyvec_basemul_acc_montgomery:", ((double) end)/CLOCKS_PER_SEC);

  start = clock();
  for(i=0;i<NTESTS;i++) {
    poly_tomsg(ct,&ap);
  }
  end = clock() - start;
  print("poly_tomsg:", ((double) end)/CLOCKS_PER_SEC);
  
  start = clock();
  for(i=0;i<NTESTS;i++) {
    poly_frommsg(&ap,ct);
  }
  end = clock() - start;
  print("poly_frommsg:", ((double) end)/CLOCKS_PER_SEC);

  start = clock();
  for(i=0;i<NTESTS;i++) {
    poly_compress(ct,&ap);
  }
  end = clock() - start;
  print("poly_compress:", ((double) end)/CLOCKS_PER_SEC);
  
  start = clock();
  for(i=0;i<NTESTS;i++) {
    poly_decompress(&ap,ct);
  }
  end = clock() - start;
  print("poly_decompress:", ((double) end)/CLOCKS_PER_SEC);

  start = clock();
  for(i=0;i<NTESTS;i++) {
    polyvec_compress(ct,&matrix[0]);
  }
  end = clock() - start;
  print("polyvec_compress:", ((double) end)/CLOCKS_PER_SEC);

  start = clock();
  for(i=0;i<NTESTS;i++) {
    polyvec_decompress(&matrix[0],ct);
  }
  end = clock() - start;
  print("polyvec_decompress:", ((double) end)/CLOCKS_PER_SEC);

  start = clock();
  for(i=0;i<NTESTS;i++) {
    indcpa_keypair(pk, sk);
  }
  end = clock() - start;
  print("indcpa_keypair:", ((double) end)/CLOCKS_PER_SEC);
  
  start = clock();
  for(i=0;i<NTESTS;i++) {
    indcpa_enc(ct, key, pk, seed);
  }
  end = clock() - start;
  print("indcpa_enc:", ((double) end)/CLOCKS_PER_SEC);

  start = clock();
  for(i=0;i<NTESTS;i++) {
    indcpa_dec(key, ct, sk);
  }
  end = clock() - start;
  print("indcpa_dec:", ((double) end)/CLOCKS_PER_SEC);

  start = clock();
  for(i=0;i<NTESTS;i++) {
    crypto_kem_keypair(pk, sk);
  }
  end = clock() - start;
  print("crypto_kem_keypair:", ((double) end)/CLOCKS_PER_SEC);
  
  start = clock();
  for(i=0;i<NTESTS;i++) {
    crypto_kem_enc(ct, key, pk);
  }
  end = clock() - start;
  print("crypto_kem_enc:", ((double) end)/CLOCKS_PER_SEC);
  
  start = clock();
  for(i=0;i<NTESTS;i++) {
    crypto_kem_dec(key, ct, sk);
  }
  end = clock() - start;
  print("crypto_kem_dec:", ((double) end)/CLOCKS_PER_SEC);
  
/* 
  for(i=0;i<NTESTS;i++) {
    t[i] = cpucycles();
    kex_uake_initA(kexsenda, key, sk, pk);
  }
  

  for(i=0;i<NTESTS;i++) {
    t[i] = cpucycles();
    kex_uake_sharedB(kexsendb, kexkey, kexsenda, sk);
  }
  

  for(i=0;i<NTESTS;i++) {
    t[i] = cpucycles();
    kex_uake_sharedA(kexkey, kexsendb, key, sk);
  }
  

  for(i=0;i<NTESTS;i++) {
    t[i] = cpucycles();
    kex_ake_initA(kexsenda, key, sk, pk);
  }
  

  for(i=0;i<NTESTS;i++) {
    t[i] = cpucycles();
    kex_ake_sharedB(kexsendb, kexkey, kexsenda, sk, pk);
  }
  

  for(i=0;i<NTESTS;i++) {
    t[i] = cpucycles();
    kex_ake_sharedA(kexkey, kexsendb, key, sk, sk);
  }
  
 */
  return 0;
}
