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

#define NTESTS 10000000

uint8_t seed[KYBER_SYMBYTES] = {0};

/* Dummy randombytes for speed tests that simulates a fast randombytes implementation
 * as in SUPERCOP so that we get comparable cycle counts */
void randombytes(__attribute__((unused)) uint8_t *r, __attribute__((unused)) size_t len) {
  return;
}

#define TIME(s) clock_gettime(CLOCK_MONOTONIC_RAW, &s);
// Result is nanosecond per call 
#define  CALC(start, stop) \
  ((double) ((stop.tv_sec - start.tv_sec) * 1000000000 + (stop.tv_nsec - start.tv_nsec))) / NTESTS;


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
  struct timespec start, stop;
  long ns;

#ifndef KYBER_90S
  poly bp, cp, dp;
#endif

  TIME(start);
  for(i=0;i<NTESTS;i++) {
    gen_matrix(matrix, seed, 0);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("gen_matrix:", ns);


  TIME(start);
  for(i=0;i<NTESTS;i++) {
    poly_getnoise_eta1(&ap, seed, 0);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("poly_getnoise_eta1:", ns);

  TIME(start);
  for(i=0;i<NTESTS;i++) {
    poly_getnoise_eta2(&ap, seed, 0);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("poly_getnoise_eta2:", ns);

#ifndef KYBER_90S
  TIME(start);
  for(i=0;i<NTESTS;i++) {
    poly_getnoise_eta1_4x(&ap, &bp, &cp, &dp, seed, 0, 1, 2, 3);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("poly_getnoise_eta1_4x:", ns);
#endif

  TIME(start);
  for(i=0;i<NTESTS;i++) {
    poly_ntt(&ap);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("poly_ntt:", ns);

  TIME(start);
  for(i=0;i<NTESTS;i++) {
    poly_invntt_tomont(&ap);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("poly_invntt_tomont:", ns);

  TIME(start);
  for(i=0;i<NTESTS;i++) {
    polyvec_basemul_acc_montgomery(&ap, &matrix[0], &matrix[1]);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("polyvec_basemul_acc_montgomery:", ns);

  TIME(start);
  for(i=0;i<NTESTS;i++) {
    poly_tomsg(ct,&ap);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("poly_tomsg:", ns);
  
  TIME(start);
  for(i=0;i<NTESTS;i++) {
    poly_frommsg(&ap,ct);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("poly_frommsg:", ns);

  TIME(start);
  for(i=0;i<NTESTS;i++) {
    poly_compress(ct,&ap);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("poly_compress:", ns);
  
  TIME(start);
  for(i=0;i<NTESTS;i++) {
    poly_decompress(&ap,ct);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("poly_decompress:", ns);

  TIME(start);
  for(i=0;i<NTESTS;i++) {
    polyvec_compress(ct,&matrix[0]);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("polyvec_compress:", ns);

  TIME(start);
  for(i=0;i<NTESTS;i++) {
    polyvec_decompress(&matrix[0],ct);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("polyvec_decompress:", ns);

  TIME(start);
  for(i=0;i<NTESTS;i++) {
    indcpa_keypair(pk, sk);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("indcpa_keypair:", ns);
  
  TIME(start);
  for(i=0;i<NTESTS;i++) {
    indcpa_enc(ct, key, pk, seed);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("indcpa_enc:", ns);

  TIME(start);
  for(i=0;i<NTESTS;i++) {
    indcpa_dec(key, ct, sk);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("indcpa_dec:", ns);

  TIME(start);
  for(i=0;i<NTESTS;i++) {
    crypto_kem_keypair(pk, sk);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("crypto_kem_keypair:", ns);
  
  TIME(start);
  for(i=0;i<NTESTS;i++) {
    crypto_kem_enc(ct, key, pk);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("crypto_kem_enc:", ns);
  
  TIME(start);
  for(i=0;i<NTESTS;i++) {
    crypto_kem_dec(key, ct, sk);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("crypto_kem_dec:", ns);
  
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
