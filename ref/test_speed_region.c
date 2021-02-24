#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "kem.h"
#include "kex.h"
#include "params.h"
#include "indcpa.h"
#include "poly.h"
#include "polyvec.h"
// #include "papi.h"
#include "ntt.h"
#include <time.h>
#include "print.h"

// micro second 
#define NTESTS 1000000

uint64_t t[NTESTS];
uint8_t seed[KYBER_SYMBYTES] = {0};

int main()
{
  unsigned int i;
  unsigned char pk[CRYPTO_PUBLICKEYBYTES] = {0};
  unsigned char sk[CRYPTO_SECRETKEYBYTES] = {0};
  unsigned char ct[CRYPTO_CIPHERTEXTBYTES] = {0};
  unsigned char key[CRYPTO_BYTES] = {0};
  // unsigned char kexsenda[KEX_AKE_SENDABYTES] = {0};
  // unsigned char kexsendb[KEX_AKE_SENDBBYTES] = {0};
  // unsigned char kexkey[KEX_SSBYTES] = {0};
  unsigned char msg[KYBER_INDCPA_MSGBYTES] = {0};
  polyvec matrix[KYBER_K];
  poly ap;
  // poly bp;
  clock_t start, end; 

  // PAPI_hl_region_begin("gen_matrix");
  start = clock();
  for(i=0;i<NTESTS;i++) {
    gen_matrix(matrix, seed, 0);
  }
  end =  clock() - start;
  print("gen_matrix:", ((double) end)/CLOCKS_PER_SEC);
  // PAPI_hl_region_end("gen_matrix");

  // PAPI_hl_region_begin("poly_getnoise_eta1");
  start = clock();
  for(i=0;i<NTESTS;i++) {
    poly_getnoise_eta1(&ap, seed, 1);
  }
  end =  clock() - start;
  print("poly_getnoise_eta1:", ((double) end)/CLOCKS_PER_SEC);
  // PAPI_hl_region_end("poly_getnoise_eta1");

  // PAPI_hl_region_begin("poly_getnoise_eta2");
  start = clock();
  for(i=0;i<NTESTS;i++) {
    poly_getnoise_eta2(&ap, seed, 0);
  }
  end =  clock() - start;
  print("poly_getnoise_eta2:", ((double) end)/CLOCKS_PER_SEC);
  // PAPI_hl_region_end("poly_getnoise_eta2");

  // PAPI_hl_region_begin("poly_tomsg");
  start = clock();
  for(i=0;i<NTESTS;i++) {
    poly_tomsg(msg, &ap);
  }
  end =  clock() - start;
  print("poly_tomsg:", ((double) end)/CLOCKS_PER_SEC);
  // PAPI_hl_region_end("poly_tomsg");

  // PAPI_hl_region_begin("poly_frommsg");
  start = clock();
  for(i=0;i<NTESTS;i++) {
    poly_frommsg(&ap, msg);
  }
  end =  clock() - start;
  print("poly_frommsg:", ((double) end)/CLOCKS_PER_SEC);
  // PAPI_hl_region_end("poly_frommsg");


  // PAPI_hl_region_begin("ref_ntt");
  start = clock();
  for(i=0;i<NTESTS;i++) {
    ntt(ap.coeffs);
  }
  end =  clock() - start;
  print("ref_ntt:", ((double) end)/CLOCKS_PER_SEC);
  // PAPI_hl_region_end("ref_ntt");

  // PAPI_hl_region_begin("ref_invntt");
  start = clock();
  for(i=0;i<NTESTS;i++) {
    invntt(ap.coeffs);
  }
  end =  clock() - start;
  print("ref_invntt:", ((double) end)/CLOCKS_PER_SEC);
  // PAPI_hl_region_end("ref_invntt");

  // PAPI_hl_region_begin("crypto_kem_keypair");
  start = clock();
  for(i=0;i<NTESTS;i++) {
    crypto_kem_keypair(pk, sk);
  }
  end =  clock() - start;
  print("crypto_kem_keypair:", ((double) end)/CLOCKS_PER_SEC);
  // PAPI_hl_region_end("crypto_kem_keypair");

  // PAPI_hl_region_begin("crypto_kem_enc");
  start = clock();
  for(i=0;i<NTESTS;i++) {
    crypto_kem_enc(ct, key, pk);
  }
  end =  clock() - start;
  print("crypto_kem_enc:", ((double) end)/CLOCKS_PER_SEC);
  // PAPI_hl_region_end("crypto_kem_enc");

  // PAPI_hl_region_begin("crypto_kem_dec");
  start = clock();
  for(i=0;i<NTESTS;i++) {
    crypto_kem_dec(key, ct, sk);
  }
  end =  clock() - start;
  print("crypto_kem_dec:", ((double) end)/CLOCKS_PER_SEC);
  // PAPI_hl_region_end("crypto_kem_dec");

  /*
  PAPI_hl_region_begin("kex_uake_initA");
  for(i=0;i<NTESTS;i++) {
    kex_uake_initA(kexsenda, key, sk, pk);
  }
  PAPI_hl_region_end("kex_uake_initA");

  PAPI_hl_region_begin("kex_uake_sharedB");
  for(i=0;i<NTESTS;i++) {
    kex_uake_sharedB(kexsendb, kexkey, kexsenda, sk);
  }
  PAPI_hl_region_end("kex_uake_sharedB");

  PAPI_hl_region_begin("kex_uake_sharedA");
  for(i=0;i<NTESTS;i++) {
    kex_uake_sharedA(kexkey, kexsendb, key, sk);
  }
  PAPI_hl_region_end("kex_uake_sharedA");

  PAPI_hl_region_begin("kex_ake_initA");
  for(i=0;i<NTESTS;i++) {
    kex_ake_initA(kexsenda, key, sk, pk);
  }
  PAPI_hl_region_end("kex_ake_initA");

  PAPI_hl_region_begin("kex_ake_sharedB");
  for(i=0;i<NTESTS;i++) {
    kex_ake_sharedB(kexsendb, kexkey, kexsenda, sk, pk);
  }
  PAPI_hl_region_end("kex_ake_sharedB");

  PAPI_hl_region_begin("kex_ake_sharedA");
  for(i=0;i<NTESTS;i++) {
    kex_ake_sharedA(kexkey, kexsendb, key, sk, sk);
  }
  PAPI_hl_region_end("kex_ake_sharedA");
  */
  return 0;
}
