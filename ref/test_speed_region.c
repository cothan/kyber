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
#include "papi.h"
#include "ntt.h"

#define NTESTS 100000

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

  PAPI_hl_region_begin("gen_matrix");
  for(i=0;i<NTESTS;i++) {
    gen_matrix(matrix, seed, 0);
  }
  PAPI_hl_region_end("gen_matrix");

  PAPI_hl_region_begin("poly_getnoise_eta1");
  for(i=0;i<NTESTS;i++) {
    poly_getnoise_eta1(&ap, seed, 1);
  }
  PAPI_hl_region_end("poly_getnoise_eta1");

  PAPI_hl_region_begin("poly_getnoise_eta2");
  for(i=0;i<NTESTS;i++) {
    poly_getnoise_eta2(&ap, seed, 0);
  }
  PAPI_hl_region_end("poly_getnoise_eta2");

  PAPI_hl_region_begin("poly_tomsg");
  for(i=0;i<NTESTS;i++) {
    poly_tomsg(msg, &ap);
  }
  PAPI_hl_region_end("poly_tomsg");

  PAPI_hl_region_begin("poly_frommsg");
  for(i=0;i<NTESTS;i++) {
    poly_frommsg(&ap, msg);
  }
  PAPI_hl_region_end("poly_frommsg");


  PAPI_hl_region_begin("ref_ntt");
  for(i=0;i<NTESTS;i++) {
    ntt(ap.coeffs);
  }
  PAPI_hl_region_end("ref_ntt");

  PAPI_hl_region_begin("ref_invntt");
  for(i=0;i<NTESTS;i++) {
    invntt(ap.coeffs);
  }
  PAPI_hl_region_end("ref_invntt");

  PAPI_hl_region_begin("crypto_kem_keypair");
  for(i=0;i<NTESTS;i++) {
    crypto_kem_keypair(pk, sk);
  }
  PAPI_hl_region_end("crypto_kem_keypair");

  PAPI_hl_region_begin("crypto_kem_enc");
  for(i=0;i<NTESTS;i++) {
    crypto_kem_enc(ct, key, pk);
  }
  PAPI_hl_region_end("crypto_kem_enc");

  PAPI_hl_region_begin("crypto_kem_dec");
  for(i=0;i<NTESTS;i++) {
    crypto_kem_dec(key, ct, sk);
  }
  PAPI_hl_region_end("crypto_kem_dec");

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
