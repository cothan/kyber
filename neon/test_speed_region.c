#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "kex.h"
#include "kem.h"
#include "params.h"
#include "indcpa.h"
#include "poly.h"
#include "polyvec.h"
#include "neon_ntt.h"
#include "cpucycles.h"

#define NTESTS 1000000

#define BEGIN(funcname) PAPI_hl_region_begin(funcname);
#define END(funcname) PAPI_hl_region_end(funcname);

uint8_t seed[KYBER_SYMBYTES] = {0};

static void VectorVectorMul(poly *mp, polyvec *b, polyvec *skpv)
{
  neon_polyvec_ntt(b);
  neon_polyvec_acc_montgomery(mp, skpv, b, 0);
  neon_invntt(mp->coeffs);
}

static void MatrixVectorMul(polyvec at[KYBER_K], polyvec *sp, polyvec *b)
{
  neon_polyvec_ntt(sp);
  // matrix-vector multiplication
  for (int i = 0; i < KYBER_K; i++)
    neon_polyvec_acc_montgomery(&b->vec[i], &at[i], sp, 0);

  neon_polyvec_invntt_to_mont(b);
}

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
  polyvec sp, b;
  poly ap, bp;
  polyvec sp, b;

  long long start, stop;
  long long ns;

  START("gen_matrix");
  for (i = 0; i < NTESTS; i++)
  {
    gen_matrix(matrix, seed, 0);
  }
  END("gen_matrix");

  START("neon_poly_getnoise_eta1_2x");
  for (i = 0; i < NTESTS; i++)
  {
    neon_poly_getnoise_eta1_2x(&ap, &bp, seed, 0, 1);
  }
  END("neon_poly_getnoise_eta1_2x");

  START("neon_poly_getnoise_eta2");
  for (i = 0; i < NTESTS; i++)
  {
    neon_poly_getnoise_eta2(&ap, seed, 0);
  }
  END("neon_poly_getnoise_eta2");

  START("poly_tomsg");
  for (i = 0; i < NTESTS; i++)
  {
    poly_tomsg(msg, &ap);
  }
  END("poly_tomsg");

  START("poly_frommsg");
  for (i = 0; i < NTESTS; i++)
  {
    poly_frommsg(&ap, msg);
  }
  END("poly_frommsg");

  START("neon_ntt");
  for (i = 0; i < NTESTS; i++)
  {
    neon_ntt(ap.coeffs);
  }
  END("neon_ntt");

  START("neon_invntt");
  for (i = 0; i < NTESTS; i++)
  {
    neon_invntt(ap.coeffs);
  }
  END("neon_invntt");

  START("crypto_kem_keypair");
  for (i = 0; i < NTESTS; i++)
  {
    crypto_kem_keypair(pk, sk);
  }
  END("crypto_kem_keypair");

  START("crypto_kem_enc");
  for (i = 0; i < NTESTS; i++)
  {
    crypto_kem_enc(ct, key, pk);
  }
  END("crypto_kem_enc");

  START("crypto_kem_dec");
  for (i = 0; i < NTESTS; i++)
  {
    crypto_kem_dec(key, ct, sk);
  }
  END("crypto_kem_dec");

  START("VectorVectorMul");
  for (i = 0; i < NTESTS; i++)
  {
    VectorVectorMul(&ap, &sp, &b);
  }
  END("VectorVectorMul");

  START("MatrixVectorMul");
  for (i = 0; i < NTESTS; i++)
  {
    MatrixVectorMul(matrix, &sp, &b);
  }
  END("MatrixVectorMul");

  /*
  START("kex_uake_initA");
  for(i=0;i<NTESTS;i++) {
    kex_uake_initA(kexsenda, key, sk, pk);
  }
  END("kex_uake_initA");
  */

  START("VectorVectorMul");
  for (i = 0; i < NTESTS; i++)
  {
    VectorVectorMul(&ap, &sp, &b);
  }
  END("VectorVectorMul");

  START("MatrixVectorMul");
  for (i = 0; i < NTESTS; i++)
  {
    MatrixVectorMul(matrix, &sp, &b);
  }
  END("MatrixVectorMul");

  return 0;
}
