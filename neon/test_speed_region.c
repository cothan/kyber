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
#include <time.h>
#include "print.h"
#include "neon_ntt.h"

#define NTESTS 1000000

uint8_t seed[KYBER_SYMBYTES] = {0};

#define TIME(s) clock_gettime(CLOCK_MONOTONIC_RAW, &s);
// Result is nanosecond per call 
#define  CALC(start, stop) \
  ((double) ((stop.tv_sec - start.tv_sec) * 1000000000 + (stop.tv_nsec - start.tv_nsec))) / NTESTS;


static 
void VectorVectorMul(poly *mp, polyvec *b, polyvec *skpv)
{
  neon_polyvec_ntt(b);
  neon_polyvec_acc_montgomery(mp, skpv, b, 0);
  neon_invntt(mp->coeffs);
}

static 
void MatrixVectorMul(polyvec at[KYBER_K], polyvec *sp, polyvec *b)
{
  neon_polyvec_ntt(sp);
  // matrix-vector multiplication
  for(int i=0;i<KYBER_K;i++)
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
  poly ap, bp;
  polyvec sp, b;

  struct timespec start, stop;
  long ns;


  TIME(start);
  for(i=0;i<NTESTS;i++) {
    gen_matrix(matrix, seed, 0);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("gen_matrix:", ns);


  TIME(start);
  for(i=0;i<NTESTS;i++) {
    neon_poly_getnoise_eta1_2x(&ap, &bp, seed, 0, 1);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("neon_poly_getnoise_eta1_2x:", ns);



  TIME(start);
  for(i=0;i<NTESTS;i++) {
    neon_poly_getnoise_eta2(&ap, seed, 0);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("neon_poly_getnoise_eta2:", ns);



  TIME(start);
  for(i=0;i<NTESTS;i++) {
    poly_tomsg(msg, &ap);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("poly_tomsg:", ns);



  TIME(start); 
  for(i=0;i<NTESTS;i++) {
    poly_frommsg(&ap, msg);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("poly_frommsg:", ns);




  TIME(start);
  for(i=0;i<NTESTS;i++) {
    neon_ntt(ap.coeffs);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("neon_ntt:", ns);



  TIME(start);
  for(i=0;i<NTESTS;i++) {
    neon_invntt(ap.coeffs);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("neon_invntt:", ns);



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


  TIME(start);
  for(i=0;i<NTESTS;i++) {
    VectorVectorMul(&ap, &sp, &b);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("VectorVectorMul", ns);

  
  TIME(start);
  for(i=0;i<NTESTS;i++) {
    MatrixVectorMul(matrix, &sp, &b);
  }
  TIME(stop);
  ns = CALC(start, stop);
  print("MatrixVectorMul", ns);




  return 0;
}
