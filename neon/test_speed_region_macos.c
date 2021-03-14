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

// For Apple M1: https://gist.github.com/dougallj/5bafb113492047c865c0c8cfbc930155#file-m1_robsize-c-L390
#include <assert.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#include <libkern/OSCacheControl.h>


#define KPERF_LIST                                                             \
  /*  ret, name, params */                                                     \
  F(int, kpc_get_counting, void)                                               \
  F(int, kpc_force_all_ctrs_set, int)                                          \
  F(int, kpc_set_counting, uint32_t)                                           \
  F(int, kpc_set_thread_counting, uint32_t)                                    \
  F(int, kpc_set_config, uint32_t, void *)                                     \
  F(int, kpc_get_config, uint32_t, void *)                                     \
  F(int, kpc_set_period, uint32_t, void *)                                     \
  F(int, kpc_get_period, uint32_t, void *)                                     \
  F(uint32_t, kpc_get_counter_count, uint32_t)                                 \
  F(uint32_t, kpc_get_config_count, uint32_t)                                  \
  F(int, kperf_sample_get, int *)                                              \
  F(int, kpc_get_thread_counters, int, unsigned int, void *)

#define F(ret, name, ...)                                                      \
  typedef ret name##proc(__VA_ARGS__);                                         \
  static name##proc *name;
KPERF_LIST
#undef F

#define CFGWORD_EL0A32EN_MASK (0x10000)
#define CFGWORD_EL0A64EN_MASK (0x20000)
#define CFGWORD_EL1EN_MASK (0x40000)
#define CFGWORD_EL3EN_MASK (0x80000)
#define CFGWORD_ALLMODES_MASK (0xf0000)

#define CPMU_NONE 0
#define CPMU_CORE_CYCLE 0x02
#define CPMU_INST_A64 0x8c
#define CPMU_INST_BRANCH 0x8d
#define CPMU_SYNC_DC_LOAD_MISS 0xbf
#define CPMU_SYNC_DC_STORE_MISS 0xc0
#define CPMU_SYNC_DTLB_MISS 0xc1
#define CPMU_SYNC_ST_HIT_YNGR_LD 0xc4
#define CPMU_SYNC_BR_ANY_MISP 0xcb
#define CPMU_FED_IC_MISS_DEM 0xd3
#define CPMU_FED_ITLB_MISS 0xd4

#define KPC_CLASS_FIXED (0)
#define KPC_CLASS_CONFIGURABLE (1)
#define KPC_CLASS_POWER (2)
#define KPC_CLASS_RAWPMU (3)
#define KPC_CLASS_FIXED_MASK (1u << KPC_CLASS_FIXED)
#define KPC_CLASS_CONFIGURABLE_MASK (1u << KPC_CLASS_CONFIGURABLE)
#define KPC_CLASS_POWER_MASK (1u << KPC_CLASS_POWER)
#define KPC_CLASS_RAWPMU_MASK (1u << KPC_CLASS_RAWPMU)

#define COUNTERS_COUNT 10
#define CONFIG_COUNT 8
#define KPC_MASK (KPC_CLASS_CONFIGURABLE_MASK | KPC_CLASS_FIXED_MASK)
uint64_t g_counters[COUNTERS_COUNT];
uint64_t g_config[COUNTERS_COUNT];

static void configure_rdtsc() {
  if (kpc_set_config(KPC_MASK, g_config)) {
    printf("kpc_set_config failed\n");
    return;
  }

  if (kpc_force_all_ctrs_set(1)) {
    printf("kpc_force_all_ctrs_set failed\n");
    return;
  }

  if (kpc_set_counting(KPC_MASK)) {
    printf("kpc_set_counting failed\n");
    return;
  }

  if (kpc_set_thread_counting(KPC_MASK)) {
    printf("kpc_set_thread_counting failed\n");
    return;
  }
}

static void init_rdtsc() {
  void *kperf = dlopen(
      "/System/Library/PrivateFrameworks/kperf.framework/Versions/A/kperf",
      RTLD_LAZY);
  if (!kperf) {
    printf("kperf = %p\n", kperf);
    return;
  }
#define F(ret, name, ...)                                                      \
  name = (name##proc *)(dlsym(kperf, #name));                                  \
  if (!name) {                                                                 \
    printf("%s = %p\n", #name, (void *)name);                                  \
    return;                                                                    \
  }
  KPERF_LIST
#undef F

  // TODO: KPC_CLASS_RAWPMU_MASK

  if (kpc_get_counter_count(KPC_MASK) != COUNTERS_COUNT) {
    printf("wrong fixed counters count\n");
    return;
  }

  if (kpc_get_config_count(KPC_MASK) != CONFIG_COUNT) {
    printf("wrong fixed config count\n");
    return;
  }

  // Not all counters can count all things:

  // CPMU_CORE_CYCLE           {0-7}
  // CPMU_FED_IC_MISS_DEM      {0-7}
  // CPMU_FED_ITLB_MISS        {0-7}

  // CPMU_INST_BRANCH          {3, 4, 5}
  // CPMU_SYNC_DC_LOAD_MISS    {3, 4, 5}
  // CPMU_SYNC_DC_STORE_MISS   {3, 4, 5}
  // CPMU_SYNC_DTLB_MISS       {3, 4, 5}
  // CPMU_SYNC_BR_ANY_MISP     {3, 4, 5}
  // CPMU_SYNC_ST_HIT_YNGR_LD  {3, 4, 5}
  // CPMU_INST_A64             {5}

  // using "CFGWORD_ALLMODES_MASK" is much noisier
  g_config[0] = CPMU_CORE_CYCLE | CFGWORD_EL0A64EN_MASK;
  // configs[3] = CPMU_SYNC_DC_LOAD_MISS | CFGWORD_EL0A64EN_MASK;
  // configs[4] = CPMU_SYNC_DTLB_MISS | CFGWORD_EL0A64EN_MASK;
  // configs[5] = CPMU_INST_A64 | CFGWORD_EL0A64EN_MASK;

  configure_rdtsc();
}

static unsigned long long int rdtsc() {
  if (kpc_get_thread_counters(0, COUNTERS_COUNT, g_counters)) {
    printf("kpc_get_thread_counters failed\n");
    return 1;
  }
  return g_counters[2];
}

//  End 


#define NTESTS 1000000

uint8_t seed[KYBER_SYMBYTES] = {0};

#define TIME(s) s = rdtsc();
// Result is clock cycles 
#define  CALC(start, stop) (stop - start) / NTESTS;


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
  // unsigned char kexsenda[KEX_AKE_SENDABYTES] = {0};
  // unsigned char kexsendb[KEX_AKE_SENDBBYTES] = {0};
  // unsigned char kexkey[KEX_SSBYTES] = {0};
  unsigned char msg[KYBER_INDCPA_MSGBYTES] = {0};
  polyvec matrix[KYBER_K];
  poly ap, bp;
  polyvec sp, b;

//   struct timespec start, stop;
  long long ns;
  long long start, stop;


// Init performance counter 
  int test_high_perf_cores = 1;

  if (test_high_perf_cores) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  } else {
    pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
  }
  init_rdtsc();  
  configure_rdtsc();


// 



  TIME(start);
  for(i=0;i<NTESTS;i++) {
    gen_matrix(matrix, seed, 0);
  }
  TIME(stop);
  ns = CALC(start, stop);
  printf("gen_matrix: %lld\n", ns);


  TIME(start);
  for(i=0;i<NTESTS;i++) {
    neon_poly_getnoise_eta1_2x(&ap, &bp, seed, 0, 1);
  }
  TIME(stop);
  ns = CALC(start, stop);
  printf("neon_poly_getnoise_eta1_2x: %lld\n", ns);



  TIME(start);
  for(i=0;i<NTESTS;i++) {
    neon_poly_getnoise_eta2(&ap, seed, 0);
  }
  TIME(stop);
  ns = CALC(start, stop);
  printf("neon_poly_getnoise_eta2: %lld\n", ns);



  TIME(start);
  for(i=0;i<NTESTS;i++) {
    poly_tomsg(msg, &ap);
  }
  TIME(stop);
  ns = CALC(start, stop);
  printf("poly_tomsg: %lld\n", ns);



  TIME(start); 
  for(i=0;i<NTESTS;i++) {
    poly_frommsg(&ap, msg);
  }
  TIME(stop);
  ns = CALC(start, stop);
  printf("poly_frommsg: %lld\n", ns);




  TIME(start);
  for(i=0;i<NTESTS;i++) {
    neon_ntt(ap.coeffs);
  }
  TIME(stop);
  ns = CALC(start, stop);
  printf("neon_ntt: %lld\n", ns);



  TIME(start);
  for(i=0;i<NTESTS;i++) {
    neon_invntt(ap.coeffs);
  }
  TIME(stop);
  ns = CALC(start, stop);
  printf("neon_invntt: %lld\n", ns);



  TIME(start);
  for(i=0;i<NTESTS;i++) {
    crypto_kem_keypair(pk, sk);
  }
  TIME(stop);
  ns = CALC(start, stop);
  printf("crypto_kem_keypair: %lld\n", ns);



  TIME(start); 
  for(i=0;i<NTESTS;i++) {
    crypto_kem_enc(ct, key, pk);
  }
  TIME(stop);
  ns = CALC(start, stop);
  printf("crypto_kem_enc: %lld\n", ns);



  TIME(start); 
  for(i=0;i<NTESTS;i++) {
    crypto_kem_dec(key, ct, sk);
  }
  TIME(stop);
  ns = CALC(start, stop);
  printf("crypto_kem_dec: %lld\n", ns);


  TIME(start);
  for(i=0;i<NTESTS;i++) {
    VectorVectorMul(&ap, &sp, &b);
  }
  TIME(stop);
  ns = CALC(start, stop);
  printf("VectorVectorMul %lld\n", ns);

  
  TIME(start);
  for(i=0;i<NTESTS;i++) {
    MatrixVectorMul(matrix, &sp, &b);
  }
  TIME(stop);
  ns = CALC(start, stop);
  printf("MatrixVectorMul %lld\n", ns);




  return 0;
}
