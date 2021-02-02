#include <papi.h>
#include <stdio.h>
#include <arm_neon.h>
#include "polyvec.h"
#include "params.h"
#include "ntt.h"
#include "reduce.h"

// clang ntt.c reduce.c neon_ntt.c speed_ntt.c -o neon_ntt -O3 -g3 -Wall -Werror -Wextra -Wpedantic -lpapi
// gcc   ntt.c reduce.c neon_ntt.c speed_ntt.c -o neon_ntt -O3 -g3 -Wall -Werror -Wextra -Wpedantic -lpapi

static const int16_t ref_zetas[128] = {
    2285, 2571, 2970, 1812, 1493, 1422, 287, 202,
    3158, 622, 1577, 182, 962, 2127, 1855, 1468,
    573, 2004, 264, 383, 2500, 1458, 1727, 3199,
    2648, 1017, 732, 608, 1787, 411, 3124, 1758,
    1223, 652, 2777, 1015, 2036, 1491, 3047, 1785,
    516, 3321, 3009, 2663, 1711, 2167, 126, 1469,
    2476, 3239, 3058, 830, 107, 1908, 3082, 2378,
    2931, 961, 1821, 2604, 448, 2264, 677, 2054,
    2226, 430, 555, 843, 2078, 871, 1550, 105,
    422, 587, 177, 3094, 3038, 2869, 1574, 1653,
    3083, 778, 1159, 3182, 2552, 1483, 2727, 1119,
    1739, 644, 2457, 349, 418, 329, 3173, 3254,
    817, 1097, 603, 610, 1322, 2044, 1864, 384,
    2114, 3193, 1218, 1994, 2455, 220, 2142, 1670,
    2144, 1799, 2051, 794, 1819, 2475, 2459, 478,
    3221, 3021, 996, 991, 958, 1869, 1522, 1628};

static const int16_t ref_zetas_inv[128] = {
    1701, 1807, 1460, 2371, 2338, 2333, 308, 108, 2851, 870, 854, 1510, 2535,
    1278, 1530, 1185, 1659, 1187, 3109, 874, 1335, 2111, 136, 1215, 2945, 1465,
    1285, 2007, 2719, 2726, 2232, 2512, 75, 156, 3000, 2911, 2980, 872, 2685,
    1590, 2210, 602, 1846, 777, 147, 2170, 2551, 246, 1676, 1755, 460, 291, 235,
    3152, 2742, 2907, 3224, 1779, 2458, 1251, 2486, 2774, 2899, 1103, 1275, 2652,
    1065, 2881, 725, 1508, 2368, 398, 951, 247, 1421, 3222, 2499, 271, 90, 853,
    1860, 3203, 1162, 1618, 666, 320, 8, 2813, 1544, 282, 1838, 1293, 2314, 552,
    2677, 2106, 1571, 205, 2918, 1542, 2721, 2597, 2312, 681, 130, 1602, 1871,
    829, 2946, 3065, 1325, 2756, 1861, 1474, 1202, 2367, 3147, 1752, 2707, 171,
    3127, 3042, 1907, 1836, 1517, 359, 758, 1441};

static int compare(int16_t *a, int16_t *b, int length, const char *string)
{
  int i, j, count = 0;
  int16_t aa, bb;
  for (i = 0; i < length; i += 8)
  {
    for (j = i; j < i + 8; j++)
    {
      aa = a[j]; // % KYBER_Q;
      bb = b[j]; // % KYBER_Q;
      if (aa != bb)
      {
        if ((aa + KYBER_Q == bb) || (aa - KYBER_Q == bb))
        {
          printf("%d: %d != %d: %d != %d ", j, a[j], b[j], aa, bb);
          printf(": OK\n");
        }
        else
        {
          printf("%d: %d != %d: %d != %d: Error\n", j, a[j], b[j], aa, bb);
          count++;
        }
      }
      if (count > 16)
      {
        printf("%s Incorrect!!\n", string);
        return 1;
      }
    }
  }
  if (count)
  {
    printf("%s Incorrect!!\n", string);
    return 1;
  }

  return 0;
}

static int16_t fqmul1(int16_t a, int16_t b)
{
  return montgomery_reduce((int32_t)a * b);
}

static void ntt(int16_t r[256])
{
  unsigned int len, start, j, k;
  int16_t t, zeta;

  k = 1;
  for (len = 128; len >= 2; len >>= 1)
  {
    for (start = 0; start < 256; start = j + len)
    {
      zeta = ref_zetas[k++];
      for (j = start; j < start + len; ++j)
      {
        t = fqmul1(zeta, r[j + len]);
        r[j + len] = r[j] - t;
        r[j] = r[j] + t;
      }
    }
  }
}

static void invntt(int16_t r[256])
{
  unsigned int start, len, j, k;
  int16_t t, zeta;

  k = 0;
  for (len = 2; len <= 128; len <<= 1)
  {
    for (start = 0; start < 256; start = j + len)
    {
      zeta = ref_zetas_inv[k++];
      for (j = start; j < start + len; ++j)
      {
        t = r[j];
        r[j] = barrett_reduce(t + r[j + len]);
        r[j + len] = t - r[j + len];
        r[j + len] = fqmul1(zeta, r[j + len]);
      }
    }
  }

  for (j = 0; j < 256; ++j)
    r[j] = fqmul1(r[j], ref_zetas_inv[127]);
}

static
void poly_reduce(poly *r)
{
  unsigned int i;
  for(i=0;i<KYBER_N;i++)
    r->coeffs[i] = barrett_reduce(r->coeffs[i]);
}


static
void poly_ntt(poly *r)
{
  ntt(r->coeffs);
  poly_reduce(r);
}

static
void poly_neon_ntt(poly *r)
{
  neon_ntt(r->coeffs);
  // poly_reduce(r);
}



#include <sys/random.h>
#include <string.h>

#define TEST1 0
#define TEST2 10000

int main(void)
{
  int16_t r_gold[256], r1[256], r2[256];
  int retval;

  getrandom(r_gold, sizeof(r_gold), 0);
  for (int i = 0; i < 256; i++)
  {
    r_gold[i] %= KYBER_Q;
  }
  memcpy(r1, r_gold, sizeof(r_gold));
  memcpy(r2, r_gold, sizeof(r_gold));

  // Test NTT
  retval = PAPI_hl_region_begin("c_ntt");
  for (int j = 0; j < TEST1; j++)
  {
    ntt(r_gold);
  }
  retval = PAPI_hl_region_end("c_ntt");

  retval = PAPI_hl_region_begin("merged_neon_ntt");
  for (int j = 0; j < TEST1; j++)
  {
    neon_ntt(r1);
  }
  retval = PAPI_hl_region_end("merged_neon_ntt");

  // Test INTT
  retval = PAPI_hl_region_begin("c_invntt");
  for (int j = 0; j < TEST1; j++)
  {
    invntt(r_gold);
  }
  retval = PAPI_hl_region_end("c_invntt");

  retval = PAPI_hl_region_begin("merged_neon_invntt");
  for (int j = 0; j < TEST1; j++)
  {
    neon_invntt(r1);
  }
  retval = PAPI_hl_region_end("merged_neon_invntt");

  int comp = 0;
  comp = compare(r_gold, r1, 256, "r_gold vs neon_invntt");
  if (comp)
    return 1;

  polyvec rvec_gold;
  polyvec rvec_test;

  for (int k = 0; k < TEST2; k++)
  {
    for (int j = 0; j < KYBER_K; j++)
    {
      getrandom(rvec_gold.vec[j].coeffs, 256, 0);
      for (int i = 0; i < 256; i++)
      {
        rvec_gold.vec[j].coeffs[i] %= KYBER_Q;
      }
      memcpy(rvec_test.vec[j].coeffs, rvec_gold.vec[j].coeffs, sizeof(r_gold));

      poly_ntt(&rvec_gold.vec[j]);
      poly_neon_ntt(&rvec_test.vec[j]);

      comp |= compare(&rvec_gold.vec[j].coeffs, &rvec_test.vec[j].coeffs, 256, "rvec_gold vs rvec_test");

      if (comp)
      {
        printf("ERROR poly_ntt\n");
        return 1;
      }
    }
  }

  return 0;
