#include <stdint.h>
#include "params.h"
#include "ntt.h"
#include "reduce.h"

/* Code to generate zetas and zetas_inv used in the number-theoretic transform:#define KYBER_ROOT_OF_UNITY 17

static const uint16_t tree[128] = {
  0, 64, 32, 96, 16, 80, 48, 112, 8, 72, 40, 104, 24, 88, 56, 120,
  4, 68, 36, 100, 20, 84, 52, 116, 12, 76, 44, 108, 28, 92, 60, 124,
  2, 66, 34, 98, 18, 82, 50, 114, 10, 74, 42, 106, 26, 90, 58, 122,
  6, 70, 38, 102, 22, 86, 54, 118, 14, 78, 46, 110, 30, 94, 62, 126,
  1, 65, 33, 97, 17, 81, 49, 113, 9, 73, 41, 105, 25, 89, 57, 121,
  5, 69, 37, 101, 21, 85, 53, 117, 13, 77, 45, 109, 29, 93, 61, 125,
  3, 67, 35, 99, 19, 83, 51, 115, 11, 75, 43, 107, 27, 91, 59, 123,
  7, 71, 39, 103, 23, 87, 55, 119, 15, 79, 47, 111, 31, 95, 63, 127
};

void init_ntt() {
  unsigned int i, j, k;
  int16_t tmp[128];

  tmp[0] = MONT;
  for(i = 1; i < 128; ++i)
    tmp[i] = fqmul(tmp[i-1], KYBER_ROOT_OF_UNITY*MONT % KYBER_Q);

  for(i = 0; i < 128; ++i)
    zetas[i] = tmp[tree[i]];

  k = 0;
  for(i = 64; i >= 1; i >>= 1)
    for(j = i; j < 2*i; ++j)
      zetas_inv[k++] = -tmp[128 - tree[j]];

  zetas_inv[127] = MONT * (MONT * (KYBER_Q - 1) * ((KYBER_Q - 1)/128) % KYBER_Q) % KYBER_Q;
}

*/

// This array is generate specifically for this ASIMD implementation 
const int16_t neon_zetas_inv[280] = {
  1701,1807,1460,2371,2338,2333,308,108,// 1
  2851,870,854,1510,2535,1278,1530,1185,// 1
  1659,1187,3109,874,1335,2111,136,1215,// 1
  2945,1465,1285,2007,2719,2726,2232,2512,// 1

  1275,1275,1275,1275,1065,1065,1065,1065,// 2
  2652,2652,2652,2652,2881,2881,2881,2881,// 2
  725,725,725,725,2368,2368,2368,2368,// 2
  1508,1508,1508,1508,398,398,398,398,// 2

  951,951,951,951,1421,1421,1421,1421,// 2
  247,247,247,247,3222,3222,3222,3222,// 2
  2499,2499,2499,2499,90,90,90,90,// 2
  271,271,271,271,853,853,853,853,// 2

  1571,1571,1571,1571,205,205,205,205,// 3
  2918,2918,2918,2918,1542,1542,1542,1542,// 3
  2721,2721,2721,2721,2597,2597,2597,2597,// 3
  2312,2312,2312,2312,681,681,681,681,// 3

  1861, // 4
  1474, // 4
  1202, // 4
  2367, // 4
  3127, // 5
  3042, // 5
  1517,1517, // 6

  75,156,3000,2911,2980,872,2685,1590,// 1
  2210,602,1846,777,147,2170,2551,246,// 1
  1676,1755,460,291,235,3152,2742,2907,// 1
  3224,1779,2458,1251,2486,2774,2899,1103,// 1

  1860,1860,1860,1860,1162,1162,1162,1162,// 2
  3203,3203,3203,3203,1618,1618,1618,1618,// 2
  666,666,666,666,8,8,8,8,// 2
  320,320,320,320,2813,2813,2813,2813,// 2

  1544,1544,1544,1544,1838,1838,1838,1838,// 2
  282,282,282,282,1293,1293,1293,1293,// 2
  2314,2314,2314,2314,2677,2677,2677,2677,// 2
  552,552,552,552,2106,2106,2106,2106,// 2

  130,130,130,130,1602,1602,1602,1602,// 3
  1871,1871,1871,1871,829,829,829,829,// 3
  2946,2946,2946,2946,3065,3065,3065,3065,// 3
  1325,1325,1325,1325,2756,2756,2756,2756,// 3

  3147, // 4
  1752, // 4
  2707, // 4
  171,  // 4
  1907, // 5
  1836, // 5
  359,359, // 6

  758, // 7
  1441,1441,1441,1441,1441,1441,1441, // last
};

// This array is generate specifically for this ASIMD implementation 
const int16_t neon_zetas[224] = {
  2970, // 6
  1493, // 5
  1422, // 5
  3158, // 4
  622,  // 4
  1577, // 4
  182,  // 4
  573,  // 3
  2004, // 3
  264,  // 3
  383,  // 3
  2500, // 3
  1458, // 3
  1727, // 3
  3199, 2571, // 3, 7
  1223,1223,1223,1223,2777,2777,2777,2777,// 2
  652,652,652,652,1015,1015,1015,1015,// 2
  2036,2036,2036,2036,3047,3047,3047,3047,// 2
  1491,1491,1491,1491,1785,1785,1785,1785,// 2
  516,516,516,516,3009,3009,3009,3009,// 2
  3321,3321,3321,3321,2663,2663,2663,2663,// 2
  1711,1711,1711,1711,126,126,126,126,// 2
  2167,2167,2167,2167,1469,1469,1469,1469,// 2
  2226,430,555,843,2078,871,1550,105,// 1
  422,587,177,3094,3038,2869,1574,1653,// 1
  3083,778,1159,3182,2552,1483,2727,1119,// 1
  1739,644,2457,349,418,329,3173,3254,// 1

  1812, // 6
  287,  // 5
  202,  // 5
  962,  // 4
  2127, // 4
  1855, // 4
  1468, // 4
  2648, // 3
  1017, // 3
  732,  // 3
  608,  // 3
  1787, // 3
  411,  // 3
  3124, // 3
  1758, 1758, // 3

  2476,2476,2476,2476,3058,3058,3058,3058,// 2
  3239,3239,3239,3239,830,830,830,830,// 2
  107,107,107,107,3082,3082,3082,3082,// 2
  1908,1908,1908,1908,2378,2378,2378,2378,// 2
  2931,2931,2931,2931,1821,1821,1821,1821,// 2
  961,961,961,961,2604,2604,2604,2604,// 2
  448,448,448,448,677,677,677,677,// 2
  2264,2264,2264,2264,2054,2054,2054,2054,// 2
  817,1097,603,610,1322,2044,1864,384,// 1
  2114,3193,1218,1994,2455,220,2142,1670,// 1
  2144,1799,2051,794,1819,2475,2459,478,// 1
  3221,3021,996,991,958,1869,1522,1628,// 1
};
