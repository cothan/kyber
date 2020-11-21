#include <arm_neon.h>
#include <stdint.h>
#include "params.h"
#include "cbd.h"

#define vload(c, ptr) c = vld1q_u32(ptr);
#define vload3(c, ptr) c = vld1q_u8_x3(ptr);
#define vstore(ptr, c) vst1q_s16_x4(ptr, c);

// c = a & b
#define vand(c, a, b) c = vandq_u32(a, b);

// c = a & b
#define vand8(c, a, b) c = vandq_u8(a, b);

// c = a >> n
#define vsr(c, a, n) c = vshrq_n_u32(a, n);

// c = a >> n
#define vsr8(c, a, n) c = vshrq_n_u8(a, n);

// c = a >> n
#define vsri8(c, a, b, n) c = vsriq_n_u8(a, b, n);

// c = a >> n
#define vsl8(c, a, n) c = vshlq_n_u8(a, n);

// c = a >> n
#define vsr16(c, a, n) c = vshrq_n_u16(a, n);

// c = a + b
#define vadd(c, a, b) c = vaddq_u32(a, b);

// c = a + b
#define vadd16(c, a, b) c = vaddq_u16(a, b);

// c = a - b
#define vsub(c, a, b) c = vsubq_s32(a, b);

/*
 * Improved in-place tranpose, minimal spill to memory.
 * Input: Memory int16_t M[8*8]
Before Transpose
  0   1   2   3   4   5   6   7 
  0   1   2   3   4   5   6   7 
  0   1   2   3   4   5   6   7 
  0   1   2   3   4   5   6   7 
  0   1   2   3   4   5   6   7 
  0   1   2   3   4   5   6   7 
  0   1   2   3   4   5   6   7 
  0   1   2   3   4   5   6   7 
After Transpose
  0   0   0   0   0   0   0   0 
  1   1   1   1   1   1   1   1 
  2   2   2   2   2   2   2   2 
  3   3   3   3   3   3   3   3 
  4   4   4   4   4   4   4   4 
  5   5   5   5   5   5   5   5 
  6   6   6   6   6   6   6   6 
  7   7   7   7   7   7   7   7 
-----------
Use 18 SIMD registers 
 */
#define transpose8x8(x, y)                                                         \
    uint16x8_t y8, y9, y12, y13, y16, y17, y18, y19, y20, y21, y24, y25, y26, y27; \
    y16 = vtrn1q_u16((uint16x8_t)y[0], (uint16x8_t)y[1]);                          \
    y17 = vtrn2q_u16((uint16x8_t)y[0], (uint16x8_t)y[1]);                          \
    y18 = vtrn1q_u16((uint16x8_t)y[2], (uint16x8_t)y[3]);                          \
    y19 = vtrn2q_u16((uint16x8_t)y[2], (uint16x8_t)y[3]);                          \
    y24 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y16, (uint32x4_t)y17);                \
    y25 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y16, (uint32x4_t)y17);                \
    y26 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y18, (uint32x4_t)y19);                \
    y27 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y18, (uint32x4_t)y19);                \
    y8 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y24, (uint32x4_t)y26);                 \
    y9 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y25, (uint32x4_t)y27);                 \
    y16 = vtrn1q_u16((uint16x8_t)y[4], (uint16x8_t)y[5]);                          \
    y17 = vtrn2q_u16((uint16x8_t)y[4], (uint16x8_t)y[5]);                          \
    y18 = vtrn1q_u16((uint16x8_t)y[6], (uint16x8_t)y[7]);                          \
    y19 = vtrn2q_u16((uint16x8_t)y[6], (uint16x8_t)y[7]);                          \
    y24 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y16, (uint32x4_t)y17);                \
    y25 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y16, (uint32x4_t)y17);                \
    y26 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y18, (uint32x4_t)y19);                \
    y27 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y18, (uint32x4_t)y19);                \
    y12 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y24, (uint32x4_t)y26);                \
    y13 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y25, (uint32x4_t)y27);                \
                                                                                   \
    y16 = (uint16x8_t)vtrn1q_u64((uint64x2_t)y8, (uint64x2_t)y12);                 \
    y17 = (uint16x8_t)vtrn2q_u64((uint64x2_t)y8, (uint64x2_t)y12);                 \
    y20 = (uint16x8_t)vtrn1q_u64((uint64x2_t)y9, (uint64x2_t)y13);                 \
    y21 = (uint16x8_t)vtrn2q_u64((uint64x2_t)y9, (uint64x2_t)y13);                 \
    x.val[0] = (int16x8_t)y16;                                                     \
    x.val[1] = (int16x8_t)y20;                                                     \
    x.val[2] = (int16x8_t)y17;                                                     \
    x.val[3] = (int16x8_t)y21;

static 
void neon_cbd2(poly *r, const uint8_t buf[2 * KYBER_N / 4])
{
    uint32x4_t t, d;   // 2
    uint32x4_t a, b;   // 2
    int32x4_t c[8];    // 8
    int16x8x4_t out_c; // 4
    uint32x4_t const_0x55555555, const_0x3;
    const_0x55555555 = vdupq_n_u32(0x55555555);
    const_0x3 = vdupq_n_u32(0x3);

    // Total SIMD register: 20
    int j = 0;
    for (int i = 0; i < KYBER_N / 2; i += 16)
    {
        vload(t, (uint32_t *)&buf[i]);
        // d = t & 0x55555555
        vand(d, t, const_0x55555555);
        // t = (t >> 1) & 0x55555555
        vsr(t, t, 1);
        vand(t, t, const_0x55555555);
        // d += t
        vadd(d, d, t);

        // C1
        // 1st iter
        vsr(b, d, 2);
        vand(a, d, const_0x3);
        vand(b, b, const_0x3);
        vsub(c[0], (int32x4_t)a, (int32x4_t)b);

        // 2nd iter
        vsr(a, d, 4);
        vsr(b, d, 6);
        vand(a, a, const_0x3);
        vand(b, b, const_0x3);
        vsub(c[1], (int32x4_t)a, (int32x4_t)b);

        // 3rd iter
        vsr(a, d, 8);
        vsr(b, d, 10);
        vand(a, a, const_0x3);
        vand(b, b, const_0x3);
        vsub(c[2], (int32x4_t)a, (int32x4_t)b);

        // 3rd iter
        vsr(a, d, 12);
        vsr(b, d, 14);
        vand(a, a, const_0x3);
        vand(b, b, const_0x3);
        vsub(c[3], (int32x4_t)a, (int32x4_t)b);

        // 4th iter
        vsr(a, d, 16);
        vsr(b, d, 18);
        vand(a, a, const_0x3);
        vand(b, b, const_0x3);
        vsub(c[4], (int32x4_t)a, (int32x4_t)b);

        // 5th iter
        vsr(a, d, 20);
        vsr(b, d, 22);
        vand(a, a, const_0x3);
        vand(b, b, const_0x3);
        vsub(c[5], (int32x4_t)a, (int32x4_t)b);

        // 6th iter
        vsr(a, d, 24);
        vsr(b, d, 26);
        vand(a, a, const_0x3);
        vand(b, b, const_0x3);
        vsub(c[6], (int32x4_t)a, (int32x4_t)b);

        // 7th iter
        vsr(a, d, 28);
        vsr(b, d, 30);
        vand(a, a, const_0x3);
        vand(b, b, const_0x3);
        vsub(c[7], (int32x4_t)a, (int32x4_t)b);

        transpose8x8(out_c, c);

        vstore(&r->coeffs[j], out_c);
        j += 32;
    }
}


/*************************************************
* Name:        load24_littleendian
*
* Description: load 3 bytes into a 32-bit integer
*              in little-endian order
*              This function is only needed for Kyber-512
*
* Arguments:   - const uint8_t *x: pointer to input byte array
*
* Returns 32-bit unsigned integer loaded from x (most significant byte is zero)
**************************************************/
#if KYBER_ETA1 == 3
static uint32_t load24_littleendian(const uint8_t x[3])
{
  uint32_t r;
  r  = (uint32_t)x[0];
  r |= (uint32_t)x[1] << 8;
  r |= (uint32_t)x[2] << 16;
  return r;
}
#endif


#if KYBER_ETA1 == 3
static void cbd3(poly *r, const uint8_t buf[3*KYBER_N/4])
{
  unsigned int i,j;
  uint32_t t,d;
  int16_t a,b;

  for(i=0;i<KYBER_N/4;i++) {
    t  = load24_littleendian(buf+3*i);
    d  = t & 0x00249249;
    d += (t>>1) & 0x00249249;
    d += (t>>2) & 0x00249249;

    for(j=0;j<4;j++) {
      a = (d >> (6*j+0)) & 0x7;
      b = (d >> (6*j+3)) & 0x7;
      r->coeffs[4*i+j] = a - b;
    }
  }
}
#endif


void cbd_eta1(poly *r, const uint8_t buf[KYBER_ETA1 * KYBER_N / 4])
{
#if KYBER_ETA1 == 2
    neon_cbd2(r, buf);
#elif KYBER_ETA1 == 3
    cbd3(r, buf);
#else
#error "This implementation requires eta1 in {2,3}"
#endif
}

void cbd_eta2(poly *r, const uint8_t buf[KYBER_ETA1 * KYBER_N / 4])
{
#if KYBER_ETA2 != 2
#error "This implementation requires eta2 = 2"
#else
    neon_cbd2(r, buf);
#endif
}

/* 
#include <string.h>
#include <stdio.h>
#include <sys/random.h>

void compare(poly *r_gold, poly *r, uint8_t *buf, const char *string)
{
    printf("%s\n", string);
    int16_t a, b, count = 0;
    for (int i = 0; i < 32; i++)
    {
        a = r_gold->coeffs[i];
        b = r->coeffs[i];
        if (a != b)
        {
            printf("%2d: %2x | %d != %d\n", i, buf[i / 2], a, b);
            count++;
        }
        if (count == 24)
            break;
    }
}

#define SIZE1 (KYBER_ETA1 * KYBER_N / 4)
#define SIZE2 (KYBER_ETA2 * KYBER_N / 4)

static uint32_t load24_littleendian(const uint8_t x[3])
{
    uint32_t r;
    r = (uint32_t)x[0];
    r |= (uint32_t)x[1] << 8;
    r |= (uint32_t)x[2] << 16;
    return r;
}

uint32_t test(uint8_t *buf)
{
    // uint8_t buf[3];
    // getrandom(buf, sizeof(buf), 0);
    uint32_t t, d;
    t = load24_littleendian(buf);
    d = t & 0x00249249;
    d += (t >> 1) & 0x00249249;
    d += (t >> 2) & 0x00249249;

    uint8_t a, b, c;
    uint8_t cc[3], aa[3], bb[3];
    a = buf[0];
    b = buf[1];
    c = buf[2];

    aa[0] = a & 0x49;
    bb[0] = b & 0x92;
    cc[0] = c & 0x24;

    aa[1] = (a & 0x92) >> 1;
    bb[1] = (b & 0x24) >> 1;
    cc[1] = (c & 0x48) >> 1;

    bb[1] |= (c & 0x1) << 7;

    aa[2] = (a & 0x24) >> 2;
    bb[2] = (b & 0x48) >> 2;
    cc[2] = (c & 0x90) >> 2;

    aa[2] |= (b & 0x1) << 6;
    bb[2] |= (c & 0x2) << 6;

    printf("aa,bb,cc: %3u, %3u, %3u\n", cc[0], bb[0], aa[0]);
    printf("aa,bb,cc: %3u, %3u, %3u\n", cc[1], bb[1], aa[1]);
    printf("aa,bb,cc: %3u, %3u, %3u\n", cc[2], bb[2], aa[2]);

    uint32_t s, sum0, sum1, sum2, t2, t3;
    sum0 = aa[0] + aa[1] + aa[2];
    sum1 = bb[0] + bb[1] + bb[2];
    sum2 = cc[0] + cc[1] + cc[2];
    printf("sum: %3u, %3u, %3u\n", sum2, sum1, sum0);

    s = sum0 & 0xff;
    // get high bit of sum0, plus with sum1
    t2 = (sum0 >> 8) + sum1;
    // get high bit of t2, plus with sum2
    t3 = (t2 >> 8) + sum2;
    printf("car: %2x,", t3);
    printf("%2x,", t2);
    printf("%2x\n", sum0 & 0xff);

    s |= t2 << 8;
    s |= t3 << 16;

    printf("d,s: %06u %06u\n", d, s);
    if (s != d)
        return 1;
    return 0;
}

int main(void)
{
    uint8_t buf1[SIZE1], buf2[SIZE2];
    poly r_gold, r;
    for (int i = 0; i < 1000; i++)
    {
        getrandom(buf1, sizeof(buf1), 0);
        getrandom(buf2, sizeof(buf2), 0);

        cbd_eta1(&r_gold, buf1);
        neon_cbd_eta1(&r, buf1);

        compare(&r_gold, &r, buf1, "CBD_ETA1");

        if (memcmp(r_gold.coeffs, r.coeffs, KYBER_N))
            return 1;

        cbd_eta2(&r_gold, buf2);
        neon_cbd_eta2(&r, buf2);

        compare(&r_gold, &r, buf2, "CBD_ETA2");

        if (memcmp(r_gold.coeffs, r.coeffs, KYBER_N))
            return 1;
    }
    return 0;
}

//gcc cbd.c neon_cbd.c -o neon_cbd -g3 -O0 -Wall -Wextra -Wpedantic
*/
