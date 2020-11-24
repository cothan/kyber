#include <arm_neon.h>
#include <stdint.h>
#include "params.h"
#include "cbd.h"

#define vload2(c, ptr) c = vld2q_u8(ptr);

#define vstore4(ptr, c) vst4q_s16(ptr, c);

// c = a >> n
#define vsr8(c, a, n) c = vshrq_n_u8(a, n);

// c = a & b
#define vand8(c, a, b) c = vandq_u8(a, b);

// c = a + b
#define vadd8(c, a, b) c = vaddq_u8(a, b);

// long c = a - b
#define vsubll8(c, a, b) c = (int16x8_t) vsubl_u8(a, b);

// long c = a - b
#define vsublh8(c, a, b) c = (int16x8_t) vsubl_high_u8(a, b);
 

static
void neon_cbd2(poly *r, const uint8_t buf[2 * KYBER_N / 4])
{
    uint8x16x2_t t, d;   // 4
    uint8x16x2_t a, b;   // 4
    int16x8x4_t res1, res2; // 4
    
    uint8x16_t const_0x55, const_0x3; // 2
    const_0x55 = vdupq_n_u8(0x55);
    const_0x3  = vdupq_n_u8(0x3);

    // Total SIMD register: 18
    unsigned int j = 0;
    for (unsigned int i = 0; i < KYBER_N / 2; i += 16*2)
    {
        // 0, 2, 4 , 6,... 
        // 1, 3, 5 , 7,... 
        vload2(t, &buf[i]);
        // d = t & 0x55555555
        vand8(d.val[0], t.val[0], const_0x55);
        vand8(d.val[1], t.val[1], const_0x55);
        // t = (t >> 1) & 0x55555555
        vsr8(t.val[0], t.val[0], 1);
        vsr8(t.val[1], t.val[1], 1);
        vand8(t.val[0], t.val[0], const_0x55);
        vand8(t.val[1], t.val[1], const_0x55);
    
        // d += t
        vadd8(d.val[0], d.val[0], t.val[0]);
        vadd8(d.val[1], d.val[1], t.val[1]);
    
        // Get a0, a2
        vand8(a.val[0], d.val[0], const_0x3);
        vand8(a.val[1], d.val[1], const_0x3);
    
        // Get b0, b2
        vsr8(b.val[0], d.val[0], 2);
        vsr8(b.val[1], d.val[1], 2);
        
        vand8(b.val[0], b.val[0], const_0x3);
        vand8(b.val[1], b.val[1], const_0x3);
        
        // 0 2  4 6  -- 8 10 12 14 | 0 4  8 12 -- 16 20 24 28 | 0
        // 1 3  5 7  -- 9 11 13 15 | 2 6 10 14 -- 18 22 26 30 | 2
        vsubll8(res1.val[0], vget_low_u8(a.val[0]), vget_low_u8(b.val[0]));
        vsubll8(res1.val[2], vget_low_u8(a.val[1]), vget_low_u8(b.val[1]));
        
        // 16 18 20 22  -- 24 26 28 30 | 32 36 40 44 -- 48 52 56 60 | 0
        // 17 19 21 23  -- 25 27 29 31 | 34 38 42 46 -- 50 54 58 62 | 2
        vsublh8(res2.val[0], a.val[0], b.val[0]);
        vsublh8(res2.val[2], a.val[1], b.val[1]);

        
        // Get a1, a3
        vsr8(a.val[0], d.val[0], 4);
        vsr8(a.val[1], d.val[1], 4);
    
        vand8(a.val[0], a.val[0], const_0x3);
        vand8(a.val[1], a.val[1], const_0x3);
        
        // Get b1, b3
        vsr8(b.val[0], d.val[0], 6);
        vsr8(b.val[1], d.val[1], 6);
        
        // 0 2  4 6  -- 8 10 12 14 | 1 5  9 13 -- 17 21 25 29 | 1
        // 1 3  5 7  -- 9 11 13 15 | 3 7 11 15 -- 19 23 27 31 | 3
        vsubll8(res1.val[1], vget_low_u8(a.val[0]), vget_low_u8(b.val[0]));
        vsubll8(res1.val[3], vget_low_u8(a.val[1]), vget_low_u8(b.val[1]));
        
        // 16 18 20 22  -- 24 26 28 30 | 33 37 41 45 -- 49 53 57 61 | 1
        // 17 19 21 23  -- 25 27 29 31 | 35 39 43 47 -- 51 55 59 63 | 3
        vsublh8(res2.val[1], a.val[0], b.val[0]);
        vsublh8(res2.val[3], a.val[1], b.val[1]);
        

        // 0 2  4 6  -- 8 10 12 14 | 0 4  8 12 -- 16 20 24 28 | 1-0
        // 0 2  4 6  -- 8 10 12 14 | 1 5  9 13 -- 17 21 25 29 | 1-1
        // 1 3  5 7  -- 9 11 13 15 | 2 6 10 14 -- 18 22 26 30 | 1-2
        // 1 3  5 7  -- 9 11 13 15 | 3 7 11 15 -- 19 23 27 31 | 1-3
        // 16 18 20 22  -- 24 26 28 30 | 32 36 40 44 -- 48 52 56 60 | 2-0
        // 16 18 20 22  -- 24 26 28 30 | 33 37 41 45 -- 49 53 57 61 | 2-1
        // 17 19 21 23  -- 25 27 29 31 | 34 38 42 46 -- 50 54 58 62 | 2-2
        // 17 19 21 23  -- 25 27 29 31 | 35 39 43 47 -- 51 55 59 63 | 2-3
        vstore4(&r->coeffs[j], res1);
        j+=32;
        vstore4(&r->coeffs[j], res2);
        j+=32;
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
    for (int i = 0; i < 64; i++)
    {
        a = r_gold->coeffs[i];
        b = r->coeffs[i];
        if (a != b)
        {
            printf("%2d: %2x | %d != %d\n", i, buf[i / 2], a, b);
            count++;
        }
        if (count == 64)
            break;
    }
}

#define SIZE1 (KYBER_ETA1 * KYBER_N / 4)
#define SIZE2 (KYBER_ETA2 * KYBER_N / 4)

void genrand(uint8_t *buf, unsigned int buflen)
{
    for (uint8_t i = 0; i < buflen; i++)
    {
        buf[i] = i;
    }
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

void test_neon_new(uint16_t *c, uint8_t *buf, unsigned int buflen)
{
    uint8_t a[8], b[8], t[4], d[4];
    for (unsigned int i = 0; i < buflen; i+=4)
    {
        t[0] = buf[i];
        t[1] = buf[i+ 1];
        t[2] = buf[i+ 2];
        t[3] = buf[i+ 3];

        d[0] = t[0] & 0x55;
        d[1] = t[1] & 0x55;
        d[2] = t[2] & 0x55;
        d[3] = t[3] & 0x55;

        d[0] += (t[0]>>1) & 0x55;
        d[1] += (t[1]>>1) & 0x55;
        d[2] += (t[2]>>1) & 0x55;
        d[3] += (t[3]>>1) & 0x55;

        a[0] = (d[0] >> 0) & 0x3;
        b[0] = (d[0] >> 2) & 0x3;
        a[2] = (d[1] >> 0) & 0x3;
        b[2] = (d[1] >> 2) & 0x3;
        a[4] = (d[2] >> 0) & 0x3;
        b[4] = (d[2] >> 2) & 0x3;
        a[6] = (d[3] >> 0) & 0x3;
        b[6] = (d[3] >> 2) & 0x3;

        c[0] = a[0] - b[0];
        c[2] = a[2] - b[2];
        c[4] = a[4] - b[4];
        c[6] = a[6] - b[6];

        a[1] = (d[0] >> 4) & 0x3;
        b[1] = (d[0] >> 6) & 0x3;
        a[3] = (d[1] >> 4) & 0x3;
        b[3] = (d[1] >> 6) & 0x3;
        a[5] = (d[2] >> 4) & 0x3;
        b[5] = (d[2] >> 6) & 0x3;
        a[7] = (d[3] >> 4) & 0x3;
        b[7] = (d[3] >> 6) & 0x3;

        c[1] = a[1] - b[1];
        c[3] = a[3] - b[3];
        c[5] = a[5] - b[5];
        c[7] = a[7] - b[7];
        c+=8;
    }
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

        test_neon_new(&r_gold.coeffs, buf2, SIZE2);
        neon_cbd2(&r, buf2);

        compare(&r_gold, &r, buf2, "CBD_ETA2");

        if (memcmp(r_gold.coeffs, r.coeffs, KYBER_N))
            return 1;
    }
    return 0;
}

//gcc cbd.c -o neon_cbd -g3 -O0 -Wall -Wextra -Wpedantic
*/