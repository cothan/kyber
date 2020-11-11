#include <arm_neon.h>
#include <stdint.h>
#include "params.h"
#include "cbd.h"

#define vload(c, ptr) c = vld1q_u32(ptr);
#define vstore(ptr, c) vst1q_s16_x4(ptr, c);

// c = a & b
#define vand(c, a, b) c = vandq_u32(a, b);

// c = a >> n
#define vsr(c, a, n) c = vshrq_n_u32(a, n);

// c = a + b
#define vadd(c, a, b) c = vaddq_u32(a, b);

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

#define transpose4x8(x, y)                                         \
    uint16x8_t y16, y18;                                           \
    int16x8_t y10, y11, y12, y13, y24, y25;                        \
    y16 = vtrn1q_u16((uint16x8_t)y[0], (uint16x8_t)y[1]);          \
    y18 = vtrn1q_u16((uint16x8_t)y[2], (uint16x8_t)y[3]);          \
    y24 = (int16x8_t)vtrn1q_u32((uint32x4_t)y16, (uint32x4_t)y18); \
    y25 = (int16x8_t)vtrn2q_u32((uint32x4_t)y16, (uint32x4_t)y18); \
                                                                   \
    y10 = (int16x8_t)vtrn1q_u64((uint64x2_t)y24, (uint64x2_t)y25); \
    y11 = (int16x8_t)vtrn2q_u64((uint64x2_t)y24, (uint64x2_t)y25); \
    y16 = vtrn1q_u16((uint16x8_t)y[4], (uint16x8_t)y[5]);          \
    y18 = vtrn1q_u16((uint16x8_t)y[6], (uint16x8_t)y[7]);          \
    y24 = (int16x8_t)vtrn1q_u32((uint32x4_t)y16, (uint32x4_t)y18); \
    y25 = (int16x8_t)vtrn2q_u32((uint32x4_t)y16, (uint32x4_t)y18); \
                                                                   \
    y12 = (int16x8_t)vtrn1q_u64((uint64x2_t)y24, (uint64x2_t)y25); \
    y13 = (int16x8_t)vtrn2q_u64((uint64x2_t)y24, (uint64x2_t)y25); \
    x.val[0] = y10;                                                \
    x.val[1] = y11;                                                \
    x.val[2] = y12;                                                \
    x.val[3] = y13;

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

        // c[0]: x | 24 | x | 16 -- x | 8  | x | 0
        // c[1]: x | 25 | x | 17 -- x | 9  | x | 1
        // c[2]: x | 26 | x | 18 -- x | 10 | x | 2
        // c[3]: x | 27 | x | 19 -- x | 11 | x | 3
        // c[4]: x | 28 | x | 20 -- x | 12 | x | 4
        // c[5]: x | 29 | x | 21 -- x | 13 | x | 5
        // c[6]: x | 30 | x | 22 -- x | 14 | x | 6
        // c[7]: x | 31 | x | 23 -- x | 15 | x | 7
        // Transpose
        // c[0]: x | x | x | x -- x | x | x | x
        // c[1]: 24 ... 31
        // c[2]: x | x | x | x -- x | x | x | x
        // c[3]: 16 ... 23
        // c[4]: x | x | x | x -- x | x | x | x
        // c[5]: 8  ... 15
        // c[6]: x | x | x | x -- x | x | x | x
        // c[7]: 0  ... 7
        transpose8x8(out_c, c);

        vstore(&r->coeffs[j], out_c);
        j += 32;
    }
}

static 
void neon_cbd3(poly *r, const uint8_t buf[3 * KYBER_N / 4])
{
    uint32x4_t t1, t2, tt1, tt2, d1, d2; // 6
    uint32x4_t a, b;                     // 2
    int32x4_t c[8];                      // 8
    int16x8x4_t c_out;                   // 4
    uint32x4_t const_0x00249249, const_0x7;
    const_0x00249249 = vdupq_n_u32(0x00249249);
    const_0x7 = vdupq_n_u32(0x7);

    // Total SIMD register: 20
    uint8_t local_buf[KYBER_N];
    for (int i = 0, j = 0; i < 3 * KYBER_N / 4; i += 6)
    {
        local_buf[j + 0] = buf[i + 0];
        local_buf[j + 1] = buf[i + 1];
        local_buf[j + 2] = buf[i + 2];
        local_buf[j + 3] = 0;
        local_buf[j + 4] = buf[i + 3];
        local_buf[j + 5] = buf[i + 4];
        local_buf[j + 6] = buf[i + 5];
        local_buf[j + 7] = 0;
        j += 8;
    }

    for (int i = 0; i < KYBER_N; i += 32)
    {
        vload(t1, (uint32_t *)&local_buf[i]);
        vload(t2, (uint32_t *)&local_buf[i + 16]);

        // d = t & 0x00249249
        vand(d1, t1, const_0x00249249);
        vand(d2, t2, const_0x00249249);
        // t = (t >> 1) & 0x00249249
        vsr(tt1, t1, 1);
        vsr(tt2, t2, 1);
        vand(tt1, tt1, const_0x00249249);
        vand(tt2, tt2, const_0x00249249);
        // d += t
        vadd(d1, d1, tt1);
        vadd(d2, d2, tt2);
        // t = (t >> 2) & 0x00249249
        vsr(tt1, t1, 2);
        vsr(tt2, t2, 2);
        vand(tt1, tt1, const_0x00249249);
        vand(tt2, tt2, const_0x00249249);
        // d += t
        vadd(d1, d1, tt1);
        vadd(d2, d2, tt2);

        // C1
        // 1st iter
        vsr(b, d1, 3);
        vand(a, d1, const_0x7);
        vand(b, b, const_0x7);
        vsub(c[0], (int32x4_t)a, (int32x4_t)b);

        // 2nd iter
        vsr(a, d1, 6);
        vsr(b, d1, 9);
        vand(a, a, const_0x7);
        vand(b, b, const_0x7);
        vsub(c[1], (int32x4_t)a, (int32x4_t)b);

        // 3rd iter
        vsr(a, d1, 12);
        vsr(b, d1, 15);
        vand(a, a, const_0x7);
        vand(b, b, const_0x7);
        vsub(c[2], (int32x4_t)a, (int32x4_t)b);

        // 3rd iter
        vsr(a, d1, 18);
        vsr(b, d1, 21);
        vand(a, a, const_0x7);
        vand(b, b, const_0x7);
        vsub(c[3], (int32x4_t)a, (int32x4_t)b);

        // 4th iter
        vsr(b, d2, 3);
        vand(a, d2, const_0x7);
        vand(b, b, const_0x7);
        vsub(c[4], (int32x4_t)a, (int32x4_t)b);

        // 5th iter
        vsr(a, d2, 6);
        vsr(b, d2, 9);
        vand(a, a, const_0x7);
        vand(b, b, const_0x7);
        vsub(c[5], (int32x4_t)a, (int32x4_t)b);

        // 6th iter
        vsr(a, d2, 12);
        vsr(b, d2, 15);
        vand(a, a, const_0x7);
        vand(b, b, const_0x7);
        vsub(c[6], (int32x4_t)a, (int32x4_t)b);

        // 7th iter
        vsr(a, d2, 18);
        vsr(b, d2, 21);
        vand(a, a, const_0x7);
        vand(b, b, const_0x7);
        vsub(c[7], (int32x4_t)a, (int32x4_t)b);

        // c[0]: x | 12 | x |  8 -- x | 4  | x | 0
        // c[1]: x | 13 | x |  9 -- x | 5  | x | 1
        // c[2]: x | 14 | x | 10 -- x | 6  | x | 2
        // c[3]: x | 15 | x | 11 -- x | 7  | x | 3
        // c[4]: x | 28 | x | 24 -- x | 20 | x | 16
        // c[5]: x | 29 | x | 25 -- x | 21 | x | 17
        // c[6]: x | 30 | x | 26 -- x | 22 | x | 19
        // c[7]: x | 31 | x | 27 -- x | 23 | x | 19
        transpose4x8(c_out, c);

        vstore(&r->coeffs[i], c_out);
    }
}

void neon_cbd_eta2(poly *r, const uint8_t buf[KYBER_ETA1 * KYBER_N / 4])
{
#if KYBER_ETA2 != 2
#error "This implementation requires eta2 = 2"
#else
    neon_cbd2(r, buf);
#endif
}

void neon_cbd_eta1(poly *r, const uint8_t buf[KYBER_ETA1*KYBER_N/4])
{
#if KYBER_ETA1 == 2
  neon_cbd2(r, buf);
#elif KYBER_ETA1 == 3
  neon_cbd3(r, buf);
#else
#error "This implementation requires eta1 in {2,3}"
#endif
}

/*
#include <string.h>
#include <stdio.h>
#include <sys/random.h>


void compare(poly *r_gold, poly *r, uint8_t *buf, const char *string)
{
    printf("%s\n", string);
    int16_t a, b, count =0;
    for (int i = 0; i < 32; i++)
    {
        a = r_gold->coeffs[i];
        b = r->coeffs[i];
        if (a != b){
            printf("%2d: %2x | %d != %d\n", i, buf[i/2] , a, b);
            count++;
        }
        if (count == 24)
            break;
    }
}


#define SIZE1 (KYBER_ETA1 * KYBER_N / 4)
#define SIZE2 (KYBER_ETA2 * KYBER_N / 4)


int main(void)
{
    uint8_t buf1[SIZE1], buf2[SIZE2];
    getrandom(buf1, sizeof(buf1), 0);
    getrandom(buf2, sizeof(buf2), 0);

    poly r_gold, r;

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


    return 0;
}

//gcc cbd.c neon_cbd.c -o neon_cbd -g3 -O0 -Wall -Wextra -Wpedantic
*/