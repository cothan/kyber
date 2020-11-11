#include <arm_neon.h>
#include <stdint.h>
#include "params.h"
#include "cbd.h"

#include <string.h>
#include <stdio.h>
#include <sys/random.h>

#define vload(c, ptr) c = vld1q_u32(ptr);

// c = a & b
#define vand(c, a, b) c = vandq_u32(a, b);

// c = a >> n 
#define vsr(c, a, n) c = vshrq_n_u32(a, n);

// c = a + b
#define vadd(c, a, b) c = vaddq_u32(a, b);

// c = a - b 
#define vsub(c, a, b) c = vsubq_s32(a, b);

// WCZD = zip1(ABCD, XYWZ)
#define vzip1(c, a, b) c = vzip1q_u8(a, b);

// XAYB = zip2(ABCD, XYWZ)
#define vzip2(c, a, b) c = vzip2q_u8(a, b);

#define vstore(ptr, c) vst1q_s16_x4(ptr, c);

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
#define transpose8x8(x, y) \
    uint16x8_t y8, y9, y12, y13, y16, y17, y18, y19, y20, y21, y24, y25, y26, y27; \
    y16 = vtrn1q_u16((uint16x8_t) y[0], (uint16x8_t) y[1]);\
    y17 = vtrn2q_u16((uint16x8_t) y[0], (uint16x8_t) y[1]);\
    y18 = vtrn1q_u16((uint16x8_t) y[2], (uint16x8_t) y[3]);\
    y19 = vtrn2q_u16((uint16x8_t) y[2], (uint16x8_t) y[3]);\
    y24 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y16, (uint32x4_t)y17);\
    y25 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y16, (uint32x4_t)y17);\
    y26 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y18, (uint32x4_t)y19);\
    y27 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y18, (uint32x4_t)y19);\
    y8  = (uint16x8_t)vtrn1q_u32((uint32x4_t)y24, (uint32x4_t)y26);\
    y9  = (uint16x8_t)vtrn1q_u32((uint32x4_t)y25, (uint32x4_t)y27);\
    y16 = vtrn1q_u16((uint16x8_t) y[4], (uint16x8_t) y[5]);\
    y17 = vtrn2q_u16((uint16x8_t) y[4], (uint16x8_t) y[5]);\
    y18 = vtrn1q_u16((uint16x8_t) y[6], (uint16x8_t) y[7]);\
    y19 = vtrn2q_u16((uint16x8_t) y[6], (uint16x8_t) y[7]);\
    y24 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y16, (uint32x4_t)y17);\
    y25 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y16, (uint32x4_t)y17);\
    y26 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y18, (uint32x4_t)y19);\
    y27 = (uint16x8_t)vtrn2q_u32((uint32x4_t)y18, (uint32x4_t)y19);\
    y12 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y24, (uint32x4_t)y26);\
    y13 = (uint16x8_t)vtrn1q_u32((uint32x4_t)y25, (uint32x4_t)y27);\
    y16 = (uint16x8_t)vtrn1q_u64((uint64x2_t)y8,  (uint64x2_t)y12);\
    \
    y17 = (uint16x8_t)vtrn2q_u64((uint64x2_t)y8,  (uint64x2_t)y12);\
    y20 = (uint16x8_t)vtrn1q_u64((uint64x2_t)y9,  (uint64x2_t)y13);\
    y21 = (uint16x8_t)vtrn2q_u64((uint64x2_t)y9,  (uint64x2_t)y13);\
    x.val[0] = (int16x8_t) y16;\
    x.val[1] = (int16x8_t) y20;\
    x.val[2] = (int16x8_t) y17;\
    x.val[3] = (int16x8_t) y21;

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
    for (int i = 0; i < KYBER_N/2; i+= 16)
    {
        vload(t, (uint32_t *) &buf[i]);
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
        vsub(c[0], (int32x4_t) a, (int32x4_t) b);
        // vsub(c[0], a, b);

        // 2nd iter
        vsr(a, d, 4);
        vsr(b, d, 6);
        vand(a, a, const_0x3);
        vand(b, b, const_0x3);
        vsub(c[1], (int32x4_t) a, (int32x4_t) b);
        // vsub(c[1], a, b);

        // 3rd iter
        vsr(a, d, 8);
        vsr(b, d, 10);
        vand(a, a, const_0x3);
        vand(b, b, const_0x3);
        vsub(c[2], (int32x4_t) a, (int32x4_t) b);
        // vsub(c[2], a, b);

        // 3rd iter
        vsr(a, d, 12);
        vsr(b, d, 14);
        vand(a, a, const_0x3);
        vand(b, b, const_0x3);
        vsub(c[3], (int32x4_t) a, (int32x4_t) b);
        // vsub(c[3], a, b);

        // 4th iter
        vsr(a, d, 16);
        vsr(b, d, 18);
        vand(a, a, const_0x3);
        vand(b, b, const_0x3);
        vsub(c[4], (int32x4_t) a, (int32x4_t) b);
        // vsub(c[4], a, b);

        // 5th iter
        vsr(a, d, 20);
        vsr(b, d, 22);
        vand(a, a, const_0x3);
        vand(b, b, const_0x3);
        vsub(c[5], (int32x4_t) a, (int32x4_t) b);
        // vsub(c[5], a, b);

        // 6th iter
        vsr(a, d, 24);
        vsr(b, d, 26);
        vand(a, a, const_0x3);
        vand(b, b, const_0x3);
        vsub(c[6], (int32x4_t) a, (int32x4_t) b);
        // vsub(c[6], a, b);

        // 7th iter
        vsr(a, d, 28);
        vsr(b, d, 30);
        vand(a, a, const_0x3);
        vand(b, b, const_0x3);
        vsub(c[7], (int32x4_t) a, (int32x4_t) b);
        // vsub(c[7], a, b);

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
        j+=32;
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


#define SIZE (KYBER_ETA1 * KYBER_N / 4)

int main(void)
{
    uint8_t buf[SIZE];
    getrandom(buf, sizeof(buf), 0);

    poly r_gold, r;

    cbd_eta2(&r_gold, buf);
    neon_cbd_eta2(&r, buf);

    int16_t a, b, count =0;
    for (int i = 0; i < 32; i++)
    {
        a = r_gold.coeffs[i];
        b = r.coeffs[i];
        if (a != b){
            printf("%2d: %2x | %d != %d\n", i, buf[i/2] , a, b);
            count++;
        }
        if (count == 24)
            break;
    }

    if (memcmp(r_gold.coeffs, r.coeffs, KYBER_N))
        return 1;

    return 0;
}
