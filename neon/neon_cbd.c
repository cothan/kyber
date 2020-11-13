#include <arm_neon.h>
#include <stdint.h>
#include "params.h"
#include "cbd.h"

#define vload(c, ptr) c = vld1q_u32(ptr);
#define vload3(c, ptr) c = vld3q_u8(ptr);
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

#define transpose4x8(x, y)                                         \
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
    uint32x4_t a, b, d1, d2, za16_tmp, zb16_tmp; // 4
    int32x4_t c[8];                      // 8
    int16x8x4_t c_out;                   // 4
    uint32x4_t const_0x7;
    const_0x7 = vdupq_n_u32(0x7);

    uint8x16x3_t neon_buf; // 3
    uint8x16_t aa[3], bb[3], cc[3]; // 9
    uint8x16_t a_tmp, b_tmp;
    uint8x16_t  const_0x49, const_0x92, const_0x24, 
                const_0x48, const_0x01, const_0x90, 
                const_0x02; // 7
    uint16x8_t sum[6], a16_tmp, b16_tmp; // 8
    uint16x8_t y16, y18;                                           \
    int16x8_t y10, y11, y12, y13, y24, y25;  
    const_0x49 = vdupq_n_u8(0x49);
    const_0x92 = vdupq_n_u8(0x92);
    const_0x24 = vdupq_n_u8(0x24);
    const_0x48 = vdupq_n_u8(0x48);
    const_0x01 = vdupq_n_u8(0x01);
    const_0x02 = vdupq_n_u8(0x02);
    const_0x90 = vdupq_n_u8(0x90);

    int j = 0;
    // Total SIMD registers: 29
    for (int i = 0; i < 3 * KYBER_N / 4; i += 16*3)
    {
        // 0, 3, 6, ... a
        // 1, 4, 7, ... b
        // 2, 5, 8, ... c
        vload3(neon_buf, &buf[i]);
        // aa[0] = a & 0x49;
        // bb[0] = b & 0x92; 
        // cc[0] = c & 0x24;
        vand8(aa[0], neon_buf.val[0], const_0x49);
        vand8(bb[0], neon_buf.val[1], const_0x92);
        vand8(cc[0], neon_buf.val[2], const_0x24);

        // aa[1] = (a & 0x92) >> 1;
        // bb[1] = (b & 0x24) >> 1;
        // cc[1] = (c & 0x48) >> 1;
        // bb[1] |= (c & 0x1) << 7;
        vand8(aa[1], neon_buf.val[0], const_0x92);
        vand8(bb[1], neon_buf.val[1], const_0x24);
        vand8(cc[1], neon_buf.val[2], const_0x48);
        vsr8(aa[1], aa[1], 1);
        vsr8(cc[1], cc[1], 1);
        // (bb[1] |= (c & 0x1) << 7;
        vand8(b_tmp, neon_buf.val[2], const_0x01);
        vsl8(b_tmp, b_tmp, 7);
        vsri8(bb[1], b_tmp, bb[1], 1);

        // aa[2] =  (a & 0x24) >> 2;
        // bb[2] =  (b & 0x48) >> 2;
        // cc[2] =  (c & 0x90) >> 2;
        // aa[2] |= (b & 0x1) << 6;
        // bb[2] |= ( (c & 0x2) >> 1 ) << 7;
        vand8(aa[2], neon_buf.val[0], const_0x24);
        vand8(bb[2], neon_buf.val[1], const_0x48);
        vand8(cc[2], neon_buf.val[2], const_0x90);
        vsr8(cc[2], cc[2], 2);

        // aa[2] |= (b & 0x1) << 6;
        // bb[2] |= (c & 0x2) << 6;
        vand8(a_tmp, neon_buf.val[1], const_0x01);
        vand8(b_tmp, neon_buf.val[2], const_0x02);
        vsl8(a_tmp, a_tmp, 6);
        vsl8(b_tmp, b_tmp, 6);
        vsri8(aa[2], a_tmp, aa[2], 2);
        vsri8(bb[2], b_tmp, bb[2], 2);
        
        // sum[0,3] = aa[0] + aa[1] + aa[2];
        // sum[1,4] = bb[0] + bb[1] + bb[2];
        // sum[2,5] = cc[0] + cc[1] + cc[2];
        sum[0] = vaddl_u8( vget_low_u8(aa[0]), vget_low_u8(aa[1]) );
        sum[1] = vaddl_u8( vget_low_u8(bb[0]), vget_low_u8(bb[1]) );
        sum[2] = vaddl_u8( vget_low_u8(cc[0]), vget_low_u8(cc[1]) );
        sum[3] = vaddl_high_u8(aa[0], aa[1]);
        sum[4] = vaddl_high_u8(bb[0], bb[1]);
        sum[5] = vaddl_high_u8(cc[0], cc[1]);
        
        sum[0] = vaddw_u8(sum[0], vget_low_u8(aa[2]));
        sum[1] = vaddw_u8(sum[1], vget_low_u8(bb[2]));
        sum[2] = vaddw_u8(sum[2], vget_low_u8(cc[2]));
        sum[3] = vaddw_high_u8(sum[3], aa[2]);
        sum[4] = vaddw_high_u8(sum[4], bb[2]);
        sum[5] = vaddw_high_u8(sum[5], cc[2]);

        // sum[1,4] = (sum[0,3] >> 8) + sum[1,4];
        vsr16(a16_tmp, sum[0], 8);
        vsr16(b16_tmp, sum[3], 8);
        vadd16(sum[1], sum[1], a16_tmp);
        vadd16(sum[4], sum[4], b16_tmp);
        // sum[2,5] = (sum[1,4] >> 8) + sum2;
        vsr16(a16_tmp, sum[1], 8);
        vsr16(b16_tmp, sum[4], 8);
        vadd16(sum[2], sum[2], a16_tmp);
        vadd16(sum[5], sum[5], b16_tmp);
        
        // sum[0] | (sum[1] << 8)
        // sum[3] | (sum[4] << 8)
        a_tmp = (uint8x16_t) sum[0];
        b_tmp = (uint8x16_t) sum[1];
        a16_tmp = (uint16x8_t) vtrn1q_u8( (uint8x16_t) sum[0], (uint8x16_t) sum[1]);
        b16_tmp = (uint16x8_t) vtrn1q_u8( (uint8x16_t) sum[3], (uint8x16_t) sum[4]);

        // sum[0] | (sum[1] << 8) | (sum[2] << 16)
        // sum[3] | (sum[4] << 8) | (sum[5] << 16)
        za16_tmp = (uint32x4_t) vtrn1q_u16(a16_tmp, sum[2]);
        zb16_tmp = (uint32x4_t) vtrn2q_u16(a16_tmp, sum[2]);
        
        d1 = vzip1q_u32(za16_tmp, zb16_tmp);
        d2 = vzip2q_u32(za16_tmp, zb16_tmp);

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

        vstore(&r->coeffs[j], c_out);
        j+=32;

        za16_tmp = (uint32x4_t) vtrn1q_u16(b16_tmp, sum[5]);
        zb16_tmp = (uint32x4_t) vtrn2q_u16(b16_tmp, sum[5]);
        
        d1 = vzip1q_u32(za16_tmp, zb16_tmp);
        d2 = vzip2q_u32(za16_tmp, zb16_tmp);

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

        vstore(&r->coeffs[j], c_out);
        j+=32;
    }
}

void neon_cbd_eta1(poly *r, const uint8_t buf[KYBER_ETA1 * KYBER_N/4])
{
#if KYBER_ETA1 == 2
  neon_cbd2(r, buf);
#elif KYBER_ETA1 == 3
  neon_cbd3(r, buf);
#else
#error "This implementation requires eta1 in {2,3}"
#endif
}

void neon_cbd_eta2(poly *r, const uint8_t buf[KYBER_ETA1 * KYBER_N / 4])
{
#if KYBER_ETA2 != 2
#error "This implementation requires eta2 = 2"
#else
    neon_cbd2(r, buf);
#endif
}


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

static uint32_t load24_littleendian(const uint8_t x[3])
{
  uint32_t r;
  r  = (uint32_t)x[0];
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
    d  = t & 0x00249249;
    d += (t>>1) & 0x00249249;
    d += (t>>2) & 0x00249249;

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


    aa[2]  = (a & 0x24) >> 2;
    bb[2]  = (b & 0x48) >> 2;
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
    printf("%2x\n",sum0 & 0xff);

    s |= t2 << 8; 
    s |= t3 << 16;
    
    printf("d,s: %06u %06u\n", d, s);
    if (s != d) return 1;
    return 0;
}

int main(void)
{
    uint8_t buf1[SIZE1], buf2[SIZE2];
    // getrandom(buf1, sizeof(buf1), 0);
    getrandom(buf2, sizeof(buf2), 0);

    poly r_gold, r;
    for (int i = 0; i < SIZE1; i++)
    {
        buf1[i] = i;
    }

    for (int i = 0; i < 16*3; i+=3){
        if (test(&buf1[i])) 
            return 1;
        printf("-------------\n");
    }

    cbd_eta1(&r_gold, buf1);
    neon_cbd_eta1(&r, buf1);
 
    // compare(&r_gold, &r, buf1, "CBD_ETA1");

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
