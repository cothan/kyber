#include <arm_neon.h>
#include "params.h"
#include <stdio.h>

#define vload(c, ptr) c = vld1q_u32(ptr);
#define vload3(c, ptr) c = vld3q_u8(ptr);
#define vstore(ptr, c) vst1q_s16_x4(ptr, c);

#define vand16(c, a, b) c = vandq_u16(a, b);

// c = a >> n
#define vsr16(c, a, n) c = vshrq_n_u16(a, n);


unsigned int rej_uniform(int16_t *r,
                         unsigned int len,
                         const uint8_t *buf,
                         unsigned int buflen)
{
    unsigned int ctr, pos;
    uint16_t val0, val1;

    ctr = pos = 0;
    while (ctr < len && pos + 3 <= buflen)
    {
        val0 = ((buf[pos + 0] >> 0) | ((uint16_t)buf[pos + 1] << 8)) & 0xFFF;
        val1 = ((buf[pos + 1] >> 4) | ((uint16_t)buf[pos + 2] << 4)) & 0xFFF;
        printf("%d, %d, ", val0, val1);
        pos += 3;

        if (val0 < KYBER_Q)
            r[ctr++] = val0;
        if (ctr < len && val1 < KYBER_Q)
            r[ctr++] = val1;
    }

    return ctr;
}


unsigned int neon_rej_uniform(int16_t *r,
                         unsigned int len,
                         const uint8_t *buf,
                         unsigned int buflen)
{
    uint8x16x3_t neon_buf;
    uint16x8x2_t val0, val1;
    uint16x8_t neon_q; 
    uint16x8x4_t sign;
    neon_q = vdupq_n_u16(KYBER_Q - 1);
    for (unsigned int i = 0; i < buflen; i+=16*3)
    {
        vload3(neon_buf, &buf[i]);


        val0.val[0] = (uint16x8_t) vzip1q_u8(neon_buf.val[0], neon_buf.val[1]);
        val0.val[1] = (uint16x8_t) vzip2q_u8(neon_buf.val[0], neon_buf.val[1]);

        vand16(val0.val[0], val0.val[0], vdupq_n_u16(0xfff));
        vand16(val0.val[1], val0.val[1], vdupq_n_u16(0xfff));

        val1.val[0] = (uint16x8_t) vtrn1q_u8(neon_buf.val[1], neon_buf.val[2]);
        val1.val[1] = (uint16x8_t) vtrn2q_u8(neon_buf.val[1], neon_buf.val[2]);

        vand16(val1.val[0], val1.val[0], vdupq_n_u16(0xfff0));
        vand16(val1.val[1], val1.val[1], vdupq_n_u16(0xfff0));

        vsr16(val1.val[0], val1.val[0], 4);
        vsr16(val1.val[1], val1.val[1], 4);

        // compare unsigned less than equal  
        sign.val[0] = vcleq_u16(val0.val[0], neon_q); 
        sign.val[1] = vcleq_u16(val0.val[1], neon_q); 
        sign.val[2] = vcleq_u16(val1.val[0], neon_q); 
        sign.val[3] = vcleq_u16(val1.val[1], neon_q); 
    }
}



#include <string.h>
#include <stdio.h>
#include <sys/random.h>

void compare(int16_t *r_gold, int16_t *r, uint8_t *buf, const char *string)
{
    printf("%s\n", string);
    int16_t a, b, count = 0;
    for (int i = 0; i < 32; i++)
    {
        a = r_gold[i];
        b = r[i];
        if (a != b)
        {
            printf("%2d: %2x | %d != %d\n", i, buf[i / 2], a, b);
            count++;
        }
        if (count == 24)
            break;
    }
}

#define SIZE 504

int main(void)
{
    uint8_t buf[SIZE];
    getrandom(buf, sizeof(buf), 0);

    for (int i = 0; i < SIZE; i++)
    {
        buf[i] = i;
    }

    int16_t r_gold[KYBER_N], r[KYBER_N];

    rej_uniform(r_gold, KYBER_N, buf, SIZE);
    neon_rej_uniform(r, KYBER_N, buf, SIZE);

    compare(r_gold, r, buf, "NEON_SAMPLE");

    if (memcmp(r_gold, r, KYBER_N))
        return 1;

    return 0;
}

//gcc cbd.c neon_sample.c -o neon_sample -g3 -O0 -Wall -Wextra -Wpedantic
