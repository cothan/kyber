#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include "neon_ntt.h"
#include "m1cycles.h"

#define NTESTS 1000000

#define TIME(s) s = rdtsc();
// Result is clock cycles
#define CALC(start, stop) (stop - start) / NTESTS;

static int compare_array(int16_t a[KYBER_N], int16_t b[KYBER_N], const char *string, const int enable)
{
    if (enable)
        printf("%s\n", string);
    int ret = 0;
    for (int i = 0; i < KYBER_N; i++)
    {
        if ((a[i] + KYBER_Q) % KYBER_Q != (b[i] + KYBER_Q) % KYBER_Q)
        {
            printf("%d: %d != %d\n", i, a[i], b[i]);
            ret = 1;
            break;
        }
    }
    return ret;
}

static
int check_ntt(void)
{
    int16_t test_ntt[KYBER_N], test_ntt_qinv[KYBER_N], gold_ntt[KYBER_N], test_ntt_qinv_lane[KYBER_N];
    int16_t temp;
    int ret = 0;

    for (int j = 0; j < NTESTS; j++)
    {
        for (int i = 0; i < KYBER_N; i++)
        {
            temp = rand() % KYBER_Q;
            test_ntt[i] = temp;
            gold_ntt[i] = temp;
            test_ntt_qinv[i] = temp;
            test_ntt_qinv_lane[i] = temp;
        }

        test_neon_ntt(test_ntt);
        test_neon_ntt_qinv(test_ntt_qinv);
        neon_ntt(gold_ntt);
        test_neon_ntt_qinv_lane(test_ntt_qinv_lane);

        ret = compare_array(test_ntt, gold_ntt, "test_neon_ntt vs neon_ntt", 0);
        ret |= compare_array(test_ntt_qinv, gold_ntt, "test_neon_ntt_qinv vs neon_ntt", 0);
        ret |= compare_array(test_ntt_qinv_lane, gold_ntt, "test_neon_ntt_qinv_lane vs neon_ntt", 0);
        if (ret)
        {
            printf("%d: Error\n", j);
            break;
        }
    }

    return ret;
}

static
int check_invntt(void)
{
    int16_t test_ntt[KYBER_N], test_ntt_qinv[KYBER_N], gold_ntt[KYBER_N], test_invntt_qinv_lane[KYBER_N];
    int16_t temp;
    int ret = 0;
    for (int j = 0; j < NTESTS; j++)
    {
        for (int i = 0; i < KYBER_N; i++)
        {
            temp = rand() % KYBER_Q;
            test_ntt[i] = temp;
            gold_ntt[i] = temp;
            test_ntt_qinv[i] = temp;
            test_invntt_qinv_lane[i] = temp;
        }

        test_neon_invntt(test_ntt);
        test_neon_invntt_qinv(test_ntt_qinv);
        test_neon_invntt_qinv_lane(test_invntt_qinv_lane);
        neon_invntt(gold_ntt);

        ret = compare_array(test_ntt, gold_ntt, "test_neon_invntt vs neon_invntt", 0);
        ret |= compare_array(test_ntt_qinv, gold_ntt, "test_neon_invntt_qinv vs neon_invntt", 0);
        ret |= compare_array(test_invntt_qinv_lane, gold_ntt, "test_neon_ntt_qinv_lane vs neon_invntt", 0);
        if (ret)
        {
            printf("%d: Error\n", j);
            break;
        }
    }

    return ret;
}

int main()
{
    int16_t test_ntt[KYBER_N], gold_ntt[KYBER_N];
    int ret = 0;
    long long start, stop, ns;

    srand(0);

    if (check_ntt())
    {
        printf("ERROR NTT\n");
        return 1;
    }
    else
    {
        printf("NTT is OKAY\n");
    }

    if (check_invntt())
    {
        printf("ERROR INV_NTT\n");
        return 1;
    }
    else
    {
        printf("INV_NTT is OKAY\n");
    }


    // Benchmark clock cycles
    setup_rdtsc();

    printf("\nBenchmark NTT:\n");
    TIME(start);
    for (int i = 0; i < NTESTS; i++)
    {
        test_neon_ntt(test_ntt);
    }
    TIME(stop);
    ns = CALC(start, stop);
    printf("test_neon_ntt: %lld\n", ns);

    TIME(start);
    for (int i = 0; i < NTESTS; i++)
    {
        test_neon_ntt_qinv(test_ntt);
    }
    TIME(stop);
    ns = CALC(start, stop);
    printf("test_neon_ntt_qinv: %lld\n", ns);

    TIME(start);
    for (int i = 0; i < NTESTS; i++)
    {
        test_neon_ntt_qinv_lane(test_ntt);
    }
    TIME(stop);
    ns = CALC(start, stop);
    printf("test_neon_ntt_qinv_lane: %lld\n", ns);

    TIME(start);
    for (int i = 0; i < NTESTS; i++)
    {
        neon_ntt(gold_ntt);
    }
    TIME(stop);
    ns = CALC(start, stop);
    printf("neon_ntt: %lld\n", ns);

    // ret = compare_array(test_ntt, gold_ntt, "test_neon_ntt vs neon_ntt", 0);

    printf("\nBenchmark INV_NTT:\n");
    TIME(start);
    for (int i = 0; i < NTESTS; i++)
    {
        test_neon_invntt(test_ntt);
    }
    TIME(stop);
    ns = CALC(start, stop);
    printf("test_neon_invntt: %lld\n", ns);

    TIME(start);
    for (int i = 0; i < NTESTS; i++)
    {
        test_neon_invntt_qinv(test_ntt);
    }
    TIME(stop);
    ns = CALC(start, stop);
    printf("test_neon_invntt_qinv: %lld\n", ns);

    TIME(start);
    for (int i = 0; i < NTESTS; i++)
    {
        test_neon_invntt_qinv_lane(test_ntt);
    }
    TIME(stop);
    ns = CALC(start, stop);
    printf("test_neon_invntt_qinv_lane: %lld\n", ns);

    TIME(start);
    for (int i = 0; i < NTESTS; i++)
    {
        neon_invntt(gold_ntt);
    }
    TIME(stop);
    ns = CALC(start, stop);
    printf("neon_invntt: %lld\n", ns);

    return 0;
}
