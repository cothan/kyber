#include "../polyvec.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define LEGNTH 16
#define TESTS 10000

int test1(polyvec *a, polyvec *b)
{
    poly r, r_test;
    int count = 0; 

    polyvec_pointwise_acc_montgomery(&r, a, b);

    neon_polyvec_acc_montgomery(&r_test, a, b, 0);

    for (int i = 0; i < KYBER_N; i+=LEGNTH)
    {
        for (int j = 0; j < LEGNTH; j++)
        {
            if (r.coeffs[i + j] != r_test.coeffs[i + j])
            {
                printf("%d: %d != %d \n", i+j, r.coeffs[i+j], r_test.coeffs[i+j]);
                count += 1;
            }

            if (count == LEGNTH) return 1;
        }
    }

    return 0;
}

int main()
{
    polyvec a, b;
    int16_t t;

    // srand(0);
    srand(time(0));
    printf("KYBER_K: %d\n", KYBER_K);

    // Initialize data
    for (int i = 0; i < KYBER_K; i++)
    {
        for (int j = 0; j < KYBER_N; j++)
        {
            t = rand();
            a.vec[i].coeffs[j] = t % KYBER_Q;
            b.vec[i].coeffs[j] = rand() % KYBER_Q;
        }
    }

    for (int i = 0; i < TESTS; i++)
        if (test1(&a, &b))
            return 1;

    return 0;
}