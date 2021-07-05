#!/bin/bash

gcc m1cycles.c neon_ntt.c reduce.c improve-inv-ntt.c improve-ntt.c test_new_ntt.c -o test_new -Wall -Wextra -Wpedantic -Wmissing-prototypes -Wredundant-decls -Wshadow -Wpointer-arith -fomit-frame-pointer -mtune=native -O3
sudo ./test_new
