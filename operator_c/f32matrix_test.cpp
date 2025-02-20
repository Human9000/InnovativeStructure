#include "f32matrix.h"

void test_f32_u8_1_vec_in_prod() {
    f32 a[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    u8_1 b[2] = {0xff, 0x0f};
    f32 res = f32_u8_1_vec_in_prod(a, b, 16);
    printf("res = %.3f\n", res);
}

void test_f32_f32_vec_in_prod() {
    f32 a[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    f32 b[16] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0};
    f32 res = f32_f32_vec_in_prod(a, b, 16);
    printf("res = %.3f\n", res);
}

void test_f32_u8_1_mul() {
    f32 a[8] = {1, 2,
        3, 4,
        5, 6,
        7, 8,
    };
    u8_1 b[2] = {0xf1, 0xf0};
    f32 c[100];
    int m = 2, n = 2, k = 4;
    f32_u8_1_mat_mul(a, b, c, m, n, k, true);
    printmat(a, m, k);
    printmat(b, n, k);
    printmat(c, m, n);
}

int main() {
//    test_f32_f32_vec_in_prod();
//    test_f32_u8_1_vec_in_prod();
    test_f32_u8_1_mul();
    return 1;
}
