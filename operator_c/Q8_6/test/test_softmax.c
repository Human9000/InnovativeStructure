#include<stdio.h>
#include<inttypes.h>

typedef int32_t i32; // if32: 1位符号位 + 25位整数位 + 6位小数位 = 32 位
typedef int64_t i64;
typedef float f32;
typedef double f64;

f64 lbit(f64 x, int bit) {
    f64 factor = 1;
    for (int i = 0; i < bit; ++i) {
        factor *= 2.0;
    }
    return (f64) (i32) (x * factor) / factor;
}


i32 expi32_4n_f(i32 x) {
    // i32: 1位符号位 + 25位整数位 + 6位小数位 = 32 位
    const i64 n = 6; // 小数位的数量
    const i64 w3 = 1 * 4 << n;      // 6 位小数
    const i64 w2 = w3 * 3 << n;     // 12 位小数
    const i64 w1 = w2 * 2 << n;     // 18 位小数
    const i64 w0 = w1 << n;         // 24 位小数
    const i64 scale_one = w0 << n;  // 30 位小数

    i64 x1 = x;         // i64: 6 位小数
    i64 x2 = x1 * x1;   // i64: 12 位小数
    i64 x3 = x2 * x1;   // i64: 18 位小数
    i64 x4 = x3 * x1;   // i64: 24 位小数

    i64 exp = w0 + (x1 * w1) + (x2 * w2) + (x3 * w3) + x4;  // 24 位小数
    return (scale_one < exp) ? 0 : (i32) (scale_one / exp); // 6 位小数
}

i32 expi32x4n_z(i32 x) {
    // i32: 1位符号位 + 25位整数位 + 6位小数位 = 32 位
    const i64 n = 6; // 小数位的数量
    const i64 w3 = 1 * 4 << n;      // 6 位小数
    const i64 w2 = w3 * 3 << n;     // 12 位小数
    const i64 w1 = w2 * 2 << n;     // 18 位小数
    const i64 w0 = w1 << n;         // 24 位小数

    i64 x1 = x;   // i64: 1位符号位 + n整数位 + 6 位小数位
    i64 x2 = x1 * x1;           // i64: 1位符号位 + n整数位 + 12 位小数位
    i64 x3 = x2 * x1;           // i64: 1位符号位 + n整数位 + 18 位小数位
    i64 x4 = x3 * x1;           // i64: 1位符号位 + n整数位 + 24 位小数位

    i64 exp = w0 + (x1 * w1) + (x2 * w2) + (x3 * w3) + x4; // 24 位小数位
    return (i32) (exp >> (n * 3));  // 6 位小数位
}

i32 expi32x4n(i32 x) {
    if (x >= 0) return expi32x4n_z(x);
    else return expi32_4n_f(-x);
}

f64 expf32_f(f32 x) {
    f64 x1 = x; // 6
    f64 x2 = x1 * x1 / 2;       // 12， x1 * x1  / 2
    f64 x3 = x2 * x1 / 3;     // 18， x2 * x1  / 3
    f64 x4 = x3 * x1 / 4;   // 18， x3 * x1  / 4
    f64 temp = 1 + x1 + x2 + x3 + x4;
    f64 res = lbit(1 / temp, 6); // (64 * 64)  / (x0 + x1 + x2 + x3 + x4);
    return res; // (64 * 64)  / (x0 + x1 + x2 + x3 + x4);
}

int main() {
    for (int i = 64 * 1; i < 64 * 64; ++i) {
        f32 x = i / 64.;
        f64 yf = expf32_f(x);
        f64 yi = expi32x4n(-i) / 64.;
        printf("%f, %f, %f, %f\n", x, yf, yi, yf - yi);

    }
}
