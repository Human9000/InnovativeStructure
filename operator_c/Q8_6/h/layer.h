#ifndef H_LAYER_H
#define H_LAYER_H

#include "opt.h"
#include <malloc.h>


typedef struct {
    i8 *weight;
    int in_channels;
    int out_channels;
    int kernel;
    int stride;
} ConvI8;

typedef struct {
    b8 *weight;
    int in_channels;
    int out_channels;
    int kernel;
    int stride;
} ConvB8;

ConvI8 CI8(i8 *w, int cin, int cout, int k, int s) {
    ConvI8 res = {w, cin, cout, k, s};
    return res;
}

ConvB8 CB8(b8 *w, int cin, int cout, int k, int s) {
    ConvB8 res = {w, cin, cout, k, s};
    return res;
}

MatI32 avg_pool1d_i32(MatI32 in, int K, int S) {
    // 解析输入数组的形状
    i32 *idata = in.data;
    int in_length = in.length;
    int channels = in.channels;
    int out_length = (in_length - K) / S + 1;

    if (ERR(out_length * channels < 1, "out_length * out_channels < 1") ||
        ERR(idata == NULL, "idata == NULL")) {
        MatI32 res = {NULL, 0, 0};
        return res;
    }
    i32 *odata = (i32 *) malloc(sizeof(i32) * out_length * channels);
//    static i32 odata[1152];
    MatI32 res = {odata, out_length, channels};
    printMI32("res", res);
    // do
    for (int c = 0; c < channels; c++) {
        i32 *o = odata + c;
        for (int l = 0; l <= in_length - K; l += S, o += channels) {
            i32 avg = 0;
            for (int k = 0; k < K; k++)
                avg += idata[(l + k) * channels + c];
            *o = avg / K;
        }
    }

    // free input
    // free(idata);
    // in.data = NULL;
    // return
    return res;
}

MatI32 conv1d_i32_i8(MatI32 in, ConvI8 p) {
    // unzip input
    i32 *idata = in.data;
    int length = in.length;
    int in_channels = in.channels;
    int out_channels = p.out_channels;
    i8 *weight = p.weight;
    int kernel = p.kernel;
    int stride = p.stride;
    int out_length = (length - kernel) / stride + 1;

    if (ERR(in_channels != p.in_channels, "in_channels != p.in_channels") ||
        ERR(out_length * out_channels < 1, "out_length * out_channels < 1") ||
        ERR(idata == NULL, "idata == NULL") ||
        ERR(weight == NULL, "weight == NULL")) {
        MatI32 res = {NULL, 0, 0};
        return res;
    }

    // clamp
    for (int i = 0; i < in.length * in.channels; ++i) {
        idata[i] = idata[i] > 64 ? 64 : idata[i];
        idata[i] = idata[i] < -64 ? -64 : idata[i];
    }


    // make output
    i32 *odata = (i32 *) malloc(sizeof(i32) * out_length * out_channels);
//    static i32 odata[380];
//    printf("conv i32 i8:%d\n", out_channels * out_length);
    MatI32 res = {odata, out_length, out_channels};
    i32 *o = odata;
    i32 *i = idata;
    // do
    for (int l = 0; l <= length - kernel;
         l += stride,
         i += in_channels * stride,
         o += out_channels) {
        i32_i8_mul(i,  // shape = 1, kernel*in
                   weight, // shape= o, kernel*in
                   o,  // shape = 1, o
                   1,
                   kernel * in_channels,
                   out_channels,
                   false);
    }

    // free input
//    free(in.data);
//    in.data = NULL;
    return res;
}

MatI32 conv1d_i32_b8(MatI32 in, ConvB8 p) {
    // unzip input
    i32 *idata = in.data;
    int length = in.length;
    int in_channels = in.channels;
    int out_channels = p.out_channels;
    b8 *weight = p.weight;
    int kernel = p.kernel;
    int stride = p.stride;
    int out_length = (length - kernel) / stride + 1;

    if (ERR(in_channels != p.in_channels, "in_channels != p.in_channels") ||
        ERR(out_length * out_channels < 1, "out_length * out_channels < 1") ||
        ERR(idata == NULL, "idata == NULL") ||
        ERR(weight == NULL, "weight == NULL")) {
        MatI32 res = {NULL, 0, 0};
        return res;
    }

    // clamp
    for (int i = 0; i < in.length * in.channels; ++i) {
        idata[i] = idata[i] > 64 ? 64 : idata[i];
        idata[i] = idata[i] < -64 ? -64 : idata[i];
    }

    // make output
    i32 *odata = (i32 *) malloc(sizeof(i32) * out_length * out_channels);
//    static i32 odata[5824];

//    printf("conv1d_i32_b8 length：%d\n", out_channels*out_length);
    i32 *i = idata;
    i32 *o = odata;
    MatI32 res = {odata, out_length, out_channels};

    for (int l = 0; l <= length - kernel;
         l += stride,
         i += in_channels,
         o += out_channels) {
        i32_b8_mul(i,  // shape = 1, kernel*in
                   weight, // shape= o, kernel*in
                   o,  // shape = 1, o
                   1,
                   kernel * in_channels,
                   out_channels,
                   false);
    }

    // free input
//    free(in.data);
//    in.data = NULL;

    return res;
}

MatI32 top_density(MatI32 in_data_mat) {
    i32 *idata = in_data_mat.data;
    int length = in_data_mat.length;
    int channels = in_data_mat.channels;

    if (ERR(channels * channels < 1, "out_length * out_channels < 1") ||
        ERR(idata == NULL, "idata == NULL")) {
        MatI32 res = {NULL, 0, 0};
        return res;
    }


    for (int l = 0; l < length; l++) {
        i32 mean = 0;
        i32 *p = idata;
        i32 c;
        i32 sum = 0;

        // 求均值
        for (c = 0; c < channels; c++, p++)
            mean += *p;

        // 裁剪+求和
        for (p = idata, c = 0; c < channels; c++, p++) {
            *p = *p > mean ? *p - mean : 0;
            sum += *p;
        }

        // 概率密度化
        for (c = 0; c < channels; c++, idata++)
            *idata /= sum;
    }
    return in_data_mat;
}

MatI32 upsample_linear(MatI32 in_data_mat, int factor, bool_ free_in_put) {

    // unzip input
    i32 *idata = in_data_mat.data;
    int in_length = in_data_mat.length;
    int channels = in_data_mat.channels;
    int out_length;


    out_length = in_length * factor - factor;


    if (ERR(out_length * channels < 1, "out_length * out_channels < 1") ||
        ERR(idata == NULL, "idata == NULL")) {
        return I32(NULL, 0, 0);
    }
    static i32 odata[3840];
    MatI32 res = {odata, out_length, channels};


    // 首位两端采用跳过策略，中间的部分用插值生成
    if (factor % 2 == 0) { // 偶数的话，所有数都不在插值的中心点，插值之后的长度为 length*factor - factor
        for (int l = 0; l < in_length - 1; l++) {
            for (int c = 0; c < channels; c++) {
                i32 left = idata[l * channels + c];
                i32 right = idata[(l + 1) * channels + c];
                i32 d = right - left;
                for (int f = 0; f < factor; ++f) {
                    odata[l * factor * channels + f * channels + c] =
                            (left * 2 * factor + d * (1 + 2 * f)) / (2 * factor);
                }
            }
        }
    } else { // 奇数的话，所有数都在插值的中心点，插值之后的长度为 length*factor - factor + 1
        for (int l = 0; l < in_length - 1; l++) {
            for (int c = 0; c < channels; c++) {
                i32 left = idata[l * channels + c];
                i32 right = idata[(l + 1) * channels + c];
                i32 d = right - left;
                for (int f = 0; f < factor; ++f) {
                    odata[l * factor * channels + f * channels + c] = (left * factor + d * f) / factor;
                }
            }
        }
    }

    return res;
}

MatI32 pad_length_I32(MatI32 in_data_mat, int left, int right) {
    // unzip input
    i32 *idata = in_data_mat.data;
    int channels = in_data_mat.channels;
    int in_length = in_data_mat.length;
    int out_length = left + right + in_data_mat.length;

    if (ERR(out_length * channels < 1, "out_length * out_channels < 1") ||
        ERR(idata == NULL, "idata == NULL")) {
        return I32(NULL, 0, 0);
    }

    // make output
    i32 *odata = (i32 *) malloc(sizeof(i32) * out_length * channels);
//    static i32 odata[4572];
//    printf("pad:%d\n", out_length * channels);
    MatI32 res = {odata, out_length, channels};

    // do
    int i = 0;
    i32 *po = odata;
    i32 *pi = idata;
    for (i = 0; i < left * channels; i++, po++)
        *po = 0;
    for (i = 0; i < in_length * channels; i++, pi++, po++)
        *po = *pi;
    for (i = 0; i < right * channels; i++, po++)
        *po = 0;

    return res;
}

i32 expi32_4n(i32 x) {
    i64 x1 = x; // 6
    i64 x2 = x1 * x1;       // 12， x1 * x1 / 2
    i64 x3 = x2 * x1;       // 18， x2 * x1 / 3
    i64 x4 = x3 * x1;       // 24， x3 * x1 / 4
    i64 x0 = (i64) 64 * 64 * 64 * 64 * 4 * 3 * 2;
    x1 = x1 * 64 * 64 * 64 * 4 * 3 * 2;
    x2 = x2 * 64 * 64 * 4 * 3;
    x3 = x3 * 64 * 4;
    i64 right = (x0 + x1 + x2 + x3 + x4);
    i64 left = x0 * 64;
    i64 res = (left / right);
    return (i32) res; // (i32) res; // (64 * 64)  / (x0 + x1 + x2 + x3 + x4);
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

i32 expi32x5n(i32 x) {
    const static i64 fractional_scale = 64; // 小数部分的缩放因子
    const static i64 w5 = 1;
    const static i64 w4 = w5 * fractional_scale * 5;
    const static i64 w3 = w4 * fractional_scale * 4;
    const static i64 w2 = w3 * fractional_scale * 3;
    const static i64 w1 = w2 * fractional_scale * 2;
    const static i64 w0 = w1 * fractional_scale;
    const static i64 scale_one = w0 * fractional_scale;
    i64 x1 = x >= 0 ? x : -x;  // 6
    i64 x2 = x1 * x1;       // 12， x1 * x1 / 2
    i64 x3 = x2 * x1;       // 18， x2 * x1 / 3
    i64 x4 = x3 * x1;       // 24， x3 * x1 / 4
    i64 x5 = x4 * x1;       // 30， x4 * x1 / 5
    i64 exp = w0 + x1 * w1 + x2 * w2 + x3 * w3 + x4 * w4 + x5 * w5;
    i64 res = x >= 0 ? exp : (scale_one / exp);
    return (i32) res; // (i32) res; // (64 * 64)  / (x0 + x1 + x2 + x3 + x4);
}

MatI32 softmax(MatI32 in) {
    // unzip input
    i32 *idata = in.data;
    int channels = in.channels;
    int length = in.length;


    if (ERR(length * channels < 1, "out_length * out_channels < 1") ||
        ERR(idata == NULL, "idata == NULL")) {
        return I32(NULL, 0, 0);
    }

    // make output
    i32 *odata = (i32 *) malloc(sizeof(i32) * length * channels);
//    static i32 odata[1152];
//    printf("softmax %d \n", length*channels);
    for (int l = 0; l < length; ++l) {
        i32 *o;
        i32 *i;
        i32 sum_channels = 0;

        // 求通道最大值
        i = idata + l * channels;
        i32 max = *i;
        for (int c = 0; c < channels; ++c, i++) if (*i > max) max = *i;

        o = odata + l * channels;
        i = idata + l * channels;
        // 最大值 减去输入
        for (int c = 0; c < channels; ++c, o++, i++) *o = max - *i;

        o = odata + l * channels;
        // 求exp, 和sum
        for (int c = 0; c < channels; ++c, o++) {
            *o = expi32_4n_f(*o); // 求 EXP(-*o)的4阶泰勒近似结果;
            sum_channels += *o;
        }
        // 求概率密度
        o = odata + l * channels;
        for (int c = 0; c < channels; ++c, o++) *o = (64 * *o) / sum_channels; // *o = *o * 64 / sum_channels;
    }

    return I32(odata, length, channels);
}

#endif
