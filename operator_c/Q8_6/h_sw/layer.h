#ifndef H_LAYER_H
#define H_LAYER_H

#include "../h/opt.h"


typedef struct {
    i32 *data;
    int length; // 包含 mem_length 在内的数据总长度
    int channels;
    int mem_length; // 记录的 mem_length 长度
} Buffer;

typedef struct {
    i32 *weight;
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

typedef struct {
    int in_channels;
    int kernel;
    int stride;
} Pool1d;

typedef struct {
    int in_channels;
    int factor;
} Upsample;

void avg_pool1d_i32(Buffer in, Pool1d p, Buffer out) {
    // 解析输入数组的形状
    i32 *idata = in.data;
    int in_length = in.length;
    int channels = in.channels;
    int kernel = p.kernel;
    int stride = p.stride;

    int out_length = (in_length - kernel) / stride + 1;

    if (ERR(out_length * channels < 1, "out_length * out_channels < 1") ||
        ERR(idata == NULL, "idata == NULL")) {
        out.length = out.mem_length; // 不输出数据
        return;
    }

    i32 *odata = out.data + out.mem_length * out.channels;
    out.length = out.mem_length + out_length;

    // do write out buffer data
    for (int c = 0; c < channels; c++) {
        i32 *o = odata + c;
        for (int l = 0; l <= in_length - kernel; l += stride, o += channels) {
            i32 avg = 0;
            for (int k = 0; k < kernel; k++)
                avg += idata[(l + k) * channels + c];
            *o = avg / kernel;
        }
    }

    // do write in buffer memory
    i32 *mw = in.data;
    i32 *mr = in.data + in.channels * (in.length - in.mem_length);
    for (int i = 0; i < in.channels * in.mem_length; i++, mr++, mw++)
        *mw = *mr;

}

void conv1d_i32_i8(Buffer in, ConvI8 p, Buffer out) {
    // unzip input
    i32 *idata = in.data;
    int length = in.length;
    int in_channels = in.channels;
    int out_channels = p.out_channels;
    i32 *weight = p.weight;
    int kernel = p.kernel;
    int stride = p.stride;
    int out_length = (length - kernel) / stride + 1;

    // clamp
    for (int i = 0; i < in.length * in.channels; ++i) {
        idata[i] = idata[i] > 64 ? 64 : idata[i];
        idata[i] = idata[i] < -64 ? -64 : idata[i];
    }

    i32 *o = out.data + out.mem_length * out.channels;
    i32 *i = in.data;
//    printf("%d,%d\n", out_length, out.length - out.mem_length);
    // do
    for (int l = 0; l <= length - kernel;
         l += stride,
         i += in_channels * stride,
         o += out_channels * stride) {

        i32_i8_mul(i,  // shape = 1, kernel*in
                   weight, // shape= o, kernel*in
                   o,  // shape = 1, o
                   1,
                   kernel * in_channels,
                   out_channels,
                   false);
    }

    // do write in buffer memory
    i32 *mw = in.data;
    i32 *mr = in.data + in.channels * (in.length - in.mem_length);
    for (int i = 0; i < in.channels * in.mem_length; i++, mr++, mw++)
        *mw = *mr;
}

void conv1d_i32_b8(Buffer in, ConvB8 p, Buffer out) {
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
        return;
    }

    // clamp
    for (int i = 0; i < in.length * in.channels; ++i) {
        idata[i] = idata[i] > 64 ? 64 : idata[i];
        idata[i] = idata[i] < -64 ? -64 : idata[i];
    }

    i32 *o = out.data + out.mem_length * out.channels;
    i32 *i = in.data;

    for (int l = 0; l < out_length;
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

    // do write in buffer memory
    i32 *mw = in.data;
    i32 *mr = in.data + in.channels * (in.length - in.mem_length);
    for (int i = 0; i < in.channels * in.mem_length; i++, mr++, mw++)
        *mw = *mr;
}

void upsample_linear(Buffer in, Upsample p, Buffer out) {
    i32 *idata = in.data;
    int in_length = in.length;
    int channels = in.channels;
    int out_length;
    int factor = p.factor;

    out_length = in_length * factor - factor;


    if (ERR(out_length * channels < 1, "out_length * out_channels < 1") ||
        ERR(idata == NULL, "idata == NULL")) {
        return ;
    }

    i32 * odata = out.data + out.mem_length * out.channels;

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


    // do write in buffer memory
    i32 *mw = in.data;
    i32 *mr = in.data + in.channels * (in.length - in.mem_length);
    for (int i = 0; i < in.channels * in.mem_length; i++, mr++, mw++)
        *mw = *mr;
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


void softmax(Buffer in, Buffer out) {
    // unzip input
    i32 *idata = in.data;
    int channels = in.channels;
    int length = in.length;
    if (ERR(length * channels < 1, "out_length * out_channels < 1") ||
        ERR(idata == NULL, "idata == NULL")) {
        return ;
    }

    i32 *odata = out.data + out.mem_length * out.channels;
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
            *o = expi32_4n_f(*o); // (64 * 64)  / (x0 + x1 + x2 + x3 + x4);
            sum_channels += *o;
        }
        // 求概率密度
        o = odata + l * channels;
        for (int c = 0; c < channels; ++c, o++) *o = 64 * *o / sum_channels; // *o = *o * 64 / sum_channels;
    }

}

#endif
