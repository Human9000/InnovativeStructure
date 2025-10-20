//
// Created by Administrator on 25-9-12.
//

#ifndef Q8_6_OPT_H
#define Q8_6_OPT_H

#include <stdint.h>


typedef struct {
    int32_t dims[10];
    int32_t length;
}TensorShape;

typedef struct {
    uint8_t *data;
    TensorShape shape;
} TensorQ1Zip;

typedef struct {
    int8_t *data;
    TensorShape shape;
} TensorQ6Zip;

typedef struct {
    int32_t *data;
    TensorShape shape;
} TensorQ6;


typedef struct {
    int8_t *data;
    uint16_t n_data;
    uint16_t *dims;
    uint16_t n_dim;
    uint8_t q_bits;
} QTensor8;


typedef struct {
    int8_t *data;
    uint16_t n_data;
    uint16_t *dims;
    uint16_t n_dim;
    uint8_t q_bits;
} QTensor32;


QTensor8 qTensor8 = {.data=NULL, .n_data=0, .dims=NULL, .n_dim=0, .q_bits=0};

int32_t q32_q8_vec_in_prod(int32_t *A, int8_t *B, int32_t len, int32_t B_quantize_bits)
{
    int32_t res = 0;								  // 初始化结果为0, 结果要除以64,解析成(-1,1)的8字节数
    for (int32_t i = 0; i < len; i++, A++, B++) // 遍历向量元素
        res += *A * *B;
    return res >> B_quantize_bits; // 返回内积结果
}


// 浮点矩阵 和 bit 矩阵做 矩阵乘法 得到一个新的矩阵，支持转置输出
// 浮点矩阵与浮点矩阵的矩阵乘法函数
void q32_q8_mul(int32_t *X, int8_t *W, int32_t *Y, int32_t left_dim, int32_t mid_dim, int32_t right_dim, int32_t W_quantize_bits) {
    int32_t *_X = X;
    int32_t *_Y = Y;
    for (int i = 0; i < left_dim; i++, _X += mid_dim) { // 遍历矩阵A的行
        int8_t *_W = W;
        for (int j = 0; j < right_dim; j++, _Y++, _W += mid_dim) // 遍历矩阵B的行
            *_Y = q32_q8_vec_in_prod(_X, _W, mid_dim, W_quantize_bits); // 计算内积并存储到输出矩阵
    }
}
void q32_q8_mul_T(int32_t *X, int8_t *W, int32_t *Y, int32_t left_dim, int32_t mid_dim, int32_t right_dim, int32_t W_quantize_bits) {
    int32_t *_Y = Y;
    int8_t *_W = W;
    for (int32_t r = 0; r < right_dim; r++, _W += mid_dim) {// 遍历矩阵B的行
        int32_t *_X = X;
        for (int32_t l = 0; l < left_dim; l++, _Y++, _X += mid_dim) // 遍历矩阵A的行
            *_Y = q32_q8_vec_in_prod(_X, _W, mid_dim, W_quantize_bits);          // 计算内积并存储到输出矩阵
    }
}

static inline int32_t acc_i32_mve_8(const uint8_t wbyte,
                                    const int32_t *x)
{
    /* 手动展开 8 行，编译器生成 8 条 32-bit ALU */
    int32_t sum = 0;
    int32_t b0 =  ((wbyte >> 0) & 1) ^ 1  ;
    int32_t b1 =  ((wbyte >> 1) & 1) ^ 1 ;
    int32_t b2 =  ((wbyte >> 2) & 1) ^ 1 ;
    int32_t b3 =  ((wbyte >> 3) & 1) ^ 1 ;
    int32_t b4 =  ((wbyte >> 4) & 1) ^ 1 ;
    int32_t b5 =  ((wbyte >> 5) & 1) ^ 1 ;
    int32_t b6 =  ((wbyte >> 6) & 1) ^ 1 ;
    int32_t b7 =  ((wbyte >> 7) & 1) ^ 1 ;

    sum += (x[0] ^ (~b0)) + b0;
    sum += (x[1] ^ (~b1)) + b1;
    sum += (x[2] ^ (~b2)) + b2;
    sum += (x[3] ^ (~b3)) + b3;
    sum += (x[4] ^ (~b4)) + b4;
    sum += (x[5] ^ (~b5)) + b5;
    sum += (x[6] ^ (~b6)) + b6;
    sum += (x[7] ^ (~b7)) + b7;

    return sum;
}
// i32 与 b8 的内积函数
int32_t q32_b8_vec_in_prod(int32_t *data, uint8_t *weight, int32_t length)
{
    int32_t res = 0; // 初始化结果为0, 结果要除以127,解析成(-1,1) 的 8 字节数
    int32_t n = 0; // 记录已经处理过的data数量

//    // 处理开头不满足8位的部分
//    if (b_start > 0) {
//        for (uint8_t k = b_start; k < 8; k++,n++, data++) {
//            int32_t neg = ((*weight >> k) & 1) ^ 1; // 取出weight的第k位, 0表示-1, 1表示+1, 若wk=0,则neg=1,否则neg=0
//            res += (*data ^ (~neg)) + neg;
//        }
//    }

    // 处理满足8位的部分
    for (weight += 1; n < length - 7; n += 8, weight++){
        res += acc_i32_mve_8(*weight, data); // 使用acc_i32_mve函数计算8个元素的累加和
    }

    // 处理结尾不满足8位的部分
    for (uint8_t k = 0; n < length; n++, k++, data++) {
        int32_t neg = ((*weight >> k) & 1) ^ 1; // 取出weight的第k位, 0表示-1, 1表示+1, 若wk=0,则neg=1,否则neg=0
        res += (*data ^ (~neg)) + neg;
    }

    return res; // 返回内积结果, bit 位解析为-1，+1,不需要变成结果的精度
}

// 浮点矩阵 和 bit 矩阵做 矩阵乘法 得到一个新的矩阵，支持转置输出
// 浮点矩阵与浮点矩阵的矩阵乘法函数
void i32_b8_mul(int32_t *data, b8 *weight, int32_t *out, int m, int k, int n, uint8_t Transpose  )
{
    if (Transpose)
    {
        for (int j = 0; j < n; j++)
        {
            int32_t *d = data;
            b8 *w = weight + (j * k) / 8;
            int32_t s = (j * k) % 8;
            for (int i = 0; i < m; i++, out++, d += k)
                *out = i32_b8_vec_in_prod(d, w, k, s);
        }
    }
    else
    {
        for (int i = 0; i < m; i++, data += k)
            for (int j = 0; j < n; j++, out++)
                *out = i32_b8_vec_in_prod(data,
                                          weight + (j * k) / 8,
                                          k,
                                          (j * k) % 8);
    }
}

#endif //Q8_6_OPT_H


