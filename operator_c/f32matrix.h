#include<stdio.h>
#include <stdint.h>

#define f32 float
#define u8_1 unsigned char


#define f32 float                // 定义浮点数类型别名
#define u8_1 unsigned char       // 定义无符号字符类型别名

void printmat(f32 *a, int m, int n);  // 声明打印矩阵函数
void printmat(u8_1 *a, int m, int n);  // 声明打印矩阵函数

/**
 * 计算浮点向量的内积
 * 
 * @param data    浮点向量指针
 * @param weight  浮点向量指针
 * @param n       向量长度
 * @return        向量内积结果（浮点数）
 */
f32 f32_f32_vec_in_prod(f32 *data, f32 *weight, int n);

/**
 * 浮点矩阵与浮点矩阵的矩阵乘法
 * 
 * @param data    输入矩阵A，按(m, k)存储，k物理空间连续
 * @param weight  输入矩阵B，按(n, k)存储，k物理空间连续
 * @param c       输出矩阵C
 * @param m       矩阵A的行数
 * @param n       矩阵B的行数
 * @param k       矩阵A的列数和矩阵B的列数
 * @param Transpose 是否转置输出，默认为false
 *                  - false: 输出矩阵C按(m, n)存储，n物理空间连续
 *                  - true: 输出矩阵C按(n, m)存储，m物理空间连续
 */
void f32_f32_mat_mul(f32 *data, f32 *weight, f32 *out, int m, int n, int k, bool Transpose = false);

/**
 * 浮点向量与bit向量的内积
 * 
 * @param data    浮点向量指针
 * @param weight  bit向量指针
 * @param n       向量长度
 * @param b_start bit向量的起始位偏移量，默认为0
 * @return        内积结果（浮点数）
 */
f32 f32_u8_1_vec_in_prod(f32 *data, u8_1 *weight, int n, int b_start = 0);

/**
 * 浮点矩阵与bit矩阵的矩阵乘法
 * 
 * @param data    输入矩阵A，浮点矩阵，按(m, k)存储，k物理空间连续
 * @param weight  输入矩阵B，bit矩阵，按(n, k)存储，k物理空间连续
 * @param c       输出矩阵C
 * @param m       矩阵A的行数
 * @param n       矩阵B的行数
 * @param k       矩阵A的列数和矩阵B的列数
 * @param Transpose 是否转置输出，默认为false
 *                  - false: 输出矩阵C按(m, n)存储，n物理空间连续
 *                  - true: 输出矩阵C按(n, m)存储，m物理空间连续
 */
void f32_u8_1_mat_mul(f32 *data, u8_1 *weight, f32 *out, int m, int n, int k, bool Transpose = false);


// 打印矩阵函数
void printmat(f32 *a, int m, int n) {
    printf("\n=> Mat [%d, %d, f32]: \n", m, n);  // 打印矩阵维度
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f \t", a[i * n + j]);  // 打印矩阵元素，保留两位小数
        }
        printf("\n");
    }
}
// 打印矩阵函数
void printmat(u8_1 *a, int m, int n) {
    printf("\n=> Mat [%d, %d, u8_1]: \n", m, n);  // 打印矩阵维度
    int i,j,k=0;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++,k++) {
            printf("%d \t", (a[k/8]<<(k%8) & 128)>0);  // 打印矩阵元素，保留两位小数
        }
        printf("\n");
    }
}

// 浮点向量与浮点向量的内积函数
f32 f32_f32_vec_in_prod(f32 *data, f32 *weight, int n) {
    int i = 0;
    f32 res = 0;  // 初始化结果为0
    for (; i < n; i++, data++, weight++)  // 遍历向量元素
        res += (*data) * (*weight);  // 累加乘积
    return res;  // 返回内积结果
}

// 浮点矩阵与浮点矩阵的矩阵乘法函数
void f32_f32_mat_mul(f32 *data, f32 *weight, f32 *out, int m, int n, int k, bool Transpose) {
    int i, j;  // 索引变量
    int bk = (k + 7) / 8;  // 计算bit向量中k占用的字节数
    float *p;
    
    if (Transpose) {  // 如果需要转置输出
        for (j = 0; j < n; j++)  // 遍历矩阵B的行
            for (i = 0, p = data; i < m; i++, weight += k, p += k, out++)  // 遍历矩阵A的行
                *out = f32_f32_vec_in_prod(p, weight, k);  // 计算内积并存储到输出矩阵
    } else {  // 不转置输出
        for (i = 0; i < m; i++, data += k)  // 遍历矩阵A的行
            for (j = 0; j < n; j++, weight += k, p += k, out++)  // 遍历矩阵B的行
                *out = f32_f32_vec_in_prod(p, weight, k);  // 计算内积并存储到输出矩阵
    }
}

// 浮点向量与bit向量的内积函数
f32 f32_u8_1_vec_in_prod(f32 *data, u8_1 *weight, int n, int b_start) {
    int i = 0, j = 0;
    f32 res = 0;  // 初始化结果为0
    u8_1 temp = (*weight) << b_start;  // 将bit向量左移b_start位
    int n0 = (8 - b_start) < n ? (8 - b_start) : n;  // 计算开头不满足8位的部分长度
    
    // 处理开头不满足8位的部分
    for (; i < n0; i++, data++, temp <<= 1)
        res += (temp & 128) ? *data : 0.;  // 如果bit为1，则累加对应的浮点数
    
    // 处理中间满足8位的部分
    for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
        for (j = 0, temp = *weight; j < 8; j++, data++, temp <<= 1)  // 遍历8位
            res += (temp & 128) ? *data : 0.;  // 如果bit为1，则累加对应的浮点数
    
    // 处理结尾不满足8位的部分
    for (temp = *weight; i < n; i++, data++, temp <<= 1)
        res += (temp & 128) ? *data : 0.;  // 如果bit为1，则累加对应的浮点数
    
    return res;  // 返回内积结果
}

//// 打印矩阵
//void printm(f32 *a, int m, int n) {
//    printf("Matrix(%d,%d,f32): \n", m, n);
//    for (int i = 0; i < m; i++) {
//        for (int j = 0; j < n; j++) {
//            printf("%.2f \t", a[i * n + j]);
//        }
//        printf("\n");
//    }
//}
//
//// 浮点向量 和 bit 向量 做 点乘 得到一个浮点数
//f32 f32_f32_vec_in_prod(f32 *data, f32 *weight, int n) {
//    int  i = 0;
//    f32 res = 0;
//    for (; i < n; i++, data++, weight++)
//        res += (*data) * (*weight); // 最左侧
//    return res;
//}
//
//// 浮点矩阵 和 bit 矩阵做 矩阵乘法 得到一个新的矩阵，支持转置输出
//void f32_f32_mat_mul(f32 *data, f32 *weight, f32 *c, int m, int n, int k, bool Transpose) {
//    int i, j; // 对应 m,n
//    int bk = (k + 7) / 8; // 求 b 中 k 占用几个Btye, 用来控制 b 移动 k 步的时候 Byte 数组后移的步长
//    float *p;
//
//    if (Transpose) {
//        for (j = 0; j < n; j++)
//            for (i = 0, p = data; i < m; i++, weight += k, p += k, c++)
//                * c = f32_f32_vec_in_prod(p, weight, k );
//    } else {
//        for (i = 0; i < m; i++, data += k)
//            for (j = 0; j < n; j++, weight += k, p += k, c++)
//                * c = f32_f32_vec_in_prod(p, weight, k );
//    }
//}
//
//// 浮点向量 和 bit 向量 做 点乘 得到一个浮点数
//f32 f32_u8_1_vec_in_prod(f32 *data, u8_1 *weight, int n, int b_start) {
//    int  i = 0, j = 0;
//    f32 res = 0;
//    u8_1 temp = (*weight) << b_start;
//    int n0 = (8 - b_start) < n ? (8 - b_start) : n;
//    // 开头不恰好满足8 bit的部分，先处理
//    for (; i < n0; i++, data++, temp <<= 1)
//        res += (temp & 128) ? *data : 0.; // 最左侧
//
//    // 中间都是8 bit 的部分，循环处理
//    for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1Byte(8 bit)的乘法，Byte数组b，后移一个单位
//        for (j = 0, temp = *weight; j < 8; j++, data++, temp <<= 1)  // 每次处理一个float数，float数组a，后移
//            res += (temp & 128) ? *data : 0.; // 最左侧
//
//    // 结尾不满足8 bit 的部分，单独处理
//    for (temp = *weight; i < n; i++, data++, temp <<= 1)
//        res += (temp & 128) ? *data : 0.;
//
//    return res;
//}

// 浮点矩阵 和 bit 矩阵做 矩阵乘法 得到一个新的矩阵，支持转置输出
void f32_u8_1_mat_mul(f32 *data, u8_1 *weight, f32 *out, int m, int n, int k, bool Transpose) {
    int i, j; // 对应 m,k,n
    
    int bk = (k + 7) / 8; // 求 b 中 k 占用几个Btye, 用来控制 b 移动 k 步的时候 Byte 数组后移的步长
    float *p;
    
    if (Transpose) {
        for (j = 0; j < n; j++)
            for (i = 0, p = data; i < m; i++, p += k, c++)
                * c = f32_u8_1_vec_in_prod(p, weight + (j * k) / 8, k, (j * k) % 8);
    } else {
        for (i = 0; i < m; i++, data += k)
            for (j = 0; j < n; j++,  c++ )
                *c = f32_u8_1_vec_in_prod(data, weight + (j * k) / 8, k, (j * k) % 8);
    }
}

