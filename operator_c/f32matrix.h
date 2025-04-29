#include<stdio.h>
#include <stdint.h>

//#define f32 float
//#define u8_1 unsigned char

typedef float f32;                // 定义浮点数类型别名
typedef unsigned char b8;         // 定义无符号8位类型别名
typedef char i8;				  // 定义有符号8位整形
typedef int16_t i16;				  // 定义有符号32位整形
typedef int32_t i32;				  // 定义有符号32位整形

void printmat(f32 *a, int m, int n);  // 声明打印矩阵函数
void printmat(b8 *a, int m, int n);  // 声明打印矩阵函数

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
f32 f32_u8_1_vec_in_prod(f32 *data, b8 *weight, int n, int b_start = 0);

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
void f32_u8_1_mat_mul(f32 *data, b8 *weight, f32 *out, int m, int n, int k, bool Transpose = false);


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
void printmat(b8 *a, int m, int n) {
	printf("\n=> Mat [%d, %d, u8_1]: \n", m, n);  // 打印矩阵维度
	int i, j, k = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++, k++) {
			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0); // 打印矩阵元素，保留两位小数
		}
		printf("\n");
	}
}
// 打印矩阵函数
void printmat(i32 *a, int m, int n) {
	printf("\n=> Mat [%d, %d, i32]: \n", m, n);  // 打印矩阵维度
	int i, j, k = 0;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++, k++) {
			printf("%d\t", a[i*n+j]); // 打印矩阵元素，保留两位小数
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
	if (Transpose) {  // 如果需要转置输出
		for (int j = 0; j < n; j++, weight += k) { // 遍历矩阵B的行
			f32 * p = data;
			for (int i = 0; i < m; i++, p += k, out++)  // 遍历矩阵A的行
				*out = f32_f32_vec_in_prod(p, weight, k);  // 计算内积并存储到输出矩阵
		}
	} else {  // 不转置输出
		for (int i = 0; i < m; i++, data += k) {		 // 遍历矩阵A的行
			f32 *p = weight;
			for (int j = 0; j < n; j++, p += k, out++) // 遍历矩阵B的行
				*out = f32_f32_vec_in_prod(data, p, k);  // 计算内积并存储到输出矩阵
		}
	}
}

// 浮点向量与bit向量的内积函数
f32 f32_u8_1_vec_in_prod(f32 *data, b8 *weight, int n, int b_start) {
	int i = 0, j = 0;
	f32 res = 0;  // 初始化结果为0
	b8 p = (*weight) << b_start;  // 将bit向量左移b_start位
	int n0 = (8 - b_start) < n ? (8 - b_start) : n;  // 计算开头不满足8位的部分长度

	// 处理开头不满足8位的部分
	for (; i < n0; i++, data++, p <<= 1)
		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数

	// 处理中间满足8位的部分
	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数

	// 处理结尾不满足8位的部分
	for (p = *weight; i < n; i++, data++, p <<= 1)
		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数

	return res;  // 返回内积结果
}

// 浮点矩阵 和 bit 矩阵做 矩阵乘法 得到一个新的矩阵，支持转置输出
void f32_u8_1_mat_mul(f32 *data, b8 *weight, f32 *out, int m, int n, int k, bool Transpose) {
	if (Transpose) {
		for (int j = 0; j < n; j++) {
			float *p = data; //
			b8  *q = weight + (j * k) / 8; // 移动 weight 的索引
			int start_q = (j * k) % 8; //

			for (int i = 0; i < m; i++, p += k, out++) {
				*out = f32_u8_1_vec_in_prod(p, q, k, start_q);
			}
		}

	} else {
		for (int i = 0; i < m; i++, data += k)
			for (int j = 0; j < n; j++,  out++)
				*out = f32_u8_1_vec_in_prod(data,  weight + (j * k) / 8,  k,  (j * k) % 8);
	}
}




// 浮点向量与浮点向量的内积函数
i32 i8_i8_vec_in_prod(i8 *data, i8 *weight, int n) {
	int i = 0;
	i32 res = 0;  // 初始化结果为0, 结果要除以127,解析成(-1,1)的8字节数
	for (; i < n; i++, data++, weight++)  // 遍历向量元素
		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	return (res/127);  // 返回内积结果
}



// 浮点向量与浮点向量的内积函数
i32 i8_b8_vec_in_prod(i8 *data, b8 *weight, int n, int b_start) {
	int i = 0, j=0;
	i32 res = 0;  // 初始化结果为0, 结果要除以127,解析成(-1,1)的8字节数
	
	b8 p = (*weight) << b_start;  // 将bit向量左移b_start位
	int n0 = (8 - b_start) < n ? (8 - b_start) : n;  // 计算开头不满足8位的部分长度
	
	// 处理开头不满足8位的部分
	for (; i < n0; i++, data++, p <<= 1)
		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	
	// 处理中间满足8位的部分
	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	
	// 处理结尾不满足8位的部分
	for (p = *weight; i < n; i++, data++, p <<= 1)
		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	
	return res;  // 返回内积结果
}

