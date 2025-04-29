#ifndef H_OPT_H
#define H_OPT_H 

#include <stdio.h>
#include <stdint.h>
typedef uint8_t b8;	 // 定义无符号8位类型别名
typedef int8_t i8;	 // 定义有符号8位整形
typedef int32_t i32; // 定义有符号32位整形
typedef int64_t i64; // 定义有符号64位整形
typedef float f32;	 // 定义有符号32位整形
typedef uint8_t bool_; // 定义无符号8位整形
#define true 1
#define false 0


typedef struct {
    i32 *data;
    int length;
    int channels;
} MatI32;

typedef struct {
    f32 *data;
    int length;
    int channels;
} MatF32;

MatI32 I32(i32 *data, int length, int channels) {
    MatI32 res = {data, length, channels};
    return res;
}

MatF32 F32(f32 *data, int length, int channels) {
    MatF32 res = {data, length, channels};
    return res;
}


void printMF32(const char *name, MatF32 mat) {
    printf("\n=> Mat %s %p [%d, %d]: \n", name, mat.data, mat.length, mat.channels); // 打印矩阵维度
    int l, c, k = 0;
    for (l = 0; l < mat.length; l++) {
        for (c = 0; c < mat.channels; c++)
            printf("%.7f,\t", mat.data[l * mat.channels + c]); // 打印矩阵元素，保留两位小数
        printf("\n");
    }
}

void printMI32(const char *name, MatI32 mat) {
    printf("\n=> Mat %s %p [%d, %d]: \n", name, mat.data, mat.length, mat.channels); // 打印矩阵维度
    int l, c;
    for (l = 0; l < mat.length; l++) {
        for (c = 0; c < mat.channels; c++)
        {
            i32 temp = (i32)(mat.data[l * mat.channels + c]);
            f32 temp2 = (f32) temp / 64;
            printf("%f \t",  temp2); // 打印矩阵元素，保留两位小数
        }
        printf("\n");
    }
}

void f32_2_i32(f32 *in, i32 *out, int length)
{ 
	for (int i = 0; i < length; i++, in++, out++)
	{ 
		*out = (i32) (*in * 64);
	}
}

void i32_2_f32(i32 *in, f32 *out, int length)
{
	for (int i = 0; i < length; i++, in++, out++)
	{
		*out = (f32)*in / 64.;
	}
}

void i8_2_f32(i8 *in, f32 *out, int length)
{
	for (int i = 0; i < length; i++, in++, out++)
	{
		*out = (f32)*in / 64.;
	}
}

void b8_2_f32(b8 *in, f32 *out, int length)
{
	for (int i = 0; i < length; i++, in++, out++)
	{
		*out = in[i/8] & (128>>(i%8)) ? 1:0;
	}
}


MatF32 I32_2_F32(MatI32 imat) {
    f32 *odata = (f32 *) imat.data;
    MatF32 res = {odata, imat.length, imat.channels};
    i32_2_f32(imat.data, odata, imat.length * imat.channels);
    return res;
}

MatI32 F32_2_I32(MatF32 imat) {
    i32 *odata = (i32 *) imat.data;
    MatI32 res = {odata, imat.length, imat.channels};
    f32_2_i32(imat.data, odata, imat.length * imat.channels);
    return res;
}

// i32 和 i8 的内积函数
f32 f32_i8_vec_in_prod(f32 *data, i8 *weight, int n)
{
	f32 res = 0;								  // 初始化结果为0, 结果要除以64,解析成(-1,1)的8字节数
	for (int i = 0; i < n; i++, data++, weight++) // 遍历向量元素
		res += *data * *weight;
	return res / 64; // 返回内积结果
}

// 浮点矩阵 和 bit 矩阵做 矩阵乘法 得到一个新的矩阵，支持转置输出
// 浮点矩阵与浮点矩阵的矩阵乘法函数
void f32_i8_mul(f32 *data, i8 *weight, f32 *out, int m, int k, int n, bool_ Transpose)
{
	if (Transpose)
	{ // 如果需要转置输出
		for (i32 j = 0; j < n; j++, weight += k)
		{ // 遍历矩阵B的行
			f32 *d = data;
			for (i32 i = 0; i < m; i++, out++, d += k)	 // 遍历矩阵A的行
				*out = f32_i8_vec_in_prod(d, weight, k); // 计算内积并存储到输出矩阵
		}
	}
	else
	{ // 不转置输出
		for (int i = 0; i < m; i++, data += k)
		{ // 遍历矩阵A的行
			i8 *w = weight;
			for (int j = 0; j < n; j++, out++, w += k) // 遍历矩阵B的行
				*out = f32_i8_vec_in_prod(data, w, k); // 计算内积并存储到输出矩阵
		}
	}
}

// i32 和 i8 的内积函数
i32 i32_i8_vec_in_prod(i32 *data, i8 *weight, int n)
{
	int res = 0;								  // 初始化结果为0, 结果要除以64,解析成(-1,1)的8字节数
//    printMI32("in:",I32(data, 1, n));
	for (int i = 0; i < n; i++, data++, weight++) // 遍历向量元素
        res += *data * *weight;
    res = res / 64;
//    printf("out: %d \n", res);
    return res ; // 返回内积结果
}

// 浮点矩阵 和 bit 矩阵做 矩阵乘法 得到一个新的矩阵，支持转置输出
// 浮点矩阵与浮点矩阵的矩阵乘法函数
void i32_i8_mul(i32 *data, i8 *weight, i32 *out, int m, int k, int n, bool_ Transpose  )
{
	if (Transpose)
	{ // 如果需要转置输出
		for (i32 j = 0; j < n; j++, weight += k)
		{														  // 遍历矩阵B的行
			for (i32 i = 0, *d = data; i < m; i++, out++, d += k) // 遍历矩阵A的行
				*out = i32_i8_vec_in_prod(d, weight, k);		  // 计算内积并存储到输出矩阵
		}
	}
	else
	{ // 不转置输出
		for (int i = 0; i < m; i++, data += k)
		{ // 遍历矩阵A的行
			i8 *w = weight;
			for (int j = 0; j < n; j++, out++, w += k) // 遍历矩阵B的行
            {
                *out = i32_i8_vec_in_prod(data, w, k); // 计算内积并存储到输出矩阵
            }

		}
	}
}

// i32 与 b8 的内积函数
i32 i32_b8_vec_in_prod(i32 *data, b8 *weight, int n, int b_start)
{
	int i = 0, j = 0;
	i32 res = 0; // 初始化结果为0, 结果要除以127,解析成(-1,1)的 8字节数

	b8 p = (*weight) << b_start;					// 将bit向量左移b_start位
	int n0 = (8 - b_start) < n ? (8 - b_start) : n; // 计算开头不满足8位的部分长度

	// 处理开头不满足8位的部分
	for (; i < n0; i++, data++, p <<= 1)
		res += (p & 128) ? *data : -*data; // 如果bit为1，则累加对应的浮点数

	// 处理中间满足8位的部分
	for (weight += 1; i < n - 7; i += 8, weight++)			  // 每次处理1字节（8位）
		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1) // 遍历8位
			res += (p & 128) ? *data : -*data;				  // 如果bit为1，则累加对应的浮点数

	// 处理结尾不满足8位的部分
	for (p = *weight; i < n; i++, data++, p <<= 1)
		res += (p & 128) ? *data : -*data; // 如果bit为1，则累加对应的浮点数

	return res; // 返回内积结果, bit 位解析为-1，+1,不需要变成结果的精度
}

// 浮点矩阵 和 bit 矩阵做 矩阵乘法 得到一个新的矩阵，支持转置输出
// 浮点矩阵与浮点矩阵的矩阵乘法函数
void i32_b8_mul(i32 *data, b8 *weight, i32 *out, int m, int k, int n, bool_ Transpose  )
{
	if (Transpose)
	{
		for (int j = 0; j < n; j++)
		{
			i32 *d = data;
			b8 *w = weight + (j * k) / 8;
			i32 s = (j * k) % 8;
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



// 定义一个名为 ERR 的函数，该函数返回一个布尔值，并接受一个布尔值和一个可选的字符串参数
bool_ ERR(bool_ state, const char *info  )
{
    if (state){
        printf(info);
        printf("\n");
    }
    return state;
}

#endif
