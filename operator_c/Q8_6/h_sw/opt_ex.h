#include "opt.h"

// i32 和 i8 的内积函数
f32 f32_f32_vec_in_prod(f32 *data, f32 *weight, int n)
{
	f32 res = 0;								  // 初始化结果为0, 结果要除以64,解析成(-1,1)的8字节数
	for (int i = 0; i < n; i++, data++, weight++) // 遍历向量元素
		res += *data * *weight;
	return res; // 返回内积结果
}

// 浮点矩阵 和 bit 矩阵做 矩阵乘法 得到一个新的矩阵，支持转置输出
// 浮点矩阵与浮点矩阵的矩阵乘法函数
void f32_f32_mul(f32 *data, f32 *weight, f32 *out, int m, int k, int n, bool Transpose = false)
{
	if (Transpose)
	{ // 如果需要转置输出
		for (i32 j = 0; j < n; j++, weight += k)
		{ // 遍历矩阵B的行
			f32 *d = data;
			for (i32 i = 0; i < m; i++, out++, d += k)	  // 遍历矩阵A的行
				*out = f32_f32_vec_in_prod(d, weight, k); // 计算内积并存储到输出矩阵
		}
	}
	else
	{ // 不转置输出
		for (int i = 0; i < m; i++, data += k)
		{ // 遍历矩阵A的行
			f32 *w = weight;
			for (int j = 0; j < n; j++, out++, w += k)	// 遍历矩阵B的行
				*out = f32_f32_vec_in_prod(data, w, k); // 计算内积并存储到输出矩阵
		}
	}
}