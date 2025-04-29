#include "f32matrix.h"


void glinear(f32 *in_data, f32 *out_data, f32 *weight, f32 *bias,
             int length,  int c_in, int c_out,  int group = 1);

void glinear(f32 *in_data, f32 *out_data, f32 *weight, f32 *bias,
             int length,  int c_in, int c_out,  int group ) {
	//    in_data:   [length, c_in]
	//    weight: [c_out, c_in//g]
	//    in_data:   [length, g, c_in//g]
	//    weight: [c_out//g, g, c_in//g]
	// 	  out_data
	int i = 0, j = 0, g = 0;
	int g_in = c_in / group;
	int g_out = c_out / group;
	if (group == 1) {
		f32_f32_mat_mul(in_data, weight, out_data, length, c_out, c_in);
	} else {
		for (i = 0; i < length; i++, in_data += c_in) // 遍历length
			for (g = 0; g < group; g++ ) { // 遍历分组数量
				f32* group_weight = weight + g * g_out * g_in; // 重置 weight 偏移
				f32* group_out = out_data + i * c_out + g * g_out; // 重置 out_data 偏移
				for (j = 0; j < g_out; j++, weight += g_in, out_data++) { // 组内遍历
					*out_data = f32_f32_vec_in_prod(in_data, weight, g_in);

				}
			}
	}
}

void f32topdinsty(f32 *in_data, int length,  int channel) {
	for (int i = 0; i < length; i++) {
		f32 e = 0;
		f32 s = 0;

		//	1 求平均值
		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
		e  = e / channel;

		// 2 裁剪平均值以下的部分，并减去平均值求和
		for (int j = 0; j < channel; j++) {
			if (in_data[i * channel + j] <= e) {
				in_data[i * channel + j] = 0.0;
			} else {
				in_data[i * channel + j] -= e;
				s += in_data[i * channel + j];
			}
		}

		// 3 所有元素都除以和，变成概率密度
		for (int j = 0; j < channel; j++) {
			in_data[i * channel + j] /= s;
		}
	}
}

void i32topdinsty(i32 *io_data, int length,  int channel) {
//	return;
	for (int i = 0; i < length; i++) {
		i32 e = 0;
		i32 s = 0;

		//	1 求平均值
		for (int j = 0; j < channel; j++) e += io_data[i * channel + j];
		e  = e / channel;

		// 2 裁剪平均值以下的部分，并减去平均值求和
		for (int j = 0; j < channel; j++) {
			if (io_data[i * channel + j] <= e) {
				io_data[i * channel + j] = 0;
			} else {
				io_data[i * channel + j] -= e;
				s += io_data[i * channel + j];
			}
		}

		// 3 所有元素都除以和，变成概率密度,1=127,0=0
		for (int j = 0; j < channel; j++) {
			io_data[i * channel + j] = io_data[i * channel + j] *128/ s;
		}
	}
}
