#include "opt_ex.h"
#include "layer.h"
#include <malloc.h>

MatF32 conv1d_f32_i8(MatF32 in_data_mat, ConvI8 params) {
    // unzip input
    f32 *idata = in_data_mat.data;
    int length = in_data_mat.length;
    int in_channels = in_data_mat.channels;
    int out_channels = params.out_channels;
    i8 *weight = params.weight;
    int kernel = params.kernel;
    int stride = params.stride;
    int out_length = (length - kernel + 1) / stride;

    if (ERR(in_channels != params.in_channels, "in_channels != params.in_channels") ||
        ERR(out_length * out_channels < 1, "out_length * out_channels < 1") ||
        ERR(idata == NULL, "idata == NULL") ||
        ERR(weight == NULL, "weight == NULL")) {
        MatF32 res = {NULL, 0, 0};
        return res;
    }

    // make output
    f32 *odata = (f32 *) malloc(sizeof(f32) * out_length * out_channels);
    printf("%d\n", out_channels * out_length);
    MatF32 res = {odata, out_length, out_channels};

    for (int i = 0; i <= length - kernel;
         i += stride, idata += in_channels, odata += out_channels) {
        f32_i8_mul(idata,  // shape = 1, kernel*in
                   weight, // shape= o, kernel*in
                   odata,  // shape = 1, o
                   1,
                   kernel * in_channels,
                   out_channels,
                   false);
    }

    return res;
}

MatF32 avg_pool1d_f32(MatF32 in_data_mat, int kernel, int stride) {
    // unzip input data
    f32 *idata = in_data_mat.data;
    int in_length = in_data_mat.length;
    int channels = in_data_mat.channels;
    int out_length = (in_length - (kernel - stride)) / stride;


    printf("input data.shape = [%d,%d] \n", in_data_mat.length, in_data_mat.channels);
    printf("avg_pool1d params = [%d,%d] \n", kernel, stride);

    if (ERR(out_length * channels < 1, "out_length * out_channels < 1") ||
        ERR(idata == NULL, "idata == NULL")) {
        MatF32 res = {NULL, 0, 0};
        return res;

    }
    // make output data
    f32 *odata = (f32 *) malloc(sizeof(f32) * out_length * channels);
//    printf("%d\n", out_length * channels);
    MatF32 res = {odata, out_length, channels};

    // do
    for (int c = 0; c < channels; c++) {
        f32 *o = odata + c;
        for (int l = 0; l <= in_length - kernel; l += stride, o += channels) {
            f32 avg = 0;
            for (int k = 0; k < kernel; k++)
                avg += idata[(l + k) * channels + c];
            *o = avg / kernel;
        }
    }

    // free input
//    free(in_data_mat.data);
//    in_data_mat.data = NULL;


    printf("output data.shape=[%d, %d] \n\n", out_length, channels);
    // return output
    return res;
}

void printCI8(const char *name, ConvI8 mat) {
    printf("\n=> Mat %stride %p [%d, %d, %d]: \n", name,
           mat.weight,
           mat.out_channels,
           mat.kernel,
           mat.in_channels
    ); // 打印矩阵维度
    int l, c;
    i8 *w = mat.weight;
    for (l = 0; l < mat.out_channels; l++) {
        for (c = 0; c < mat.kernel * mat.in_channels; c++, w++)
            printf("%.7f\t", *w / 64.0); // 打印矩阵元素，保留两位小数
        printf("\n");
    }
}
int upsample_linear(f32 *data, f32 *out, int length, int channels, int factor)
{
	// 首位两端采用跳过策略，中间的部分用插值生成
	if (factor % 2 == 0)
	{ // 偶数的话，所有数都不在插值的中心点，插值之后的长度为 length*factor - factor
		for (int c = 0; c < channels; c++)
		{
			f32 *o = out + c;
			f32 *d = data + c;
			for (int l = 0; l < length - 1; l++, d += channels)
			{
				f32 left = d[0], right = d[channels];
				f32 d = right - left;
				for (int f = 0; f < factor; f++, o += channels)
				{
					*o = left + d * (1 + 2 * f) / (2 * factor);
				}
			}
		}
		return length * factor - factor;
	}
	else
	{ // 奇数的话，所有数都在插值的中心点，插值之后的长度为 length*factor - factor + 1
		for (int c = 0; c < channels; c++)
		{
			f32 *o = out + c;
			f32 *d = data + c;
			for (int l = 0; l < length - 1; l++, d += channels)
			{
				f32 left = d[0], right = d[channels];
				f32 d = right - left;
				for (int f = 0; f < factor; f++, o += channels)
				{
					*o = left + d * f / factor;
				}
			}
			*o = *d;
		}
		return length * factor - factor + 1;
	}
}
int conv1d(i32 *data, b8 *weight, i32 *bias, i32 *out, int length, int in_channels, int out_channels, int kernel, int stride)
{
	int res = (length - kernel + 1) / stride;
	if (res < 1)
		return res;

	for (int i = 0; i < length; i += stride, data += in_channels)
	{
		// 在滑动窗口内进行矩阵乘法
		i32_b8_mul(data,   // shape = 1, kernel*in
				   weight, // shape= o, kernel*in
				   out,	   // shape = 1, o
				   1,
				   kernel * in_channels,
				   out_channels);

		// 添加偏执项
		if (bias == NULL)
		{
			out += out_channels;
			continue;
		}
		i32 *b = bias;
		for (int j = 0; j < out_channels; j++, out++, b++)
		{
			*out += *b;
		}
	}
	return res;
}

int conv1d(i32 *data, i8 *weight, i32 *bias, i32 *out, int length, int in_channels, int out_channels, int kernel, int stride)
{
	int res = (length - kernel + 1) / stride;
	if (res < 1)
		return res;

	for (int i = 0; i < length; i += stride, data += in_channels)
	{
		// 在滑动窗口内进行矩阵乘法
		i32_i8_mul(data,   // shape = 1, kernel*in
				   weight, // shape= o, kernel*in
				   out,	   // shape = 1, o
				   1,
				   kernel * in_channels,
				   out_channels);

		// 添加偏执项
		if (bias == NULL)
		{
			out += out_channels;
			continue;
		}
		// 添加偏执项
		i32 *b = bias;
		for (int j = 0; j < out_channels; j++, out++, b++)
		{
			*out += *b;
		}
	}
	return res;
}

int conv1d(f32 *data, f32 *weight, f32 *bias, f32 *out, int length, int in_channels, int out_channels, int kernel, int stride)
{
	int res = (length - kernel + 1) / stride;
	if (res < 1)
		return res;

	for (int i = 0; i < length; i += stride, data += in_channels)
	{
		f32_f32_mul(data,	// shape = 1, kernel*in
					weight, // shape= o, kernel*in
					out,	// shape = 1, o
					1,
					kernel * in_channels,
					out_channels);
		// 添加偏执项
		if (bias == NULL)
		{
			out += out_channels;
			continue;
		}
		f32 *b = bias;
		for (int j = 0; j < out_channels; j++, out++, b++)
		{
			*out += *b;
		}
	}
	return res;
}