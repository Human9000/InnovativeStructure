#include "f32matrix.h"

void conv1d(f32 *in_data, f32 *out_data, f32 *weight, f32 *bias,
    int length,  int c_in, int c_out,
    int k_size = 1, int stride = 1, int group = 1);

void linear(f32 *in_data, f32 *out_data, f32 *weight, f32 *bias,
    int length,  int c_in, int c_out,  int group = 1);
 
void linear(f32 *in_data, f32 *out_data, f32 *weight, f32 *bias,
    int length,  int c_in, int c_out,  int group ){
    //    in_data:   [length, c_in]
    //    weight: [c_out, c_in]
    //    in_data:   [length * g, c_in//g]
    //    weight: [c_out, c_in//g]
    int i=0,j=0;
    int c_in_g = c_in / group;
    if (group == 1){
        f32_f32_mat_mul(in_data, weight, out_data, length, c_out, c_in);
    }else{
        for(i=0; i<length; i++, in_data += c_in_g)
            for(j=0;j<c_out;j++, weight+=c_in_g, out_data++){ 
                *out_data = f32_f32_vec_in_prod(in_data, weight, c_in / group);
        }
    }
}
