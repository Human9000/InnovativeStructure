#ifndef H_NET_H
#define H_NET_H

#include "layer.h"

i8 d1_w[48] = {64, 41, 11, -64, 14, 37, -64, -64, -41, 55, 64, 64, 23, 9, 4, 8, 64, 53, -22, -5, -5, 6, 20, 28, 64, -10,
               17, -55, -18, 25, -18, 21, 9, 12, 22, 29, -51, 43, -14, 17, 30, -17, 22, 28, 7, -15, -16, -9};
i8 d2_w[48] = {-13, 64, 64, -64, -5, -64, -64, 64, 14, -64, -6, 64, -3, -64, 25, -44, -29, 62, -21, -13, -27, 64, -23,
               -42, 47, -64, -64, 64, -27, -44, -1, 29, -8, 64, 64, -28, 5, -52, -19, 5, -20, 50, -23, 0, -10, -64, 21,
               12};
b8 d3_w[96] = {25, 182, 104, 73, 145, 153, 17, 17, 55, 119, 57, 200, 220, 137, 54, 1, 219, 228, 238, 204, 59, 174, 143,
               255, 73, 177, 21, 156, 141, 220, 186, 132, 27, 81, 46, 236, 49, 17, 185, 255, 244, 19, 102, 224, 85, 51,
               214, 108, 187, 94, 238, 255, 249, 200, 51, 45, 136, 12, 223, 255, 102, 203, 244, 137, 115, 185, 179, 101,
               3, 51, 94, 230, 155, 108, 153, 170, 166, 25, 110, 131, 181, 75, 177, 25, 85, 15, 255, 216, 136, 205, 17,
               182, 108, 238, 206, 230};
b8 d6_w[32] = {10, 51, 74, 180, 137, 4, 32, 192, 86, 80, 133, 139, 192, 219, 150, 69, 230, 95, 139, 125, 201, 253, 218,
               207, 254, 247, 238, 250, 190, 222, 183, 252};
i8 d7_w[48] = {-8, 32, 38, -8, -47, 11, 3, 15, 48, -23, -35, -9, 53, -27, 15, 20, -15, 64, 36, 16, -24, -61, -64, -30,
               52, 37, -25, -17, 8, -25, 3, -23, -64, -7, 31, 36, -30, 18, 25, 14, 58, -10, -28, 42, -42, 0, 9, -64};
b8 d8_w[96] = {158, 128, 255, 40, 144, 253, 27, 116, 40, 69, 17, 230, 69, 57, 114, 137, 163, 0, 151, 41, 46, 227, 56,
               154, 43, 142, 51, 127, 180, 81, 16, 73, 44, 127, 51, 135, 43, 141, 236, 255, 123, 46, 151, 46, 51, 16,
               223, 93, 143, 247, 51, 159, 251, 174, 151, 38, 24, 43, 136, 154, 239, 50, 253, 36, 100, 70, 210, 130, 16,
               161, 63, 0, 179, 62, 120, 45, 211, 174, 1, 218, 86, 151, 44, 110, 43, 133, 14, 153, 105, 22, 143, 254,
               243, 119, 219, 46};
b8 d11_w[32] = {244, 127, 72, 110, 169, 11, 142, 58, 2, 206, 67, 28, 88, 178, 135, 117, 90, 150, 82, 55, 240, 207, 232,
                13, 233, 107, 195, 174, 75, 194, 190, 164};
i8 d12_w[48] = {-49, -13, -13, 29, 64, -8, -36, -26, 15, 64, 30, 13, -3, 29, 12, -36, -3, -64, -13, -25, -9, 56, 20, 64,
                -45, -22, 31, 18, 64, 9, 13, 11, -16, 45, -21, -64, 38, -53, -52, -15, -59, 37, -12, 29, 11, -22, 64,
                -9};
b8 d13_w[96] = {32, 38, 61, 225, 22, 141, 137, 180, 227, 28, 168, 179, 115, 217, 222, 244, 13, 96, 155, 239, 160, 217,
                139, 94, 240, 78, 253, 12, 129, 26, 76, 161, 14, 137, 180, 173, 9, 190, 64, 147, 52, 202, 254, 6, 173,
                244, 26, 147, 72, 238, 189, 2, 170, 163, 40, 175, 224, 181, 14, 45, 73, 184, 177, 17, 103, 246, 144,
                247, 242, 224, 84, 200, 68, 48, 153, 55, 230, 127, 95, 35, 19, 242, 1, 12, 244, 6, 127, 9, 138, 207,
                157, 227, 46, 95, 40, 163};
b8 u1_w[64] = {207, 203, 126, 49, 120, 253, 253, 127, 89, 130, 195, 167, 209, 194, 84, 21, 172, 84, 2, 234, 28, 192, 78,
               33, 96, 178, 132, 106, 71, 66, 50, 198, 166, 255, 253, 94, 239, 63, 191, 222, 112, 62, 160, 202, 7, 131,
               203, 129, 82, 201, 124, 20, 233, 9, 176, 124, 253, 254, 146, 175, 59, 255, 206, 163};
i8 u2_w[64] = {37, 64, -64, 64, -64, -64, 64, 64, -64, 64, 64, 30, -64, 64, -64, 64, 64, -64, 64, -64, -39, -64, -64,
               64, 64, -64, 64, 64, -64, -64, 64, -64, -64, -64, 64, 64, -64, 64, -64, -64, -64, -64, 64, -64, -64, 64,
               64, -64, -64, -64, -64, -64, -64, 2, 64, -64, 64, 64, -64, -64, -64, -64, -64, 64};
b8 u3_w[64] = {241, 52, 89, 116, 220, 209, 50, 100, 210, 205, 96, 205, 126, 166, 115, 221, 72, 203, 147, 225, 48, 229,
               48, 72, 229, 229, 127, 46, 209, 180, 55, 115, 221, 220, 203, 1, 72, 48, 210, 219, 149, 41, 102, 50, 122,
               206, 127, 45, 220, 255, 177, 54, 210, 41, 206, 219, 147, 46, 62, 72, 54, 205, 91, 72};
b8 u6_w[64] = {253, 167, 223, 243, 254, 238, 166, 27, 78, 80, 43, 22, 88, 241, 228, 247, 239, 199, 207, 210, 76, 127,
               150, 91, 45, 217, 65, 88, 107, 154, 1, 21, 17, 41, 177, 0, 158, 241, 206, 245, 175, 213, 46, 239, 251,
               190, 57, 254, 114, 137, 143, 84, 57, 247, 168, 51, 194, 7, 66, 198, 231, 19, 52, 6};
i8 u7_w[32] = {16, 64, 1, -64, -14, 10, 20, 12, -16, -10, -30, -2, 50, 64, -5, -12, 22, -64, 22, -15, 28, 8, -64, -1,
               12, 8, 35, 11, 34, 3, -3, -26};

MatI32 net(f32 *data, int length, int channel) {
    // 输入的数据必须是 length * 12
    MatF32 mf;
    MatI32 mi;
    // pad
    mf = pad_length(F32(data, length, channel), 232, 100, false); //	x = F.pad(x, (232, 100))

    // down
    mf = avg_pool1d_f32(mf, 4, 4);               // nn.AvgPool1d(4, stride=4)
    mf = conv1d_f32_i8(mf,
                       CI8(d1_w, 12, 4, 1, 1));  // QConv1d(12, 4, 1, 1, q_input_bias=False, b_weight=False),
    mi = F32_2_I32(mf);                       // q_input_bias = True  ↓
    mi = conv1d_i32_i8(mi, CI8(d2_w, 4, 4, 3, 1));   // QConv1d(4, 4, 3, 1, b_weight=False),
    mi = conv1d_i32_b8(mi, CB8(d3_w, 4, 4, 3, 1), true);   // QConv1d(4, 64, 3, 1,)
    mi = avg_pool1d_i32(mi, 7, 5);               // nn.AvgPool1d(7, stride=5),
    mi = top_density(mi);                   // TopDensity(dim=1),
    mi = conv1d_i32_b8(mi, {d6_w, 64, 4, 1, 1});  // QConv1d(64, 4, 1, 1, ),
    mi = conv1d_i32_b8(mi, {d7_w, 4, 4, 3, 1});   // QConv1d(4, 4, 3, 1, b_weight=False),
    mi = conv1d_i32_b8(mi, {d8_w, 4, 64, 3, 1});  // QConv1d(4, 64, 3, 1, ),
    mi = avg_pool1d(mi, 3, 1);               // nn.AvgPool1d(3, stride=1),
    mi = top_density(mi);                   // TopDensity(dim=1),
    mi = conv1d_i32_b8(mi, {d11_w, 64, 4, 1, 1}); // QConv1d(64, 4, 1, 1, ),
    mi = conv1d_i32_b8(mi, {d12_w, 4, 4, 3, 1});  // QConv1d(4, 4, 3, 1, b_weight=False),
    mi = conv1d_i32_b8(mi, {d13_w, 4, 64, 3, 1}); // QConv1d(4, 64, 3, 1,),
    mi = avg_pool1d(mi, 3, 1);               // nn.AvgPool1d(3, stride=1),
    mi = top_density(mi);                   // TopDensity(dim=1),

    // up
    mi = upsample_linear(mi, 5);          // Upsample(5),
    mi = conv1d_i32_b8(mi, {u1_w, 64, 8, 1, 1}); // QConv1d(64, 8, 1 ),
    mi = conv1d_i32_b8(mi, {u2_w, 8, 8, 1, 1});  // QConv1d(4, 4, 3, 1, b_weight=False),
    mi = conv1d_i32_b8(mi, {u3_w, 8, 64, 1, 1}); // QConv1d(4, 64, 3, 1,),
    mi = top_density(mi);                  // TopDensity(dim=1),
    mi = upsample_linear(mi, 4);          // Upsample(4),
    mi = conv1d_i32_b8(mi, {u6_w, 64, 8, 1, 1}); // QConv1d(64, 8, 1 ),
    mi = conv1d_i32_b8(mi, {u7_w, 8, 4, 1, 1});  // QConv1d(8, 8, 1, b_weight=False),

    return mi;
}

#endif
