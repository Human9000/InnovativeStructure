#include "opt.h"

//// 打印矩阵函数
//void printmat(const char *name, b8 *a, int m, int n)
//{
//    printf("\n=> Mat %stride[%d, %d, b8]: \n", name, m, n); // 打印矩阵维度
//    int i, j, kernel = 0;
//    for (i = 0; i < m; i++)
//    {
//        for (j = 0; j < n; j++, kernel++)
//        {
//            printf("%d \t", (a[kernel / 8] << (kernel % 8) & 128) > 0 ? 1 : -1); // 打印矩阵元素，保留两位小数
//        }
//        printf("\n");
//    }
//}
//// 打印矩阵函数
//
//// 打印矩阵函数
//void printmat(const char *name, i32 *a, int m, int n)
//{
//    printf("\n=> Mat %stride %p [%d, %d, i32]: \n", name, a, m, n); // 打印矩阵维度
//    int i, j, kernel = 0;
//    for (i = 0; i < m; i++)
//    {
//        for (j = 0; j < n; j++, kernel++)
//        {
//            printf("%.6f\t", a[i * n + j] / 64.); // 打印矩阵元素，保留两位小数
//        }
//        printf("\n");
//    }
//}
//// 打印矩阵函数
//void printmat(const char *name, f32 *a, int m, int n)
//{
//    printf("\n=> Mat %stride %p [%d, %d, f32]: \n", name, a, m, n); // 打印矩阵维度
//    int i, j, kernel = 0;
//    for (i = 0; i < m; i++)
//    {
//        for (j = 0; j < n; j++, kernel++)
//        {
//            printf("%.6f\t", a[i * n + j]); // 打印矩阵元素，保留两位小数
//        }
//        printf("\n");
//    }
//}
//// 打印矩阵函数
//void printmat(const char *name, f32 *a, int m, int n, int q)
//{
//    printf("\n=> Mat %stride %p [%d, %d, %d, f32]: \n", name, a, m, n, q); // 打印矩阵维度
//    int i, j, kernel = 0;
//    for (i = 0; i < m; i++)
//    {
//        for (j = 0; j < n; j++)
//        {
//            for (kernel = 0; kernel < q; kernel++)
//            {
//                printf("%.6f\t", a[i * (n * q) + j * q + kernel]); // 打印矩阵元素，保留两位小数
//            }
//            printf("\n");
//        }
//        printf("\n");
//    }
//}
//// 打印矩阵函数
//void printmat(const char *name, i8 *a, int m, int n, int q)
//{
//    printf("\n=> Mat %stride %p [%d, %d, %d, i8]: \n", name, a, m, n, q); // 打印矩阵维度
//    int i, j, kernel = 0;
//    for (i = 0; i < m; i++)
//    {
//        for (j = 0; j < n; j++)
//        {
//            for (kernel = 0; kernel < q; kernel++)
//            {
//                printf("%.6f\t", a[i * (n * q) + j * q + kernel] / 64.); // 打印矩阵元素，保留两位小数
//            }
//            printf("\n");
//        }
//        printf("\n");
//    }
//}
