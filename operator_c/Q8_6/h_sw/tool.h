#include "opt.h"

//// 打印矩阵函数
//void printmat(const char *name, b8 *a, int m, int n)
//{
//    printf("\n=> Mat %s[%d, %d, b8]: \n", name, m, n); // 打印矩阵维度
//    int i, j, k = 0;
//    for (i = 0; i < m; i++)
//    {
//        for (j = 0; j < n; j++, k++)
//        {
//            printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0 ? 1 : -1); // 打印矩阵元素，保留两位小数
//        }
//        printf("\n");
//    }
//}
//// 打印矩阵函数
//
//// 打印矩阵函数
//void printmat(const char *name, i32 *a, int m, int n)
//{
//    printf("\n=> Mat %s %p [%d, %d, i32]: \n", name, a, m, n); // 打印矩阵维度
//    int i, j, k = 0;
//    for (i = 0; i < m; i++)
//    {
//        for (j = 0; j < n; j++, k++)
//        {
//            printf("%.6f\t", a[i * n + j] / 64.); // 打印矩阵元素，保留两位小数
//        }
//        printf("\n");
//    }
//}
//// 打印矩阵函数
//void printmat(const char *name, f32 *a, int m, int n)
//{
//    printf("\n=> Mat %s %p [%d, %d, f32]: \n", name, a, m, n); // 打印矩阵维度
//    int i, j, k = 0;
//    for (i = 0; i < m; i++)
//    {
//        for (j = 0; j < n; j++, k++)
//        {
//            printf("%.6f\t", a[i * n + j]); // 打印矩阵元素，保留两位小数
//        }
//        printf("\n");
//    }
//}
//// 打印矩阵函数
//void printmat(const char *name, f32 *a, int m, int n, int q)
//{
//    printf("\n=> Mat %s %p [%d, %d, %d, f32]: \n", name, a, m, n, q); // 打印矩阵维度
//    int i, j, k = 0;
//    for (i = 0; i < m; i++)
//    {
//        for (j = 0; j < n; j++)
//        {
//            for (k = 0; k < q; k++)
//            {
//                printf("%.6f\t", a[i * (n * q) + j * q + k]); // 打印矩阵元素，保留两位小数
//            }
//            printf("\n");
//        }
//        printf("\n");
//    }
//}
//// 打印矩阵函数
//void printmat(const char *name, i8 *a, int m, int n, int q)
//{
//    printf("\n=> Mat %s %p [%d, %d, %d, i8]: \n", name, a, m, n, q); // 打印矩阵维度
//    int i, j, k = 0;
//    for (i = 0; i < m; i++)
//    {
//        for (j = 0; j < n; j++)
//        {
//            for (k = 0; k < q; k++)
//            {
//                printf("%.6f\t", a[i * (n * q) + j * q + k] / 64.); // 打印矩阵元素，保留两位小数
//            }
//            printf("\n");
//        }
//        printf("\n");
//    }
//}
