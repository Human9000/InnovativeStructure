	.file	"f32matrix_test.cpp"
 # GNU C++17 (x86_64-posix-seh, Built by MinGW-Builds project) version 11.4.0 (x86_64-w64-mingw32)
 #	compiled by GNU C version 11.4.0, GMP version 6.2.1, MPFR version 4.1.0, MPC version 1.2.1, isl version isl-0.25-GMP

 # GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
 # options passed: -mtune=core2 -march=nocona -fno-asynchronous-unwind-tables
	.text
	.section	.text$_Z6printfPKcz,"x"
	.linkonce discard
	.globl	_Z6printfPKcz
	.def	_Z6printfPKcz;	.scl	2;	.type	32;	.endef
_Z6printfPKcz:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	pushq	%rbx	 #
	subq	$56, %rsp	 #,
	movq	%rcx, 16(%rbp)	 # __format, __format
	movq	%rdx, 24(%rbp)	 #,
	movq	%r8, 32(%rbp)	 #,
	movq	%r9, 40(%rbp)	 #,
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:374:   __builtin_va_list __local_argv; __builtin_va_start( __local_argv, __format );
	leaq	24(%rbp), %rax	 #, tmp86
	movq	%rax, -32(%rbp)	 # tmp86, MEM[(char * *)&__local_argv]
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:375:   __retval = __mingw_vfprintf( stdout, __format, __local_argv );
	movq	-32(%rbp), %rbx	 # __local_argv, __local_argv.0_1
	movl	$1, %ecx	 #,
	movq	__imp___acrt_iob_func(%rip), %rax	 #, tmp87
	call	*%rax	 # tmp87
	movq	%rbx, %r8	 # __local_argv.0_1,
	movq	16(%rbp), %rdx	 # __format,
	movq	%rax, %rcx	 # _2,
	call	__mingw_vfprintf	 #
	movl	%eax, -20(%rbp)	 # tmp88, __retval
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:377:   return __retval;
	movl	-20(%rbp), %eax	 # __retval, _11
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:378: }
	movq	-8(%rbp), %rbx	 #,
	leave	
	ret	
	.section .rdata,"dr"
.LC0:
	.ascii "\12=> Mat [%d, %d, f32]: \12\0"
.LC1:
	.ascii "%.2f \11\0"
.LC2:
	.ascii "\12\0"
	.text
	.globl	_Z8printmatPfii
	.def	_Z8printmatPfii;	.scl	2;	.type	32;	.endef
_Z8printmatPfii:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	subq	$48, %rsp	 #,
	movq	%rcx, 16(%rbp)	 # a, a
	movl	%edx, 24(%rbp)	 # m, m
	movl	%r8d, 32(%rbp)	 # n, n
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:68:     printf("\n=> Mat [%d, %d, f32]: \n", m, n);  // 打印矩阵维度
	movl	32(%rbp), %edx	 # n, tmp89
	movl	24(%rbp), %eax	 # m, tmp90
	movl	%edx, %r8d	 # tmp89,
	movl	%eax, %edx	 # tmp90,
	leaq	.LC0(%rip), %rax	 #, tmp91
	movq	%rax, %rcx	 # tmp91,
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:69:     for (int i = 0; i < m; i++) {
	movl	$0, -4(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:69:     for (int i = 0; i < m; i++) {
	jmp	.L4	 #
.L7:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:70:         for (int j = 0; j < n; j++) {
	movl	$0, -8(%rbp)	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:70:         for (int j = 0; j < n; j++) {
	jmp	.L5	 #
.L6:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:71:             printf("%.2f \t", a[i * n + j]);  // 打印矩阵元素，保留两位小数
	movl	-4(%rbp), %eax	 # i, tmp92
	imull	32(%rbp), %eax	 # n, _1
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:71:             printf("%.2f \t", a[i * n + j]);  // 打印矩阵元素，保留两位小数
	movl	-8(%rbp), %edx	 # j, tmp93
	addl	%edx, %eax	 # tmp93, _2
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:71:             printf("%.2f \t", a[i * n + j]);  // 打印矩阵元素，保留两位小数
	leaq	0(,%rax,4), %rdx	 #, _4
	movq	16(%rbp), %rax	 # a, tmp94
	addq	%rdx, %rax	 # _4, _5
	movss	(%rax), %xmm0	 # *_5, _6
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:71:             printf("%.2f \t", a[i * n + j]);  // 打印矩阵元素，保留两位小数
	cvtss2sd	%xmm0, %xmm0	 # _6, _7
	movq	%xmm0, %rax	 # _7, tmp95
	movq	%rax, %rdx	 # tmp95, tmp96
	movq	%rdx, %xmm0	 # tmp96, tmp98
	movapd	%xmm0, %xmm1	 # tmp98,
	movq	%rax, %rdx	 # tmp99,
	leaq	.LC1(%rip), %rax	 #, tmp100
	movq	%rax, %rcx	 # tmp100,
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:70:         for (int j = 0; j < n; j++) {
	addl	$1, -8(%rbp)	 #, j
.L5:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:70:         for (int j = 0; j < n; j++) {
	movl	-8(%rbp), %eax	 # j, tmp101
	cmpl	32(%rbp), %eax	 # n, tmp101
	jl	.L6	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:73:         printf("\n");
	leaq	.LC2(%rip), %rax	 #, tmp102
	movq	%rax, %rcx	 # tmp102,
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:69:     for (int i = 0; i < m; i++) {
	addl	$1, -4(%rbp)	 #, i
.L4:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:69:     for (int i = 0; i < m; i++) {
	movl	-4(%rbp), %eax	 # i, tmp103
	cmpl	24(%rbp), %eax	 # m, tmp103
	jl	.L7	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:75: }
	nop	
	nop	
	leave	
	ret	
	.section .rdata,"dr"
.LC3:
	.ascii "\12=> Mat [%d, %d, u8_1]: \12\0"
.LC4:
	.ascii "%d \11\0"
	.text
	.globl	_Z8printmatPhii
	.def	_Z8printmatPhii;	.scl	2;	.type	32;	.endef
_Z8printmatPhii:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	subq	$48, %rsp	 #,
	movq	%rcx, 16(%rbp)	 # a, a
	movl	%edx, 24(%rbp)	 # m, m
	movl	%r8d, 32(%rbp)	 # n, n
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:78:     printf("\n=> Mat [%d, %d, u8_1]: \n", m, n);  // 打印矩阵维度
	movl	32(%rbp), %edx	 # n, tmp92
	movl	24(%rbp), %eax	 # m, tmp93
	movl	%edx, %r8d	 # tmp92,
	movl	%eax, %edx	 # tmp93,
	leaq	.LC3(%rip), %rax	 #, tmp94
	movq	%rax, %rcx	 # tmp94,
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:79:     int i,j,k=0;
	movl	$0, -12(%rbp)	 #, k
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:80:     for (i = 0; i < m; i++) {
	movl	$0, -4(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:80:     for (i = 0; i < m; i++) {
	jmp	.L9	 #
.L12:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:81:         for (j = 0; j < n; j++,k++) {
	movl	$0, -8(%rbp)	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:81:         for (j = 0; j < n; j++,k++) {
	jmp	.L10	 #
.L11:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:82:             printf("%d \t", (a[k/8]<<(k%8) & 128)>0);  // 打印矩阵元素，保留两位小数
	movl	-12(%rbp), %eax	 # k, tmp95
	leal	7(%rax), %edx	 #, tmp97
	testl	%eax, %eax	 # tmp96
	cmovs	%edx, %eax	 # tmp97,, tmp96
	sarl	$3, %eax	 #, tmp98
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:82:             printf("%d \t", (a[k/8]<<(k%8) & 128)>0);  // 打印矩阵元素，保留两位小数
	movq	16(%rbp), %rdx	 # a, tmp99
	addq	%rdx, %rax	 # tmp99, _3
	movzbl	(%rax), %eax	 # *_3, _4
	movzbl	%al, %edx	 # _4, _5
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:82:             printf("%d \t", (a[k/8]<<(k%8) & 128)>0);  // 打印矩阵元素，保留两位小数
	movl	-12(%rbp), %eax	 # k, tmp100
	andl	$7, %eax	 #, _6
	movl	%eax, %ecx	 # _6, tmp107
	sall	%cl, %edx	 # tmp107, _5
	movl	%edx, %eax	 # _5, _7
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:82:             printf("%d \t", (a[k/8]<<(k%8) & 128)>0);  // 打印矩阵元素，保留两位小数
	andl	$128, %eax	 #, _8
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:82:             printf("%d \t", (a[k/8]<<(k%8) & 128)>0);  // 打印矩阵元素，保留两位小数
	testl	%eax, %eax	 # _8
	setg	%al	 #, _9
	movzbl	%al, %eax	 # _9, _10
	movl	%eax, %edx	 # _10,
	leaq	.LC4(%rip), %rax	 #, tmp101
	movq	%rax, %rcx	 # tmp101,
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:81:         for (j = 0; j < n; j++,k++) {
	addl	$1, -8(%rbp)	 #, j
	addl	$1, -12(%rbp)	 #, k
.L10:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:81:         for (j = 0; j < n; j++,k++) {
	movl	-8(%rbp), %eax	 # j, tmp102
	cmpl	32(%rbp), %eax	 # n, tmp102
	jl	.L11	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:84:         printf("\n");
	leaq	.LC2(%rip), %rax	 #, tmp103
	movq	%rax, %rcx	 # tmp103,
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:80:     for (i = 0; i < m; i++) {
	addl	$1, -4(%rbp)	 #, i
.L9:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:80:     for (i = 0; i < m; i++) {
	movl	-4(%rbp), %eax	 # i, tmp104
	cmpl	24(%rbp), %eax	 # m, tmp104
	jl	.L12	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:86: }
	nop	
	nop	
	leave	
	ret	
	.globl	_Z19f32_f32_vec_in_prodPfS_i
	.def	_Z19f32_f32_vec_in_prodPfS_i;	.scl	2;	.type	32;	.endef
_Z19f32_f32_vec_in_prodPfS_i:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	subq	$16, %rsp	 #,
	movq	%rcx, 16(%rbp)	 # data, data
	movq	%rdx, 24(%rbp)	 # weight, weight
	movl	%r8d, 32(%rbp)	 # n, n
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:90:     int i = 0;
	movl	$0, -4(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:91:     f32 res = 0;  // 初始化结果为0
	pxor	%xmm0, %xmm0	 # tmp87
	movss	%xmm0, -8(%rbp)	 # tmp87, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:92:     for (; i < n; i++, data++, weight++)  // 遍历向量元素
	jmp	.L14	 #
.L15:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:93:         res += (*data) * (*weight);  // 累加乘积
	movq	16(%rbp), %rax	 # data, tmp88
	movss	(%rax), %xmm1	 # *data_4, _1
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:93:         res += (*data) * (*weight);  // 累加乘积
	movq	24(%rbp), %rax	 # weight, tmp89
	movss	(%rax), %xmm0	 # *weight_5, _2
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:93:         res += (*data) * (*weight);  // 累加乘积
	mulss	%xmm1, %xmm0	 # _1, _3
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:93:         res += (*data) * (*weight);  // 累加乘积
	movss	-8(%rbp), %xmm1	 # res, tmp91
	addss	%xmm1, %xmm0	 # tmp91, tmp90
	movss	%xmm0, -8(%rbp)	 # tmp90, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:92:     for (; i < n; i++, data++, weight++)  // 遍历向量元素
	addl	$1, -4(%rbp)	 #, i
	addq	$4, 16(%rbp)	 #, data
	addq	$4, 24(%rbp)	 #, weight
.L14:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:92:     for (; i < n; i++, data++, weight++)  // 遍历向量元素
	movl	-4(%rbp), %eax	 # i, tmp92
	cmpl	32(%rbp), %eax	 # n, tmp92
	jl	.L15	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:94:     return res;  // 返回内积结果
	movss	-8(%rbp), %xmm0	 # res, _13
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:95: }
	leave	
	ret	
	.globl	_Z15f32_f32_mat_mulPfS_S_iiib
	.def	_Z15f32_f32_mat_mulPfS_S_iiib;	.scl	2;	.type	32;	.endef
_Z15f32_f32_mat_mulPfS_S_iiib:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	subq	$64, %rsp	 #,
	movq	%rcx, 16(%rbp)	 # data, data
	movq	%rdx, 24(%rbp)	 # weight, weight
	movq	%r8, 32(%rbp)	 # out, out
	movl	%r9d, 40(%rbp)	 # m, m
	movl	64(%rbp), %eax	 # Transpose, tmp94
	movb	%al, -20(%rbp)	 # tmp95, Transpose
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:102:     if (Transpose) {  // 如果需要转置输出
	cmpb	$0, -20(%rbp)	 #, Transpose
	je	.L18	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:103:         for (j = 0; j < n; j++)  // 遍历矩阵B的行
	movl	$0, -8(%rbp)	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:103:         for (j = 0; j < n; j++)  // 遍历矩阵B的行
	jmp	.L19	 #
.L22:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104:             for (i = 0, p = data; i < m; i++, weight += k, p += k, out++)  // 遍历矩阵A的行
	movl	$0, -4(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104:             for (i = 0, p = data; i < m; i++, weight += k, p += k, out++)  // 遍历矩阵A的行
	movq	16(%rbp), %rax	 # data, tmp96
	movq	%rax, -16(%rbp)	 # tmp96, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104:             for (i = 0, p = data; i < m; i++, weight += k, p += k, out++)  // 遍历矩阵A的行
	jmp	.L20	 #
.L21:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:105:                 *out = f32_f32_vec_in_prod(p, weight, k);  // 计算内积并存储到输出矩阵
	movl	56(%rbp), %ecx	 # k, tmp97
	movq	24(%rbp), %rdx	 # weight, tmp98
	movq	-16(%rbp), %rax	 # p, tmp99
	movl	%ecx, %r8d	 # tmp97,
	movq	%rax, %rcx	 # tmp99,
	call	_Z19f32_f32_vec_in_prodPfS_i	 #
	movd	%xmm0, %eax	 #, _1
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:105:                 *out = f32_f32_vec_in_prod(p, weight, k);  // 计算内积并存储到输出矩阵
	movq	32(%rbp), %rdx	 # out, tmp100
	movl	%eax, (%rdx)	 # _1, *out_18
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104:             for (i = 0, p = data; i < m; i++, weight += k, p += k, out++)  // 遍历矩阵A的行
	addl	$1, -4(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104:             for (i = 0, p = data; i < m; i++, weight += k, p += k, out++)  // 遍历矩阵A的行
	movl	56(%rbp), %eax	 # k, tmp101
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104:             for (i = 0, p = data; i < m; i++, weight += k, p += k, out++)  // 遍历矩阵A的行
	salq	$2, %rax	 #, _3
	addq	%rax, 24(%rbp)	 # _3, weight
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104:             for (i = 0, p = data; i < m; i++, weight += k, p += k, out++)  // 遍历矩阵A的行
	movl	56(%rbp), %eax	 # k, tmp102
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104:             for (i = 0, p = data; i < m; i++, weight += k, p += k, out++)  // 遍历矩阵A的行
	salq	$2, %rax	 #, _5
	addq	%rax, -16(%rbp)	 # _5, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104:             for (i = 0, p = data; i < m; i++, weight += k, p += k, out++)  // 遍历矩阵A的行
	addq	$4, 32(%rbp)	 #, out
.L20:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104:             for (i = 0, p = data; i < m; i++, weight += k, p += k, out++)  // 遍历矩阵A的行
	movl	-4(%rbp), %eax	 # i, tmp103
	cmpl	40(%rbp), %eax	 # m, tmp103
	jl	.L21	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:103:         for (j = 0; j < n; j++)  // 遍历矩阵B的行
	addl	$1, -8(%rbp)	 #, j
.L19:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:103:         for (j = 0; j < n; j++)  // 遍历矩阵B的行
	movl	-8(%rbp), %eax	 # j, tmp104
	cmpl	48(%rbp), %eax	 # n, tmp104
	jl	.L22	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:111: }
	jmp	.L28	 #
.L18:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:107:         for (i = 0; i < m; i++, data += k)  // 遍历矩阵A的行
	movl	$0, -4(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:107:         for (i = 0; i < m; i++, data += k)  // 遍历矩阵A的行
	jmp	.L24	 #
.L27:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:108:             for (j = 0; j < n; j++, weight += k, p += k, out++)  // 遍历矩阵B的行
	movl	$0, -8(%rbp)	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:108:             for (j = 0; j < n; j++, weight += k, p += k, out++)  // 遍历矩阵B的行
	jmp	.L25	 #
.L26:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:109:                 *out = f32_f32_vec_in_prod(p, weight, k);  // 计算内积并存储到输出矩阵
	movl	56(%rbp), %ecx	 # k, tmp105
	movq	24(%rbp), %rdx	 # weight, tmp106
	movq	-16(%rbp), %rax	 # p, tmp107
	movl	%ecx, %r8d	 # tmp105,
	movq	%rax, %rcx	 # tmp107,
	call	_Z19f32_f32_vec_in_prodPfS_i	 #
	movd	%xmm0, %eax	 #, _6
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:109:                 *out = f32_f32_vec_in_prod(p, weight, k);  // 计算内积并存储到输出矩阵
	movq	32(%rbp), %rdx	 # out, tmp108
	movl	%eax, (%rdx)	 # _6, *out_20
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:108:             for (j = 0; j < n; j++, weight += k, p += k, out++)  // 遍历矩阵B的行
	addl	$1, -8(%rbp)	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:108:             for (j = 0; j < n; j++, weight += k, p += k, out++)  // 遍历矩阵B的行
	movl	56(%rbp), %eax	 # k, tmp109
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:108:             for (j = 0; j < n; j++, weight += k, p += k, out++)  // 遍历矩阵B的行
	salq	$2, %rax	 #, _8
	addq	%rax, 24(%rbp)	 # _8, weight
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:108:             for (j = 0; j < n; j++, weight += k, p += k, out++)  // 遍历矩阵B的行
	movl	56(%rbp), %eax	 # k, tmp110
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:108:             for (j = 0; j < n; j++, weight += k, p += k, out++)  // 遍历矩阵B的行
	salq	$2, %rax	 #, _10
	addq	%rax, -16(%rbp)	 # _10, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:108:             for (j = 0; j < n; j++, weight += k, p += k, out++)  // 遍历矩阵B的行
	addq	$4, 32(%rbp)	 #, out
.L25:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:108:             for (j = 0; j < n; j++, weight += k, p += k, out++)  // 遍历矩阵B的行
	movl	-8(%rbp), %eax	 # j, tmp111
	cmpl	48(%rbp), %eax	 # n, tmp111
	jl	.L26	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:107:         for (i = 0; i < m; i++, data += k)  // 遍历矩阵A的行
	addl	$1, -4(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:107:         for (i = 0; i < m; i++, data += k)  // 遍历矩阵A的行
	movl	56(%rbp), %eax	 # k, tmp112
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:107:         for (i = 0; i < m; i++, data += k)  // 遍历矩阵A的行
	salq	$2, %rax	 #, _12
	addq	%rax, 16(%rbp)	 # _12, data
.L24:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:107:         for (i = 0; i < m; i++, data += k)  // 遍历矩阵A的行
	movl	-4(%rbp), %eax	 # i, tmp113
	cmpl	40(%rbp), %eax	 # m, tmp113
	jl	.L27	 #,
.L28:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:111: }
	nop	
	leave	
	ret	
	.globl	_Z20f32_u8_1_vec_in_prodPfPhii
	.def	_Z20f32_u8_1_vec_in_prodPfPhii;	.scl	2;	.type	32;	.endef
_Z20f32_u8_1_vec_in_prodPfPhii:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	andq	$-16, %rsp	 #,
	subq	$32, %rsp	 #,
	movq	%rcx, 16(%rbp)	 # data, data
	movq	%rdx, 24(%rbp)	 # weight, weight
	movl	%r8d, 32(%rbp)	 # n, n
	movl	%r9d, 40(%rbp)	 # b_start, b_start
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:115:     int i = 0, j = 0;
	movl	$0, 28(%rsp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:115:     int i = 0, j = 0;
	movl	$0, 24(%rsp)	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:116:     f32 res = 0;  // 初始化结果为0
	pxor	%xmm0, %xmm0	 # tmp98
	movss	%xmm0, 20(%rsp)	 # tmp98, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:117:     u8_1 temp = (*weight) << b_start;  // 将bit向量左移b_start位
	movq	24(%rbp), %rax	 # weight, tmp99
	movzbl	(%rax), %eax	 # *weight_35(D), _1
	movzbl	%al, %edx	 # _1, _2
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:117:     u8_1 temp = (*weight) << b_start;  // 将bit向量左移b_start位
	movl	40(%rbp), %eax	 # b_start, tmp100
	movl	%eax, %ecx	 # tmp100, tmp129
	sall	%cl, %edx	 # tmp129, _2
	movl	%edx, %eax	 # _2, _3
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:117:     u8_1 temp = (*weight) << b_start;  // 将bit向量左移b_start位
	movb	%al, 19(%rsp)	 # _3, temp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:118:     int n0 = (8 - b_start) < n ? (8 - b_start) : n;  // 计算开头不满足8位的部分长度
	movl	$8, %eax	 #, tmp101
	subl	40(%rbp), %eax	 # b_start, _4
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:118:     int n0 = (8 - b_start) < n ? (8 - b_start) : n;  // 计算开头不满足8位的部分长度
	movl	32(%rbp), %edx	 # n, tmp103
	cmpl	%eax, %edx	 # _4, tmp103
	cmovle	%edx, %eax	 # tmp103,, tmp102
	movl	%eax, 12(%rsp)	 # tmp102, n0
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:121:     for (; i < n0; i++, data++, temp <<= 1)
	jmp	.L30	 #
.L33:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:122:         res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movzbl	19(%rsp), %eax	 # temp, temp.2_5
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:122:         res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	testb	%al, %al	 # temp.2_5
	jns	.L31	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:122:         res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movq	16(%rbp), %rax	 # data, tmp104
	movss	(%rax), %xmm0	 # *data_12, iftmp.1_28
	jmp	.L32	 #
.L31:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:122:         res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movq	16(%rbp), %rax	 # data, tmp105
	movss	(%rax), %xmm0	 # *data_12, _6
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:122:         res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movss	.LC6(%rip), %xmm1	 #, tmp106
	xorps	%xmm1, %xmm0	 # tmp106, iftmp.1_28
.L32:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:122:         res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movss	20(%rsp), %xmm1	 # res, tmp108
	addss	%xmm1, %xmm0	 # tmp108, tmp107
	movss	%xmm0, 20(%rsp)	 # tmp107, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:121:     for (; i < n0; i++, data++, temp <<= 1)
	addl	$1, 28(%rsp)	 #, i
	addq	$4, 16(%rbp)	 #, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:121:     for (; i < n0; i++, data++, temp <<= 1)
	salb	19(%rsp)	 # temp
.L30:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:121:     for (; i < n0; i++, data++, temp <<= 1)
	movl	28(%rsp), %eax	 # i, tmp109
	cmpl	12(%rsp), %eax	 # n0, tmp109
	jl	.L33	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:125:     for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	addq	$1, 24(%rbp)	 #, weight
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:125:     for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	jmp	.L34	 #
.L39:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:126:         for (j = 0, temp = *weight; j < 8; j++, data++, temp <<= 1)  // 遍历8位
	movl	$0, 24(%rsp)	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:126:         for (j = 0, temp = *weight; j < 8; j++, data++, temp <<= 1)  // 遍历8位
	movq	24(%rbp), %rax	 # weight, tmp110
	movzbl	(%rax), %eax	 # *weight_16, tmp111
	movb	%al, 19(%rsp)	 # tmp111, temp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:126:         for (j = 0, temp = *weight; j < 8; j++, data++, temp <<= 1)  // 遍历8位
	jmp	.L35	 #
.L38:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:127:             res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movzbl	19(%rsp), %eax	 # temp, temp.4_7
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:127:             res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	testb	%al, %al	 # temp.4_7
	jns	.L36	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:127:             res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movq	16(%rbp), %rax	 # data, tmp112
	movss	(%rax), %xmm0	 # *data_13, iftmp.3_29
	jmp	.L37	 #
.L36:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:127:             res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movq	16(%rbp), %rax	 # data, tmp113
	movss	(%rax), %xmm0	 # *data_13, _8
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:127:             res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movss	.LC6(%rip), %xmm1	 #, tmp114
	xorps	%xmm1, %xmm0	 # tmp114, iftmp.3_29
.L37:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:127:             res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movss	20(%rsp), %xmm1	 # res, tmp116
	addss	%xmm1, %xmm0	 # tmp116, tmp115
	movss	%xmm0, 20(%rsp)	 # tmp115, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:126:         for (j = 0, temp = *weight; j < 8; j++, data++, temp <<= 1)  // 遍历8位
	addl	$1, 24(%rsp)	 #, j
	addq	$4, 16(%rbp)	 #, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:126:         for (j = 0, temp = *weight; j < 8; j++, data++, temp <<= 1)  // 遍历8位
	salb	19(%rsp)	 # temp
.L35:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:126:         for (j = 0, temp = *weight; j < 8; j++, data++, temp <<= 1)  // 遍历8位
	cmpl	$7, 24(%rsp)	 #, j
	jle	.L38	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:125:     for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	addl	$8, 28(%rsp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:125:     for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	addq	$1, 24(%rbp)	 #, weight
.L34:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:125:     for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	movl	32(%rbp), %eax	 # n, tmp117
	subl	$7, %eax	 #, _9
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:125:     for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	cmpl	%eax, 28(%rsp)	 # _9, i
	jl	.L39	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:130:     for (temp = *weight; i < n; i++, data++, temp <<= 1)
	movq	24(%rbp), %rax	 # weight, tmp118
	movzbl	(%rax), %eax	 # *weight_16, tmp119
	movb	%al, 19(%rsp)	 # tmp119, temp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:130:     for (temp = *weight; i < n; i++, data++, temp <<= 1)
	jmp	.L40	 #
.L43:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:131:         res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movzbl	19(%rsp), %eax	 # temp, temp.6_10
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:131:         res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	testb	%al, %al	 # temp.6_10
	jns	.L41	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:131:         res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movq	16(%rbp), %rax	 # data, tmp120
	movss	(%rax), %xmm0	 # *data_15, iftmp.5_30
	jmp	.L42	 #
.L41:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:131:         res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movq	16(%rbp), %rax	 # data, tmp121
	movss	(%rax), %xmm0	 # *data_15, _11
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:131:         res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movss	.LC6(%rip), %xmm1	 #, tmp122
	xorps	%xmm1, %xmm0	 # tmp122, iftmp.5_30
.L42:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:131:         res += (temp & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movss	20(%rsp), %xmm1	 # res, tmp124
	addss	%xmm1, %xmm0	 # tmp124, tmp123
	movss	%xmm0, 20(%rsp)	 # tmp123, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:130:     for (temp = *weight; i < n; i++, data++, temp <<= 1)
	addl	$1, 28(%rsp)	 #, i
	addq	$4, 16(%rbp)	 #, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:130:     for (temp = *weight; i < n; i++, data++, temp <<= 1)
	salb	19(%rsp)	 # temp
.L40:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:130:     for (temp = *weight; i < n; i++, data++, temp <<= 1)
	movl	28(%rsp), %eax	 # i, tmp125
	cmpl	32(%rbp), %eax	 # n, tmp125
	jl	.L43	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:133:     return res;  // 返回内积结果
	movss	20(%rsp), %xmm0	 # res, _43
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:134: } 
	leave	
	ret	
	.globl	_Z16f32_u8_1_mat_mulPfPhS_iiib
	.def	_Z16f32_u8_1_mat_mulPfPhS_iiib;	.scl	2;	.type	32;	.endef
_Z16f32_u8_1_mat_mulPfPhS_iiib:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	subq	$64, %rsp	 #,
	movq	%rcx, 16(%rbp)	 # data, data
	movq	%rdx, 24(%rbp)	 # weight, weight
	movq	%r8, 32(%rbp)	 # out, out
	movl	%r9d, 40(%rbp)	 # m, m
	movl	64(%rbp), %eax	 # Transpose, tmp100
	movb	%al, -20(%rbp)	 # tmp101, Transpose
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:142:     if (Transpose) {
	cmpb	$0, -20(%rbp)	 #, Transpose
	je	.L46	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:143:         for (j = 0; j < n; j++)
	movl	$0, -8(%rbp)	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:143:         for (j = 0; j < n; j++)
	jmp	.L47	 #
.L50:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:144:             for (i = 0, p = data; i < m; i++, p += k, out++)
	movl	$0, -4(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:144:             for (i = 0, p = data; i < m; i++, p += k, out++)
	movq	16(%rbp), %rax	 # data, tmp102
	movq	%rax, -16(%rbp)	 # tmp102, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:144:             for (i = 0, p = data; i < m; i++, p += k, out++)
	jmp	.L48	 #
.L49:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:145:                 *out = f32_u8_1_vec_in_prod(p, weight + (j * k) / 8, k, (j * k) % 8);
	movl	-8(%rbp), %eax	 # j, tmp103
	imull	56(%rbp), %eax	 # k, _1
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:145:                 *out = f32_u8_1_vec_in_prod(p, weight + (j * k) / 8, k, (j * k) % 8);
	cltd
	shrl	$29, %edx	 #, tmp105
	addl	%edx, %eax	 # tmp105, tmp106
	andl	$7, %eax	 #, tmp107
	subl	%edx, %eax	 # tmp105, tmp108
	movl	%eax, %r8d	 # tmp108, _2
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:145:                 *out = f32_u8_1_vec_in_prod(p, weight + (j * k) / 8, k, (j * k) % 8);
	movl	-8(%rbp), %eax	 # j, tmp109
	imull	56(%rbp), %eax	 # k, _3
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:145:                 *out = f32_u8_1_vec_in_prod(p, weight + (j * k) / 8, k, (j * k) % 8);
	leal	7(%rax), %edx	 #, tmp111
	testl	%eax, %eax	 # tmp110
	cmovs	%edx, %eax	 # tmp111,, tmp110
	sarl	$3, %eax	 #, tmp112
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:145:                 *out = f32_u8_1_vec_in_prod(p, weight + (j * k) / 8, k, (j * k) % 8);
	movq	24(%rbp), %rdx	 # weight, tmp113
	addq	%rax, %rdx	 # _5, _6
	movl	56(%rbp), %ecx	 # k, tmp114
	movq	-16(%rbp), %rax	 # p, tmp115
	movl	%r8d, %r9d	 # _2,
	movl	%ecx, %r8d	 # tmp114,
	movq	%rax, %rcx	 # tmp115,
	call	_Z20f32_u8_1_vec_in_prodPfPhii	 #
	movd	%xmm0, %eax	 #, _7
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:145:                 *out = f32_u8_1_vec_in_prod(p, weight + (j * k) / 8, k, (j * k) % 8);
	movq	32(%rbp), %rdx	 # out, tmp116
	movl	%eax, (%rdx)	 # _7, *out_20
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:144:             for (i = 0, p = data; i < m; i++, p += k, out++)
	addl	$1, -4(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:144:             for (i = 0, p = data; i < m; i++, p += k, out++)
	movl	56(%rbp), %eax	 # k, tmp117
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:144:             for (i = 0, p = data; i < m; i++, p += k, out++)
	salq	$2, %rax	 #, _9
	addq	%rax, -16(%rbp)	 # _9, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:144:             for (i = 0, p = data; i < m; i++, p += k, out++)
	addq	$4, 32(%rbp)	 #, out
.L48:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:144:             for (i = 0, p = data; i < m; i++, p += k, out++)
	movl	-4(%rbp), %eax	 # i, tmp118
	cmpl	40(%rbp), %eax	 # m, tmp118
	jl	.L49	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:143:         for (j = 0; j < n; j++)
	addl	$1, -8(%rbp)	 #, j
.L47:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:143:         for (j = 0; j < n; j++)
	movl	-8(%rbp), %eax	 # j, tmp119
	cmpl	48(%rbp), %eax	 # n, tmp119
	jl	.L50	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:151: }
	jmp	.L56	 #
.L46:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:147:         for (i = 0; i < m; i++, data += k)
	movl	$0, -4(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:147:         for (i = 0; i < m; i++, data += k)
	jmp	.L52	 #
.L55:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:148:             for (j = 0; j < n; j++,  out++ )
	movl	$0, -8(%rbp)	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:148:             for (j = 0; j < n; j++,  out++ )
	jmp	.L53	 #
.L54:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:149:                 *out = f32_u8_1_vec_in_prod(data, weight + (j * k) / 8, k, (j * k) % 8);
	movl	-8(%rbp), %eax	 # j, tmp120
	imull	56(%rbp), %eax	 # k, _10
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:149:                 *out = f32_u8_1_vec_in_prod(data, weight + (j * k) / 8, k, (j * k) % 8);
	cltd
	shrl	$29, %edx	 #, tmp122
	addl	%edx, %eax	 # tmp122, tmp123
	andl	$7, %eax	 #, tmp124
	subl	%edx, %eax	 # tmp122, tmp125
	movl	%eax, %ecx	 # tmp125, _11
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:149:                 *out = f32_u8_1_vec_in_prod(data, weight + (j * k) / 8, k, (j * k) % 8);
	movl	-8(%rbp), %eax	 # j, tmp126
	imull	56(%rbp), %eax	 # k, _12
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:149:                 *out = f32_u8_1_vec_in_prod(data, weight + (j * k) / 8, k, (j * k) % 8);
	leal	7(%rax), %edx	 #, tmp128
	testl	%eax, %eax	 # tmp127
	cmovs	%edx, %eax	 # tmp128,, tmp127
	sarl	$3, %eax	 #, tmp129
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:149:                 *out = f32_u8_1_vec_in_prod(data, weight + (j * k) / 8, k, (j * k) % 8);
	movq	24(%rbp), %rdx	 # weight, tmp130
	addq	%rdx, %rax	 # tmp130, _15
	movl	56(%rbp), %edx	 # k, tmp131
	movl	%ecx, %r9d	 # _11,
	movl	%edx, %r8d	 # tmp131,
	movq	%rax, %rdx	 # _15,
	movq	16(%rbp), %rcx	 # data,
	call	_Z20f32_u8_1_vec_in_prodPfPhii	 #
	movd	%xmm0, %eax	 #, _16
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:149:                 *out = f32_u8_1_vec_in_prod(data, weight + (j * k) / 8, k, (j * k) % 8);
	movq	32(%rbp), %rdx	 # out, tmp132
	movl	%eax, (%rdx)	 # _16, *out_22
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:148:             for (j = 0; j < n; j++,  out++ )
	addl	$1, -8(%rbp)	 #, j
	addq	$4, 32(%rbp)	 #, out
.L53:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:148:             for (j = 0; j < n; j++,  out++ )
	movl	-8(%rbp), %eax	 # j, tmp133
	cmpl	48(%rbp), %eax	 # n, tmp133
	jl	.L54	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:147:         for (i = 0; i < m; i++, data += k)
	addl	$1, -4(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:147:         for (i = 0; i < m; i++, data += k)
	movl	56(%rbp), %eax	 # k, tmp134
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:147:         for (i = 0; i < m; i++, data += k)
	salq	$2, %rax	 #, _18
	addq	%rax, 16(%rbp)	 # _18, data
.L52:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:147:         for (i = 0; i < m; i++, data += k)
	movl	-4(%rbp), %eax	 # i, tmp135
	cmpl	40(%rbp), %eax	 # m, tmp135
	jl	.L55	 #,
.L56:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:151: }
	nop	
	leave	
	ret	
	.section .rdata,"dr"
.LC23:
	.ascii "res = %.3f\12\0"
	.text
	.globl	_Z25test_f32_u8_1_vec_in_prodv
	.def	_Z25test_f32_u8_1_vec_in_prodv;	.scl	2;	.type	32;	.endef
_Z25test_f32_u8_1_vec_in_prodv:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	andq	$-16, %rsp	 #,
	addq	$-128, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:4:     f32 a[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	movss	.LC7(%rip), %xmm0	 #, tmp83
	movss	%xmm0, 48(%rsp)	 # tmp83, a[0]
	movss	.LC8(%rip), %xmm0	 #, tmp84
	movss	%xmm0, 52(%rsp)	 # tmp84, a[1]
	movss	.LC9(%rip), %xmm0	 #, tmp85
	movss	%xmm0, 56(%rsp)	 # tmp85, a[2]
	movss	.LC10(%rip), %xmm0	 #, tmp86
	movss	%xmm0, 60(%rsp)	 # tmp86, a[3]
	movss	.LC11(%rip), %xmm0	 #, tmp87
	movss	%xmm0, 64(%rsp)	 # tmp87, a[4]
	movss	.LC12(%rip), %xmm0	 #, tmp88
	movss	%xmm0, 68(%rsp)	 # tmp88, a[5]
	movss	.LC13(%rip), %xmm0	 #, tmp89
	movss	%xmm0, 72(%rsp)	 # tmp89, a[6]
	movss	.LC14(%rip), %xmm0	 #, tmp90
	movss	%xmm0, 76(%rsp)	 # tmp90, a[7]
	movss	.LC15(%rip), %xmm0	 #, tmp91
	movss	%xmm0, 80(%rsp)	 # tmp91, a[8]
	movss	.LC16(%rip), %xmm0	 #, tmp92
	movss	%xmm0, 84(%rsp)	 # tmp92, a[9]
	movss	.LC17(%rip), %xmm0	 #, tmp93
	movss	%xmm0, 88(%rsp)	 # tmp93, a[10]
	movss	.LC18(%rip), %xmm0	 #, tmp94
	movss	%xmm0, 92(%rsp)	 # tmp94, a[11]
	movss	.LC19(%rip), %xmm0	 #, tmp95
	movss	%xmm0, 96(%rsp)	 # tmp95, a[12]
	movss	.LC20(%rip), %xmm0	 #, tmp96
	movss	%xmm0, 100(%rsp)	 # tmp96, a[13]
	movss	.LC21(%rip), %xmm0	 #, tmp97
	movss	%xmm0, 104(%rsp)	 # tmp97, a[14]
	movss	.LC22(%rip), %xmm0	 #, tmp98
	movss	%xmm0, 108(%rsp)	 # tmp98, a[15]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:5:     u8_1 b[2] = {0xff, 0x0f};
	movw	$4095, 46(%rsp)	 #, b
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:6:     f32 res = f32_u8_1_vec_in_prod(a, b, 16);
	leaq	46(%rsp), %rdx	 #, tmp99
	leaq	48(%rsp), %rax	 #, tmp100
	movl	$0, %r9d	 #,
	movl	$16, %r8d	 #,
	movq	%rax, %rcx	 # tmp100,
	call	_Z20f32_u8_1_vec_in_prodPfPhii	 #
	movd	%xmm0, %eax	 #, tmp101
	movl	%eax, 124(%rsp)	 # tmp101, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:7:     printf("res = %.3f\n", res);
	pxor	%xmm0, %xmm0	 # _1
	cvtss2sd	124(%rsp), %xmm0	 # res, _1
	movq	%xmm0, %rax	 # _1, tmp102
	movq	%rax, %rdx	 # tmp102, tmp103
	movq	%rdx, %xmm0	 # tmp103, tmp105
	movapd	%xmm0, %xmm1	 # tmp105,
	movq	%rax, %rdx	 # tmp106,
	leaq	.LC23(%rip), %rax	 #, tmp107
	movq	%rax, %rcx	 # tmp107,
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:8: }
	nop	
	leave	
	ret	
	.globl	_Z24test_f32_f32_vec_in_prodv
	.def	_Z24test_f32_f32_vec_in_prodv;	.scl	2;	.type	32;	.endef
_Z24test_f32_f32_vec_in_prodv:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	andq	$-16, %rsp	 #,
	subq	$176, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:11:     f32 a[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
	movss	.LC7(%rip), %xmm0	 #, tmp83
	movss	%xmm0, 96(%rsp)	 # tmp83, a[0]
	movss	.LC8(%rip), %xmm0	 #, tmp84
	movss	%xmm0, 100(%rsp)	 # tmp84, a[1]
	movss	.LC9(%rip), %xmm0	 #, tmp85
	movss	%xmm0, 104(%rsp)	 # tmp85, a[2]
	movss	.LC10(%rip), %xmm0	 #, tmp86
	movss	%xmm0, 108(%rsp)	 # tmp86, a[3]
	movss	.LC11(%rip), %xmm0	 #, tmp87
	movss	%xmm0, 112(%rsp)	 # tmp87, a[4]
	movss	.LC12(%rip), %xmm0	 #, tmp88
	movss	%xmm0, 116(%rsp)	 # tmp88, a[5]
	movss	.LC13(%rip), %xmm0	 #, tmp89
	movss	%xmm0, 120(%rsp)	 # tmp89, a[6]
	movss	.LC14(%rip), %xmm0	 #, tmp90
	movss	%xmm0, 124(%rsp)	 # tmp90, a[7]
	movss	.LC15(%rip), %xmm0	 #, tmp91
	movss	%xmm0, 128(%rsp)	 # tmp91, a[8]
	movss	.LC16(%rip), %xmm0	 #, tmp92
	movss	%xmm0, 132(%rsp)	 # tmp92, a[9]
	movss	.LC17(%rip), %xmm0	 #, tmp93
	movss	%xmm0, 136(%rsp)	 # tmp93, a[10]
	movss	.LC18(%rip), %xmm0	 #, tmp94
	movss	%xmm0, 140(%rsp)	 # tmp94, a[11]
	movss	.LC19(%rip), %xmm0	 #, tmp95
	movss	%xmm0, 144(%rsp)	 # tmp95, a[12]
	movss	.LC20(%rip), %xmm0	 #, tmp96
	movss	%xmm0, 148(%rsp)	 # tmp96, a[13]
	movss	.LC21(%rip), %xmm0	 #, tmp97
	movss	%xmm0, 152(%rsp)	 # tmp97, a[14]
	movss	.LC22(%rip), %xmm0	 #, tmp98
	movss	%xmm0, 156(%rsp)	 # tmp98, a[15]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:12:     f32 b[16] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0};
	movss	.LC7(%rip), %xmm0	 #, tmp99
	movss	%xmm0, 32(%rsp)	 # tmp99, b[0]
	movss	.LC7(%rip), %xmm0	 #, tmp100
	movss	%xmm0, 36(%rsp)	 # tmp100, b[1]
	movss	.LC7(%rip), %xmm0	 #, tmp101
	movss	%xmm0, 40(%rsp)	 # tmp101, b[2]
	movss	.LC7(%rip), %xmm0	 #, tmp102
	movss	%xmm0, 44(%rsp)	 # tmp102, b[3]
	movss	.LC7(%rip), %xmm0	 #, tmp103
	movss	%xmm0, 48(%rsp)	 # tmp103, b[4]
	movss	.LC7(%rip), %xmm0	 #, tmp104
	movss	%xmm0, 52(%rsp)	 # tmp104, b[5]
	movss	.LC7(%rip), %xmm0	 #, tmp105
	movss	%xmm0, 56(%rsp)	 # tmp105, b[6]
	movss	.LC7(%rip), %xmm0	 #, tmp106
	movss	%xmm0, 60(%rsp)	 # tmp106, b[7]
	movss	.LC7(%rip), %xmm0	 #, tmp107
	movss	%xmm0, 64(%rsp)	 # tmp107, b[8]
	movss	.LC7(%rip), %xmm0	 #, tmp108
	movss	%xmm0, 68(%rsp)	 # tmp108, b[9]
	movss	.LC7(%rip), %xmm0	 #, tmp109
	movss	%xmm0, 72(%rsp)	 # tmp109, b[10]
	movss	.LC7(%rip), %xmm0	 #, tmp110
	movss	%xmm0, 76(%rsp)	 # tmp110, b[11]
	movss	.LC7(%rip), %xmm0	 #, tmp111
	movss	%xmm0, 80(%rsp)	 # tmp111, b[12]
	movss	.LC7(%rip), %xmm0	 #, tmp112
	movss	%xmm0, 84(%rsp)	 # tmp112, b[13]
	movss	.LC7(%rip), %xmm0	 #, tmp113
	movss	%xmm0, 88(%rsp)	 # tmp113, b[14]
	pxor	%xmm0, %xmm0	 # tmp114
	movss	%xmm0, 92(%rsp)	 # tmp114, b[15]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:13:     f32 res = f32_f32_vec_in_prod(a, b, 16);
	leaq	32(%rsp), %rdx	 #, tmp115
	leaq	96(%rsp), %rax	 #, tmp116
	movl	$16, %r8d	 #,
	movq	%rax, %rcx	 # tmp116,
	call	_Z19f32_f32_vec_in_prodPfS_i	 #
	movd	%xmm0, %eax	 #, tmp117
	movl	%eax, 172(%rsp)	 # tmp117, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:14:     printf("res = %.3f\n", res);
	pxor	%xmm0, %xmm0	 # _1
	cvtss2sd	172(%rsp), %xmm0	 # res, _1
	movq	%xmm0, %rax	 # _1, tmp118
	movq	%rax, %rdx	 # tmp118, tmp119
	movq	%rdx, %xmm0	 # tmp119, tmp121
	movapd	%xmm0, %xmm1	 # tmp121,
	movq	%rax, %rdx	 # tmp122,
	leaq	.LC23(%rip), %rax	 #, tmp123
	movq	%rax, %rcx	 # tmp123,
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:15: }
	nop	
	leave	
	ret	
	.globl	_Z17test_f32_u8_1_mulv
	.def	_Z17test_f32_u8_1_mulv;	.scl	2;	.type	32;	.endef
_Z17test_f32_u8_1_mulv:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	andq	$-16, %rsp	 #,
	subq	$528, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:18:     f32 a[8] = {1, 2,
	movss	.LC7(%rip), %xmm0	 #, tmp82
	movss	%xmm0, 480(%rsp)	 # tmp82, a[0]
	movss	.LC8(%rip), %xmm0	 #, tmp83
	movss	%xmm0, 484(%rsp)	 # tmp83, a[1]
	movss	.LC9(%rip), %xmm0	 #, tmp84
	movss	%xmm0, 488(%rsp)	 # tmp84, a[2]
	movss	.LC10(%rip), %xmm0	 #, tmp85
	movss	%xmm0, 492(%rsp)	 # tmp85, a[3]
	movss	.LC11(%rip), %xmm0	 #, tmp86
	movss	%xmm0, 496(%rsp)	 # tmp86, a[4]
	movss	.LC12(%rip), %xmm0	 #, tmp87
	movss	%xmm0, 500(%rsp)	 # tmp87, a[5]
	movss	.LC13(%rip), %xmm0	 #, tmp88
	movss	%xmm0, 504(%rsp)	 # tmp88, a[6]
	movss	.LC14(%rip), %xmm0	 #, tmp89
	movss	%xmm0, 508(%rsp)	 # tmp89, a[7]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:23:     u8_1 b[2] = {0xf1, 0xf0};
	movw	$-3855, 478(%rsp)	 #, b
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:25:     int m = 2, n = 2, k = 4;
	movl	$2, 524(%rsp)	 #, m
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:25:     int m = 2, n = 2, k = 4;
	movl	$2, 520(%rsp)	 #, n
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:25:     int m = 2, n = 2, k = 4;
	movl	$4, 516(%rsp)	 #, k
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:26:     f32_u8_1_mat_mul(a, b, c, m, n, k, true);
	movl	524(%rsp), %r9d	 # m, tmp90
	leaq	64(%rsp), %r8	 #, tmp91
	leaq	478(%rsp), %rdx	 #, tmp92
	leaq	480(%rsp), %rax	 #, tmp93
	movl	$1, 48(%rsp)	 #,
	movl	516(%rsp), %ecx	 # k, tmp94
	movl	%ecx, 40(%rsp)	 # tmp94,
	movl	520(%rsp), %ecx	 # n, tmp95
	movl	%ecx, 32(%rsp)	 # tmp95,
	movq	%rax, %rcx	 # tmp93,
	call	_Z16f32_u8_1_mat_mulPfPhS_iiib	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:27:     printmat(a, m, k);
	movl	516(%rsp), %ecx	 # k, tmp96
	movl	524(%rsp), %edx	 # m, tmp97
	leaq	480(%rsp), %rax	 #, tmp98
	movl	%ecx, %r8d	 # tmp96,
	movq	%rax, %rcx	 # tmp98,
	call	_Z8printmatPfii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:28:     printmat(b, n, k);
	movl	516(%rsp), %ecx	 # k, tmp99
	movl	520(%rsp), %edx	 # n, tmp100
	leaq	478(%rsp), %rax	 #, tmp101
	movl	%ecx, %r8d	 # tmp99,
	movq	%rax, %rcx	 # tmp101,
	call	_Z8printmatPhii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:29:     printmat(c, m, n);
	movl	520(%rsp), %ecx	 # n, tmp102
	movl	524(%rsp), %edx	 # m, tmp103
	leaq	64(%rsp), %rax	 #, tmp104
	movl	%ecx, %r8d	 # tmp102,
	movq	%rax, %rcx	 # tmp104,
	call	_Z8printmatPfii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:30: }
	nop	
	leave	
	ret	
	.def	__main;	.scl	2;	.type	32;	.endef
	.globl	main
	.def	main;	.scl	2;	.type	32;	.endef
main:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	subq	$32, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:32: int main() {
	call	__main	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:35:     test_f32_u8_1_mul();
	call	_Z17test_f32_u8_1_mulv	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:36:     return 1;
	movl	$1, %eax	 #, _3
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix_test.cpp:37: }
	leave	
	ret	
	.section .rdata,"dr"
	.align 16
.LC6:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
	.align 4
.LC7:
	.long	1065353216
	.align 4
.LC8:
	.long	1073741824
	.align 4
.LC9:
	.long	1077936128
	.align 4
.LC10:
	.long	1082130432
	.align 4
.LC11:
	.long	1084227584
	.align 4
.LC12:
	.long	1086324736
	.align 4
.LC13:
	.long	1088421888
	.align 4
.LC14:
	.long	1090519040
	.align 4
.LC15:
	.long	1091567616
	.align 4
.LC16:
	.long	1092616192
	.align 4
.LC17:
	.long	1093664768
	.align 4
.LC18:
	.long	1094713344
	.align 4
.LC19:
	.long	1095761920
	.align 4
.LC20:
	.long	1096810496
	.align 4
.LC21:
	.long	1097859072
	.align 4
.LC22:
	.long	1098907648
	.ident	"GCC: (x86_64-posix-seh, Built by MinGW-Builds project) 11.4.0"
	.def	__mingw_vfprintf;	.scl	2;	.type	32;	.endef
