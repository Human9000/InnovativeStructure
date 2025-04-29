	.file	"test.cpp"
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
	.ascii "\12=> Mat %s[%d, %d, b8]: \12\0"
.LC1:
	.ascii "%d \11\0"
.LC2:
	.ascii "\12\0"
	.text
	.globl	_Z8printmatPKcPhii
	.def	_Z8printmatPKcPhii;	.scl	2;	.type	32;	.endef
_Z8printmatPKcPhii:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	subq	$48, %rsp	 #,
	movq	%rcx, 16(%rbp)	 # name, name
	movq	%rdx, 24(%rbp)	 # a, a
	movl	%r8d, 32(%rbp)	 # m, m
	movl	%r9d, 40(%rbp)	 # n, n
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:11: 	printf("\n=> Mat %s[%d, %d, b8]: \n", name, m, n);  // 打印矩阵维度
	movl	40(%rbp), %edx	 # n, tmp92
	movl	32(%rbp), %eax	 # m, tmp93
	movl	%edx, %r9d	 # tmp92,
	movl	%eax, %r8d	 # tmp93,
	movq	16(%rbp), %rdx	 # name,
	leaq	.LC0(%rip), %rax	 #, tmp94
	movq	%rax, %rcx	 # tmp94,
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:12: 	int i, j, k = 0;
	movl	$0, -12(%rbp)	 #, k
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:13: 	for (i = 0; i < m; i++) {
	movl	$0, -4(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:13: 	for (i = 0; i < m; i++) {
	jmp	.L4	 #
.L7:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:14: 		for (j = 0; j < n; j++, k++) {
	movl	$0, -8(%rbp)	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:14: 		for (j = 0; j < n; j++, k++) {
	jmp	.L5	 #
.L6:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:15: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0); // 打印矩阵元素，保留两位小数
	movl	-12(%rbp), %eax	 # k, tmp95
	leal	7(%rax), %edx	 #, tmp97
	testl	%eax, %eax	 # tmp96
	cmovs	%edx, %eax	 # tmp97,, tmp96
	sarl	$3, %eax	 #, tmp98
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:15: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0); // 打印矩阵元素，保留两位小数
	movq	24(%rbp), %rdx	 # a, tmp99
	addq	%rdx, %rax	 # tmp99, _3
	movzbl	(%rax), %eax	 # *_3, _4
	movzbl	%al, %edx	 # _4, _5
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:15: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0); // 打印矩阵元素，保留两位小数
	movl	-12(%rbp), %eax	 # k, tmp100
	andl	$7, %eax	 #, _6
	movl	%eax, %ecx	 # _6, tmp107
	sall	%cl, %edx	 # tmp107, _5
	movl	%edx, %eax	 # _5, _7
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:15: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0); // 打印矩阵元素，保留两位小数
	andl	$128, %eax	 #, _8
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:15: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0); // 打印矩阵元素，保留两位小数
	testl	%eax, %eax	 # _8
	setg	%al	 #, _9
	movzbl	%al, %eax	 # _9, _10
	movl	%eax, %edx	 # _10,
	leaq	.LC1(%rip), %rax	 #, tmp101
	movq	%rax, %rcx	 # tmp101,
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:14: 		for (j = 0; j < n; j++, k++) {
	addl	$1, -8(%rbp)	 #, j
	addl	$1, -12(%rbp)	 #, k
.L5:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:14: 		for (j = 0; j < n; j++, k++) {
	movl	-8(%rbp), %eax	 # j, tmp102
	cmpl	40(%rbp), %eax	 # n, tmp102
	jl	.L6	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:17: 		printf("\n");
	leaq	.LC2(%rip), %rax	 #, tmp103
	movq	%rax, %rcx	 # tmp103,
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:13: 	for (i = 0; i < m; i++) {
	addl	$1, -4(%rbp)	 #, i
.L4:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:13: 	for (i = 0; i < m; i++) {
	movl	-4(%rbp), %eax	 # i, tmp104
	cmpl	32(%rbp), %eax	 # m, tmp104
	jl	.L7	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:19: }
	nop	
	nop	
	leave	
	ret	
	.section .rdata,"dr"
.LC3:
	.ascii "\12=> Mat %s %x [%d, %d, i8]: \12\0"
.LC5:
	.ascii "%.6f\11\0"
	.text
	.globl	_Z8printmatPKcPaii
	.def	_Z8printmatPKcPaii;	.scl	2;	.type	32;	.endef
_Z8printmatPKcPaii:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	subq	$64, %rsp	 #,
	movq	%rcx, 16(%rbp)	 # name, name
	movq	%rdx, 24(%rbp)	 # a, a
	movl	%r8d, 32(%rbp)	 # m, m
	movl	%r9d, 40(%rbp)	 # n, n
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:22: 	printf("\n=> Mat %s %x [%d, %d, i8]: \n", name, a, m, n); // 打印矩阵维度
	movl	32(%rbp), %ecx	 # m, tmp89
	movq	24(%rbp), %rdx	 # a, tmp90
	movl	40(%rbp), %eax	 # n, tmp91
	movl	%eax, 32(%rsp)	 # tmp91,
	movl	%ecx, %r9d	 # tmp89,
	movq	%rdx, %r8	 # tmp90,
	movq	16(%rbp), %rdx	 # name,
	leaq	.LC3(%rip), %rax	 #, tmp92
	movq	%rax, %rcx	 # tmp92,
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:23: 	int i, j, k = 0;
	movl	$0, -12(%rbp)	 #, k
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:24: 	for (i = 0; i < m; i++) {
	movl	$0, -4(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:24: 	for (i = 0; i < m; i++) {
	jmp	.L9	 #
.L12:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:25: 		for (j = 0; j < n; j++, k++) {
	movl	$0, -8(%rbp)	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:25: 		for (j = 0; j < n; j++, k++) {
	jmp	.L10	 #
.L11:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:26: 			printf("%.6f\t", a[i * n + j] / 64. ); // 打印矩阵元素，保留两位小数
	movl	-4(%rbp), %eax	 # i, tmp93
	imull	40(%rbp), %eax	 # n, _1
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:26: 			printf("%.6f\t", a[i * n + j] / 64. ); // 打印矩阵元素，保留两位小数
	movl	-8(%rbp), %edx	 # j, tmp94
	addl	%edx, %eax	 # tmp94, _2
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:26: 			printf("%.6f\t", a[i * n + j] / 64. ); // 打印矩阵元素，保留两位小数
	movq	24(%rbp), %rdx	 # a, tmp95
	addq	%rdx, %rax	 # tmp95, _4
	movzbl	(%rax), %eax	 # *_4, _5
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:26: 			printf("%.6f\t", a[i * n + j] / 64. ); // 打印矩阵元素，保留两位小数
	movsbl	%al, %eax	 # _5, tmp96
	pxor	%xmm0, %xmm0	 # _6
	cvtsi2sdl	%eax, %xmm0	 # tmp96, _6
	movsd	.LC4(%rip), %xmm1	 #, tmp97
	divsd	%xmm1, %xmm0	 # tmp97, _7
	movq	%xmm0, %rax	 # _7, tmp98
	movq	%rax, %rdx	 # tmp98, tmp99
	movq	%rdx, %xmm0	 # tmp99, tmp101
	movapd	%xmm0, %xmm1	 # tmp101,
	movq	%rax, %rdx	 # tmp102,
	leaq	.LC5(%rip), %rax	 #, tmp103
	movq	%rax, %rcx	 # tmp103,
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:25: 		for (j = 0; j < n; j++, k++) {
	addl	$1, -8(%rbp)	 #, j
	addl	$1, -12(%rbp)	 #, k
.L10:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:25: 		for (j = 0; j < n; j++, k++) {
	movl	-8(%rbp), %eax	 # j, tmp104
	cmpl	40(%rbp), %eax	 # n, tmp104
	jl	.L11	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:28: 		printf("\n");
	leaq	.LC2(%rip), %rax	 #, tmp105
	movq	%rax, %rcx	 # tmp105,
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:24: 	for (i = 0; i < m; i++) {
	addl	$1, -4(%rbp)	 #, i
.L9:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:24: 	for (i = 0; i < m; i++) {
	movl	-4(%rbp), %eax	 # i, tmp106
	cmpl	32(%rbp), %eax	 # m, tmp106
	jl	.L12	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:30: }
	nop	
	nop	
	leave	
	ret	
	.section .rdata,"dr"
	.align 8
.LC6:
	.ascii "\12=> Mat %s %x [%d, %d, i32]: \12\0"
	.text
	.globl	_Z8printmatPKcPiii
	.def	_Z8printmatPKcPiii;	.scl	2;	.type	32;	.endef
_Z8printmatPKcPiii:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	subq	$64, %rsp	 #,
	movq	%rcx, 16(%rbp)	 # name, name
	movq	%rdx, 24(%rbp)	 # a, a
	movl	%r8d, 32(%rbp)	 # m, m
	movl	%r9d, 40(%rbp)	 # n, n
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:34: 	printf("\n=> Mat %s %x [%d, %d, i32]: \n", name, a, m, n);  // 打印矩阵维度
	movl	32(%rbp), %ecx	 # m, tmp90
	movq	24(%rbp), %rdx	 # a, tmp91
	movl	40(%rbp), %eax	 # n, tmp92
	movl	%eax, 32(%rsp)	 # tmp92,
	movl	%ecx, %r9d	 # tmp90,
	movq	%rdx, %r8	 # tmp91,
	movq	16(%rbp), %rdx	 # name,
	leaq	.LC6(%rip), %rax	 #, tmp93
	movq	%rax, %rcx	 # tmp93,
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:35: 	int i, j, k = 0;
	movl	$0, -12(%rbp)	 #, k
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:36: 	for (i = 0; i < m; i++) {
	movl	$0, -4(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:36: 	for (i = 0; i < m; i++) {
	jmp	.L14	 #
.L17:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:37: 		for (j = 0; j < n; j++, k++) {
	movl	$0, -8(%rbp)	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:37: 		for (j = 0; j < n; j++, k++) {
	jmp	.L15	 #
.L16:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:38: 			printf("%.6f\t", a[i * n + j] / 64. ); // 打印矩阵元素，保留两位小数
	movl	-4(%rbp), %eax	 # i, tmp94
	imull	40(%rbp), %eax	 # n, _1
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:38: 			printf("%.6f\t", a[i * n + j] / 64. ); // 打印矩阵元素，保留两位小数
	movl	-8(%rbp), %edx	 # j, tmp95
	addl	%edx, %eax	 # tmp95, _2
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:38: 			printf("%.6f\t", a[i * n + j] / 64. ); // 打印矩阵元素，保留两位小数
	leaq	0(,%rax,4), %rdx	 #, _4
	movq	24(%rbp), %rax	 # a, tmp96
	addq	%rdx, %rax	 # _4, _5
	movl	(%rax), %eax	 # *_5, _6
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:38: 			printf("%.6f\t", a[i * n + j] / 64. ); // 打印矩阵元素，保留两位小数
	pxor	%xmm0, %xmm0	 # _7
	cvtsi2sdl	%eax, %xmm0	 # _6, _7
	movsd	.LC4(%rip), %xmm1	 #, tmp97
	divsd	%xmm1, %xmm0	 # tmp97, _8
	movq	%xmm0, %rax	 # _8, tmp98
	movq	%rax, %rdx	 # tmp98, tmp99
	movq	%rdx, %xmm0	 # tmp99, tmp101
	movapd	%xmm0, %xmm1	 # tmp101,
	movq	%rax, %rdx	 # tmp102,
	leaq	.LC5(%rip), %rax	 #, tmp103
	movq	%rax, %rcx	 # tmp103,
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:37: 		for (j = 0; j < n; j++, k++) {
	addl	$1, -8(%rbp)	 #, j
	addl	$1, -12(%rbp)	 #, k
.L15:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:37: 		for (j = 0; j < n; j++, k++) {
	movl	-8(%rbp), %eax	 # j, tmp104
	cmpl	40(%rbp), %eax	 # n, tmp104
	jl	.L16	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:40: 		printf("\n");
	leaq	.LC2(%rip), %rax	 #, tmp105
	movq	%rax, %rcx	 # tmp105,
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:36: 	for (i = 0; i < m; i++) {
	addl	$1, -4(%rbp)	 #, i
.L14:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:36: 	for (i = 0; i < m; i++) {
	movl	-4(%rbp), %eax	 # i, tmp106
	cmpl	32(%rbp), %eax	 # m, tmp106
	jl	.L17	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:42: }
	nop	
	nop	
	leave	
	ret	
	.globl	_Z18i32_i8_vec_in_prodPiPai
	.def	_Z18i32_i8_vec_in_prodPiPai;	.scl	2;	.type	32;	.endef
_Z18i32_i8_vec_in_prodPiPai:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	subq	$16, %rsp	 #,
	movq	%rcx, 16(%rbp)	 # data, data
	movq	%rdx, 24(%rbp)	 # weight, weight
	movl	%r8d, 32(%rbp)	 # n, n
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	i32 res = 0;  // 初始化结果为0, 结果要除以64,解析成(-1,1)的8字节数
	movl	$0, -4(%rbp)	 #, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 	for (int i = 0; i < n; i++) // 遍历向量元素
	movl	$0, -8(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 	for (int i = 0; i < n; i++) // 遍历向量元素
	jmp	.L19	 #
.L20:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:48: 		res += data[i] * weight[i];
	movl	-8(%rbp), %eax	 # i, tmp93
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:48: 		res += data[i] * weight[i];
	leaq	0(,%rax,4), %rdx	 #, _2
	movq	16(%rbp), %rax	 # data, tmp94
	addq	%rdx, %rax	 # _2, _3
	movl	(%rax), %edx	 # *_3, _4
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:48: 		res += data[i] * weight[i];
	movl	-8(%rbp), %eax	 # i, tmp95
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:48: 		res += data[i] * weight[i];
	movq	24(%rbp), %rcx	 # weight, tmp96
	addq	%rcx, %rax	 # tmp96, _6
	movzbl	(%rax), %eax	 # *_6, _7
	movsbl	%al, %eax	 # _7, _8
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:48: 		res += data[i] * weight[i];
	imull	%edx, %eax	 # _4, _9
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:48: 		res += data[i] * weight[i];
	addl	%eax, -4(%rbp)	 # _9, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 	for (int i = 0; i < n; i++) // 遍历向量元素
	addl	$1, -8(%rbp)	 #, i
.L19:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 	for (int i = 0; i < n; i++) // 遍历向量元素
	movl	-8(%rbp), %eax	 # i, tmp97
	cmpl	32(%rbp), %eax	 # n, tmp97
	jl	.L20	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:51: 	return res >> 6; // 返回内积结果
	movl	-4(%rbp), %eax	 # res, tmp98
	sarl	$6, %eax	 #, _15
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:52: }
	leave	
	ret	
	.globl	_Z18i32_b8_vec_in_prodPiPhii
	.def	_Z18i32_b8_vec_in_prodPiPhii;	.scl	2;	.type	32;	.endef
_Z18i32_b8_vec_in_prodPiPhii:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	subq	$32, %rsp	 #,
	movq	%rcx, 16(%rbp)	 # data, data
	movq	%rdx, 24(%rbp)	 # weight, weight
	movl	%r8d, 32(%rbp)	 # n, n
	movl	%r9d, 40(%rbp)	 # b_start, b_start
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:56: 	int i = 0, j = 0;
	movl	$0, -4(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:56: 	int i = 0, j = 0;
	movl	$0, -8(%rbp)	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:57: 	i32 res = 0;  // 初始化结果为0, 结果要除以127,解析成(-1,1)的8字节数
	movl	$0, -12(%rbp)	 #, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:59: 	b8 p = (*weight) << b_start;  // 将bit向量左移b_start位
	movq	24(%rbp), %rax	 # weight, tmp98
	movzbl	(%rax), %eax	 # *weight_35(D), _1
	movzbl	%al, %edx	 # _1, _2
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:59: 	b8 p = (*weight) << b_start;  // 将bit向量左移b_start位
	movl	40(%rbp), %eax	 # b_start, tmp99
	movl	%eax, %ecx	 # tmp99, tmp120
	sall	%cl, %edx	 # tmp120, _2
	movl	%edx, %eax	 # _2, _3
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:59: 	b8 p = (*weight) << b_start;  // 将bit向量左移b_start位
	movb	%al, -13(%rbp)	 # _3, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:60: 	int n0 = (8 - b_start) < n ? (8 - b_start) : n;  // 计算开头不满足8位的部分长度
	movl	$8, %eax	 #, tmp100
	subl	40(%rbp), %eax	 # b_start, _4
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:60: 	int n0 = (8 - b_start) < n ? (8 - b_start) : n;  // 计算开头不满足8位的部分长度
	movl	32(%rbp), %edx	 # n, tmp102
	cmpl	%eax, %edx	 # _4, tmp102
	cmovle	%edx, %eax	 # tmp102,, tmp101
	movl	%eax, -20(%rbp)	 # tmp101, n0
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:63: 	for (; i < n0; i++, data++, p <<= 1)
	jmp	.L23	 #
.L26:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:64: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movzbl	-13(%rbp), %eax	 # p, p.2_5
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:64: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	testb	%al, %al	 # p.2_5
	jns	.L24	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:64: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movq	16(%rbp), %rax	 # data, tmp103
	movl	(%rax), %eax	 # *data_12, iftmp.1_28
	jmp	.L25	 #
.L24:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:64: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movq	16(%rbp), %rax	 # data, tmp104
	movl	(%rax), %eax	 # *data_12, _6
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:64: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	negl	%eax	 # iftmp.1_28
.L25:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:64: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	addl	%eax, -12(%rbp)	 # iftmp.1_28, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:63: 	for (; i < n0; i++, data++, p <<= 1)
	addl	$1, -4(%rbp)	 #, i
	addq	$4, 16(%rbp)	 #, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:63: 	for (; i < n0; i++, data++, p <<= 1)
	salb	-13(%rbp)	 # p
.L23:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:63: 	for (; i < n0; i++, data++, p <<= 1)
	movl	-4(%rbp), %eax	 # i, tmp105
	cmpl	-20(%rbp), %eax	 # n0, tmp105
	jl	.L26	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:67: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	addq	$1, 24(%rbp)	 #, weight
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:67: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	jmp	.L27	 #
.L32:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	movl	$0, -8(%rbp)	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	movq	24(%rbp), %rax	 # weight, tmp106
	movzbl	(%rax), %eax	 # *weight_16, tmp107
	movb	%al, -13(%rbp)	 # tmp107, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	jmp	.L28	 #
.L31:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:69: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movzbl	-13(%rbp), %eax	 # p, p.4_7
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:69: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	testb	%al, %al	 # p.4_7
	jns	.L29	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:69: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movq	16(%rbp), %rax	 # data, tmp108
	movl	(%rax), %eax	 # *data_13, iftmp.3_29
	jmp	.L30	 #
.L29:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:69: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movq	16(%rbp), %rax	 # data, tmp109
	movl	(%rax), %eax	 # *data_13, _8
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:69: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	negl	%eax	 # iftmp.3_29
.L30:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:69: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addl	%eax, -12(%rbp)	 # iftmp.3_29, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	addl	$1, -8(%rbp)	 #, j
	addq	$4, 16(%rbp)	 #, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	salb	-13(%rbp)	 # p
.L28:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	cmpl	$7, -8(%rbp)	 #, j
	jle	.L31	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:67: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	addl	$8, -4(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:67: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	addq	$1, 24(%rbp)	 #, weight
.L27:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:67: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	movl	32(%rbp), %eax	 # n, tmp110
	subl	$7, %eax	 #, _9
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:67: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	cmpl	%eax, -4(%rbp)	 # _9, i
	jl	.L32	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:72: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	movq	24(%rbp), %rax	 # weight, tmp111
	movzbl	(%rax), %eax	 # *weight_16, tmp112
	movb	%al, -13(%rbp)	 # tmp112, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:72: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	jmp	.L33	 #
.L36:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:73: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movzbl	-13(%rbp), %eax	 # p, p.6_10
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:73: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	testb	%al, %al	 # p.6_10
	jns	.L34	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:73: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movq	16(%rbp), %rax	 # data, tmp113
	movl	(%rax), %eax	 # *data_15, iftmp.5_30
	jmp	.L35	 #
.L34:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:73: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movq	16(%rbp), %rax	 # data, tmp114
	movl	(%rax), %eax	 # *data_15, _11
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:73: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	negl	%eax	 # iftmp.5_30
.L35:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:73: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	addl	%eax, -12(%rbp)	 # iftmp.5_30, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:72: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	addl	$1, -4(%rbp)	 #, i
	addq	$4, 16(%rbp)	 #, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:72: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	salb	-13(%rbp)	 # p
.L33:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:72: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	movl	-4(%rbp), %eax	 # i, tmp115
	cmpl	32(%rbp), %eax	 # n, tmp115
	jl	.L36	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:75: 	return res >> 6;  // 返回内积结果, 乘法的结果要除以 6
	movl	-12(%rbp), %eax	 # res, tmp116
	sarl	$6, %eax	 #, _43
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:76: }
	leave	
	ret	
	.globl	_Z10i32_i8_mulPiPaS_iiib
	.def	_Z10i32_i8_mulPiPaS_iiib;	.scl	2;	.type	32;	.endef
_Z10i32_i8_mulPiPaS_iiib:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	subq	$64, %rsp	 #,
	movq	%rcx, 16(%rbp)	 # data, data
	movq	%rdx, 24(%rbp)	 # weight, weight
	movq	%r8, 32(%rbp)	 # out, out
	movl	%r9d, 40(%rbp)	 # m, m
	movl	64(%rbp), %eax	 # Transpose, tmp98
	movb	%al, -20(%rbp)	 # tmp99, Transpose
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:83: 	if (Transpose) {  // 如果需要转置输出
	cmpb	$0, -20(%rbp)	 #, Transpose
	je	.L39	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:84: 		for (int j = 0; j < n; j++) { // 遍历矩阵B的行
	movl	$0, -4(%rbp)	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:84: 		for (int j = 0; j < n; j++) { // 遍历矩阵B的行
	jmp	.L40	 #
.L43:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:85: 			for (int i = 0; i < m; i++,  out++)  // 遍历矩阵A的行
	movl	$0, -8(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:85: 			for (int i = 0; i < m; i++,  out++)  // 遍历矩阵A的行
	jmp	.L41	 #
.L42:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:86: 				*out = i32_i8_vec_in_prod(data + i * k, weight + j * k, k); // 计算内积并存储到输出矩阵
	movl	-4(%rbp), %eax	 # j, tmp100
	imull	48(%rbp), %eax	 # k, _1
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:86: 				*out = i32_i8_vec_in_prod(data + i * k, weight + j * k, k); // 计算内积并存储到输出矩阵
	movq	24(%rbp), %rdx	 # weight, tmp101
	addq	%rax, %rdx	 # _2, _3
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:86: 				*out = i32_i8_vec_in_prod(data + i * k, weight + j * k, k); // 计算内积并存储到输出矩阵
	movl	-8(%rbp), %eax	 # i, tmp102
	imull	48(%rbp), %eax	 # k, _4
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:86: 				*out = i32_i8_vec_in_prod(data + i * k, weight + j * k, k); // 计算内积并存储到输出矩阵
	leaq	0(,%rax,4), %rcx	 #, _6
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:86: 				*out = i32_i8_vec_in_prod(data + i * k, weight + j * k, k); // 计算内积并存储到输出矩阵
	movq	16(%rbp), %rax	 # data, tmp103
	addq	%rcx, %rax	 # _6, _7
	movl	48(%rbp), %ecx	 # k, tmp104
	movl	%ecx, %r8d	 # tmp104,
	movq	%rax, %rcx	 # _7,
	call	_Z18i32_i8_vec_in_prodPiPai	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:86: 				*out = i32_i8_vec_in_prod(data + i * k, weight + j * k, k); // 计算内积并存储到输出矩阵
	movq	32(%rbp), %rdx	 # out, tmp105
	movl	%eax, (%rdx)	 # _8, *out_17
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:85: 			for (int i = 0; i < m; i++,  out++)  // 遍历矩阵A的行
	addl	$1, -8(%rbp)	 #, i
	addq	$4, 32(%rbp)	 #, out
.L41:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:85: 			for (int i = 0; i < m; i++,  out++)  // 遍历矩阵A的行
	movl	-8(%rbp), %eax	 # i, tmp106
	cmpl	40(%rbp), %eax	 # m, tmp106
	jl	.L42	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:84: 		for (int j = 0; j < n; j++) { // 遍历矩阵B的行
	addl	$1, -4(%rbp)	 #, j
.L40:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:84: 		for (int j = 0; j < n; j++) { // 遍历矩阵B的行
	movl	-4(%rbp), %eax	 # j, tmp107
	cmpl	56(%rbp), %eax	 # n, tmp107
	jl	.L43	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:94: }
	jmp	.L49	 #
.L39:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:89: 		for (int i = 0; i < m; i++) {		 // 遍历矩阵A的行
	movl	$0, -12(%rbp)	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:89: 		for (int i = 0; i < m; i++) {		 // 遍历矩阵A的行
	jmp	.L45	 #
.L48:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:90: 			for (int j = 0; j < n; j++, out++) // 遍历矩阵B的行
	movl	$0, -16(%rbp)	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:90: 			for (int j = 0; j < n; j++, out++) // 遍历矩阵B的行
	jmp	.L46	 #
.L47:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:91: 				*out = i32_i8_vec_in_prod(data + i * k, weight + j * k, k); // 计算内积并存储到输出矩阵
	movl	-16(%rbp), %eax	 # j, tmp108
	imull	48(%rbp), %eax	 # k, _9
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:91: 				*out = i32_i8_vec_in_prod(data + i * k, weight + j * k, k); // 计算内积并存储到输出矩阵
	movq	24(%rbp), %rdx	 # weight, tmp109
	addq	%rax, %rdx	 # _10, _11
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:91: 				*out = i32_i8_vec_in_prod(data + i * k, weight + j * k, k); // 计算内积并存储到输出矩阵
	movl	-12(%rbp), %eax	 # i, tmp110
	imull	48(%rbp), %eax	 # k, _12
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:91: 				*out = i32_i8_vec_in_prod(data + i * k, weight + j * k, k); // 计算内积并存储到输出矩阵
	leaq	0(,%rax,4), %rcx	 #, _14
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:91: 				*out = i32_i8_vec_in_prod(data + i * k, weight + j * k, k); // 计算内积并存储到输出矩阵
	movq	16(%rbp), %rax	 # data, tmp111
	addq	%rcx, %rax	 # _14, _15
	movl	48(%rbp), %ecx	 # k, tmp112
	movl	%ecx, %r8d	 # tmp112,
	movq	%rax, %rcx	 # _15,
	call	_Z18i32_i8_vec_in_prodPiPai	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:91: 				*out = i32_i8_vec_in_prod(data + i * k, weight + j * k, k); // 计算内积并存储到输出矩阵
	movq	32(%rbp), %rdx	 # out, tmp113
	movl	%eax, (%rdx)	 # _16, *out_19
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:90: 			for (int j = 0; j < n; j++, out++) // 遍历矩阵B的行
	addl	$1, -16(%rbp)	 #, j
	addq	$4, 32(%rbp)	 #, out
.L46:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:90: 			for (int j = 0; j < n; j++, out++) // 遍历矩阵B的行
	movl	-16(%rbp), %eax	 # j, tmp114
	cmpl	56(%rbp), %eax	 # n, tmp114
	jl	.L47	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:89: 		for (int i = 0; i < m; i++) {		 // 遍历矩阵A的行
	addl	$1, -12(%rbp)	 #, i
.L45:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:89: 		for (int i = 0; i < m; i++) {		 // 遍历矩阵A的行
	movl	-12(%rbp), %eax	 # i, tmp115
	cmpl	40(%rbp), %eax	 # m, tmp115
	jl	.L48	 #,
.L49:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:94: }
	nop	
	leave	
	ret	
	.section .rdata,"dr"
.LC7:
	.ascii "a\0"
.LC8:
	.ascii "b\0"
.LC9:
	.ascii "res\0"
	.text
	.globl	_Z16test_i32_i8_prodv
	.def	_Z16test_i32_i8_prodv;	.scl	2;	.type	32;	.endef
_Z16test_i32_i8_prodv:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	subq	$64, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:4: 	i32 a[3] = {0, 1, 2}; // 0, 1/64 , 2/64
	movl	$0, -12(%rbp)	 #, a[0]
	movl	$1, -8(%rbp)	 #, a[1]
	movl	$2, -4(%rbp)	 #, a[2]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:5: 	i8 b[3] = {64, 64, 64};
	movw	$16448, -15(%rbp)	 #, b
	movb	$64, -13(%rbp)	 #, b
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:6: 	printmat("a", a, 1, 3);
	leaq	-12(%rbp), %rax	 #, tmp83
	movl	$3, %r9d	 #,
	movl	$1, %r8d	 #,
	movq	%rax, %rdx	 # tmp83,
	leaq	.LC7(%rip), %rax	 #, tmp84
	movq	%rax, %rcx	 # tmp84,
	call	_Z8printmatPKcPiii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:7: 	printmat("b", b, 3, 1);
	leaq	-15(%rbp), %rax	 #, tmp85
	movl	$1, %r9d	 #,
	movl	$3, %r8d	 #,
	movq	%rax, %rdx	 # tmp85,
	leaq	.LC8(%rip), %rax	 #, tmp86
	movq	%rax, %rcx	 # tmp86,
	call	_Z8printmatPKcPaii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:8: 	i32 res = i32_i8_vec_in_prod(a, b, 3);
	leaq	-15(%rbp), %rdx	 #, tmp87
	leaq	-12(%rbp), %rax	 #, tmp88
	movl	$3, %r8d	 #,
	movq	%rax, %rcx	 # tmp88,
	call	_Z18i32_i8_vec_in_prodPiPai	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:8: 	i32 res = i32_i8_vec_in_prod(a, b, 3);
	movl	%eax, -20(%rbp)	 # _1, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:9: 	printmat("res", &res, 1, 1);
	leaq	-20(%rbp), %rax	 #, tmp89
	movl	$1, %r9d	 #,
	movl	$1, %r8d	 #,
	movq	%rax, %rdx	 # tmp89,
	leaq	.LC9(%rip), %rax	 #, tmp90
	movq	%rax, %rcx	 # tmp90,
	call	_Z8printmatPKcPiii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:11: }
	nop	
	leave	
	ret	
	.section .rdata,"dr"
.LC10:
	.ascii "c\0"
	.text
	.globl	_Z15test_i32_i8_mulv
	.def	_Z15test_i32_i8_mulv;	.scl	2;	.type	32;	.endef
_Z15test_i32_i8_mulv:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	andq	$-16, %rsp	 #,
	addq	$-128, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:13: 	i32 a[6] = {1, 1, 2,
	movl	$1, 96(%rsp)	 #, a[0]
	movl	$1, 100(%rsp)	 #, a[1]
	movl	$2, 104(%rsp)	 #, a[2]
	movl	$1, 108(%rsp)	 #, a[3]
	movl	$1, 112(%rsp)	 #, a[4]
	movl	$4, 116(%rsp)	 #, a[5]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:16: 	i8 b[6] = {64, 64, 32,
	movl	$2113600, 90(%rsp)	 #, b
	movw	$16448, 94(%rsp)	 #, b
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:20: 	printmat("a", a, 2, 3);
	leaq	96(%rsp), %rax	 #, tmp82
	movl	$3, %r9d	 #,
	movl	$2, %r8d	 #,
	movq	%rax, %rdx	 # tmp82,
	leaq	.LC7(%rip), %rax	 #, tmp83
	movq	%rax, %rcx	 # tmp83,
	call	_Z8printmatPKcPiii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:21: 	printmat("b", b, 2, 3);
	leaq	90(%rsp), %rax	 #, tmp84
	movl	$3, %r9d	 #,
	movl	$2, %r8d	 #,
	movq	%rax, %rdx	 # tmp84,
	leaq	.LC8(%rip), %rax	 #, tmp85
	movq	%rax, %rcx	 # tmp85,
	call	_Z8printmatPKcPaii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:22: 	i32_i8_mul(a, b, c, 2, 3, 2, false);
	leaq	76(%rsp), %rcx	 #, tmp86
	leaq	90(%rsp), %rdx	 #, tmp87
	leaq	96(%rsp), %rax	 #, tmp88
	movl	$0, 48(%rsp)	 #,
	movl	$2, 40(%rsp)	 #,
	movl	$3, 32(%rsp)	 #,
	movl	$2, %r9d	 #,
	movq	%rcx, %r8	 # tmp86,
	movq	%rax, %rcx	 # tmp88,
	call	_Z10i32_i8_mulPiPaS_iiib	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:23: 	printmat("c", c, 2, 2);
	leaq	76(%rsp), %rax	 #, tmp89
	movl	$2, %r9d	 #,
	movl	$2, %r8d	 #,
	movq	%rax, %rdx	 # tmp89,
	leaq	.LC10(%rip), %rax	 #, tmp90
	movq	%rax, %rcx	 # tmp90,
	call	_Z8printmatPKcPiii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:25: 	printmat("a", a, 2, 3);
	leaq	96(%rsp), %rax	 #, tmp91
	movl	$3, %r9d	 #,
	movl	$2, %r8d	 #,
	movq	%rax, %rdx	 # tmp91,
	leaq	.LC7(%rip), %rax	 #, tmp92
	movq	%rax, %rcx	 # tmp92,
	call	_Z8printmatPKcPiii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:26: 	printmat("b", b, 2, 3);
	leaq	90(%rsp), %rax	 #, tmp93
	movl	$3, %r9d	 #,
	movl	$2, %r8d	 #,
	movq	%rax, %rdx	 # tmp93,
	leaq	.LC8(%rip), %rax	 #, tmp94
	movq	%rax, %rcx	 # tmp94,
	call	_Z8printmatPKcPaii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:27: 	i32_i8_mul(a, b, c, 2, 3, 2, false);
	leaq	76(%rsp), %rcx	 #, tmp95
	leaq	90(%rsp), %rdx	 #, tmp96
	leaq	96(%rsp), %rax	 #, tmp97
	movl	$0, 48(%rsp)	 #,
	movl	$2, 40(%rsp)	 #,
	movl	$3, 32(%rsp)	 #,
	movl	$2, %r9d	 #,
	movq	%rcx, %r8	 # tmp95,
	movq	%rax, %rcx	 # tmp97,
	call	_Z10i32_i8_mulPiPaS_iiib	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:28: 	printmat("c", c, 2, 2);
	leaq	76(%rsp), %rax	 #, tmp98
	movl	$2, %r9d	 #,
	movl	$2, %r8d	 #,
	movq	%rax, %rdx	 # tmp98,
	leaq	.LC10(%rip), %rax	 #, tmp99
	movq	%rax, %rcx	 # tmp99,
	call	_Z8printmatPKcPiii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:30: }
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
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:33: int main() {
	call	__main	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:35: 	test_i32_i8_mul();
	call	_Z15test_i32_i8_mulv	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test.cpp:36: }
	movl	$0, %eax	 #, _3
	leave	
	ret	
	.section .rdata,"dr"
	.align 8
.LC4:
	.long	0
	.long	1078984704
	.ident	"GCC: (x86_64-posix-seh, Built by MinGW-Builds project) 11.4.0"
	.def	__mingw_vfprintf;	.scl	2;	.type	32;	.endef
