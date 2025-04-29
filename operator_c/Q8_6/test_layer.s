	.file	"test_layer.cpp"
 # GNU C++17 (x86_64-posix-seh, Built by MinGW-Builds project) version 11.4.0 (x86_64-w64-mingw32)
 #	compiled by GNU C version 11.4.0, GMP version 6.2.1, MPFR version 4.1.0, MPC version 1.2.1, isl version isl-0.25-GMP

 # GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
 # options passed: -mtune=core2 -march=nocona -O3 -fno-asynchronous-unwind-tables
	.text
	.section .rdata,"dr"
.LC0:
	.ascii "\12\0"
	.text
	.p2align 4
	.def	_Z6printfPKcz.constprop.0;	.scl	3;	.type	32;	.endef
_Z6printfPKcz.constprop.0:
	pushq	%rbx	 #
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:375:   __retval = __mingw_vfprintf( stdout, __format, __local_argv );
	movl	$1, %ecx	 #,
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:371: int printf (const char *__format, ...)
	subq	$48, %rsp	 #,
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:374:   __builtin_va_list __local_argv; __builtin_va_start( __local_argv, __format );
	leaq	72(%rsp), %rbx	 #, tmp86
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:371: int printf (const char *__format, ...)
	movq	%rdx, 72(%rsp)	 #,
	movq	%r8, 80(%rsp)	 #,
	movq	%r9, 88(%rsp)	 #,
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:374:   __builtin_va_list __local_argv; __builtin_va_start( __local_argv, __format );
	movq	%rbx, 40(%rsp)	 # tmp86, MEM[(char * *)&__local_argv]
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:375:   __retval = __mingw_vfprintf( stdout, __format, __local_argv );
	call	*__imp___acrt_iob_func(%rip)	 #
	leaq	.LC0(%rip), %rdx	 #, tmp88
	movq	%rbx, %r8	 # tmp86,
	movq	%rax, %rcx	 # tmp90, _2
	call	__mingw_vfprintf	 #
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:378: }
	addq	$48, %rsp	 #,
	popq	%rbx	 #
	ret	
	.section .rdata,"dr"
.LC1:
	.ascii "%.6f\11\0"
	.text
	.p2align 4
	.def	_Z6printfPKcz.constprop.1;	.scl	3;	.type	32;	.endef
_Z6printfPKcz.constprop.1:
	pushq	%rbx	 #
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:375:   __retval = __mingw_vfprintf( stdout, __format, __local_argv );
	movl	$1, %ecx	 #,
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:371: int printf (const char *__format, ...)
	subq	$48, %rsp	 #,
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:374:   __builtin_va_list __local_argv; __builtin_va_start( __local_argv, __format );
	leaq	72(%rsp), %rbx	 #, tmp86
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:371: int printf (const char *__format, ...)
	movq	%rdx, 72(%rsp)	 #,
	movq	%r8, 80(%rsp)	 #,
	movq	%r9, 88(%rsp)	 #,
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:374:   __builtin_va_list __local_argv; __builtin_va_start( __local_argv, __format );
	movq	%rbx, 40(%rsp)	 # tmp86, MEM[(char * *)&__local_argv]
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:375:   __retval = __mingw_vfprintf( stdout, __format, __local_argv );
	call	*__imp___acrt_iob_func(%rip)	 #
	leaq	.LC1(%rip), %rdx	 #, tmp88
	movq	%rbx, %r8	 # tmp86,
	movq	%rax, %rcx	 # tmp90, _2
	call	__mingw_vfprintf	 #
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:378: }
	addq	$48, %rsp	 #,
	popq	%rbx	 #
	ret	
	.section .rdata,"dr"
.LC2:
	.ascii "%d \11\0"
	.text
	.p2align 4
	.def	_Z6printfPKcz.constprop.2;	.scl	3;	.type	32;	.endef
_Z6printfPKcz.constprop.2:
	pushq	%rbx	 #
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:375:   __retval = __mingw_vfprintf( stdout, __format, __local_argv );
	movl	$1, %ecx	 #,
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:371: int printf (const char *__format, ...)
	subq	$48, %rsp	 #,
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:374:   __builtin_va_list __local_argv; __builtin_va_start( __local_argv, __format );
	leaq	72(%rsp), %rbx	 #, tmp86
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:371: int printf (const char *__format, ...)
	movq	%rdx, 72(%rsp)	 #,
	movq	%r8, 80(%rsp)	 #,
	movq	%r9, 88(%rsp)	 #,
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:374:   __builtin_va_list __local_argv; __builtin_va_start( __local_argv, __format );
	movq	%rbx, 40(%rsp)	 # tmp86, MEM[(char * *)&__local_argv]
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:375:   __retval = __mingw_vfprintf( stdout, __format, __local_argv );
	call	*__imp___acrt_iob_func(%rip)	 #
	leaq	.LC2(%rip), %rdx	 #, tmp88
	movq	%rbx, %r8	 # tmp86,
	movq	%rax, %rcx	 # tmp90, _2
	call	__mingw_vfprintf	 #
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:378: }
	addq	$48, %rsp	 #,
	popq	%rbx	 #
	ret	
	.section	.text$_Z6printfPKcz,"x"
	.linkonce discard
	.p2align 4
	.globl	_Z6printfPKcz
	.def	_Z6printfPKcz;	.scl	2;	.type	32;	.endef
_Z6printfPKcz:
	pushq	%r12	 #
	movq	%rcx, %r12	 # tmp89, __format
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:375:   __retval = __mingw_vfprintf( stdout, __format, __local_argv );
	movl	$1, %ecx	 #,
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:372: {
	pushq	%rbx	 #
	subq	$56, %rsp	 #,
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:374:   __builtin_va_list __local_argv; __builtin_va_start( __local_argv, __format );
	leaq	88(%rsp), %rbx	 #, tmp86
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:372: {
	movq	%rdx, 88(%rsp)	 #,
	movq	%r8, 96(%rsp)	 #,
	movq	%r9, 104(%rsp)	 #,
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:374:   __builtin_va_list __local_argv; __builtin_va_start( __local_argv, __format );
	movq	%rbx, 40(%rsp)	 # tmp86, MEM[(char * *)&__local_argv]
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:375:   __retval = __mingw_vfprintf( stdout, __format, __local_argv );
	call	*__imp___acrt_iob_func(%rip)	 #
	movq	%rbx, %r8	 # tmp86,
	movq	%r12, %rdx	 # __format,
	movq	%rax, %rcx	 # tmp90, _5
	call	__mingw_vfprintf	 #
 # D:/Program Files/RedPanda-Cpp/MinGW64/x86_64-w64-mingw32/include/stdio.h:378: }
	addq	$56, %rsp	 #,
	popq	%rbx	 #
	popq	%r12	 #
	ret	
	.section .rdata,"dr"
.LC3:
	.ascii "\12=> Mat %s[%d, %d, b8]: \12\0"
	.text
	.p2align 4
	.globl	_Z8printmatPKcPhii
	.def	_Z8printmatPKcPhii;	.scl	2;	.type	32;	.endef
_Z8printmatPKcPhii:
	pushq	%r15	 #
	leaq	.LC0(%rip), %r15	 #, tmp110
	pushq	%r14	 #
	movl	%r8d, %r14d	 # tmp114, m
	pushq	%r13	 #
	movl	%r9d, %r13d	 # tmp115, n
	pushq	%r12	 #
	xorl	%r12d, %r12d	 # i
	pushq	%rbp	 #
	leaq	.LC2(%rip), %rbp	 #, tmp111
	pushq	%rdi	 #
	movq	%rdx, %rdi	 # tmp113, a
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:10: 	printf("\n=> Mat %s[%d, %d, b8]: \n", name, m, n);  // 打印矩阵维度
	movq	%rcx, %rdx	 # name,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:9: void printmat(const char *name, b8 *a, int m, int n) {
	pushq	%rsi	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:10: 	printf("\n=> Mat %s[%d, %d, b8]: \n", name, m, n);  // 打印矩阵维度
	leaq	.LC3(%rip), %rcx	 #, tmp100
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:9: void printmat(const char *name, b8 *a, int m, int n) {
	pushq	%rbx	 #
	xorl	%ebx, %ebx	 # k
	subq	$40, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:10: 	printf("\n=> Mat %s[%d, %d, b8]: \n", name, m, n);  // 打印矩阵维度
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:12: 	for (i = 0; i < m; i++) {
	testl	%r14d, %r14d	 # m
	jle	.L10	 #,
	.p2align 4,,10
	.p2align 3
.L11:
	leal	0(%r13,%rbx), %esi	 #, _46
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:13: 		for (j = 0; j < n; j++, k++) {
	testl	%r13d, %r13d	 # n
	jle	.L18	 #,
	.p2align 4,,10
	.p2align 3
.L16:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:14: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0 ? 1: -1); // 打印矩阵元素，保留两位小数
	movl	%ebx, %eax	 # k, tmp101
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:14: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0 ? 1: -1); // 打印矩阵元素，保留两位小数
	movl	%ebx, %ecx	 # k, tmp104
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:14: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0 ? 1: -1); // 打印矩阵元素，保留两位小数
	movl	$1, %edx	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:14: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0 ? 1: -1); // 打印矩阵元素，保留两位小数
	sarl	$3, %eax	 #, tmp101
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:14: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0 ? 1: -1); // 打印矩阵元素，保留两位小数
	andl	$7, %ecx	 #, tmp104
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:14: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0 ? 1: -1); // 打印矩阵元素，保留两位小数
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:14: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0 ? 1: -1); // 打印矩阵元素，保留两位小数
	movzbl	(%rdi,%rax), %eax	 # *_3, *_3
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:14: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0 ? 1: -1); // 打印矩阵元素，保留两位小数
	sall	%cl, %eax	 # tmp104, tmp105
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:14: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0 ? 1: -1); // 打印矩阵元素，保留两位小数
	testb	$-128, %al	 #, tmp105
	jne	.L24	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:14: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0 ? 1: -1); // 打印矩阵元素，保留两位小数
	movl	$-1, %edx	 #,
.L24:
	movq	%rbp, %rcx	 # tmp111,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:13: 		for (j = 0; j < n; j++, k++) {
	addl	$1, %ebx	 #, k
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:14: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0 ? 1: -1); // 打印矩阵元素，保留两位小数
	call	_Z6printfPKcz.constprop.2	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:13: 		for (j = 0; j < n; j++, k++) {
	cmpl	%esi, %ebx	 # _46, k
	jne	.L16	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:9: void printmat(const char *name, b8 *a, int m, int n) {
	movl	%esi, %ebx	 # _46, k
.L18:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:16: 		printf("\n");
	movq	%r15, %rcx	 # tmp110,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:12: 	for (i = 0; i < m; i++) {
	addl	$1, %r12d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:16: 		printf("\n");
	call	_Z6printfPKcz.constprop.0	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:12: 	for (i = 0; i < m; i++) {
	cmpl	%r12d, %r14d	 # i, m
	jne	.L11	 #,
.L10:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:18: }
	addq	$40, %rsp	 #,
	popq	%rbx	 #
	popq	%rsi	 #
	popq	%rdi	 #
	popq	%rbp	 #
	popq	%r12	 #
	popq	%r13	 #
	popq	%r14	 #
	popq	%r15	 #
	ret	
	.section .rdata,"dr"
.LC4:
	.ascii "\12=> Mat %s %p [%d, %d, i8]: \12\0"
	.text
	.p2align 4
	.globl	_Z8printmatPKcPaii
	.def	_Z8printmatPKcPaii;	.scl	2;	.type	32;	.endef
_Z8printmatPKcPaii:
	pushq	%r15	 #
	pushq	%r14	 #
	movq	%rdx, %r14	 # tmp126, a
	pushq	%r13	 #
	pushq	%r12	 #
	movl	%r9d, %r12d	 # tmp128, n
	pushq	%rbp	 #
	pushq	%rdi	 #
	pushq	%rsi	 #
	pushq	%rbx	 #
	subq	$88, %rsp	 #,
	movl	%r8d, 176(%rsp)	 # tmp127, m
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:21: 	printf("\n=> Mat %s %p [%d, %d, i8]: \n", name,  a, m, n); // 打印矩阵维度
	movl	%r9d, 32(%rsp)	 # n,
	movl	%r8d, %r9d	 # tmp127,
	movq	%rdx, %r8	 # a,
	movq	%rcx, %rdx	 # name,
	leaq	.LC4(%rip), %rcx	 #, tmp103
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:20: void printmat(const char *name, i8 *a, int m, int n) {
	movaps	%xmm6, 64(%rsp)	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:21: 	printf("\n=> Mat %s %p [%d, %d, i8]: \n", name,  a, m, n); // 打印矩阵维度
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:23: 	for (i = 0; i < m; i++) {
	movl	176(%rsp), %eax	 # m,
	testl	%eax, %eax	 #
	jle	.L25	 #,
	leal	-1(%r12), %eax	 #, tmp123
	xorl	%ebp, %ebp	 # ivtmp.57
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:23: 	for (i = 0; i < m; i++) {
	xorl	%edi, %edi	 # i
	leaq	.LC0(%rip), %r15	 #, tmp120
	addq	%r14, %rax	 # a, tmp124
	movq	%rax, 56(%rsp)	 # tmp124, %sfp
	.p2align 4,,10
	.p2align 3
.L27:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:24: 		for (j = 0; j < n; j++, k++) {
	testl	%r12d, %r12d	 # n
	jle	.L30	 #,
	movq	56(%rsp), %rdx	 # %sfp, tmp124
	movslq	%ebp, %rax	 # ivtmp.57, _39
	movsd	.LC5(%rip), %xmm6	 #, tmp119
	leaq	(%r14,%rax), %r13	 #, ivtmp.52
	leaq	.LC1(%rip), %rbx	 #, tmp121
	leaq	1(%rax,%rdx), %rsi	 #, _23
	.p2align 4,,10
	.p2align 3
.L28:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:25: 			printf("%.6f\t", a[i * n + j] / 64. ); // 打印矩阵元素，保留两位小数
	movsbl	0(%r13), %eax	 # MEM[(i8 *)_37], MEM[(i8 *)_37]
	pxor	%xmm1, %xmm1	 # tmp108
	movq	%rbx, %rcx	 # tmp121,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:24: 		for (j = 0; j < n; j++, k++) {
	addq	$1, %r13	 #, ivtmp.52
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:25: 			printf("%.6f\t", a[i * n + j] / 64. ); // 打印矩阵元素，保留两位小数
	cvtsi2sdl	%eax, %xmm1	 # MEM[(i8 *)_37], tmp108
	mulsd	%xmm6, %xmm1	 # tmp119, tmp110
	movq	%xmm1, %rdx	 # tmp110,
	call	_Z6printfPKcz.constprop.1	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:24: 		for (j = 0; j < n; j++, k++) {
	cmpq	%r13, %rsi	 # ivtmp.52, _23
	jne	.L28	 #,
.L30:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:27: 		printf("\n");
	movq	%r15, %rcx	 # tmp120,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:23: 	for (i = 0; i < m; i++) {
	addl	$1, %edi	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:23: 	for (i = 0; i < m; i++) {
	addl	%r12d, %ebp	 # n, ivtmp.57
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:27: 		printf("\n");
	call	_Z6printfPKcz.constprop.0	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:23: 	for (i = 0; i < m; i++) {
	cmpl	%edi, 176(%rsp)	 # i, m
	jne	.L27	 #,
.L25:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:29: }
	movaps	64(%rsp), %xmm6	 #,
	addq	$88, %rsp	 #,
	popq	%rbx	 #
	popq	%rsi	 #
	popq	%rdi	 #
	popq	%rbp	 #
	popq	%r12	 #
	popq	%r13	 #
	popq	%r14	 #
	popq	%r15	 #
	ret	
	.section .rdata,"dr"
	.align 8
.LC6:
	.ascii "\12=> Mat %s %p [%d, %d, i32]: \12\0"
	.text
	.p2align 4
	.globl	_Z8printmatPKcPiii
	.def	_Z8printmatPKcPiii;	.scl	2;	.type	32;	.endef
_Z8printmatPKcPiii:
	pushq	%r15	 #
	pushq	%r14	 #
	pushq	%r13	 #
	movq	%rdx, %r13	 # tmp129, a
	pushq	%r12	 #
	movl	%r9d, %r12d	 # tmp131, n
	pushq	%rbp	 #
	pushq	%rdi	 #
	pushq	%rsi	 #
	pushq	%rbx	 #
	subq	$88, %rsp	 #,
	movl	%r8d, 176(%rsp)	 # tmp130, m
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:33: 	printf("\n=> Mat %s %p [%d, %d, i32]: \n", name, a, m, n);  // 打印矩阵维度
	movl	%r9d, 32(%rsp)	 # n,
	movl	%r8d, %r9d	 # tmp130,
	movq	%rdx, %r8	 # a,
	movq	%rcx, %rdx	 # name,
	leaq	.LC6(%rip), %rcx	 #, tmp105
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:32: void printmat(const char *name, i32 *a, int m, int n) {
	movaps	%xmm6, 64(%rsp)	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:33: 	printf("\n=> Mat %s %p [%d, %d, i32]: \n", name, a, m, n);  // 打印矩阵维度
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:35: 	for (i = 0; i < m; i++) {
	movl	176(%rsp), %eax	 # m,
	testl	%eax, %eax	 #
	jle	.L33	 #,
	leal	-1(%r12), %eax	 #, tmp126
	movsd	.LC5(%rip), %xmm6	 #, tmp127
	xorl	%ebp, %ebp	 # ivtmp.71
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:35: 	for (i = 0; i < m; i++) {
	xorl	%edi, %edi	 # i
	leaq	.LC0(%rip), %r15	 #, tmp123
	movq	%rax, 56(%rsp)	 # tmp126, %sfp
	.p2align 4,,10
	.p2align 3
.L35:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:36: 		for (j = 0; j < n; j++, k++) {
	testl	%r12d, %r12d	 # n
	jle	.L38	 #,
	leaq	.LC1(%rip), %rbx	 #, tmp124
	movslq	%ebp, %rax	 # ivtmp.71, _44
	leaq	0(%r13,%rax,4), %r14	 #, ivtmp.66
	addq	56(%rsp), %rax	 # %sfp, tmp109
	leaq	4(%r13,%rax,4), %rsi	 #, _26
	.p2align 4,,10
	.p2align 3
.L36:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:37: 			printf("%.6f\t", a[i * n + j] / 64. ); // 打印矩阵元素，保留两位小数
	pxor	%xmm1, %xmm1	 # tmp112
	cvtsi2sdl	(%r14), %xmm1	 # MEM[(i32 *)_41], tmp112
	movq	%rbx, %rcx	 # tmp124,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:36: 		for (j = 0; j < n; j++, k++) {
	addq	$4, %r14	 #, ivtmp.66
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:37: 			printf("%.6f\t", a[i * n + j] / 64. ); // 打印矩阵元素，保留两位小数
	mulsd	%xmm6, %xmm1	 # tmp127, tmp113
	movq	%xmm1, %rdx	 # tmp113,
	call	_Z6printfPKcz.constprop.1	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:36: 		for (j = 0; j < n; j++, k++) {
	cmpq	%r14, %rsi	 # ivtmp.66, _26
	jne	.L36	 #,
.L38:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:39: 		printf("\n");
	movq	%r15, %rcx	 # tmp123,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:35: 	for (i = 0; i < m; i++) {
	addl	$1, %edi	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:35: 	for (i = 0; i < m; i++) {
	addl	%r12d, %ebp	 # n, ivtmp.71
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:39: 		printf("\n");
	call	_Z6printfPKcz.constprop.0	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:35: 	for (i = 0; i < m; i++) {
	cmpl	%edi, 176(%rsp)	 # i, m
	jne	.L35	 #,
.L33:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:41: }
	movaps	64(%rsp), %xmm6	 #,
	addq	$88, %rsp	 #,
	popq	%rbx	 #
	popq	%rsi	 #
	popq	%rdi	 #
	popq	%rbp	 #
	popq	%r12	 #
	popq	%r13	 #
	popq	%r14	 #
	popq	%r15	 #
	ret	
	.p2align 4
	.globl	_Z18i32_i8_vec_in_prodPiPai
	.def	_Z18i32_i8_vec_in_prodPiPai;	.scl	2;	.type	32;	.endef
_Z18i32_i8_vec_in_prodPiPai:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	testl	%r8d, %r8d	 # n
	jle	.L46	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:44: i32 i32_i8_vec_in_prod(i32 *data, i8 *weight, int n) {
	pushq	%rbp	 #
	leal	-1(%r8), %eax	 #, tmp258
	movq	%rdx, %r9	 # tmp376, weight
	movq	%rsp, %rbp	 #,
	pushq	%rbx	 #
	subq	$48, %rsp	 #,
	andq	$-16, %rsp	 #,
	cmpl	$14, %eax	 #, tmp258
	movaps	%xmm6, (%rsp)	 #,
	movaps	%xmm7, 16(%rsp)	 #,
	movaps	%xmm8, 32(%rsp)	 #,
	jbe	.L47	 #,
	movl	%r8d, %r10d	 # n, bnd.77
	movq	%rcx, %rax	 # data, ivtmp.98
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	pxor	%xmm5, %xmm5	 # vect_res_20.80
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	pxor	%xmm3, %xmm3	 # tmp266
	shrl	$4, %r10d	 #, bnd.77
	pxor	%xmm2, %xmm2	 # tmp276
	subl	$1, %r10d	 #, tmp261
	salq	$6, %r10	 #, tmp262
	leaq	64(%rcx,%r10), %r10	 #, _160
	.p2align 4,,10
	.p2align 3
.L44:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqu	(%rdx), %xmm0	 # MEM <vector(16) signed char> [(i8 *)_124], MEM <vector(16) signed char> [(i8 *)_124]
	movdqa	%xmm3, %xmm1	 # tmp266, tmp267
	movdqa	%xmm2, %xmm7	 # tmp276, tmp277
	addq	$64, %rax	 #, ivtmp.98
	addq	$16, %rdx	 #, ivtmp.101
	pcmpgtb	%xmm0, %xmm1	 # MEM <vector(16) signed char> [(i8 *)_124], tmp267
	movdqa	%xmm0, %xmm4	 # MEM <vector(16) signed char> [(i8 *)_124], tmp268
	punpcklbw	%xmm1, %xmm4	 # tmp267, tmp268
	punpckhbw	%xmm1, %xmm0	 # tmp267, tmp272
	pcmpgtw	%xmm4, %xmm7	 # tmp268, tmp277
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqu	-64(%rax), %xmm1	 # MEM <vector(4) int> [(i32 *)_100], vect__4.83
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm4, %xmm8	 # tmp268, tmp278
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm1, %xmm6	 # vect__4.83, tmp282
	psrlq	$32, %xmm1	 #, tmp284
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	punpcklwd	%xmm7, %xmm8	 # tmp277, tmp278
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	pmuludq	%xmm8, %xmm6	 # tmp278, tmp282
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	punpckhwd	%xmm7, %xmm4	 # tmp277, tmp291
	movdqa	%xmm2, %xmm7	 # tmp276, tmp303
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	psrlq	$32, %xmm8	 #, tmp285
	pmuludq	%xmm8, %xmm1	 # tmp285, tmp283
	pshufd	$8, %xmm6, %xmm6	 #, tmp282, tmp280
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	pcmpgtw	%xmm0, %xmm7	 # tmp272, tmp303
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	pshufd	$8, %xmm1, %xmm1	 #, tmp283, tmp281
	punpckldq	%xmm1, %xmm6	 # tmp281, vect__9.92
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqu	-48(%rax), %xmm1	 # MEM <vector(4) int> [(i32 *)_100 + 16B], vect__4.84
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	paddd	%xmm5, %xmm6	 # vect_res_20.80, vect_res_17.93
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm1, %xmm5	 # vect__4.84, tmp295
	psrlq	$32, %xmm1	 #, tmp297
	pmuludq	%xmm4, %xmm5	 # tmp291, tmp295
	psrlq	$32, %xmm4	 #, tmp298
	pmuludq	%xmm4, %xmm1	 # tmp298, tmp296
	pshufd	$8, %xmm5, %xmm5	 #, tmp295, tmp293
	pshufd	$8, %xmm1, %xmm1	 #, tmp296, tmp294
	punpckldq	%xmm1, %xmm5	 # tmp294, vect__9.92
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqu	-32(%rax), %xmm1	 # MEM <vector(4) int> [(i32 *)_100 + 32B], vect__4.85
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	paddd	%xmm6, %xmm5	 # vect_res_17.93, vect_res_17.93
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm0, %xmm6	 # tmp272, tmp304
	punpckhwd	%xmm7, %xmm0	 # tmp303, tmp317
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm1, %xmm4	 # vect__4.85, tmp308
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	punpcklwd	%xmm7, %xmm6	 # tmp303, tmp304
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	psrlq	$32, %xmm1	 #, tmp310
	pmuludq	%xmm6, %xmm4	 # tmp304, tmp308
	psrlq	$32, %xmm6	 #, tmp311
	pmuludq	%xmm6, %xmm1	 # tmp311, tmp309
	pshufd	$8, %xmm4, %xmm4	 #, tmp308, tmp306
	pshufd	$8, %xmm1, %xmm1	 #, tmp309, tmp307
	punpckldq	%xmm1, %xmm4	 # tmp307, vect__9.92
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqu	-16(%rax), %xmm1	 # MEM <vector(4) int> [(i32 *)_100 + 48B], vect__4.86
	cmpq	%r10, %rax	 # _160, ivtmp.98
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	paddd	%xmm5, %xmm4	 # vect_res_17.93, vect_res_17.93
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm1, %xmm5	 # vect__4.86, tmp321
	psrlq	$32, %xmm1	 #, tmp323
	pmuludq	%xmm0, %xmm5	 # tmp317, tmp321
	psrlq	$32, %xmm0	 #, tmp324
	pmuludq	%xmm0, %xmm1	 # tmp324, tmp322
	pshufd	$8, %xmm5, %xmm5	 #, tmp321, tmp319
	pshufd	$8, %xmm1, %xmm1	 #, tmp322, tmp320
	punpckldq	%xmm1, %xmm5	 # tmp320, vect__9.92
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	paddd	%xmm4, %xmm5	 # vect_res_17.93, vect_res_20.80
	jne	.L44	 #,
	movdqa	%xmm5, %xmm0	 # vect_res_20.80, tmp326
	movl	%r8d, %edx	 # n, tmp.79
	psrldq	$8, %xmm0	 #, tmp326
	andl	$-16, %edx	 #, tmp.79
	testb	$15, %r8b	 #, n
	paddd	%xmm0, %xmm5	 # tmp326, _73
	movdqa	%xmm5, %xmm0	 # _73, tmp328
	psrldq	$4, %xmm0	 #, tmp328
	paddd	%xmm0, %xmm5	 # tmp328, tmp329
	movd	%xmm5, %eax	 # tmp329, stmp_res_17.94
	je	.L45	 #,
.L43:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%edx, %rbx	 # tmp.79, _1
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movsbl	(%r9,%rbx), %r11d	 # *_6, *_6
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	0(,%rbx,4), %r10	 #, _2
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%rcx,%rbx,4), %r11d	 # *_3, tmp331
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%r11d, %eax	 # tmp331, stmp_res_17.94
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	1(%rdx), %r11d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%r11d, %r8d	 # i, n
	jle	.L45	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%r11d, %r11	 # i, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movsbl	(%r9,%r11), %r11d	 # *_81, *_81
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	4(%rcx,%r10), %r11d	 # *_79, tmp334
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%r11d, %eax	 # tmp334, stmp_res_17.94
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	2(%rdx), %r11d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%r11d, %r8d	 # i, n
	jle	.L45	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%r11d, %r11	 # i, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movsbl	(%r9,%r11), %r11d	 # *_93, *_93
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	8(%rcx,%r10), %r11d	 # *_91, tmp337
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%r11d, %eax	 # tmp337, stmp_res_17.94
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	3(%rdx), %r11d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%r11d, %r8d	 # i, n
	jle	.L45	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%r11d, %r11	 # i, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movsbl	(%r9,%r11), %r11d	 # *_105, *_105
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	12(%rcx,%r10), %r11d	 # *_103, tmp340
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%r11d, %eax	 # tmp340, stmp_res_17.94
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	4(%rdx), %r11d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%r11d, %r8d	 # i, n
	jle	.L45	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%r11d, %r11	 # i, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movsbl	(%r9,%r11), %r11d	 # *_117, *_117
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	16(%rcx,%r10), %r11d	 # *_115, tmp343
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%r11d, %eax	 # tmp343, stmp_res_17.94
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	5(%rdx), %r11d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%r11d, %r8d	 # i, n
	jle	.L45	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%r11d, %r11	 # i, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movsbl	(%r9,%r11), %r11d	 # *_129, *_129
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	20(%rcx,%r10), %r11d	 # *_127, tmp346
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%r11d, %eax	 # tmp346, stmp_res_17.94
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	6(%rdx), %r11d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%r11d, %r8d	 # i, n
	jle	.L45	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%r11d, %r11	 # i, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movsbl	(%r9,%r11), %r11d	 # *_141, *_141
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	24(%rcx,%r10), %r11d	 # *_139, tmp349
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%r11d, %eax	 # tmp349, stmp_res_17.94
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	7(%rdx), %r11d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%r11d, %r8d	 # i, n
	jle	.L45	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%r11d, %r11	 # i, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movsbl	(%r9,%r11), %r11d	 # *_153, *_153
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	28(%rcx,%r10), %r11d	 # *_151, tmp352
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%r11d, %eax	 # tmp352, stmp_res_17.94
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	8(%rdx), %r11d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%r11d, %r8d	 # i, n
	jle	.L45	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%r11d, %r11	 # i, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movsbl	(%r9,%r11), %r11d	 # *_165, *_165
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	32(%rcx,%r10), %r11d	 # *_163, tmp355
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%r11d, %eax	 # tmp355, stmp_res_17.94
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	9(%rdx), %r11d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%r11d, %r8d	 # i, n
	jle	.L45	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%r11d, %r11	 # i, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movsbl	(%r9,%r11), %r11d	 # *_177, *_177
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	36(%rcx,%r10), %r11d	 # *_175, tmp358
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%r11d, %eax	 # tmp358, stmp_res_17.94
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	10(%rdx), %r11d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%r11d, %r8d	 # i, n
	jle	.L45	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%r11d, %r11	 # i, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movsbl	(%r9,%r11), %r11d	 # *_189, *_189
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	40(%rcx,%r10), %r11d	 # *_187, tmp361
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%r11d, %eax	 # tmp361, stmp_res_17.94
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	11(%rdx), %r11d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%r11d, %r8d	 # i, n
	jle	.L45	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%r11d, %r11	 # i, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movsbl	(%r9,%r11), %r11d	 # *_201, *_201
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	44(%rcx,%r10), %r11d	 # *_199, tmp364
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%r11d, %eax	 # tmp364, stmp_res_17.94
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	12(%rdx), %r11d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%r11d, %r8d	 # i, n
	jle	.L45	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%r11d, %r11	 # i, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movsbl	(%r9,%r11), %r11d	 # *_213, *_213
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	48(%rcx,%r10), %r11d	 # *_211, tmp367
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%r11d, %eax	 # tmp367, stmp_res_17.94
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	13(%rdx), %r11d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%r11d, %r8d	 # i, n
	jle	.L45	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%r11d, %r11	 # i, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	addl	$14, %edx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movsbl	(%r9,%r11), %r11d	 # *_225, *_225
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	52(%rcx,%r10), %r11d	 # *_223, tmp370
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%r11d, %eax	 # tmp370, stmp_res_17.94
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%edx, %r8d	 # i, n
	jle	.L45	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%edx, %rdx	 # i, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movsbl	(%r9,%rdx), %edx	 # *_31, *_31
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	56(%rcx,%r10), %edx	 # *_33, tmp373
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%edx, %eax	 # tmp373, stmp_res_17.94
.L45:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:51: }
	movq	-8(%rbp), %rbx	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:50: 	return res >> 6; // 返回内积结果
	sarl	$6, %eax	 #, <retval>
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:51: }
	movaps	(%rsp), %xmm6	 #,
	movaps	16(%rsp), %xmm7	 #,
	movaps	32(%rsp), %xmm8	 #,
	leave	
	ret	
	.p2align 4,,10
	.p2align 3
.L46:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	xorl	%eax, %eax	 # <retval>
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:51: }
	ret	
.L47:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	xorl	%edx, %edx	 # tmp.79
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:45: 	i32 res = 0;  // 初始化结果为0, 结果要除以64,解析成(-1,1)的8字节数
	xorl	%eax, %eax	 # stmp_res_17.94
	jmp	.L43	 #
	.p2align 4
	.globl	_Z18i32_b8_vec_in_prodPiPhii
	.def	_Z18i32_b8_vec_in_prodPiPhii;	.scl	2;	.type	32;	.endef
_Z18i32_b8_vec_in_prodPiPhii:
	pushq	%rbp	 #
	movq	%rcx, %r11	 # tmp973, data
	movl	%r9d, %ecx	 # tmp976, b_start
	movq	%rsp, %rbp	 #,
	pushq	%r15	 #
	pushq	%r14	 #
	pushq	%r13	 #
	pushq	%r12	 #
	pushq	%rdi	 #
	pushq	%rsi	 #
	pushq	%rbx	 #
	movl	%r8d, %ebx	 # tmp975, n
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:59: 	int n0 = (8 - b_start) < n ? (8 - b_start) : n;  // 计算开头不满足8位的部分长度
	movl	$8, %r8d	 #, tmp482
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:54: i32 i32_b8_vec_in_prod(i32 *data, b8 *weight, int n, int b_start) {
	subq	$160, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:59: 	int n0 = (8 - b_start) < n ? (8 - b_start) : n;  // 计算开头不满足8位的部分长度
	subl	%r9d, %r8d	 # b_start, tmp481
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:54: i32 i32_b8_vec_in_prod(i32 *data, b8 *weight, int n, int b_start) {
	andq	$-16, %rsp	 #,
	subq	$448, %rsp	 #,
	movaps	%xmm6, 448(%rsp)	 #,
	movaps	%xmm7, 464(%rsp)	 #,
	movaps	%xmm8, 480(%rsp)	 #,
	movaps	%xmm9, 496(%rsp)	 #,
	movaps	%xmm10, 512(%rsp)	 #,
	movaps	%xmm11, 528(%rsp)	 #,
	movaps	%xmm12, 544(%rsp)	 #,
	movaps	%xmm13, 560(%rsp)	 #,
	movaps	%xmm14, 576(%rsp)	 #,
	movaps	%xmm15, 592(%rsp)	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:58: 	b8 p = (*weight) << b_start;  // 将bit向量左移b_start位
	movzbl	(%rdx), %esi	 # *weight_35(D), *weight_35(D)
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:58: 	b8 p = (*weight) << b_start;  // 将bit向量左移b_start位
	sall	%cl, %esi	 # b_start, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:59: 	int n0 = (8 - b_start) < n ? (8 - b_start) : n;  // 计算开头不满足8位的部分长度
	cmpl	%ebx, %r8d	 # n, tmp481
	cmovg	%ebx, %r8d	 # tmp481,, n, n0
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:62: 	for (; i < n0; i++, data++, p <<= 1)
	testl	%r8d, %r8d	 # n0
	jle	.L75	 #,
	movslq	%r8d, %rax	 # n0, n0
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:56: 	i32 res = 0;  // 初始化结果为0, 结果要除以127,解析成(-1,1)的8字节数
	xorl	%r9d, %r9d	 # <retval>
	leaq	(%r11,%rax,4), %r10	 #, data
	.p2align 4,,10
	.p2align 3
.L56:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:63: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movl	(%r11), %eax	 # MEM[(i32 *)data_207], pretmp_212
	movl	%eax, %ecx	 # pretmp_212, tmp953
	negl	%ecx	 # tmp953
	testb	%sil, %sil	 # p
	cmovns	%ecx, %eax	 # tmp953,, pretmp_212
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:62: 	for (; i < n0; i++, data++, p <<= 1)
	addq	$4, %r11	 #, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:62: 	for (; i < n0; i++, data++, p <<= 1)
	addl	%esi, %esi	 # p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:63: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	addl	%eax, %r9d	 # pretmp_212, <retval>
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:62: 	for (; i < n0; i++, data++, p <<= 1)
	cmpq	%r11, %r10	 # data, data
	jne	.L56	 #,
.L54:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:66: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	leal	-7(%rbx), %esi	 #, _7
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:66: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	leaq	1(%rdx), %r12	 #, weight
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:66: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	cmpl	%r8d, %esi	 # n0, _7
	jle	.L57	 #,
	leal	-8(%rbx), %eax	 #, tmp485
	subl	%r8d, %eax	 # n0, _276
	movl	%eax, %r13d	 # _276, _277
	shrl	$3, %r13d	 #, _277
	cmpl	$119, %eax	 #, _276
	leal	1(%r13), %edi	 #, tmp952
	jbe	.L76	 #,
	movl	%edi, %r11d	 # tmp952, bnd.116
	pxor	%xmm6, %xmm6	 # vect_res_93.121
	movq	%r12, %rcx	 # weight, ivtmp.242
	movq	%r10, %rax	 # data, ivtmp.245
	shrl	$4, %r11d	 #, bnd.116
	movaps	%xmm6, 352(%rsp)	 # vect_res_93.121, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm6, %xmm13	 #, tmp542
	pxor	%xmm12, %xmm12	 # tmp524
	subl	$1, %r11d	 #, tmp488
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:66: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	movaps	%xmm6, 368(%rsp)	 # vect_res_93.121, %sfp
	salq	$4, %r11	 #, tmp489
	movaps	%xmm6, 384(%rsp)	 # vect_res_93.121, %sfp
	leaq	17(%rdx,%r11), %rdx	 #, _52
	movaps	%xmm6, 400(%rsp)	 # vect_res_169.221, %sfp
	.p2align 4,,10
	.p2align 3
.L59:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqu	(%rax), %xmm1	 # MEM <vector(4) int> [(i32 *)_94], MEM <vector(4) int> [(i32 *)_94]
	addq	$16, %rcx	 #, ivtmp.242
	addq	$512, %rax	 #, ivtmp.245
	movdqu	-496(%rax), %xmm6	 # MEM <vector(4) int> [(i32 *)_94 + 16B], MEM <vector(4) int> [(i32 *)_94 + 16B]
	movdqu	-480(%rax), %xmm4	 # MEM <vector(4) int> [(i32 *)_94 + 32B], MEM <vector(4) int> [(i32 *)_94 + 32B]
	movdqa	%xmm1, %xmm3	 # MEM <vector(4) int> [(i32 *)_94], vect_perm_even_379
	movdqu	-464(%rax), %xmm2	 # MEM <vector(4) int> [(i32 *)_94 + 48B], MEM <vector(4) int> [(i32 *)_94 + 48B]
	shufps	$136, %xmm6, %xmm3	 #, MEM <vector(4) int> [(i32 *)_94 + 16B], vect_perm_even_379
	shufps	$221, %xmm6, %xmm1	 #, MEM <vector(4) int> [(i32 *)_94 + 16B], vect_perm_odd_380
	movdqu	-448(%rax), %xmm0	 # MEM <vector(4) int> [(i32 *)_94 + 64B], MEM <vector(4) int> [(i32 *)_94 + 64B]
	movdqa	%xmm4, %xmm6	 # MEM <vector(4) int> [(i32 *)_94 + 32B], vect_perm_even_381
	movdqu	-432(%rax), %xmm5	 # MEM <vector(4) int> [(i32 *)_94 + 80B], MEM <vector(4) int> [(i32 *)_94 + 80B]
	shufps	$136, %xmm2, %xmm6	 #, MEM <vector(4) int> [(i32 *)_94 + 48B], vect_perm_even_381
	shufps	$221, %xmm2, %xmm4	 #, MEM <vector(4) int> [(i32 *)_94 + 48B], vect_perm_odd_382
	movdqu	-416(%rax), %xmm8	 # MEM <vector(4) int> [(i32 *)_94 + 96B], MEM <vector(4) int> [(i32 *)_94 + 96B]
	movdqa	%xmm0, %xmm2	 # MEM <vector(4) int> [(i32 *)_94 + 64B], vect_perm_even_383
	movdqu	-400(%rax), %xmm7	 # MEM <vector(4) int> [(i32 *)_94 + 112B], MEM <vector(4) int> [(i32 *)_94 + 112B]
	shufps	$136, %xmm5, %xmm2	 #, MEM <vector(4) int> [(i32 *)_94 + 80B], vect_perm_even_383
	shufps	$221, %xmm5, %xmm0	 #, MEM <vector(4) int> [(i32 *)_94 + 80B], vect_perm_odd_384
	movdqu	-288(%rax), %xmm9	 # MEM <vector(4) int> [(i32 *)_94 + 224B], MEM <vector(4) int> [(i32 *)_94 + 224B]
	movdqa	%xmm8, %xmm5	 # MEM <vector(4) int> [(i32 *)_94 + 96B], vect_perm_even_385
	shufps	$136, %xmm7, %xmm5	 #, MEM <vector(4) int> [(i32 *)_94 + 112B], vect_perm_even_385
	shufps	$221, %xmm7, %xmm8	 #, MEM <vector(4) int> [(i32 *)_94 + 112B], vect_perm_odd_386
	movdqa	%xmm3, %xmm7	 # vect_perm_even_379, vect_perm_even_387
	shufps	$221, %xmm6, %xmm3	 #, vect_perm_even_381, vect_perm_odd_388
	shufps	$136, %xmm6, %xmm7	 #, vect_perm_even_381, vect_perm_even_387
	movdqa	%xmm2, %xmm6	 # vect_perm_even_383, vect_perm_even_389
	movdqa	%xmm7, %xmm10	 # vect_perm_even_387, vect_perm_even_395
	shufps	$221, %xmm5, %xmm2	 #, vect_perm_even_385, vect_perm_odd_390
	shufps	$136, %xmm5, %xmm6	 #, vect_perm_even_385, vect_perm_even_389
	movdqa	%xmm1, %xmm5	 # vect_perm_odd_380, vect_perm_even_391
	shufps	$136, %xmm6, %xmm10	 #, vect_perm_even_389, vect_perm_even_395
	shufps	$221, %xmm4, %xmm1	 #, vect_perm_odd_382, vect_perm_odd_392
	shufps	$136, %xmm4, %xmm5	 #, vect_perm_odd_382, vect_perm_even_391
	shufps	$221, %xmm6, %xmm7	 #, vect_perm_even_389, vect_perm_even_387
	movdqa	%xmm0, %xmm4	 # vect_perm_odd_384, vect_perm_even_393
	movdqa	%xmm5, %xmm6	 # vect_perm_even_391, vect_perm_even_397
	shufps	$136, %xmm8, %xmm4	 #, vect_perm_odd_386, vect_perm_even_393
	shufps	$136, %xmm4, %xmm6	 #, vect_perm_even_393, vect_perm_even_397
	movaps	%xmm7, 336(%rsp)	 # vect_perm_even_387, %sfp
	movdqa	%xmm6, %xmm7	 # vect_perm_even_397, vect_perm_even_397
	movdqa	%xmm3, %xmm6	 # vect_perm_odd_388, vect_perm_even_399
	shufps	$221, %xmm8, %xmm0	 #, vect_perm_odd_386, vect_perm_odd_394
	shufps	$221, %xmm4, %xmm5	 #, vect_perm_even_393, vect_perm_even_391
	movdqu	-368(%rax), %xmm8	 # MEM <vector(4) int> [(i32 *)_94 + 144B], MEM <vector(4) int> [(i32 *)_94 + 144B]
	shufps	$136, %xmm2, %xmm6	 #, vect_perm_odd_390, vect_perm_even_399
	shufps	$221, %xmm2, %xmm3	 #, vect_perm_odd_390, vect_perm_odd_388
	movaps	%xmm6, 304(%rsp)	 # vect_perm_even_399, %sfp
	movdqa	%xmm1, %xmm6	 # vect_perm_odd_392, vect_perm_even_401
	shufps	$221, %xmm0, %xmm1	 #, vect_perm_odd_394, vect_perm_odd_392
	movaps	%xmm1, 272(%rsp)	 # vect_perm_odd_392, %sfp
	movdqu	-384(%rax), %xmm1	 # MEM <vector(4) int> [(i32 *)_94 + 128B], MEM <vector(4) int> [(i32 *)_94 + 128B]
	shufps	$136, %xmm0, %xmm6	 #, vect_perm_odd_394, vect_perm_even_401
	movdqu	-352(%rax), %xmm4	 # MEM <vector(4) int> [(i32 *)_94 + 160B], MEM <vector(4) int> [(i32 *)_94 + 160B]
	movaps	%xmm3, 288(%rsp)	 # vect_perm_odd_388, %sfp
	movdqa	%xmm9, %xmm14	 # MEM <vector(4) int> [(i32 *)_94 + 224B], vect_perm_even_425
	movdqu	-336(%rax), %xmm2	 # MEM <vector(4) int> [(i32 *)_94 + 176B], MEM <vector(4) int> [(i32 *)_94 + 176B]
	movdqa	%xmm1, %xmm3	 # MEM <vector(4) int> [(i32 *)_94 + 128B], vect_perm_even_419
	movaps	%xmm5, 320(%rsp)	 # vect_perm_even_391, %sfp
	shufps	$221, %xmm8, %xmm1	 #, MEM <vector(4) int> [(i32 *)_94 + 144B], vect_perm_odd_420
	movdqu	-320(%rax), %xmm0	 # MEM <vector(4) int> [(i32 *)_94 + 192B], MEM <vector(4) int> [(i32 *)_94 + 192B]
	shufps	$136, %xmm8, %xmm3	 #, MEM <vector(4) int> [(i32 *)_94 + 144B], vect_perm_even_419
	movdqa	%xmm4, %xmm8	 # MEM <vector(4) int> [(i32 *)_94 + 160B], vect_perm_even_421
	movdqu	-304(%rax), %xmm5	 # MEM <vector(4) int> [(i32 *)_94 + 208B], MEM <vector(4) int> [(i32 *)_94 + 208B]
	shufps	$136, %xmm2, %xmm8	 #, MEM <vector(4) int> [(i32 *)_94 + 176B], vect_perm_even_421
	shufps	$221, %xmm2, %xmm4	 #, MEM <vector(4) int> [(i32 *)_94 + 176B], vect_perm_odd_422
	movdqu	-272(%rax), %xmm11	 # MEM <vector(4) int> [(i32 *)_94 + 240B], MEM <vector(4) int> [(i32 *)_94 + 240B]
	movdqa	%xmm0, %xmm2	 # MEM <vector(4) int> [(i32 *)_94 + 192B], vect_perm_even_423
	shufps	$136, %xmm5, %xmm2	 #, MEM <vector(4) int> [(i32 *)_94 + 208B], vect_perm_even_423
	shufps	$221, %xmm5, %xmm0	 #, MEM <vector(4) int> [(i32 *)_94 + 208B], vect_perm_odd_424
	shufps	$136, %xmm11, %xmm14	 #, MEM <vector(4) int> [(i32 *)_94 + 240B], vect_perm_even_425
	movdqa	%xmm14, %xmm5	 # vect_perm_even_425, vect_perm_even_425
	movdqa	%xmm3, %xmm14	 # vect_perm_even_419, vect_perm_even_427
	shufps	$221, %xmm11, %xmm9	 #, MEM <vector(4) int> [(i32 *)_94 + 240B], vect_perm_odd_426
	shufps	$221, %xmm8, %xmm3	 #, vect_perm_even_421, vect_perm_odd_428
	shufps	$136, %xmm8, %xmm14	 #, vect_perm_even_421, vect_perm_even_427
	movdqa	%xmm2, %xmm8	 # vect_perm_even_423, vect_perm_even_429
	movdqa	%xmm14, %xmm11	 # vect_perm_even_427, vect_perm_even_427
	movdqa	%xmm1, %xmm14	 # vect_perm_odd_420, vect_perm_even_431
	shufps	$136, %xmm5, %xmm8	 #, vect_perm_even_425, vect_perm_even_429
	shufps	$221, %xmm5, %xmm2	 #, vect_perm_even_425, vect_perm_odd_430
	shufps	$221, %xmm4, %xmm1	 #, vect_perm_odd_422, vect_perm_odd_432
	shufps	$136, %xmm4, %xmm14	 #, vect_perm_odd_422, vect_perm_even_431
	movdqa	%xmm0, %xmm4	 # vect_perm_odd_424, vect_perm_even_433
	movdqa	%xmm14, %xmm5	 # vect_perm_even_431, vect_perm_even_431
	shufps	$221, %xmm9, %xmm0	 #, vect_perm_odd_426, vect_perm_odd_434
	shufps	$136, %xmm9, %xmm4	 #, vect_perm_odd_426, vect_perm_even_433
	shufps	$136, %xmm4, %xmm14	 #, vect_perm_even_433, vect_perm_even_437
	movdqa	%xmm11, %xmm9	 # vect_perm_even_427, vect_perm_even_435
	movaps	%xmm14, 240(%rsp)	 # vect_perm_even_437, %sfp
	movdqa	%xmm5, %xmm14	 # vect_perm_even_431, vect_perm_even_431
	shufps	$221, %xmm8, %xmm11	 #, vect_perm_even_429, vect_perm_even_427
	shufps	$136, %xmm8, %xmm9	 #, vect_perm_even_429, vect_perm_even_435
	movaps	%xmm11, 256(%rsp)	 # vect_perm_even_427, %sfp
	movdqu	-240(%rax), %xmm11	 # MEM <vector(4) int> [(i32 *)_94 + 272B], MEM <vector(4) int> [(i32 *)_94 + 272B]
	shufps	$221, %xmm4, %xmm14	 #, vect_perm_even_433, vect_perm_even_431
	movdqa	%xmm3, %xmm4	 # vect_perm_odd_428, vect_perm_even_439
	movaps	%xmm14, 224(%rsp)	 # vect_perm_even_431, %sfp
	shufps	$136, %xmm2, %xmm4	 #, vect_perm_odd_430, vect_perm_even_439
	movaps	%xmm4, 208(%rsp)	 # vect_perm_even_439, %sfp
	movdqa	%xmm3, %xmm4	 # vect_perm_odd_428, vect_perm_odd_428
	shufps	$221, %xmm2, %xmm4	 #, vect_perm_odd_430, vect_perm_odd_428
	movdqa	%xmm1, %xmm2	 # vect_perm_odd_432, vect_perm_even_441
	movaps	%xmm4, 192(%rsp)	 # vect_perm_odd_428, %sfp
	shufps	$221, %xmm0, %xmm1	 #, vect_perm_odd_434, vect_perm_odd_432
	movdqu	-224(%rax), %xmm4	 # MEM <vector(4) int> [(i32 *)_94 + 288B], MEM <vector(4) int> [(i32 *)_94 + 288B]
	shufps	$136, %xmm0, %xmm2	 #, vect_perm_odd_434, vect_perm_even_441
	movaps	%xmm1, 160(%rsp)	 # vect_perm_odd_432, %sfp
	movaps	%xmm2, 176(%rsp)	 # vect_perm_even_441, %sfp
	movdqu	-256(%rax), %xmm1	 # MEM <vector(4) int> [(i32 *)_94 + 256B], MEM <vector(4) int> [(i32 *)_94 + 256B]
	movdqu	-208(%rax), %xmm2	 # MEM <vector(4) int> [(i32 *)_94 + 304B], MEM <vector(4) int> [(i32 *)_94 + 304B]
	movdqa	%xmm4, %xmm15	 # MEM <vector(4) int> [(i32 *)_94 + 288B], vect_perm_even_461
	movdqu	-192(%rax), %xmm0	 # MEM <vector(4) int> [(i32 *)_94 + 320B], MEM <vector(4) int> [(i32 *)_94 + 320B]
	movdqa	%xmm1, %xmm3	 # MEM <vector(4) int> [(i32 *)_94 + 256B], vect_perm_even_459
	shufps	$221, %xmm11, %xmm1	 #, MEM <vector(4) int> [(i32 *)_94 + 272B], vect_perm_odd_460
	shufps	$136, %xmm2, %xmm15	 #, MEM <vector(4) int> [(i32 *)_94 + 304B], vect_perm_even_461
	shufps	$221, %xmm2, %xmm4	 #, MEM <vector(4) int> [(i32 *)_94 + 304B], vect_perm_odd_462
	shufps	$136, %xmm11, %xmm3	 #, MEM <vector(4) int> [(i32 *)_94 + 272B], vect_perm_even_459
	movdqu	-176(%rax), %xmm5	 # MEM <vector(4) int> [(i32 *)_94 + 336B], MEM <vector(4) int> [(i32 *)_94 + 336B]
	movdqu	-160(%rax), %xmm8	 # MEM <vector(4) int> [(i32 *)_94 + 352B], MEM <vector(4) int> [(i32 *)_94 + 352B]
	movdqa	%xmm0, %xmm2	 # MEM <vector(4) int> [(i32 *)_94 + 320B], vect_perm_even_463
	shufps	$136, %xmm5, %xmm2	 #, MEM <vector(4) int> [(i32 *)_94 + 336B], vect_perm_even_463
	shufps	$221, %xmm5, %xmm0	 #, MEM <vector(4) int> [(i32 *)_94 + 336B], vect_perm_odd_464
	movdqu	-48(%rax), %xmm11	 # MEM <vector(4) int> [(i32 *)_94 + 464B], MEM <vector(4) int> [(i32 *)_94 + 464B]
	movdqu	-144(%rax), %xmm14	 # MEM <vector(4) int> [(i32 *)_94 + 368B], MEM <vector(4) int> [(i32 *)_94 + 368B]
	movdqa	%xmm8, %xmm5	 # MEM <vector(4) int> [(i32 *)_94 + 352B], vect_perm_even_465
	shufps	$136, %xmm14, %xmm5	 #, MEM <vector(4) int> [(i32 *)_94 + 368B], vect_perm_even_465
	shufps	$221, %xmm14, %xmm8	 #, MEM <vector(4) int> [(i32 *)_94 + 368B], vect_perm_odd_466
	movdqa	%xmm3, %xmm14	 # vect_perm_even_459, vect_perm_even_467
	shufps	$221, %xmm15, %xmm3	 #, vect_perm_even_461, vect_perm_odd_468
	shufps	$136, %xmm15, %xmm14	 #, vect_perm_even_461, vect_perm_even_467
	movdqa	%xmm2, %xmm15	 # vect_perm_even_463, vect_perm_even_469
	shufps	$221, %xmm5, %xmm2	 #, vect_perm_even_465, vect_perm_odd_470
	shufps	$136, %xmm5, %xmm15	 #, vect_perm_even_465, vect_perm_even_469
	movdqa	%xmm1, %xmm5	 # vect_perm_odd_460, vect_perm_even_471
	shufps	$221, %xmm4, %xmm1	 #, vect_perm_odd_462, vect_perm_odd_472
	shufps	$136, %xmm4, %xmm5	 #, vect_perm_odd_462, vect_perm_even_471
	movdqa	%xmm0, %xmm4	 # vect_perm_odd_464, vect_perm_even_473
	shufps	$221, %xmm8, %xmm0	 #, vect_perm_odd_466, vect_perm_odd_474
	shufps	$136, %xmm8, %xmm4	 #, vect_perm_odd_466, vect_perm_even_473
	movdqa	%xmm14, %xmm8	 # vect_perm_even_467, vect_perm_even_475
	shufps	$221, %xmm15, %xmm14	 #, vect_perm_even_469, vect_perm_even_467
	movaps	%xmm14, 144(%rsp)	 # vect_perm_even_467, %sfp
	movdqa	%xmm5, %xmm14	 # vect_perm_even_471, vect_perm_even_477
	shufps	$221, %xmm4, %xmm5	 #, vect_perm_even_473, vect_perm_even_471
	shufps	$136, %xmm15, %xmm8	 #, vect_perm_even_469, vect_perm_even_475
	movaps	%xmm5, 128(%rsp)	 # vect_perm_even_471, %sfp
	shufps	$136, %xmm4, %xmm14	 #, vect_perm_even_473, vect_perm_even_477
	movdqa	%xmm3, %xmm4	 # vect_perm_odd_468, vect_perm_even_479
	shufps	$221, %xmm2, %xmm3	 #, vect_perm_odd_470, vect_perm_odd_468
	movdqu	-96(%rax), %xmm5	 # MEM <vector(4) int> [(i32 *)_94 + 416B], MEM <vector(4) int> [(i32 *)_94 + 416B]
	shufps	$136, %xmm2, %xmm4	 #, vect_perm_odd_470, vect_perm_even_479
	movdqa	%xmm1, %xmm2	 # vect_perm_odd_472, vect_perm_odd_472
	movaps	%xmm4, 416(%rsp)	 # vect_perm_even_479, %sfp
	movdqa	%xmm1, %xmm4	 # vect_perm_odd_472, vect_perm_even_481
	movdqu	-128(%rax), %xmm1	 # MEM <vector(4) int> [(i32 *)_94 + 384B], MEM <vector(4) int> [(i32 *)_94 + 384B]
	shufps	$136, %xmm0, %xmm4	 #, vect_perm_odd_474, vect_perm_even_481
	shufps	$221, %xmm0, %xmm2	 #, vect_perm_odd_474, vect_perm_odd_472
	movaps	%xmm14, 432(%rsp)	 # vect_perm_even_477, %sfp
	movdqu	-112(%rax), %xmm14	 # MEM <vector(4) int> [(i32 *)_94 + 400B], MEM <vector(4) int> [(i32 *)_94 + 400B]
	movaps	%xmm3, 112(%rsp)	 # vect_perm_odd_468, %sfp
	movdqu	-64(%rax), %xmm0	 # MEM <vector(4) int> [(i32 *)_94 + 448B], MEM <vector(4) int> [(i32 *)_94 + 448B]
	movaps	%xmm2, 80(%rsp)	 # vect_perm_odd_472, %sfp
	movdqa	%xmm1, %xmm3	 # MEM <vector(4) int> [(i32 *)_94 + 384B], vect_perm_even_499
	movdqu	-80(%rax), %xmm2	 # MEM <vector(4) int> [(i32 *)_94 + 432B], MEM <vector(4) int> [(i32 *)_94 + 432B]
	shufps	$136, %xmm14, %xmm3	 #, MEM <vector(4) int> [(i32 *)_94 + 400B], vect_perm_even_499
	shufps	$221, %xmm14, %xmm1	 #, MEM <vector(4) int> [(i32 *)_94 + 400B], vect_perm_odd_500
	movaps	%xmm4, 96(%rsp)	 # vect_perm_even_481, %sfp
	movdqa	%xmm5, %xmm14	 # MEM <vector(4) int> [(i32 *)_94 + 416B], vect_perm_even_501
	movdqu	-32(%rax), %xmm4	 # MEM <vector(4) int> [(i32 *)_94 + 480B], MEM <vector(4) int> [(i32 *)_94 + 480B]
	movdqu	-16(%rax), %xmm15	 # MEM <vector(4) int> [(i32 *)_94 + 496B], MEM <vector(4) int> [(i32 *)_94 + 496B]
	shufps	$136, %xmm2, %xmm14	 #, MEM <vector(4) int> [(i32 *)_94 + 432B], vect_perm_even_501
	shufps	$221, %xmm2, %xmm5	 #, MEM <vector(4) int> [(i32 *)_94 + 432B], vect_perm_odd_502
	movdqa	%xmm0, %xmm2	 # MEM <vector(4) int> [(i32 *)_94 + 448B], vect_perm_even_503
	shufps	$136, %xmm11, %xmm2	 #, MEM <vector(4) int> [(i32 *)_94 + 464B], vect_perm_even_503
	shufps	$221, %xmm11, %xmm0	 #, MEM <vector(4) int> [(i32 *)_94 + 464B], vect_perm_odd_504
	movdqa	%xmm4, %xmm11	 # MEM <vector(4) int> [(i32 *)_94 + 480B], vect_perm_even_505
	shufps	$136, %xmm15, %xmm11	 #, MEM <vector(4) int> [(i32 *)_94 + 496B], vect_perm_even_505
	shufps	$221, %xmm15, %xmm4	 #, MEM <vector(4) int> [(i32 *)_94 + 496B], vect_perm_odd_506
	movdqa	%xmm3, %xmm15	 # vect_perm_even_499, vect_perm_even_507
	shufps	$221, %xmm14, %xmm3	 #, vect_perm_even_501, vect_perm_odd_508
	shufps	$136, %xmm14, %xmm15	 #, vect_perm_even_501, vect_perm_even_507
	movdqa	%xmm2, %xmm14	 # vect_perm_even_503, vect_perm_even_509
	shufps	$221, %xmm11, %xmm2	 #, vect_perm_even_505, vect_perm_odd_510
	shufps	$136, %xmm11, %xmm14	 #, vect_perm_even_505, vect_perm_even_509
	movdqa	%xmm1, %xmm11	 # vect_perm_odd_500, vect_perm_even_511
	shufps	$221, %xmm5, %xmm1	 #, vect_perm_odd_502, vect_perm_odd_512
	shufps	$136, %xmm5, %xmm11	 #, vect_perm_odd_502, vect_perm_even_511
	movdqa	%xmm0, %xmm5	 # vect_perm_odd_504, vect_perm_even_513
	shufps	$221, %xmm4, %xmm0	 #, vect_perm_odd_506, vect_perm_odd_514
	shufps	$136, %xmm4, %xmm5	 #, vect_perm_odd_506, vect_perm_even_513
	movdqa	%xmm15, %xmm4	 # vect_perm_even_507, vect_perm_even_515
	shufps	$221, %xmm14, %xmm15	 #, vect_perm_even_509, vect_perm_odd_516
	shufps	$136, %xmm14, %xmm4	 #, vect_perm_even_509, vect_perm_even_515
	movdqa	%xmm11, %xmm14	 # vect_perm_even_511, vect_perm_even_517
	shufps	$136, %xmm5, %xmm14	 #, vect_perm_even_513, vect_perm_even_517
	movaps	%xmm14, 64(%rsp)	 # vect_perm_even_517, %sfp
	movdqa	%xmm11, %xmm14	 # vect_perm_even_511, vect_perm_even_511
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm13, %xmm11	 # tmp542, vect_iftmp.168
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	shufps	$221, %xmm5, %xmm14	 #, vect_perm_even_513, vect_perm_even_511
	movdqa	%xmm3, %xmm5	 # vect_perm_odd_508, vect_perm_even_519
	movaps	%xmm14, 48(%rsp)	 # vect_perm_even_511, %sfp
	shufps	$221, %xmm2, %xmm3	 #, vect_perm_odd_510, vect_perm_odd_508
	shufps	$136, %xmm2, %xmm5	 #, vect_perm_odd_510, vect_perm_even_519
	movdqa	%xmm1, %xmm2	 # vect_perm_odd_512, vect_perm_even_521
	shufps	$221, %xmm0, %xmm1	 #, vect_perm_odd_514, vect_perm_odd_512
	movdqa	%xmm1, %xmm14	 # vect_perm_odd_512, vect_perm_odd_522
	movdqu	-16(%rcx), %xmm1	 # MEM <vector(16) unsigned char> [(b8 *)_91], tmp1049
	shufps	$136, %xmm0, %xmm2	 #, vect_perm_odd_514, vect_perm_even_521
	movdqa	%xmm12, %xmm0	 # tmp524, tmp525
	movaps	%xmm2, (%rsp)	 # vect_perm_even_521, %sfp
	movaps	%xmm3, 16(%rsp)	 # vect_perm_odd_508, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm13, %xmm3	 # tmp542, vect_iftmp.160
	psubd	%xmm7, %xmm11	 # vect_perm_even_397, vect_iftmp.168
	pcmpgtb	%xmm1, %xmm0	 # tmp1049, tmp525
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movaps	%xmm5, 32(%rsp)	 # vect_perm_even_519, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm10, %xmm3	 # vect_perm_even_395, vect_iftmp.160
	movdqa	%xmm0, %xmm2	 # tmp525, tmp525
	movdqa	%xmm12, %xmm0	 # tmp524, tmp530
	pcmpeqb	%xmm12, %xmm2	 # tmp524, tmp527
	pcmpgtb	%xmm2, %xmm0	 # tmp527, tmp530
	movdqa	%xmm0, %xmm1	 # tmp530, tmp530
	movdqa	%xmm2, %xmm0	 # tmp527, tmp531
	punpcklbw	%xmm1, %xmm0	 # tmp530, tmp531
	punpckhbw	%xmm1, %xmm2	 # tmp530, tmp535
	pxor	%xmm1, %xmm1	 # tmp539
	pcmpgtw	%xmm0, %xmm1	 # tmp531, tmp539
	movdqa	%xmm1, %xmm5	 # tmp539, tmp539
	movdqa	%xmm0, %xmm1	 # tmp531, tmp540
	punpcklwd	%xmm5, %xmm1	 # tmp539, tmp540
	pand	%xmm1, %xmm3	 # tmp540, tmp543
	pandn	%xmm10, %xmm1	 # vect_perm_even_395, tmp544
	por	%xmm3, %xmm1	 # tmp543, vect_patt_224.164
	movdqa	%xmm13, %xmm3	 # tmp542, vect_iftmp.160
	punpckhwd	%xmm5, %xmm0	 # tmp539, tmp549
	psubd	%xmm9, %xmm3	 # vect_perm_even_435, vect_iftmp.160
	pand	%xmm0, %xmm3	 # tmp549, tmp552
	pandn	%xmm9, %xmm0	 # vect_perm_even_435, tmp553
	pxor	%xmm9, %xmm9	 # tmp557
	pcmpgtw	%xmm2, %xmm9	 # tmp535, tmp557
	por	%xmm3, %xmm0	 # tmp552, vect_patt_224.164
	movdqa	%xmm9, %xmm5	 # tmp557, tmp557
	movdqa	%xmm2, %xmm9	 # tmp535, tmp558
	punpcklwd	%xmm5, %xmm9	 # tmp557, tmp558
	movdqa	%xmm9, %xmm3	 # tmp558, tmp558
	movdqa	%xmm13, %xmm9	 # tmp542, vect_iftmp.160
	punpckhwd	%xmm5, %xmm2	 # tmp557, tmp567
	psubd	%xmm8, %xmm9	 # vect_perm_even_475, vect_iftmp.160
	pand	%xmm3, %xmm9	 # tmp558, tmp561
	pandn	%xmm8, %xmm3	 # vect_perm_even_475, tmp562
	movdqa	%xmm3, %xmm8	 # tmp562, tmp562
	por	%xmm9, %xmm8	 # tmp561, tmp562
	movdqa	%xmm13, %xmm9	 # tmp542, vect_iftmp.160
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	368(%rsp), %xmm8	 # %sfp, vect_patt_224.164
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm4, %xmm9	 # vect_perm_even_515, vect_iftmp.160
	movdqa	%xmm9, %xmm3	 # vect_iftmp.160, vect_iftmp.160
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	400(%rsp), %xmm9	 # %sfp, vect_res_169.221
	movaps	%xmm8, 400(%rsp)	 # vect_patt_224.164, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm13, %xmm8	 # tmp542, vect_iftmp.168
	pand	%xmm2, %xmm3	 # tmp567, tmp570
	pandn	%xmm4, %xmm2	 # vect_perm_even_515, tmp571
	por	%xmm3, %xmm2	 # tmp570, vect_patt_224.164
	movdqa	%xmm12, %xmm3	 # tmp524, tmp579
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	352(%rsp), %xmm2	 # %sfp, vect_res_78.165
	paddd	%xmm1, %xmm9	 # vect_patt_224.164, vect_res_169.221
	movdqa	384(%rsp), %xmm1	 # %sfp, vect_res_93.121
	paddd	%xmm0, %xmm1	 # vect_patt_224.164, vect_res_93.121
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:67: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	movdqu	-16(%rcx), %xmm0	 # MEM <vector(16) unsigned char> [(b8 *)_91], vect_p_81.166
	cmpq	%rcx, %rdx	 # ivtmp.242, _52
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm1, %xmm4	 # vect_res_93.121, vect_res_78.165
	movdqa	%xmm12, %xmm1	 # tmp524, tmp574
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:67: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	paddb	%xmm0, %xmm0	 # tmp1066, vect_p_81.166
	pcmpgtb	%xmm0, %xmm1	 # vect_p_81.166, tmp574
	paddb	%xmm0, %xmm0	 # vect_p_81.166, vect_p_81.166
	pcmpeqb	%xmm12, %xmm1	 # tmp524, tmp576
	pcmpgtb	%xmm1, %xmm3	 # tmp576, tmp579
	movdqa	%xmm3, %xmm5	 # tmp579, tmp579
	movdqa	%xmm1, %xmm3	 # tmp576, tmp580
	punpcklbw	%xmm5, %xmm3	 # tmp579, tmp580
	punpckhbw	%xmm5, %xmm1	 # tmp579, tmp584
	pxor	%xmm5, %xmm5	 # tmp588
	pcmpgtw	%xmm3, %xmm5	 # tmp580, tmp588
	movdqa	%xmm5, %xmm10	 # tmp588, tmp588
	movdqa	%xmm3, %xmm5	 # tmp580, tmp589
	punpcklwd	%xmm10, %xmm5	 # tmp588, tmp589
	punpckhwd	%xmm10, %xmm3	 # tmp588, tmp598
	pand	%xmm5, %xmm11	 # tmp589, tmp592
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	240(%rsp), %xmm10	 # %sfp, vect_perm_even_437
	psubd	432(%rsp), %xmm8	 # %sfp, vect_iftmp.168
	pandn	%xmm7, %xmm5	 # vect_perm_even_397, tmp593
	movdqa	%xmm5, %xmm7	 # tmp593, tmp593
	movdqa	%xmm13, %xmm5	 # tmp542, vect_iftmp.168
	psubd	%xmm10, %xmm5	 # vect_perm_even_437, vect_iftmp.168
	por	%xmm11, %xmm7	 # tmp592, tmp593
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm9, %xmm7	 # vect_res_78.165, vect_res_91.173
	movdqa	%xmm12, %xmm9	 # tmp524, tmp624
	pand	%xmm3, %xmm5	 # tmp598, tmp601
	pandn	%xmm10, %xmm3	 # vect_perm_even_437, tmp602
	por	%xmm5, %xmm3	 # tmp601, vect_patt_227.172
	pxor	%xmm5, %xmm5	 # tmp606
	pcmpgtw	%xmm1, %xmm5	 # tmp584, tmp606
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm8, %xmm10	 # vect_iftmp.168, vect_iftmp.168
	movdqa	%xmm13, %xmm8	 # tmp542, vect_iftmp.168
	pcmpgtb	%xmm0, %xmm9	 # vect_p_94.174, tmp624
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm4, %xmm3	 # vect_res_78.165, vect_res_91.173
	movdqa	%xmm5, %xmm11	 # tmp606, tmp606
	movdqa	%xmm1, %xmm5	 # tmp584, tmp607
	punpcklwd	%xmm11, %xmm5	 # tmp606, tmp607
	punpckhwd	%xmm11, %xmm1	 # tmp606, tmp616
	pand	%xmm5, %xmm10	 # tmp607, tmp610
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	64(%rsp), %xmm11	 # %sfp, vect_perm_even_517
	pandn	432(%rsp), %xmm5	 # %sfp, tmp611
	por	%xmm10, %xmm5	 # tmp610, vect_patt_227.172
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	400(%rsp), %xmm5	 # %sfp, vect_res_91.173
	movaps	%xmm0, 400(%rsp)	 # vect_p_94.174, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm11, %xmm8	 # vect_perm_even_517, vect_iftmp.168
	movdqa	%xmm13, %xmm0	 # tmp542, vect_iftmp.176
	movdqa	%xmm8, %xmm10	 # vect_iftmp.168, vect_iftmp.168
	movdqa	%xmm12, %xmm8	 # tmp524, tmp629
	pand	%xmm1, %xmm10	 # tmp616, tmp619
	pandn	%xmm11, %xmm1	 # vect_perm_even_517, tmp620
	por	%xmm10, %xmm1	 # tmp619, vect_patt_227.172
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm2, %xmm1	 # vect_res_78.165, vect_res_91.173
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	304(%rsp), %xmm11	 # %sfp, vect_perm_even_399
	movdqa	%xmm9, %xmm2	 # tmp624, tmp624
	pcmpeqb	%xmm12, %xmm2	 # tmp524, tmp626
	psubd	%xmm11, %xmm0	 # vect_perm_even_399, vect_iftmp.176
	movdqa	%xmm0, %xmm10	 # vect_iftmp.176, vect_iftmp.176
	movdqa	%xmm13, %xmm0	 # tmp542, vect_iftmp.176
	pcmpgtb	%xmm2, %xmm8	 # tmp626, tmp629
	movdqa	%xmm2, %xmm9	 # tmp626, tmp630
	punpcklbw	%xmm8, %xmm9	 # tmp629, tmp630
	punpckhbw	%xmm8, %xmm2	 # tmp629, tmp634
	pxor	%xmm8, %xmm8	 # tmp638
	pcmpgtw	%xmm9, %xmm8	 # tmp630, tmp638
	movdqa	%xmm9, %xmm4	 # tmp630, tmp630
	punpcklwd	%xmm8, %xmm9	 # tmp638, tmp639
	pand	%xmm9, %xmm10	 # tmp639, tmp642
	pandn	%xmm11, %xmm9	 # vect_perm_even_399, tmp643
	por	%xmm10, %xmm9	 # tmp642, vect_patt_230.180
	movdqa	208(%rsp), %xmm10	 # %sfp, vect_perm_even_439
	punpckhwd	%xmm8, %xmm4	 # tmp638, tmp648
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm7, %xmm9	 # vect_res_91.173, vect_res_104.181
	movdqa	%xmm12, %xmm7	 # tmp524, tmp678
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm10, %xmm0	 # vect_perm_even_439, vect_iftmp.176
	movdqa	%xmm0, %xmm8	 # vect_iftmp.176, vect_iftmp.176
	pxor	%xmm0, %xmm0	 # tmp656
	pcmpgtw	%xmm2, %xmm0	 # tmp634, tmp656
	pand	%xmm4, %xmm8	 # tmp648, tmp651
	pandn	%xmm10, %xmm4	 # vect_perm_even_439, tmp652
	por	%xmm8, %xmm4	 # tmp651, vect_patt_230.180
	movdqa	%xmm2, %xmm8	 # tmp634, tmp657
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm3, %xmm4	 # vect_res_91.173, vect_res_104.181
	movdqa	%xmm0, %xmm11	 # tmp656, tmp656
	punpcklwd	%xmm0, %xmm8	 # tmp656, tmp657
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm13, %xmm0	 # tmp542, vect_iftmp.176
	psubd	416(%rsp), %xmm0	 # %sfp, vect_iftmp.176
	punpckhwd	%xmm11, %xmm2	 # tmp656, tmp666
	movdqa	32(%rsp), %xmm11	 # %sfp, vect_perm_even_519
	movdqa	%xmm0, %xmm10	 # vect_iftmp.176, vect_iftmp.176
	movdqa	%xmm13, %xmm0	 # tmp542, vect_iftmp.176
	psubd	%xmm11, %xmm0	 # vect_perm_even_519, vect_iftmp.176
	pand	%xmm8, %xmm10	 # tmp657, tmp660
	pandn	416(%rsp), %xmm8	 # %sfp, tmp661
	por	%xmm10, %xmm8	 # tmp660, vect_patt_230.180
	movdqa	%xmm0, %xmm10	 # vect_iftmp.176, vect_iftmp.176
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm5, %xmm8	 # vect_res_91.173, vect_patt_230.180
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:67: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	movdqa	400(%rsp), %xmm0	 # %sfp, vect_p_94.174
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movaps	%xmm8, 432(%rsp)	 # vect_patt_230.180, %sfp
	pand	%xmm2, %xmm10	 # tmp666, tmp669
	pandn	%xmm11, %xmm2	 # vect_perm_even_519, tmp670
	por	%xmm10, %xmm2	 # tmp669, vect_patt_230.180
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:67: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	paddb	%xmm0, %xmm0	 # vect_p_94.174, vect_p_94.174
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm1, %xmm2	 # vect_res_91.173, vect_res_104.181
	movdqa	%xmm12, %xmm1	 # tmp524, tmp675
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	96(%rsp), %xmm11	 # %sfp, vect_perm_even_481
	pcmpgtb	%xmm0, %xmm1	 # vect_p_107.182, tmp675
	movdqa	%xmm13, %xmm10	 # tmp542, vect_iftmp.184
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:67: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	paddb	%xmm0, %xmm0	 # vect_p_107.182, vect_p_120.190
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm6, %xmm10	 # vect_perm_even_401, vect_iftmp.184
	movdqa	%xmm13, %xmm8	 # tmp542, vect_iftmp.184
	psubd	%xmm11, %xmm8	 # vect_perm_even_481, vect_iftmp.184
	pcmpgtb	%xmm1, %xmm7	 # tmp675, tmp678
	movdqa	%xmm7, %xmm5	 # tmp678, tmp678
	movdqa	%xmm1, %xmm7	 # tmp675, tmp679
	punpcklbw	%xmm5, %xmm7	 # tmp678, tmp679
	movdqa	%xmm7, %xmm3	 # tmp679, tmp679
	pxor	%xmm7, %xmm7	 # tmp687
	punpckhbw	%xmm5, %xmm1	 # tmp678, tmp683
	pcmpgtw	%xmm3, %xmm7	 # tmp679, tmp687
	movdqa	%xmm3, %xmm5	 # tmp679, tmp688
	punpcklwd	%xmm7, %xmm5	 # tmp687, tmp688
	punpckhwd	%xmm7, %xmm3	 # tmp687, tmp697
	pand	%xmm5, %xmm6	 # tmp688, tmp691
	movdqa	176(%rsp), %xmm7	 # %sfp, vect_perm_even_441
	pandn	%xmm10, %xmm5	 # vect_iftmp.184, tmp692
	por	%xmm6, %xmm5	 # tmp691, vect_patt_233.188
	movdqa	%xmm13, %xmm6	 # tmp542, vect_iftmp.184
	movdqa	(%rsp), %xmm10	 # %sfp, vect_perm_even_521
	psubd	%xmm7, %xmm6	 # vect_perm_even_441, vect_iftmp.184
	pand	%xmm3, %xmm7	 # tmp697, vect_perm_even_441
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm9, %xmm5	 # vect_res_104.181, vect_res_117.189
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm13, %xmm9	 # tmp542, vect_iftmp.192
	pandn	%xmm6, %xmm3	 # vect_iftmp.184, tmp701
	por	%xmm7, %xmm3	 # tmp700, vect_patt_233.188
	pxor	%xmm7, %xmm7	 # tmp705
	pcmpgtw	%xmm1, %xmm7	 # tmp683, tmp705
	movdqa	%xmm1, %xmm6	 # tmp683, tmp706
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm4, %xmm3	 # vect_res_104.181, vect_res_117.189
	punpcklwd	%xmm7, %xmm6	 # tmp705, tmp706
	punpckhwd	%xmm7, %xmm1	 # tmp705, tmp715
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm13, %xmm7	 # tmp542, vect_iftmp.184
	pand	%xmm6, %xmm11	 # tmp706, vect_perm_even_481
	psubd	%xmm10, %xmm7	 # vect_perm_even_521, vect_iftmp.184
	pand	%xmm1, %xmm10	 # tmp715, vect_perm_even_521
	pandn	%xmm8, %xmm6	 # vect_iftmp.184, tmp710
	pxor	%xmm8, %xmm8	 # tmp737
	pandn	%xmm7, %xmm1	 # vect_iftmp.184, tmp719
	por	%xmm10, %xmm1	 # tmp718, vect_patt_233.188
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm2, %xmm1	 # vect_res_104.181, vect_res_117.189
	movdqa	%xmm12, %xmm2	 # tmp524, tmp725
	pcmpgtb	%xmm0, %xmm2	 # vect_p_120.190, tmp725
	movdqa	%xmm12, %xmm7	 # tmp524, tmp728
	por	%xmm11, %xmm6	 # tmp709, vect_patt_233.188
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	336(%rsp), %xmm11	 # %sfp, vect_perm_odd_396
	movdqa	%xmm13, %xmm10	 # tmp542, vect_iftmp.192
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:67: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	paddb	%xmm0, %xmm0	 # vect_p_120.190, vect_p_133.198
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	432(%rsp), %xmm6	 # %sfp, vect_res_117.189
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm11, %xmm9	 # vect_perm_odd_396, vect_iftmp.192
	pcmpgtb	%xmm2, %xmm7	 # tmp725, tmp728
	movdqa	%xmm2, %xmm4	 # tmp725, tmp729
	punpcklbw	%xmm7, %xmm4	 # tmp728, tmp729
	pcmpgtw	%xmm4, %xmm8	 # tmp729, tmp737
	punpckhbw	%xmm7, %xmm2	 # tmp728, tmp733
	movdqa	%xmm4, %xmm7	 # tmp729, tmp738
	punpcklwd	%xmm8, %xmm7	 # tmp737, tmp738
	pand	%xmm7, %xmm11	 # tmp738, vect_perm_odd_396
	pandn	%xmm9, %xmm7	 # vect_iftmp.192, tmp742
	por	%xmm11, %xmm7	 # tmp741, vect_patt_236.196
	movdqa	256(%rsp), %xmm11	 # %sfp, vect_perm_odd_436
	punpckhwd	%xmm8, %xmm4	 # tmp737, tmp747
	movdqa	%xmm13, %xmm8	 # tmp542, vect_iftmp.192
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm5, %xmm7	 # vect_res_117.189, vect_res_130.197
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm11, %xmm8	 # vect_perm_odd_436, vect_iftmp.192
	movdqa	%xmm11, %xmm9	 # vect_perm_odd_436, vect_perm_odd_436
	movdqa	144(%rsp), %xmm11	 # %sfp, vect_perm_odd_476
	pand	%xmm4, %xmm9	 # tmp747, vect_perm_odd_436
	pandn	%xmm8, %xmm4	 # vect_iftmp.192, tmp751
	por	%xmm9, %xmm4	 # tmp750, vect_patt_236.196
	pxor	%xmm9, %xmm9	 # tmp755
	pcmpgtw	%xmm2, %xmm9	 # tmp733, tmp755
	movdqa	%xmm2, %xmm8	 # tmp733, tmp756
	psubd	%xmm11, %xmm10	 # vect_perm_odd_476, vect_iftmp.192
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm3, %xmm4	 # vect_res_117.189, vect_res_130.197
	punpcklwd	%xmm9, %xmm8	 # tmp755, tmp756
	punpckhwd	%xmm9, %xmm2	 # tmp755, tmp765
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm13, %xmm9	 # tmp542, vect_iftmp.192
	pand	%xmm8, %xmm11	 # tmp756, vect_perm_odd_476
	psubd	%xmm15, %xmm9	 # vect_perm_odd_516, vect_iftmp.192
	pandn	%xmm10, %xmm8	 # vect_iftmp.192, tmp760
	por	%xmm11, %xmm8	 # tmp759, vect_patt_236.196
	movdqa	%xmm15, %xmm11	 # vect_perm_odd_516, vect_perm_odd_516
	pand	%xmm2, %xmm11	 # tmp765, vect_perm_odd_516
	pandn	%xmm9, %xmm2	 # vect_iftmp.192, tmp769
	movdqa	%xmm12, %xmm9	 # tmp524, tmp776
	por	%xmm11, %xmm2	 # tmp768, vect_patt_236.196
	pcmpgtb	%xmm0, %xmm9	 # vect_p_133.198, tmp776
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm1, %xmm2	 # vect_res_117.189, vect_res_130.197
	paddd	%xmm6, %xmm8	 # vect_res_117.189, vect_res_130.197
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	320(%rsp), %xmm15	 # %sfp, vect_perm_odd_398
	movdqa	224(%rsp), %xmm10	 # %sfp, vect_perm_odd_438
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:67: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	paddb	%xmm0, %xmm0	 # vect_p_133.198, vect_p_146.206
	movdqa	%xmm9, %xmm1	 # tmp776, tmp776
	movdqa	%xmm12, %xmm9	 # tmp524, tmp779
	pcmpgtb	%xmm1, %xmm9	 # tmp776, tmp779
	movdqa	%xmm9, %xmm5	 # tmp779, tmp779
	movdqa	%xmm1, %xmm9	 # tmp776, tmp780
	punpcklbw	%xmm5, %xmm9	 # tmp779, tmp780
	movdqa	%xmm9, %xmm3	 # tmp780, tmp780
	pxor	%xmm9, %xmm9	 # tmp788
	punpckhbw	%xmm5, %xmm1	 # tmp779, tmp784
	pcmpgtw	%xmm3, %xmm9	 # tmp780, tmp788
	movdqa	%xmm3, %xmm6	 # tmp780, tmp789
	movdqa	%xmm1, %xmm11	 # tmp784, tmp807
	movdqa	%xmm9, %xmm5	 # tmp788, tmp788
	punpcklwd	%xmm9, %xmm6	 # tmp788, tmp789
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm13, %xmm9	 # tmp542, vect_iftmp.200
	psubd	%xmm15, %xmm9	 # vect_perm_odd_398, vect_iftmp.200
	pand	%xmm6, %xmm15	 # tmp789, vect_perm_odd_398
	punpckhwd	%xmm5, %xmm3	 # tmp788, tmp798
	pandn	%xmm9, %xmm6	 # vect_iftmp.200, tmp793
	movdqa	%xmm13, %xmm9	 # tmp542, vect_iftmp.200
	por	%xmm15, %xmm6	 # tmp792, vect_patt_239.204
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm7, %xmm6	 # vect_res_130.197, vect_res_143.205
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm10, %xmm9	 # vect_perm_odd_438, vect_iftmp.200
	pand	%xmm3, %xmm10	 # tmp798, vect_perm_odd_438
	movdqa	%xmm12, %xmm7	 # tmp524, tmp828
	movdqa	288(%rsp), %xmm15	 # %sfp, vect_perm_odd_400
	pandn	%xmm9, %xmm3	 # vect_iftmp.200, tmp802
	pxor	%xmm9, %xmm9	 # tmp806
	pcmpgtw	%xmm1, %xmm9	 # tmp784, tmp806
	por	%xmm10, %xmm3	 # tmp801, vect_patt_239.204
	movdqa	%xmm13, %xmm10	 # tmp542, vect_iftmp.200
	pcmpgtb	%xmm0, %xmm7	 # vect_p_146.206, tmp828
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm4, %xmm3	 # vect_res_130.197, vect_res_143.205
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:67: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	paddb	%xmm0, %xmm0	 # vect_p_146.206, vect_p_159.214
	punpcklwd	%xmm9, %xmm11	 # tmp806, tmp807
	movdqa	%xmm11, %xmm5	 # tmp807, tmp807
	punpckhwd	%xmm9, %xmm1	 # tmp806, tmp816
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	128(%rsp), %xmm11	 # %sfp, vect_perm_odd_478
	movdqa	%xmm13, %xmm9	 # tmp542, vect_iftmp.200
	psubd	%xmm11, %xmm10	 # vect_perm_odd_478, vect_iftmp.200
	pand	%xmm5, %xmm11	 # tmp807, vect_perm_odd_478
	pandn	%xmm10, %xmm5	 # vect_iftmp.200, tmp811
	movdqa	48(%rsp), %xmm10	 # %sfp, vect_perm_odd_518
	por	%xmm11, %xmm5	 # tmp810, vect_patt_239.204
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm8, %xmm5	 # vect_res_130.197, vect_res_143.205
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm13, %xmm11	 # tmp542, vect_iftmp.208
	psubd	%xmm10, %xmm9	 # vect_perm_odd_518, vect_iftmp.200
	pand	%xmm1, %xmm10	 # tmp816, vect_perm_odd_518
	pandn	%xmm9, %xmm1	 # vect_iftmp.200, tmp820
	por	%xmm10, %xmm1	 # tmp819, vect_patt_239.204
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm2, %xmm1	 # vect_res_130.197, vect_res_143.205
	movdqa	%xmm7, %xmm2	 # tmp828, tmp828
	movdqa	%xmm12, %xmm7	 # tmp524, tmp831
	movdqa	%xmm2, %xmm4	 # tmp828, tmp832
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm13, %xmm9	 # tmp542, vect_iftmp.208
	pcmpgtb	%xmm2, %xmm7	 # tmp828, tmp831
	psubd	%xmm15, %xmm9	 # vect_perm_odd_400, vect_iftmp.208
	movdqa	%xmm13, %xmm10	 # tmp542, vect_iftmp.216
	punpcklbw	%xmm7, %xmm4	 # tmp831, tmp832
	punpckhbw	%xmm7, %xmm2	 # tmp831, tmp836
	pxor	%xmm7, %xmm7	 # tmp840
	pcmpgtw	%xmm4, %xmm7	 # tmp832, tmp840
	movdqa	%xmm4, %xmm8	 # tmp832, tmp841
	punpcklwd	%xmm7, %xmm8	 # tmp840, tmp841
	pand	%xmm8, %xmm15	 # tmp841, vect_perm_odd_400
	pandn	%xmm9, %xmm8	 # vect_iftmp.208, tmp845
	por	%xmm15, %xmm8	 # tmp844, vect_patt_242.212
	punpckhwd	%xmm7, %xmm4	 # tmp840, tmp850
	movdqa	%xmm13, %xmm7	 # tmp542, vect_iftmp.208
	pxor	%xmm9, %xmm9	 # tmp858
	movdqa	192(%rsp), %xmm15	 # %sfp, vect_perm_odd_440
	pcmpgtw	%xmm2, %xmm9	 # tmp836, tmp858
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm6, %xmm8	 # vect_res_143.205, vect_res_156.213
	pxor	%xmm6, %xmm6	 # tmp893
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm15, %xmm7	 # vect_perm_odd_440, vect_iftmp.208
	pand	%xmm4, %xmm15	 # tmp850, vect_perm_odd_440
	pandn	%xmm7, %xmm4	 # vect_iftmp.208, tmp854
	por	%xmm15, %xmm4	 # tmp853, vect_patt_242.212
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm3, %xmm4	 # vect_res_143.205, vect_res_156.213
	movdqa	%xmm12, %xmm3	 # tmp524, tmp881
	pcmpgtb	%xmm0, %xmm3	 # vect_p_159.214, tmp881
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	112(%rsp), %xmm15	 # %sfp, vect_perm_odd_480
	movdqa	%xmm12, %xmm0	 # tmp524, tmp884
	movdqa	%xmm2, %xmm7	 # tmp836, tmp859
	punpcklwd	%xmm9, %xmm7	 # tmp858, tmp859
	punpckhwd	%xmm9, %xmm2	 # tmp858, tmp868
	movdqa	%xmm13, %xmm9	 # tmp542, vect_iftmp.208
	psubd	%xmm15, %xmm11	 # vect_perm_odd_480, vect_iftmp.208
	pand	%xmm7, %xmm15	 # tmp859, vect_perm_odd_480
	pcmpgtb	%xmm3, %xmm0	 # tmp881, tmp884
	pandn	%xmm11, %xmm7	 # vect_iftmp.208, tmp863
	por	%xmm15, %xmm7	 # tmp862, vect_patt_242.212
	movdqa	16(%rsp), %xmm15	 # %sfp, vect_perm_odd_520
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm5, %xmm7	 # vect_res_143.205, vect_res_156.213
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm15, %xmm9	 # vect_perm_odd_520, vect_iftmp.208
	pand	%xmm2, %xmm15	 # tmp868, vect_perm_odd_520
	movdqa	%xmm0, %xmm5	 # tmp884, tmp884
	movdqa	%xmm3, %xmm0	 # tmp881, tmp885
	pandn	%xmm9, %xmm2	 # vect_iftmp.208, tmp872
	por	%xmm15, %xmm2	 # tmp871, vect_patt_242.212
	punpcklbw	%xmm5, %xmm0	 # tmp884, tmp885
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm1, %xmm2	 # vect_res_143.205, vect_res_156.213
	movdqa	%xmm0, %xmm1	 # tmp885, tmp885
	movdqa	%xmm3, %xmm0	 # tmp881, tmp881
	punpckhbw	%xmm5, %xmm0	 # tmp884, tmp881
	pcmpgtw	%xmm1, %xmm6	 # tmp885, tmp893
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm13, %xmm9	 # tmp542, vect_iftmp.216
	movdqa	%xmm1, %xmm3	 # tmp885, tmp894
	movdqa	272(%rsp), %xmm5	 # %sfp, vect_perm_odd_402
	psubd	%xmm5, %xmm9	 # vect_perm_odd_402, vect_iftmp.216
	punpcklwd	%xmm6, %xmm3	 # tmp893, tmp894
	pand	%xmm3, %xmm5	 # tmp894, vect_perm_odd_402
	punpckhwd	%xmm6, %xmm1	 # tmp893, tmp903
	pandn	%xmm9, %xmm3	 # vect_iftmp.216, tmp898
	por	%xmm5, %xmm3	 # tmp897, vect_patt_245.220
	movdqa	%xmm13, %xmm6	 # tmp542, vect_iftmp.216
	movdqa	160(%rsp), %xmm5	 # %sfp, vect_perm_odd_442
	pxor	%xmm9, %xmm9	 # tmp911
	pcmpgtw	%xmm0, %xmm9	 # tmp889, tmp911
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm3, %xmm8	 # vect_patt_245.220, vect_res_156.213
	movaps	%xmm8, 400(%rsp)	 # vect_res_156.213, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm5, %xmm6	 # vect_perm_odd_442, vect_iftmp.216
	pand	%xmm1, %xmm5	 # tmp903, vect_perm_odd_442
	pandn	%xmm6, %xmm1	 # vect_iftmp.216, tmp907
	movdqa	80(%rsp), %xmm6	 # %sfp, vect_perm_odd_482
	por	%xmm5, %xmm1	 # tmp906, vect_patt_245.220
	movdqa	%xmm0, %xmm5	 # tmp889, tmp912
	punpcklwd	%xmm9, %xmm5	 # tmp911, tmp912
	punpckhwd	%xmm9, %xmm0	 # tmp911, tmp921
	movdqa	%xmm13, %xmm9	 # tmp542, vect_iftmp.216
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm1, %xmm4	 # vect_patt_245.220, vect_res_156.213
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm6, %xmm10	 # vect_perm_odd_482, vect_iftmp.216
	psubd	%xmm14, %xmm9	 # vect_perm_odd_522, vect_iftmp.216
	pand	%xmm5, %xmm6	 # tmp912, vect_perm_odd_482
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movaps	%xmm4, 384(%rsp)	 # vect_res_156.213, %sfp
	pandn	%xmm10, %xmm5	 # vect_iftmp.216, tmp916
	por	%xmm6, %xmm5	 # tmp915, vect_patt_245.220
	movdqa	%xmm14, %xmm6	 # vect_perm_odd_522, vect_perm_odd_522
	paddd	%xmm5, %xmm7	 # vect_patt_245.220, vect_res_156.213
	pand	%xmm0, %xmm6	 # tmp921, vect_perm_odd_522
	pandn	%xmm9, %xmm0	 # vect_iftmp.216, tmp925
	por	%xmm6, %xmm0	 # tmp924, vect_patt_245.220
	paddd	%xmm0, %xmm2	 # vect_patt_245.220, vect_res_156.213
	movaps	%xmm7, 368(%rsp)	 # vect_res_156.213, %sfp
	movaps	%xmm2, 352(%rsp)	 # vect_res_156.213, %sfp
	jne	.L59	 #,
	movdqa	384(%rsp), %xmm0	 # %sfp, vect_res_169.221
	movl	%edi, %edx	 # tmp952, niters_vector_mult_vf.117
	andl	$-16, %edx	 #, niters_vector_mult_vf.117
	leal	(%r8,%rdx,8), %r14d	 #, tmp.120
	paddd	%xmm8, %xmm0	 # vect_res_156.213, vect_res_169.221
	movl	%edx, %r11d	 # niters_vector_mult_vf.117, _345
	paddd	368(%rsp), %xmm0	 # %sfp, tmp930
	paddd	352(%rsp), %xmm0	 # %sfp, _703
	movdqa	%xmm0, %xmm1	 # _703, tmp932
	psrldq	$8, %xmm1	 #, tmp932
	paddd	%xmm1, %xmm0	 # tmp932, _705
	movdqa	%xmm0, %xmm1	 # _705, tmp934
	psrldq	$4, %xmm1	 #, tmp934
	paddd	%xmm1, %xmm0	 # tmp934, tmp935
	movd	%xmm0, %eax	 # tmp935, stmp_res_169.222
	addl	%eax, %r9d	 # stmp_res_169.222, <retval>
	movq	%r11, %rax	 # _345, tmp937
	addq	%r12, %r11	 # weight, tmp.119
	salq	$5, %rax	 #, tmp937
	addq	%r10, %rax	 # data, tmp.118
	cmpl	%edx, %edi	 # niters_vector_mult_vf.117, tmp952
	je	.L61	 #,
	.p2align 4,,10
	.p2align 3
.L64:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	(%rax), %ecx	 # MEM[(i32 *)data_278], pretmp_284
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:67: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	movzbl	(%r11), %edx	 # MEM[(b8 *)weight_279], p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	%ecx, %r15d	 # pretmp_284, tmp955
	negl	%r15d	 # tmp955
	testb	%dl, %dl	 # p
	cmovns	%r15d, %ecx	 # tmp955,, pretmp_284
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addl	%ecx, %r9d	 # pretmp_284, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	4(%rax), %ecx	 # MEM[(i32 *)data_278 + 4B], iftmp.4_320
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	%ecx, %r15d	 # iftmp.4_320, tmp965
	negl	%r15d	 # tmp965
	testb	$64, %dl	 #, p
	cmove	%r15d, %ecx	 # tmp965,, iftmp.4_320
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addl	%ecx, %r9d	 # iftmp.4_320, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	8(%rax), %ecx	 # MEM[(i32 *)data_278 + 8B], iftmp.4_314
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	%ecx, %r15d	 # iftmp.4_314, tmp969
	negl	%r15d	 # tmp969
	testb	$32, %dl	 #, p
	cmove	%r15d, %ecx	 # tmp969,, iftmp.4_314
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addl	%ecx, %r9d	 # iftmp.4_314, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	12(%rax), %ecx	 # MEM[(i32 *)data_278 + 12B], iftmp.4_308
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	%ecx, %r15d	 # iftmp.4_308, tmp961
	negl	%r15d	 # tmp961
	testb	$16, %dl	 #, p
	cmove	%r15d, %ecx	 # tmp961,, iftmp.4_308
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addl	%ecx, %r9d	 # iftmp.4_308, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	16(%rax), %ecx	 # MEM[(i32 *)data_278 + 16B], iftmp.4_302
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	%ecx, %r15d	 # iftmp.4_302, tmp971
	negl	%r15d	 # tmp971
	testb	$8, %dl	 #, p
	cmove	%r15d, %ecx	 # tmp971,, iftmp.4_302
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addl	%ecx, %r9d	 # iftmp.4_302, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	20(%rax), %ecx	 # MEM[(i32 *)data_278 + 20B], iftmp.4_296
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	%ecx, %r15d	 # iftmp.4_296, tmp959
	negl	%r15d	 # tmp959
	testb	$4, %dl	 #, p
	cmove	%r15d, %ecx	 # tmp959,, iftmp.4_296
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addl	%ecx, %r9d	 # iftmp.4_296, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	24(%rax), %ecx	 # MEM[(i32 *)data_278 + 24B], iftmp.4_290
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	%ecx, %r15d	 # iftmp.4_290, tmp967
	negl	%r15d	 # tmp967
	testb	$2, %dl	 #, p
	cmove	%r15d, %ecx	 # tmp967,, iftmp.4_290
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addl	%r9d, %ecx	 # res, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	28(%rax), %r9d	 # MEM[(i32 *)data_278 + 28B], iftmp.4_285
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	%r9d, %r15d	 # iftmp.4_285, tmp957
	negl	%r15d	 # tmp957
	andl	$1, %edx	 #, p
	cmove	%r15d, %r9d	 # tmp957,, iftmp.4_285
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:66: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	addl	$8, %r14d	 #, tmp.120
	addq	$32, %rax	 #, tmp.118
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:66: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	addq	$1, %r11	 #, tmp.119
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:68: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addl	%ecx, %r9d	 # res, <retval>
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:66: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	cmpl	%r14d, %esi	 # tmp.120, _7
	jg	.L64	 #,
.L61:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:66: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	leal	8(%r8,%r13,8), %r8d	 #, n0
	movl	%edi, %edi	 # tmp952, _18
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:66: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	addq	%rdi, %r12	 # _18, weight
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:67: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	salq	$5, %rdi	 #, tmp928
	addq	%rdi, %r10	 # tmp928, data
.L57:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:71: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	cmpl	%r8d, %ebx	 # n0, n
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:71: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	movzbl	(%r12), %edx	 # *weight_56, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:71: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	jle	.L53	 #,
	subl	$1, %ebx	 #, tmp946
	subl	%r8d, %ebx	 # n0, tmp948
	leaq	4(%r10,%rbx,4), %r8	 #, _51
	.p2align 4,,10
	.p2align 3
.L74:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:72: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movl	(%r10), %eax	 # MEM[(i32 *)data_102], pretmp_139
	movl	%eax, %ecx	 # pretmp_139, tmp963
	negl	%ecx	 # tmp963
	testb	%dl, %dl	 # p
	cmovns	%ecx, %eax	 # tmp963,, pretmp_139
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:71: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	addq	$4, %r10	 #, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:71: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	addl	%edx, %edx	 # p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:72: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	addl	%eax, %r9d	 # pretmp_139, <retval>
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:71: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	cmpq	%r10, %r8	 # data, _51
	jne	.L74	 #,
.L53:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:75: }
	movaps	448(%rsp), %xmm6	 #,
	movl	%r9d, %eax	 # <retval>,
	movaps	464(%rsp), %xmm7	 #,
	movaps	480(%rsp), %xmm8	 #,
	movaps	496(%rsp), %xmm9	 #,
	movaps	512(%rsp), %xmm10	 #,
	movaps	528(%rsp), %xmm11	 #,
	movaps	544(%rsp), %xmm12	 #,
	movaps	560(%rsp), %xmm13	 #,
	movaps	576(%rsp), %xmm14	 #,
	movaps	592(%rsp), %xmm15	 #,
	leaq	-56(%rbp), %rsp	 #,
	popq	%rbx	 #
	popq	%rsi	 #
	popq	%rdi	 #
	popq	%r12	 #
	popq	%r13	 #
	popq	%r14	 #
	popq	%r15	 #
	popq	%rbp	 #
	ret	
.L75:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:62: 	for (; i < n0; i++, data++, p <<= 1)
	movq	%r11, %r10	 # data, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:56: 	i32 res = 0;  // 初始化结果为0, 结果要除以127,解析成(-1,1)的8字节数
	xorl	%r9d, %r9d	 # <retval>
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:55: 	int i = 0, j = 0;
	xorl	%r8d, %r8d	 # n0
	jmp	.L54	 #
.L76:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:66: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	movl	%r8d, %r14d	 # n0, tmp.120
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:66: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	movq	%r12, %r11	 # weight, tmp.119
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:66: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	movq	%r10, %rax	 # data, tmp.118
	jmp	.L64	 #
	.p2align 4
	.globl	_Z10i32_i8_mulPiPaS_iiib
	.def	_Z10i32_i8_mulPiPaS_iiib;	.scl	2;	.type	32;	.endef
_Z10i32_i8_mulPiPaS_iiib:
	pushq	%rbp	 #
	movq	%rcx, %r10	 # tmp857, data
	movq	%rdx, %rcx	 # tmp858, weight
	movq	%rsp, %rbp	 #,
	pushq	%r15	 #
	pushq	%r14	 #
	pushq	%r13	 #
	pushq	%r12	 #
	pushq	%rdi	 #
	pushq	%rsi	 #
	pushq	%rbx	 #
	subq	$48, %rsp	 #,
	andq	$-16, %rsp	 #,
	subq	$64, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:80: 	if (Transpose) {  // 如果需要转置输出
	cmpb	$0, 64(%rbp)	 #, Transpose
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:79: void i32_i8_mul(i32 *data, i8 *weight, i32 *out, int m, int k, int n, bool Transpose = false) {
	movl	%r9d, 40(%rbp)	 # tmp860, m
	movl	48(%rbp), %edx	 # k, k
	movaps	%xmm6, 64(%rsp)	 #,
	movaps	%xmm7, 80(%rsp)	 #,
	movaps	%xmm8, 96(%rsp)	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:80: 	if (Transpose) {  // 如果需要转置输出
	jne	.L84	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:88: 		for (int i = 0; i < m; i++) {		 // 遍历矩阵A的行
	testl	%r9d, %r9d	 #
	jle	.L83	 #,
	movl	56(%rbp), %r11d	 # n,
	testl	%r11d, %r11d	 #
	jle	.L83	 #,
	movl	56(%rbp), %eax	 # n, _600
	movslq	%edx, %r13	 # k, _614
	movl	%edx, %r14d	 # k, niters_vector_mult_vf.286
	movq	%r10, 48(%rsp)	 # data, %sfp
	andl	$-16, %r14d	 #, niters_vector_mult_vf.286
	xorl	%r11d, %r11d	 # ivtmp.356
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	pxor	%xmm4, %xmm4	 # tmp849
	pxor	%xmm3, %xmm3	 # tmp850
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:88: 		for (int i = 0; i < m; i++) {		 // 遍历矩阵A的行
	movl	$0, 36(%rsp)	 #, %sfp
	leaq	0(,%rax,4), %rdi	 #, _604
	negq	%rax	 # tmp829
	addq	%rdi, %r8	 # _604, out
	movq	%rdi, 24(%rsp)	 # _604, %sfp
	salq	$2, %rax	 #, tmp830
	movq	%r8, 56(%rsp)	 # out, %sfp
	leaq	0(,%r13,4), %rdi	 #, _618
	movl	%edx, %r8d	 # k, bnd.285
	shrl	$4, %r8d	 #, bnd.285
	movq	%rdi, 16(%rsp)	 # _618, %sfp
	leal	-1(%rdx), %edi	 #, _165
	subl	$1, %r8d	 #, _580
	movq	%rax, 8(%rsp)	 # tmp830, %sfp
	addq	$1, %r8	 #, tmp835
	movl	%edi, 40(%rsp)	 # _165, %sfp
	salq	$6, %r8	 #, tmp835
	movq	%r8, (%rsp)	 # tmp835, %sfp
	.p2align 4,,10
	.p2align 3
.L86:
	movq	8(%rsp), %rdi	 # %sfp, out
	movq	%rcx, %r12	 # weight, ivtmp.348
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:81: 		for (int j = 0; j < n; j++) { // 遍历矩阵B的行
	xorl	%r9d, %r9d	 # ivtmp.347
	movq	(%rsp), %rsi	 # %sfp, _584
	addq	56(%rsp), %rdi	 # %sfp, out
	addq	48(%rsp), %rsi	 # %sfp, _584
	.p2align 4,,10
	.p2align 3
.L97:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	testl	%edx, %edx	 # k
	jle	.L100	 #,
	cmpl	$14, 40(%rsp)	 #, %sfp
	jbe	.L101	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	movq	48(%rsp), %rax	 # %sfp, ivtmp.334
	movq	%r12, %r8	 # ivtmp.348, ivtmp.337
	pxor	%xmm0, %xmm0	 # vect_res_104.288
	.p2align 4,,10
	.p2align 3
.L95:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqu	(%r8), %xmm1	 # MEM <vector(16) signed char> [(i8 *)_578], MEM <vector(16) signed char> [(i8 *)_578]
	movdqa	%xmm4, %xmm2	 # tmp849, tmp689
	movdqa	%xmm3, %xmm7	 # tmp850, tmp699
	addq	$64, %rax	 #, ivtmp.334
	addq	$16, %r8	 #, ivtmp.337
	pcmpgtb	%xmm1, %xmm2	 # MEM <vector(16) signed char> [(i8 *)_578], tmp689
	movdqa	%xmm1, %xmm5	 # MEM <vector(16) signed char> [(i8 *)_578], tmp690
	punpcklbw	%xmm2, %xmm5	 # tmp689, tmp690
	punpckhbw	%xmm2, %xmm1	 # tmp689, tmp694
	pcmpgtw	%xmm5, %xmm7	 # tmp690, tmp699
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqu	-64(%rax), %xmm2	 # MEM <vector(4) int> [(i32 *)_574], vect__59.291
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm5, %xmm8	 # tmp690, tmp700
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm2, %xmm6	 # vect__59.291, tmp704
	psrlq	$32, %xmm2	 #, tmp706
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	punpcklwd	%xmm7, %xmm8	 # tmp699, tmp700
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	pmuludq	%xmm8, %xmm6	 # tmp700, tmp704
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	punpckhwd	%xmm7, %xmm5	 # tmp699, tmp713
	movdqa	%xmm3, %xmm7	 # tmp850, tmp725
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	psrlq	$32, %xmm8	 #, tmp707
	pmuludq	%xmm8, %xmm2	 # tmp707, tmp705
	pshufd	$8, %xmm6, %xmm6	 #, tmp704, tmp702
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	pcmpgtw	%xmm1, %xmm7	 # tmp694, tmp725
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	pshufd	$8, %xmm2, %xmm2	 #, tmp705, tmp703
	punpckldq	%xmm2, %xmm6	 # tmp703, vect__63.300
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqu	-48(%rax), %xmm2	 # MEM <vector(4) int> [(i32 *)_574 + 16B], vect__59.292
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	paddd	%xmm6, %xmm0	 # vect__63.300, vect_res_65.301
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm2, %xmm6	 # vect__59.292, tmp717
	psrlq	$32, %xmm2	 #, tmp719
	pmuludq	%xmm5, %xmm6	 # tmp713, tmp717
	psrlq	$32, %xmm5	 #, tmp720
	pmuludq	%xmm5, %xmm2	 # tmp720, tmp718
	pshufd	$8, %xmm6, %xmm5	 #, tmp717, tmp715
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm1, %xmm6	 # tmp694, tmp726
	punpckhwd	%xmm7, %xmm1	 # tmp725, tmp739
	punpcklwd	%xmm7, %xmm6	 # tmp725, tmp726
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	pshufd	$8, %xmm2, %xmm2	 #, tmp718, tmp716
	punpckldq	%xmm2, %xmm5	 # tmp716, vect__63.300
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqu	-32(%rax), %xmm2	 # MEM <vector(4) int> [(i32 *)_574 + 32B], vect__59.293
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	paddd	%xmm5, %xmm0	 # vect__63.300, vect_res_65.301
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm2, %xmm5	 # vect__59.293, tmp730
	psrlq	$32, %xmm2	 #, tmp732
	pmuludq	%xmm6, %xmm5	 # tmp726, tmp730
	psrlq	$32, %xmm6	 #, tmp733
	pmuludq	%xmm6, %xmm2	 # tmp733, tmp731
	pshufd	$8, %xmm5, %xmm5	 #, tmp730, tmp728
	pshufd	$8, %xmm2, %xmm2	 #, tmp731, tmp729
	punpckldq	%xmm2, %xmm5	 # tmp729, vect__63.300
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqu	-16(%rax), %xmm2	 # MEM <vector(4) int> [(i32 *)_574 + 48B], vect__59.294
	cmpq	%rsi, %rax	 # _584, ivtmp.334
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	paddd	%xmm0, %xmm5	 # vect_res_65.301, vect_res_65.301
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm2, %xmm0	 # vect__59.294, tmp743
	psrlq	$32, %xmm2	 #, tmp745
	pmuludq	%xmm1, %xmm0	 # tmp739, tmp743
	psrlq	$32, %xmm1	 #, tmp746
	pmuludq	%xmm1, %xmm2	 # tmp746, tmp744
	pshufd	$8, %xmm0, %xmm0	 #, tmp743, tmp741
	pshufd	$8, %xmm2, %xmm2	 #, tmp744, tmp742
	punpckldq	%xmm2, %xmm0	 # tmp742, vect__63.300
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	paddd	%xmm5, %xmm0	 # vect_res_65.301, vect_res_104.288
	jne	.L95	 #,
	movdqa	%xmm0, %xmm1	 # vect_res_104.288, tmp748
	cmpl	%r14d, %edx	 # niters_vector_mult_vf.286, k
	psrldq	$8, %xmm1	 #, tmp748
	paddd	%xmm1, %xmm0	 # tmp748, _206
	movdqa	%xmm0, %xmm1	 # _206, tmp750
	psrldq	$4, %xmm1	 #, tmp750
	paddd	%xmm1, %xmm0	 # tmp750, tmp751
	movd	%xmm0, %eax	 # tmp751, stmp_res_65.302
	je	.L96	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	movl	%r14d, %r8d	 # niters_vector_mult_vf.286, i
.L94:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%r8d, %rbx	 # i, _378
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r11), %r15	 #, tmp752
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp753
	movsbl	(%rbx,%r9), %ebx	 # *_384, *_384
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r15,4), %ebx	 # *_381, tmp756
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp756, stmp_res_65.302
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	1(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L96	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _392
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r11), %r15	 #, tmp757
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp758
	movsbl	(%rbx,%r9), %ebx	 # *_398, *_398
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r15,4), %ebx	 # *_395, tmp761
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp761, stmp_res_65.302
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	2(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L96	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _406
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r11), %r15	 #, tmp762
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp763
	movsbl	(%rbx,%r9), %ebx	 # *_412, *_412
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r15,4), %ebx	 # *_409, tmp766
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp766, stmp_res_65.302
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	3(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L96	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _420
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r11), %r15	 #, tmp767
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp768
	movsbl	(%rbx,%r9), %ebx	 # *_426, *_426
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r15,4), %ebx	 # *_423, tmp771
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp771, stmp_res_65.302
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	4(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L96	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _434
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r11), %r15	 #, tmp772
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp773
	movsbl	(%rbx,%r9), %ebx	 # *_440, *_440
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r15,4), %ebx	 # *_437, tmp776
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp776, stmp_res_65.302
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	5(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L96	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _448
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r11), %r15	 #, tmp777
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp778
	movsbl	(%rbx,%r9), %ebx	 # *_454, *_454
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r15,4), %ebx	 # *_451, tmp781
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp781, stmp_res_65.302
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	6(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L96	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _462
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r11), %r15	 #, tmp782
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp783
	movsbl	(%rbx,%r9), %ebx	 # *_468, *_468
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r15,4), %ebx	 # *_465, tmp786
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp786, stmp_res_65.302
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	7(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L96	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _476
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r11), %r15	 #, tmp787
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp788
	movsbl	(%rbx,%r9), %ebx	 # *_482, *_482
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r15,4), %ebx	 # *_479, tmp791
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp791, stmp_res_65.302
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	8(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L96	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _490
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r11), %r15	 #, tmp792
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp793
	movsbl	(%rbx,%r9), %ebx	 # *_496, *_496
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r15,4), %ebx	 # *_493, tmp796
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp796, stmp_res_65.302
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	9(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L96	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _504
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r11), %r15	 #, tmp797
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp798
	movsbl	(%rbx,%r9), %ebx	 # *_510, *_510
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r15,4), %ebx	 # *_507, tmp801
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp801, stmp_res_65.302
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	10(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L96	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _518
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r11), %r15	 #, tmp802
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp803
	movsbl	(%rbx,%r9), %ebx	 # *_524, *_524
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r15,4), %ebx	 # *_521, tmp806
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp806, stmp_res_65.302
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	11(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L96	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _532
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r11), %r15	 #, tmp807
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp808
	movsbl	(%rbx,%r9), %ebx	 # *_538, *_538
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r15,4), %ebx	 # *_535, tmp811
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp811, stmp_res_65.302
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	12(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L96	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _546
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r11), %r15	 #, tmp812
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp813
	movsbl	(%rbx,%r9), %ebx	 # *_552, *_552
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r15,4), %ebx	 # *_549, tmp816
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp816, stmp_res_65.302
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	13(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L96	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _560
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	addl	$14, %r8d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r11), %r15	 #, tmp817
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp818
	movsbl	(%rbx,%r9), %ebx	 # *_566, *_566
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r15,4), %ebx	 # *_563, tmp821
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp821, stmp_res_65.302
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%r8d, %edx	 # i, k
	jle	.L96	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%r8d, %r8	 # i, _17
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%r8,%r11), %rbx	 #, tmp822
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %r8	 # weight, tmp823
	movsbl	(%r8,%r9), %r8d	 # *_11, *_11
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%rbx,4), %r8d	 # *_14, tmp826
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%r8d, %eax	 # tmp826, stmp_res_65.302
	.p2align 4,,10
	.p2align 3
.L96:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:50: 	return res >> 6; // 返回内积结果
	sarl	$6, %eax	 #, _135
.L93:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:90: 				*out = i32_i8_vec_in_prod(data + i * k,
	movl	%eax, (%rdi)	 # _135, MEM[(i32 *)out_87]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:89: 			for (int j = 0; j < n; j++, out++) // 遍历矩阵B的行
	addq	%r13, %r9	 # _614, ivtmp.347
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:89: 			for (int j = 0; j < n; j++, out++) // 遍历矩阵B的行
	addq	$4, %rdi	 #, out
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:89: 			for (int j = 0; j < n; j++, out++) // 遍历矩阵B的行
	addq	%r13, %r12	 # _614, ivtmp.348
	cmpq	56(%rsp), %rdi	 # %sfp, out
	jne	.L97	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:88: 		for (int i = 0; i < m; i++) {		 // 遍历矩阵A的行
	addl	$1, 36(%rsp)	 #, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:88: 		for (int i = 0; i < m; i++) {		 // 遍历矩阵A的行
	addq	%r13, %r11	 # _614, ivtmp.356
	movq	24(%rsp), %rsi	 # %sfp, _604
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:88: 		for (int i = 0; i < m; i++) {		 // 遍历矩阵A的行
	movl	36(%rsp), %eax	 # %sfp, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:88: 		for (int i = 0; i < m; i++) {		 // 遍历矩阵A的行
	addq	%rsi, 56(%rsp)	 # _604, %sfp
	movq	16(%rsp), %rsi	 # %sfp, _618
	addq	%rsi, 48(%rsp)	 # _618, %sfp
	cmpl	%eax, 40(%rbp)	 # i, m
	jne	.L86	 #,
.L83:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:95: }
	movaps	64(%rsp), %xmm6	 #,
	movaps	80(%rsp), %xmm7	 #,
	movaps	96(%rsp), %xmm8	 #,
	leaq	-56(%rbp), %rsp	 #,
	popq	%rbx	 #
	popq	%rsi	 #
	popq	%rdi	 #
	popq	%r12	 #
	popq	%r13	 #
	popq	%r14	 #
	popq	%r15	 #
	popq	%rbp	 #
	ret	
.L84:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:81: 		for (int j = 0; j < n; j++) { // 遍历矩阵B的行
	movl	56(%rbp), %r9d	 # n,
	testl	%r9d, %r9d	 #
	jle	.L83	 #,
	movl	40(%rbp), %eax	 # m,
	testl	%eax, %eax	 #
	jle	.L83	 #,
	movl	40(%rbp), %r9d	 # m, _186
	movslq	%edx, %r15	 # k, _461
	movl	%edx, %r13d	 # k, niters_vector_mult_vf.267
	movq	%rcx, 48(%rsp)	 # weight, %sfp
	andl	$-16, %r13d	 #, niters_vector_mult_vf.267
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:81: 		for (int j = 0; j < n; j++) { // 遍历矩阵B的行
	movl	$0, 24(%rsp)	 #, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	pxor	%xmm5, %xmm5	 # tmp839
	pxor	%xmm4, %xmm4	 # tmp840
	leaq	0(,%r9,4), %rax	 #, _391
	negq	%r9	 # tmp682
	movq	%rax, 16(%rsp)	 # _391, %sfp
	leaq	0(,%r9,4), %rdi	 #, tmp683
	addq	%rax, %r8	 # _391, out
	leaq	0(,%r15,4), %rax	 #, _321
	movq	%r8, 56(%rsp)	 # out, %sfp
	movq	%rax, 40(%rsp)	 # _321, %sfp
	leal	-1(%rdx), %eax	 #, _132
	movl	%eax, 36(%rsp)	 # _132, %sfp
	movl	%edx, %eax	 # k, bnd.266
	shrl	$4, %eax	 #, bnd.266
	movq	%rdi, 8(%rsp)	 # tmp683, %sfp
	subl	$1, %eax	 #, _250
	addq	$1, %rax	 #, tmp836
	salq	$4, %rax	 #, tmp836
	movq	%rax, (%rsp)	 # tmp836, %sfp
	.p2align 4,,10
	.p2align 3
.L87:
	movq	48(%rsp), %rax	 # %sfp, ivtmp.331
	movq	%r10, %r12	 # data, ivtmp.321
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:79: void i32_i8_mul(i32 *data, i8 *weight, i32 *out, int m, int k, int n, bool Transpose = false) {
	xorl	%r9d, %r9d	 # ivtmp.320
	movq	(%rsp), %rsi	 # %sfp, _278
	movq	8(%rsp), %rdi	 # %sfp, out
	addq	56(%rsp), %rdi	 # %sfp, out
	movq	%rax, %r11	 # ivtmp.331, _545
	subq	%rcx, %r11	 # weight, _545
	addq	%rax, %rsi	 # ivtmp.331, _278
	.p2align 4,,10
	.p2align 3
.L92:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	testl	%edx, %edx	 # k
	jle	.L98	 #,
	cmpl	$14, 36(%rsp)	 #, %sfp
	jbe	.L99	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	movq	48(%rsp), %r8	 # %sfp, ivtmp.310
	movq	%r12, %rax	 # ivtmp.321, ivtmp.313
	pxor	%xmm0, %xmm0	 # vect_res_101.269
	.p2align 4,,10
	.p2align 3
.L90:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm5, %xmm3	 # tmp839, tmp542
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqu	(%rax), %xmm7	 # MEM <vector(4) int> [(i32 *)_134], vect__46.272
	addq	$16, %r8	 #, ivtmp.310
	addq	$64, %rax	 #, ivtmp.313
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqu	-16(%r8), %xmm1	 # MEM <vector(16) signed char> [(i8 *)_236], MEM <vector(16) signed char> [(i8 *)_236]
	movdqa	%xmm4, %xmm8	 # tmp840, tmp551
	pcmpgtb	%xmm1, %xmm3	 # MEM <vector(16) signed char> [(i8 *)_236], tmp542
	movdqa	%xmm1, %xmm2	 # MEM <vector(16) signed char> [(i8 *)_236], tmp543
	punpcklbw	%xmm3, %xmm2	 # tmp542, tmp543
	pcmpgtw	%xmm2, %xmm8	 # tmp543, tmp551
	punpckhbw	%xmm3, %xmm1	 # tmp542, tmp547
	movdqa	%xmm2, %xmm3	 # tmp543, tmp552
	punpcklwd	%xmm8, %xmm3	 # tmp551, tmp552
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm3, %xmm6	 # tmp552, tmp557
	psrlq	$32, %xmm3	 #, tmp559
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	punpckhwd	%xmm8, %xmm2	 # tmp551, tmp565
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	pmuludq	%xmm7, %xmm6	 # vect__46.272, tmp557
	psrlq	$32, %xmm7	 #, tmp560
	pmuludq	%xmm7, %xmm3	 # tmp560, tmp558
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm4, %xmm7	 # tmp840, tmp577
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	pshufd	$8, %xmm6, %xmm6	 #, tmp557, tmp555
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	pcmpgtw	%xmm1, %xmm7	 # tmp547, tmp577
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	pshufd	$8, %xmm3, %xmm3	 #, tmp558, tmp556
	punpckldq	%xmm3, %xmm6	 # tmp556, vect__50.281
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	paddd	%xmm6, %xmm0	 # vect__50.281, vect_res_52.282
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqu	-48(%rax), %xmm6	 # MEM <vector(4) int> [(i32 *)_134 + 16B], vect__46.273
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm2, %xmm3	 # tmp565, tmp570
	psrlq	$32, %xmm2	 #, tmp572
	pmuludq	%xmm6, %xmm3	 # vect__46.273, tmp570
	psrlq	$32, %xmm6	 #, tmp573
	pmuludq	%xmm6, %xmm2	 # tmp573, tmp571
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqu	-32(%rax), %xmm6	 # MEM <vector(4) int> [(i32 *)_134 + 32B], vect__46.274
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	pshufd	$8, %xmm3, %xmm3	 #, tmp570, tmp568
	pshufd	$8, %xmm2, %xmm2	 #, tmp571, tmp569
	punpckldq	%xmm2, %xmm3	 # tmp569, vect__50.281
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm1, %xmm2	 # tmp547, tmp578
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	paddd	%xmm3, %xmm0	 # vect__50.281, vect_res_52.282
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	punpcklwd	%xmm7, %xmm2	 # tmp577, tmp578
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm2, %xmm3	 # tmp578, tmp583
	psrlq	$32, %xmm2	 #, tmp585
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	punpckhwd	%xmm7, %xmm1	 # tmp577, tmp591
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	pmuludq	%xmm6, %xmm3	 # vect__46.274, tmp583
	psrlq	$32, %xmm6	 #, tmp586
	pmuludq	%xmm6, %xmm2	 # tmp586, tmp584
	pshufd	$8, %xmm3, %xmm3	 #, tmp583, tmp581
	pshufd	$8, %xmm2, %xmm2	 #, tmp584, tmp582
	punpckldq	%xmm2, %xmm3	 # tmp582, vect__50.281
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	paddd	%xmm0, %xmm3	 # vect_res_52.282, vect__50.281
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm1, %xmm0	 # tmp591, tmp596
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqa	%xmm3, %xmm2	 # vect__50.281, vect_res_52.282
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movdqu	-16(%rax), %xmm3	 # MEM <vector(4) int> [(i32 *)_134 + 48B], vect__46.275
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	psrlq	$32, %xmm1	 #, tmp598
	cmpq	%rsi, %r8	 # _278, ivtmp.310
	pmuludq	%xmm3, %xmm0	 # vect__46.275, tmp596
	psrlq	$32, %xmm3	 #, tmp599
	pmuludq	%xmm3, %xmm1	 # tmp599, tmp597
	pshufd	$8, %xmm0, %xmm0	 #, tmp596, tmp594
	pshufd	$8, %xmm1, %xmm1	 #, tmp597, tmp595
	punpckldq	%xmm1, %xmm0	 # tmp595, vect__50.281
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	paddd	%xmm2, %xmm0	 # vect_res_52.282, vect_res_101.269
	jne	.L90	 #,
	movdqa	%xmm0, %xmm1	 # vect_res_101.269, tmp601
	cmpl	%r13d, %edx	 # niters_vector_mult_vf.267, k
	psrldq	$8, %xmm1	 #, tmp601
	paddd	%xmm1, %xmm0	 # tmp601, _64
	movdqa	%xmm0, %xmm1	 # _64, tmp603
	psrldq	$4, %xmm1	 #, tmp603
	paddd	%xmm1, %xmm0	 # tmp603, tmp604
	movd	%xmm0, %eax	 # tmp604, stmp_res_52.283
	je	.L91	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	movl	%r13d, %r8d	 # niters_vector_mult_vf.267, i
.L89:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%r8d, %rbx	 # i, _41
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r9), %r14	 #, tmp605
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp606
	movsbl	(%rbx,%r11), %ebx	 # *_47, *_47
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r14,4), %ebx	 # *_45, tmp609
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp609, stmp_res_52.283
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	1(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L91	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _57
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r9), %r14	 #, tmp610
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp611
	movsbl	(%rbx,%r11), %ebx	 # *_60, *_60
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r14,4), %ebx	 # *_58, tmp614
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp614, stmp_res_52.283
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	2(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L91	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _105
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r9), %r14	 #, tmp615
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp616
	movsbl	(%rbx,%r11), %ebx	 # *_216, *_216
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r14,4), %ebx	 # *_213, tmp619
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp619, stmp_res_52.283
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	3(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L91	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _224
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r9), %r14	 #, tmp620
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp621
	movsbl	(%rbx,%r11), %ebx	 # *_230, *_230
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r14,4), %ebx	 # *_227, tmp624
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp624, stmp_res_52.283
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	4(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L91	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _238
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r9), %r14	 #, tmp625
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp626
	movsbl	(%rbx,%r11), %ebx	 # *_244, *_244
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r14,4), %ebx	 # *_241, tmp629
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp629, stmp_res_52.283
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	5(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L91	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _252
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r9), %r14	 #, tmp630
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp631
	movsbl	(%rbx,%r11), %ebx	 # *_258, *_258
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r14,4), %ebx	 # *_255, tmp634
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp634, stmp_res_52.283
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	6(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L91	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _266
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%rbx,%r9), %r14	 #, tmp635
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp636
	movsbl	(%rbx,%r11), %ebx	 # *_272, *_272
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r14,4), %ebx	 # *_269, tmp639
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp639, stmp_res_52.283
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	7(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L91	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _280
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%r9,%rbx), %r14	 #, tmp640
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp641
	movsbl	(%rbx,%r11), %ebx	 # *_286, *_286
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r14,4), %ebx	 # *_283, tmp644
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp644, stmp_res_52.283
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	8(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L91	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _294
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%r9,%rbx), %r14	 #, tmp645
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp646
	movsbl	(%rbx,%r11), %ebx	 # *_300, *_300
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r14,4), %ebx	 # *_297, tmp649
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp649, stmp_res_52.283
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	9(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L91	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _308
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%r9,%rbx), %r14	 #, tmp650
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp651
	movsbl	(%rbx,%r11), %ebx	 # *_314, *_314
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r14,4), %ebx	 # *_311, tmp654
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp654, stmp_res_52.283
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	10(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L91	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _322
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%r9,%rbx), %r14	 #, tmp655
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp656
	movsbl	(%rbx,%r11), %ebx	 # *_328, *_328
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r14,4), %ebx	 # *_325, tmp659
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp659, stmp_res_52.283
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	11(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L91	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _336
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%r9,%rbx), %r14	 #, tmp660
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp661
	movsbl	(%rbx,%r11), %ebx	 # *_342, *_342
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r14,4), %ebx	 # *_339, tmp664
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp664, stmp_res_52.283
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	12(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L91	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _350
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%r9,%rbx), %r14	 #, tmp665
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp666
	movsbl	(%rbx,%r11), %ebx	 # *_356, *_356
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r14,4), %ebx	 # *_353, tmp669
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp669, stmp_res_52.283
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	leal	13(%r8), %ebx	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%ebx, %edx	 # i, k
	jle	.L91	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%ebx, %rbx	 # i, _364
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	addl	$14, %r8d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%r9,%rbx), %r14	 #, tmp670
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %rbx	 # weight, tmp671
	movsbl	(%rbx,%r11), %ebx	 # *_370, *_370
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%r14,4), %ebx	 # *_367, tmp674
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%ebx, %eax	 # tmp674, stmp_res_52.283
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	cmpl	%r8d, %edx	 # i, k
	jle	.L91	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	movslq	%r8d, %r8	 # i, _148
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	leaq	(%r8,%r9), %rbx	 #, tmp675
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addq	%rcx, %r8	 # weight, tmp676
	movsbl	(%r8,%r11), %r8d	 # *_142, *_142
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	imull	(%r10,%rbx,4), %r8d	 # *_145, tmp679
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:47: 		res += data[i] * weight[i];
	addl	%r8d, %eax	 # tmp679, stmp_res_52.283
	.p2align 4,,10
	.p2align 3
.L91:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:50: 	return res >> 6; // 返回内积结果
	sarl	$6, %eax	 #, _159
.L88:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:83: 				*out = i32_i8_vec_in_prod(data + i * k,
	movl	%eax, (%rdi)	 # _159, MEM[(i32 *)out_84]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:82: 			for (int i = 0; i < m; i++,  out++)  // 遍历矩阵A的行
	addq	40(%rsp), %r12	 # %sfp, ivtmp.321
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:82: 			for (int i = 0; i < m; i++,  out++)  // 遍历矩阵A的行
	addq	$4, %rdi	 #, out
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:82: 			for (int i = 0; i < m; i++,  out++)  // 遍历矩阵A的行
	addq	%r15, %r9	 # _461, ivtmp.320
	cmpq	56(%rsp), %rdi	 # %sfp, out
	jne	.L92	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:81: 		for (int j = 0; j < n; j++) { // 遍历矩阵B的行
	addl	$1, 24(%rsp)	 #, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:81: 		for (int j = 0; j < n; j++) { // 遍历矩阵B的行
	movq	16(%rsp), %rsi	 # %sfp, _391
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:81: 		for (int j = 0; j < n; j++) { // 遍历矩阵B的行
	movl	24(%rsp), %eax	 # %sfp, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:81: 		for (int j = 0; j < n; j++) { // 遍历矩阵B的行
	addq	%rsi, 56(%rsp)	 # _391, %sfp
	addq	%r15, 48(%rsp)	 # _461, %sfp
	cmpl	%eax, 56(%rbp)	 # j, n
	jne	.L87	 #,
	jmp	.L83	 #
	.p2align 4,,10
	.p2align 3
.L100:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	xorl	%eax, %eax	 # _135
	jmp	.L93	 #
	.p2align 4,,10
	.p2align 3
.L98:
	xorl	%eax, %eax	 # _159
	jmp	.L88	 #
.L101:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:45: 	i32 res = 0;  // 初始化结果为0, 结果要除以64,解析成(-1,1)的8字节数
	xorl	%eax, %eax	 # stmp_res_65.302
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	xorl	%r8d, %r8d	 # i
	jmp	.L94	 #
.L99:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:45: 	i32 res = 0;  // 初始化结果为0, 结果要除以64,解析成(-1,1)的8字节数
	xorl	%eax, %eax	 # stmp_res_52.283
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:46: 	for (int i = 0; i < n; i++) // 遍历向量元素
	xorl	%r8d, %r8d	 # i
	jmp	.L89	 #
	.p2align 4
	.globl	_Z10i32_b8_mulPiPhS_iiib
	.def	_Z10i32_b8_mulPiPhS_iiib;	.scl	2;	.type	32;	.endef
_Z10i32_b8_mulPiPhS_iiib:
	pushq	%r15	 #
	movq	%r8, %r10	 # tmp158, out
	pushq	%r14	 #
	pushq	%r13	 #
	pushq	%r12	 #
	pushq	%rbp	 #
	movq	%rdx, %rbp	 # tmp157, weight
	pushq	%rdi	 #
	pushq	%rsi	 #
	pushq	%rbx	 #
	subq	$72, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:100: 	if (Transpose) {
	cmpb	$0, 192(%rsp)	 #, Transpose
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:99: void i32_b8_mul(i32 *data, b8 *weight, i32 *out, int m, int k, int n, bool Transpose = false) {
	movq	%rcx, 144(%rsp)	 # tmp156, data
	movl	176(%rsp), %r14d	 # k, k
	movl	%r9d, 168(%rsp)	 # tmp159, m
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:100: 	if (Transpose) {
	jne	.L108	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:110: 		for (int i = 0; i < m; i++)
	testl	%r9d, %r9d	 #
	jle	.L107	 #,
	movl	184(%rsp), %r8d	 # n,
	testl	%r8d, %r8d	 #
	jle	.L107	 #,
	movslq	184(%rsp), %r13	 # n, n
	movq	%rcx, %rbx	 # tmp156, ivtmp.394
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:110: 		for (int i = 0; i < m; i++)
	xorl	%r15d, %r15d	 # i
	leaq	0(,%r13,4), %rax	 #, _106
	leaq	(%r10,%rax), %rdi	 #, out
	movq	%rax, 40(%rsp)	 # _106, %sfp
	movslq	%r14d, %rax	 # k, k
	salq	$2, %rax	 #, _117
	movq	%rdi, %rsi	 # out, ivtmp.391
	movq	%rax, 48(%rsp)	 # _117, %sfp
	.p2align 4,,10
	.p2align 3
.L110:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:112: 				*out = i32_b8_vec_in_prod(data + i * k,
	movq	%r10, %r12	 # out, out
	xorl	%r13d, %r13d	 # ivtmp.387
	.p2align 4,,10
	.p2align 3
.L113:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:112: 				*out = i32_b8_vec_in_prod(data + i * k,
	movl	%r13d, %eax	 # ivtmp.387, tmp143
	movl	%r14d, %r8d	 # k,
	movq	%rbx, %rcx	 # ivtmp.394,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:113: 				                          weight + (j * k) / 8,
	leal	7(%r13), %edx	 #, tmp150
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:112: 				*out = i32_b8_vec_in_prod(data + i * k,
	sarl	$31, %eax	 #, tmp143
	shrl	$29, %eax	 #, tmp144
	leal	0(%r13,%rax), %r9d	 #, tmp145
	andl	$7, %r9d	 #, tmp146
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:113: 				                          weight + (j * k) / 8,
	testl	%r13d, %r13d	 # ivtmp.387
	cmovns	%r13d, %edx	 # tmp150,, ivtmp.387, _12
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:112: 				*out = i32_b8_vec_in_prod(data + i * k,
	subl	%eax, %r9d	 # tmp144,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:111: 			for (int j = 0; j < n; j++,  out++)
	addq	$4, %r12	 #, out
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:111: 			for (int j = 0; j < n; j++,  out++)
	addl	%r14d, %r13d	 # k, ivtmp.387
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:113: 				                          weight + (j * k) / 8,
	sarl	$3, %edx	 #, tmp151
	movslq	%edx, %rdx	 # tmp151, tmp152
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:112: 				*out = i32_b8_vec_in_prod(data + i * k,
	addq	%rbp, %rdx	 # weight, tmp153
	call	_Z18i32_b8_vec_in_prodPiPhii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:112: 				*out = i32_b8_vec_in_prod(data + i * k,
	movl	%eax, -4(%r12)	 # tmp161, MEM[(i32 *)out_65]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:111: 			for (int j = 0; j < n; j++,  out++)
	cmpq	%rsi, %r12	 # ivtmp.391, out
	jne	.L113	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:110: 		for (int i = 0; i < m; i++)
	movq	40(%rsp), %rax	 # %sfp, _106
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:110: 		for (int i = 0; i < m; i++)
	addl	$1, %r15d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:111: 			for (int j = 0; j < n; j++,  out++)
	movq	%rdi, %r10	 # out, out
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:110: 		for (int i = 0; i < m; i++)
	addq	48(%rsp), %rbx	 # %sfp, ivtmp.394
	addq	%rax, %rsi	 # _106, ivtmp.391
	cmpl	%r15d, 168(%rsp)	 # i, m
	je	.L107	 #,
	addq	%rax, %rdi	 # _106, out
	jmp	.L110	 #
	.p2align 4,,10
	.p2align 3
.L108:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:101: 		for (int j = 0; j < n; j++) {
	movl	184(%rsp), %edx	 # n,
	testl	%edx, %edx	 #
	jg	.L117	 #,
.L107:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:117: }
	addq	$72, %rsp	 #,
	popq	%rbx	 #
	popq	%rsi	 #
	popq	%rdi	 #
	popq	%rbp	 #
	popq	%r12	 #
	popq	%r13	 #
	popq	%r14	 #
	popq	%r15	 #
	ret	
	.p2align 4,,10
	.p2align 3
.L117:
	movl	168(%rsp), %eax	 # m,
	testl	%eax, %eax	 #
	jle	.L107	 #,
	movslq	168(%rsp), %r12	 # m, m
	movq	%rbp, 152(%rsp)	 # weight, weight
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:101: 		for (int j = 0; j < n; j++) {
	movl	$0, 40(%rsp)	 #, %sfp
	leaq	0(,%r12,4), %rax	 #, _24
	movslq	%r14d, %r12	 # k, k
	movq	%rax, 56(%rsp)	 # _24, %sfp
	addq	%r8, %rax	 # out, out
	salq	$2, %r12	 #, _55
	movq	%rax, 48(%rsp)	 # out, %sfp
	movq	%rax, %rdi	 # out, ivtmp.376
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:101: 		for (int j = 0; j < n; j++) {
	xorl	%eax, %eax	 # ivtmp.379
	movl	%eax, %ebp	 # ivtmp.379, ivtmp.379
	.p2align 4,,10
	.p2align 3
.L111:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:104: 				                          weight + (j * k) / 8,
	leal	7(%rbp), %ebx	 #, tmp139
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:103: 				*out = i32_b8_vec_in_prod(data + i * k,
	movl	%ebp, %eax	 # ivtmp.379, tmp132
	movq	144(%rsp), %r15	 # data, ivtmp.372
	movq	%r10, %r13	 # out, out
	sarl	$31, %eax	 #, tmp132
	shrl	$29, %eax	 #, tmp133
	leal	0(%rbp,%rax), %esi	 #, tmp134
	andl	$7, %esi	 #, tmp135
	subl	%eax, %esi	 # tmp133, tmp136
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:104: 				                          weight + (j * k) / 8,
	testl	%ebp, %ebp	 # ivtmp.379
	cmovns	%ebp, %ebx	 # tmp139,, ivtmp.379, _1
	sarl	$3, %ebx	 #, tmp140
	movslq	%ebx, %rbx	 # tmp140, tmp141
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:103: 				*out = i32_b8_vec_in_prod(data + i * k,
	addq	152(%rsp), %rbx	 # weight, _6
	.p2align 4,,10
	.p2align 3
.L112:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:103: 				*out = i32_b8_vec_in_prod(data + i * k,
	movq	%r15, %rcx	 # ivtmp.372,
	movl	%esi, %r9d	 # tmp136,
	movl	%r14d, %r8d	 # k,
	movq	%rbx, %rdx	 # _6,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:102: 			for (int i = 0; i < m; i++,  out++) 
	addq	$4, %r13	 #, out
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:103: 				*out = i32_b8_vec_in_prod(data + i * k,
	call	_Z18i32_b8_vec_in_prodPiPhii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:102: 			for (int i = 0; i < m; i++,  out++) 
	addq	%r12, %r15	 # _55, ivtmp.372
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:103: 				*out = i32_b8_vec_in_prod(data + i * k,
	movl	%eax, -4(%r13)	 # tmp160, MEM[(i32 *)out_62]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:102: 			for (int i = 0; i < m; i++,  out++) 
	cmpq	%rdi, %r13	 # ivtmp.376, out
	jne	.L112	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:101: 		for (int j = 0; j < n; j++) {
	movq	56(%rsp), %rsi	 # %sfp, _24
	addl	%r14d, %ebp	 # k, ivtmp.379
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:101: 		for (int j = 0; j < n; j++) {
	addl	$1, 40(%rsp)	 #, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:102: 			for (int i = 0; i < m; i++,  out++) 
	movq	48(%rsp), %rcx	 # %sfp, out
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:101: 		for (int j = 0; j < n; j++) {
	movl	40(%rsp), %eax	 # %sfp, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:101: 		for (int j = 0; j < n; j++) {
	addq	%rsi, %rdi	 # _24, ivtmp.376
	cmpl	%eax, 184(%rsp)	 # j, n
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:102: 			for (int i = 0; i < m; i++,  out++) 
	movq	%rcx, %r10	 # out, out
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/opt.h:101: 		for (int j = 0; j < n; j++) {
	je	.L107	 #,
	leaq	(%rcx,%rsi), %rax	 #, out
	movq	%rax, 48(%rsp)	 # out, %sfp
	jmp	.L111	 #
	.p2align 4
	.globl	_Z9avgpool1dPiS_iiii
	.def	_Z9avgpool1dPiS_iiii;	.scl	2;	.type	32;	.endef
_Z9avgpool1dPiS_iiii:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:4: 	for(int c=0; c<channels; c++){
	testl	%r9d, %r9d	 # channels
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:3: void avgpool1d(i32 *data, i32 *out, int length, int channels, int kernel, int stride){
	pushq	%r15	 #
	pushq	%r14	 #
	pushq	%r13	 #
	pushq	%r12	 #
	movq	%rcx, %r12	 # tmp130, data
	pushq	%rbp	 #
	pushq	%rdi	 #
	pushq	%rsi	 #
	pushq	%rbx	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:3: void avgpool1d(i32 *data, i32 *out, int length, int channels, int kernel, int stride){
	movl	104(%rsp), %ecx	 # kernel, kernel
	movl	112(%rsp), %r10d	 # stride, stride
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:4: 	for(int c=0; c<channels; c++){
	jle	.L118	 #,
	testl	%r8d, %r8d	 # length
	movl	%r8d, %r13d	 # tmp132, length
	jle	.L118	 #,
	movl	%r9d, %edi	 # tmp133, channels
	movslq	%r9d, %r14	 # channels, _51
	movq	%rdx, %rbp	 # tmp131, out
	imull	%r10d, %edi	 # stride, _57
	leaq	0(,%r14,4), %r8	 #, _70
	xorl	%ebx, %ebx	 # ivtmp.413
	.p2align 4,,10
	.p2align 3
.L120:
	movl	%ebx, %esi	 # ivtmp.413, c
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:7: 			for(int k=0;k<kernel;k++)
	xorl	%r11d, %r11d	 # ivtmp.411
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:5: 		for(int l=0; l<length; l+=stride){
	xorl	%r9d, %r9d	 # l
	.p2align 4,,10
	.p2align 3
.L125:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:7: 			for(int k=0;k<kernel;k++)
	xorl	%r15d, %r15d	 # _71
	testl	%ecx, %ecx	 # kernel
	jle	.L124	 #,
	movslq	%r11d, %rax	 # ivtmp.411, ivtmp.411
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:7: 			for(int k=0;k<kernel;k++)
	xorl	%edx, %edx	 # k
	addq	%rbx, %rax	 # ivtmp.413, tmp116
	leaq	(%r12,%rax,4), %r15	 #, ivtmp.405
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:6: 			i32 avg=0;
	xorl	%eax, %eax	 # avg
	.p2align 4,,10
	.p2align 3
.L121:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:7: 			for(int k=0;k<kernel;k++)
	addl	$1, %edx	 #, k
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:8: 				avg += data[(l+k)*channels + c];
	addl	(%r15), %eax	 # MEM[(i32 *)_62], avg
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:7: 			for(int k=0;k<kernel;k++)
	addq	%r8, %r15	 # _70, ivtmp.405
	cmpl	%edx, %ecx	 # k, kernel
	jne	.L121	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:9: 			out[l/stride + c] = avg / kernel; 
	cltd
	idivl	%ecx	 # kernel
	movl	%eax, %r15d	 # avg, _71
.L124:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:9: 			out[l/stride + c] = avg / kernel; 
	movl	%r9d, %eax	 # l, tmp124
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:5: 		for(int l=0; l<length; l+=stride){
	addl	%r10d, %r9d	 # stride, l
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:5: 		for(int l=0; l<length; l+=stride){
	addl	%edi, %r11d	 # _57, ivtmp.411
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:9: 			out[l/stride + c] = avg / kernel; 
	cltd
	idivl	%r10d	 # stride
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:9: 			out[l/stride + c] = avg / kernel; 
	addl	%esi, %eax	 # c, tmp126
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:5: 		for(int l=0; l<length; l+=stride){
	cmpl	%r9d, %r13d	 # l, length
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:9: 			out[l/stride + c] = avg / kernel; 
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:9: 			out[l/stride + c] = avg / kernel; 
	movl	%r15d, 0(%rbp,%rax,4)	 # _71, *_12
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:5: 		for(int l=0; l<length; l+=stride){
	jg	.L125	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:4: 	for(int c=0; c<channels; c++){
	addq	$1, %rbx	 #, ivtmp.413
	cmpq	%rbx, %r14	 # ivtmp.413, _51
	jne	.L120	 #,
.L118:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/layer.h:12: }
	popq	%rbx	 #
	popq	%rsi	 #
	popq	%rdi	 #
	popq	%rbp	 #
	popq	%r12	 #
	popq	%r13	 #
	popq	%r14	 #
	popq	%r15	 #
	ret	
	.p2align 4
	.globl	_Z14test_avgpool1dv
	.def	_Z14test_avgpool1dv;	.scl	2;	.type	32;	.endef
_Z14test_avgpool1dv:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test_layer.cpp:5: }
	ret	
	.def	__main;	.scl	2;	.type	32;	.endef
	.section	.text.startup,"x"
	.p2align 4
	.globl	main
	.def	main;	.scl	2;	.type	32;	.endef
main:
	subq	$40, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test_layer.cpp:7: int main(){
	call	__main	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/Q8_6/test_layer.cpp:9: }
	xorl	%eax, %eax	 #
	addq	$40, %rsp	 #,
	ret	
	.section .rdata,"dr"
	.align 8
.LC5:
	.long	0
	.long	1066401792
	.ident	"GCC: (x86_64-posix-seh, Built by MinGW-Builds project) 11.4.0"
	.def	__mingw_vfprintf;	.scl	2;	.type	32;	.endef
