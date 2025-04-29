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
	.ascii "%d\11\0"
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
	.section .rdata,"dr"
.LC3:
	.ascii "%.2f \11\0"
	.text
	.p2align 4
	.def	_Z6printfPKcz.constprop.3;	.scl	3;	.type	32;	.endef
_Z6printfPKcz.constprop.3:
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
	leaq	.LC3(%rip), %rdx	 #, tmp88
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
.LC4:
	.ascii "\12=> Mat [%d, %d, f32]: \12\0"
	.text
	.p2align 4
	.globl	_Z8printmatPfii
	.def	_Z8printmatPfii;	.scl	2;	.type	32;	.endef
_Z8printmatPfii:
	pushq	%r15	 #
	pushq	%r14	 #
	pushq	%r13	 #
	movq	%rcx, %r13	 # tmp123, a
	pushq	%r12	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:70: 	printf("\n=> Mat [%d, %d, f32]: \n", m, n);  // 打印矩阵维度
	leaq	.LC4(%rip), %rcx	 #, tmp103
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:69: void printmat(f32 *a, int m, int n) {
	movl	%r8d, %r12d	 # tmp125, n
	pushq	%rbp	 #
	pushq	%rdi	 #
	pushq	%rsi	 #
	pushq	%rbx	 #
	subq	$56, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:69: void printmat(f32 *a, int m, int n) {
	movl	%edx, 136(%rsp)	 # tmp124, m
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:70: 	printf("\n=> Mat [%d, %d, f32]: \n", m, n);  // 打印矩阵维度
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:71: 	for (int i = 0; i < m; i++) {
	movl	136(%rsp), %eax	 # m,
	testl	%eax, %eax	 #
	jle	.L12	 #,
	leal	-1(%r12), %eax	 #, tmp121
	xorl	%ebp, %ebp	 # ivtmp.55
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:71: 	for (int i = 0; i < m; i++) {
	xorl	%edi, %edi	 # i
	leaq	.LC0(%rip), %r15	 #, tmp118
	movq	%rax, 40(%rsp)	 # tmp121, %sfp
	leaq	.LC3(%rip), %rsi	 #, tmp122
	.p2align 4,,10
	.p2align 3
.L14:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:72: 		for (int j = 0; j < n; j++) {
	testl	%r12d, %r12d	 # n
	jle	.L17	 #,
	movslq	%ebp, %rax	 # ivtmp.55, _39
	leaq	0(%r13,%rax,4), %r14	 #, ivtmp.49
	addq	40(%rsp), %rax	 # %sfp, tmp107
	leaq	4(%r13,%rax,4), %rbx	 #, _21
	.p2align 4,,10
	.p2align 3
.L15:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:73: 			printf("%.2f \t", a[i * n + j]);  // 打印矩阵元素，保留两位小数
	movq	%rsi, %rcx	 # tmp122,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:72: 		for (int j = 0; j < n; j++) {
	addq	$4, %r14	 #, ivtmp.49
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:73: 			printf("%.2f \t", a[i * n + j]);  // 打印矩阵元素，保留两位小数
	pxor	%xmm1, %xmm1	 # tmp110
	cvtss2sd	-4(%r14), %xmm1	 # MEM[(f32 *)_36], tmp110
	movq	%xmm1, %rdx	 # tmp110,
	call	_Z6printfPKcz.constprop.3	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:72: 		for (int j = 0; j < n; j++) {
	cmpq	%r14, %rbx	 # ivtmp.49, _21
	jne	.L15	 #,
.L17:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:75: 		printf("\n");
	movq	%r15, %rcx	 # tmp118,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:71: 	for (int i = 0; i < m; i++) {
	addl	$1, %edi	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:71: 	for (int i = 0; i < m; i++) {
	addl	%r12d, %ebp	 # n, ivtmp.55
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:75: 		printf("\n");
	call	_Z6printfPKcz.constprop.0	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:71: 	for (int i = 0; i < m; i++) {
	cmpl	%edi, 136(%rsp)	 # i, m
	jne	.L14	 #,
.L12:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:77: }
	addq	$56, %rsp	 #,
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
.LC5:
	.ascii "\12=> Mat [%d, %d, u8_1]: \12\0"
	.text
	.p2align 4
	.globl	_Z8printmatPhii
	.def	_Z8printmatPhii;	.scl	2;	.type	32;	.endef
_Z8printmatPhii:
	pushq	%r15	 #
	xorl	%r15d, %r15d	 # k
	pushq	%r14	 #
	leaq	.LC0(%rip), %r14	 #, tmp112
	pushq	%r13	 #
	movl	%edx, %r13d	 # tmp115, m
	pushq	%r12	 #
	movl	%r8d, %r12d	 # tmp116, n
	pushq	%rbp	 #
	xorl	%ebp, %ebp	 # i
	pushq	%rdi	 #
	leaq	.LC2(%rip), %rdi	 #, tmp113
	pushq	%rsi	 #
	movq	%rcx, %rsi	 # tmp114, a
	pushq	%rbx	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:80: 	printf("\n=> Mat [%d, %d, u8_1]: \n", m, n);  // 打印矩阵维度
	leaq	.LC5(%rip), %rcx	 #, tmp101
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:79: void printmat(b8 *a, int m, int n) {
	subq	$40, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:80: 	printf("\n=> Mat [%d, %d, u8_1]: \n", m, n);  // 打印矩阵维度
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:82: 	for (i = 0; i < m; i++) {
	testl	%r13d, %r13d	 # m
	jle	.L20	 #,
	.p2align 4,,10
	.p2align 3
.L21:
	leal	(%r12,%r15), %ebx	 #, _58
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:83: 		for (j = 0; j < n; j++, k++) {
	testl	%r12d, %r12d	 # n
	jle	.L25	 #,
	.p2align 4,,10
	.p2align 3
.L23:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:84: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0); // 打印矩阵元素，保留两位小数
	movl	%r15d, %eax	 # k, tmp102
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:84: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0); // 打印矩阵元素，保留两位小数
	movl	%r15d, %ecx	 # k, tmp105
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:83: 		for (j = 0; j < n; j++, k++) {
	addl	$1, %r15d	 #, k
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:84: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0); // 打印矩阵元素，保留两位小数
	sarl	$3, %eax	 #, tmp102
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:84: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0); // 打印矩阵元素，保留两位小数
	andl	$7, %ecx	 #, tmp105
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:84: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0); // 打印矩阵元素，保留两位小数
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:84: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0); // 打印矩阵元素，保留两位小数
	movzbl	(%rsi,%rax), %edx	 # *_3, *_3
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:84: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0); // 打印矩阵元素，保留两位小数
	sall	%cl, %edx	 # tmp105, tmp106
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:84: 			printf("%d \t", (a[k / 8] << (k % 8) & 128) > 0); // 打印矩阵元素，保留两位小数
	movq	%rdi, %rcx	 # tmp113,
	shrb	$7, %dl	 #, tmp109
	movzbl	%dl, %edx	 # tmp109, tmp108
	call	_Z6printfPKcz.constprop.2	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:83: 		for (j = 0; j < n; j++, k++) {
	cmpl	%ebx, %r15d	 # _58, k
	jne	.L23	 #,
.L25:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:86: 		printf("\n");
	movq	%r14, %rcx	 # tmp112,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:82: 	for (i = 0; i < m; i++) {
	addl	$1, %ebp	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:86: 		printf("\n");
	call	_Z6printfPKcz.constprop.0	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:82: 	for (i = 0; i < m; i++) {
	cmpl	%ebp, %r13d	 # i, m
	jne	.L21	 #,
.L20:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:88: }
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
.LC6:
	.ascii "\12=> Mat [%d, %d, i32]: \12\0"
	.text
	.p2align 4
	.globl	_Z8printmatPiii
	.def	_Z8printmatPiii;	.scl	2;	.type	32;	.endef
_Z8printmatPiii:
	pushq	%r15	 #
	pushq	%r14	 #
	pushq	%r13	 #
	movq	%rcx, %r13	 # tmp116, a
	pushq	%r12	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:91: 	printf("\n=> Mat [%d, %d, i32]: \n", m, n);  // 打印矩阵维度
	leaq	.LC6(%rip), %rcx	 #, tmp102
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:90: void printmat(i32 *a, int m, int n) {
	movl	%r8d, %r12d	 # tmp118, n
	pushq	%rbp	 #
	pushq	%rdi	 #
	pushq	%rsi	 #
	pushq	%rbx	 #
	subq	$56, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:90: void printmat(i32 *a, int m, int n) {
	movl	%edx, 136(%rsp)	 # tmp117, m
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:91: 	printf("\n=> Mat [%d, %d, i32]: \n", m, n);  // 打印矩阵维度
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:93: 	for (i = 0; i < m; i++) {
	movl	136(%rsp), %eax	 # m,
	testl	%eax, %eax	 #
	jle	.L32	 #,
	leal	-1(%r12), %eax	 #, tmp114
	xorl	%ebp, %ebp	 # ivtmp.82
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:93: 	for (i = 0; i < m; i++) {
	xorl	%edi, %edi	 # i
	leaq	.LC0(%rip), %r15	 #, tmp111
	movq	%rax, 40(%rsp)	 # tmp114, %sfp
	leaq	.LC1(%rip), %rsi	 #, tmp115
	.p2align 4,,10
	.p2align 3
.L34:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:94: 		for (j = 0; j < n; j++, k++) {
	testl	%r12d, %r12d	 # n
	jle	.L37	 #,
	movslq	%ebp, %rax	 # ivtmp.82, _35
	leaq	0(%r13,%rax,4), %r14	 #, ivtmp.77
	addq	40(%rsp), %rax	 # %sfp, tmp106
	leaq	4(%r13,%rax,4), %rbx	 #, _8
	.p2align 4,,10
	.p2align 3
.L35:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:95: 			printf("%d\t", a[i*n+j]); // 打印矩阵元素，保留两位小数
	movl	(%r14), %edx	 # MEM[(i32 *)_32],
	movq	%rsi, %rcx	 # tmp115,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:94: 		for (j = 0; j < n; j++, k++) {
	addq	$4, %r14	 #, ivtmp.77
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:95: 			printf("%d\t", a[i*n+j]); // 打印矩阵元素，保留两位小数
	call	_Z6printfPKcz.constprop.1	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:94: 		for (j = 0; j < n; j++, k++) {
	cmpq	%r14, %rbx	 # ivtmp.77, _8
	jne	.L35	 #,
.L37:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:97: 		printf("\n");
	movq	%r15, %rcx	 # tmp111,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:93: 	for (i = 0; i < m; i++) {
	addl	$1, %edi	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:93: 	for (i = 0; i < m; i++) {
	addl	%r12d, %ebp	 # n, ivtmp.82
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:97: 		printf("\n");
	call	_Z6printfPKcz.constprop.0	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:93: 	for (i = 0; i < m; i++) {
	cmpl	%edi, 136(%rsp)	 # i, m
	jne	.L34	 #,
.L32:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:99: }
	addq	$56, %rsp	 #,
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
	.globl	_Z19f32_f32_vec_in_prodPfS_i
	.def	_Z19f32_f32_vec_in_prodPfS_i;	.scl	2;	.type	32;	.endef
_Z19f32_f32_vec_in_prodPfS_i:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	andq	$-16, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:105: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	testl	%r8d, %r8d	 # n
	jle	.L48	 #,
	leal	-1(%r8), %eax	 #, tmp126
	cmpl	$2, %eax	 #, tmp126
	jbe	.L49	 #,
	movl	%r8d, %r9d	 # n, bnd.88
	xorl	%eax, %eax	 # ivtmp.119
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104: 	f32 res = 0;  // 初始化结果为0
	pxor	%xmm0, %xmm0	 # <retval>
	shrl	$2, %r9d	 #,
	salq	$4, %r9	 #, _81
	.p2align 4,,10
	.p2align 3
.L43:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movq	(%rcx,%rax), %xmm1	 # MEM <vector(4) float> [(f32 *)data_8(D) + ivtmp.119_31 * 1], vect__1.95
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movq	(%rdx,%rax), %xmm2	 # MEM <vector(4) float> [(f32 *)weight_9(D) + ivtmp.119_31 * 1], vect__2.98
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movhps	8(%rcx,%rax), %xmm1	 # MEM <vector(4) float> [(f32 *)data_8(D) + ivtmp.119_31 * 1], vect__1.95
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movhps	8(%rdx,%rax), %xmm2	 # MEM <vector(4) float> [(f32 *)weight_9(D) + ivtmp.119_31 * 1], vect__2.98
	addq	$16, %rax	 #, ivtmp.119
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	mulps	%xmm2, %xmm1	 # vect__2.98, vect__3.99
	cmpq	%r9, %rax	 # _81, ivtmp.119
	addss	%xmm1, %xmm0	 # stmp_res_13.100, stmp_res_13.100
	movaps	%xmm1, %xmm2	 # vect__3.99, tmp131
	shufps	$85, %xmm1, %xmm2	 #, vect__3.99, tmp131
	addss	%xmm2, %xmm0	 # stmp_res_13.100, stmp_res_13.100
	movaps	%xmm1, %xmm2	 # vect__3.99, tmp132
	unpckhps	%xmm1, %xmm2	 # vect__3.99, tmp132
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	shufps	$255, %xmm1, %xmm1	 #, vect__3.99, tmp135
	addss	%xmm2, %xmm0	 # stmp_res_13.100, stmp_res_13.100
	addss	%xmm1, %xmm0	 # stmp_res_13.100, <retval>
	jne	.L43	 #,
	movl	%r8d, %r9d	 # n, niters_vector_mult_vf.89
	andl	$-4, %r9d	 #,
	movl	%r9d, %eax	 # niters_vector_mult_vf.89, niters_vector_mult_vf.89
	salq	$2, %rax	 #, _41
	leaq	(%rcx,%rax), %r10	 #, tmp.104
	addq	%rdx, %rax	 # weight, tmp.105
	cmpl	%r9d, %r8d	 # niters_vector_mult_vf.89, n
	je	.L40	 #,
.L42:
	subl	%r9d, %r8d	 # niters_vector_mult_vf.89, niters.101
	cmpl	$1, %r8d	 #, niters.101
	je	.L46	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movq	(%rcx,%r9,4), %xmm1	 # MEM <vector(2) float> [(f32 *)vectp_data.108_98], vect__6.109
	movl	%r8d, %ecx	 # niters.101, niters_vector_mult_vf.103
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movq	(%rdx,%r9,4), %xmm2	 # MEM <vector(2) float> [(f32 *)vectp_weight.111_104], vect__5.112
	andl	$-2, %ecx	 #, niters_vector_mult_vf.103
	movl	%ecx, %edx	 # niters_vector_mult_vf.103, niters_vector_mult_vf.103
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	mulps	%xmm2, %xmm1	 # vect__5.112, vect__4.113
	salq	$2, %rdx	 #, _91
	addq	%rdx, %r10	 # _91, tmp.104
	addq	%rdx, %rax	 # _91, tmp.105
	cmpl	%ecx, %r8d	 # niters_vector_mult_vf.103, niters.101
	addss	%xmm1, %xmm0	 # stmp_res_26.114, stmp_res_26.114
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movshdup	%xmm1, %xmm1	 # vect__4.113, stmp_res_26.114
	addss	%xmm1, %xmm0	 # stmp_res_26.114, <retval>
	je	.L40	 #,
.L46:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movss	(%r10), %xmm1	 # *data_83, *data_83
	mulss	(%rax), %xmm1	 # *weight_84, tmp142
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	addss	%xmm1, %xmm0	 # tmp142, <retval>
.L40:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:108: }
	leave	
	ret	
	.p2align 4,,10
	.p2align 3
.L48:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104: 	f32 res = 0;  // 初始化结果为0
	pxor	%xmm0, %xmm0	 # <retval>
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:108: }
	leave	
	ret	
.L49:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:105: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	movq	%rdx, %rax	 # weight, tmp.105
	movq	%rcx, %r10	 # data, tmp.104
	xorl	%r9d, %r9d	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104: 	f32 res = 0;  // 初始化结果为0
	pxor	%xmm0, %xmm0	 # <retval>
	jmp	.L42	 #
	.p2align 4
	.globl	_Z15f32_f32_mat_mulPfS_S_iiib
	.def	_Z15f32_f32_mat_mulPfS_S_iiib;	.scl	2;	.type	32;	.endef
_Z15f32_f32_mat_mulPfS_S_iiib:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	pushq	%r15	 #
	pushq	%r14	 #
	pushq	%r13	 #
	pushq	%r12	 #
	pushq	%rdi	 #
	pushq	%rsi	 #
	pushq	%rbx	 #
	andq	$-16, %rsp	 #,
	subq	$32, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:112: 	if (Transpose) {  // 如果需要转置输出
	cmpb	$0, 64(%rbp)	 #, Transpose
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:111: void f32_f32_mat_mul(f32 *data, f32 *weight, f32 *out, int m, int n, int k, bool Transpose) {
	movl	%r9d, 40(%rbp)	 # tmp272, m
	movl	56(%rbp), %r11d	 # k, k
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:112: 	if (Transpose) {  // 如果需要转置输出
	jne	.L56	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:119: 		for (int i = 0; i < m; i++, data += k) {		 // 遍历矩阵A的行
	testl	%r9d, %r9d	 #
	jle	.L55	 #,
	movl	48(%rbp), %r10d	 # n,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:121: 			for (int j = 0; j < n; j++, p += k, out++) // 遍历矩阵B的行
	movslq	%r11d, %r14	 # k, k
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:121: 			for (int j = 0; j < n; j++, p += k, out++) // 遍历矩阵B的行
	salq	$2, %r14	 #, _3
	testl	%r10d, %r10d	 #
	jle	.L55	 #,
	movl	48(%rbp), %eax	 # n, _54
	movl	%r11d, %r15d	 # k, niters_vector_mult_vf.163
	movq	%rdx, 24(%rbp)	 # weight, weight
	pxor	%xmm3, %xmm3	 # res
	andl	$-4, %r15d	 #, niters_vector_mult_vf.163
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:119: 		for (int i = 0; i < m; i++, data += k) {		 // 遍历矩阵A的行
	movl	$0, 24(%rsp)	 #, %sfp
	leaq	0(,%rax,4), %rdi	 #, _103
	negq	%rax	 # tmp256
	leaq	(%r8,%rdi), %r13	 #, ivtmp.240
	movl	%r11d, %r8d	 # k, bnd.162
	movq	%rdi, 8(%rsp)	 # _103, %sfp
	salq	$2, %rax	 #, tmp257
	leal	-1(%r11), %edi	 #, _195
	shrl	$2, %r8d	 #, bnd.162
	movq	%rax, (%rsp)	 # tmp257, %sfp
	leal	-1(%r8), %ebx	 #, tmp212
	movl	%r15d, %r8d	 # niters_vector_mult_vf.163, niters_vector_mult_vf.163
	movl	%edi, 28(%rsp)	 # _195, %sfp
	leaq	0(,%r8,4), %r12	 #, _204
	addq	$1, %rbx	 #, tmp213
	salq	$4, %rbx	 #, _233
	.p2align 4,,10
	.p2align 3
.L59:
	movq	(%rsp), %rax	 # %sfp, tmp257
	leaq	(%rax,%r13), %r10	 #, out
	leaq	(%rcx,%r12), %rax	 #, tmp.164
	movq	%rax, 16(%rsp)	 # tmp.164, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:120: 			f32 *p = weight;
	movq	24(%rbp), %rax	 # weight, p
	.p2align 4,,10
	.p2align 3
.L79:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:105: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	testl	%r11d, %r11d	 # k
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104: 	f32 res = 0;  // 初始化结果为0
	movaps	%xmm3, %xmm1	 # res, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:105: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	jle	.L78	 #,
	cmpl	$2, 28(%rsp)	 #, %sfp
	jbe	.L81	 #,
	xorl	%edx, %edx	 # ivtmp.221
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104: 	f32 res = 0;  // 初始化结果为0
	movaps	%xmm3, %xmm1	 # res, res
	.p2align 4,,10
	.p2align 3
.L72:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movq	(%rcx,%rdx), %xmm0	 # MEM <vector(4) float> [(f32 *)data_92 + ivtmp.221_279 * 1], vect__34.169
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movq	(%rax,%rdx), %xmm2	 # MEM <vector(4) float> [(f32 *)p_97 + ivtmp.221_279 * 1], vect__36.172
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movhps	8(%rcx,%rdx), %xmm0	 # MEM <vector(4) float> [(f32 *)data_92 + ivtmp.221_279 * 1], vect__34.169
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movhps	8(%rax,%rdx), %xmm2	 # MEM <vector(4) float> [(f32 *)p_97 + ivtmp.221_279 * 1], vect__36.172
	addq	$16, %rdx	 #, ivtmp.221
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	mulps	%xmm2, %xmm0	 # vect__36.172, vect__37.173
	cmpq	%rdx, %rbx	 # ivtmp.221, _233
	addss	%xmm0, %xmm1	 # stmp_res_39.174, stmp_res_39.174
	movaps	%xmm0, %xmm2	 # vect__37.173, tmp242
	shufps	$85, %xmm0, %xmm2	 #, vect__37.173, tmp242
	addss	%xmm2, %xmm1	 # stmp_res_39.174, stmp_res_39.174
	movaps	%xmm0, %xmm2	 # vect__37.173, tmp243
	unpckhps	%xmm0, %xmm2	 # vect__37.173, tmp243
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	shufps	$255, %xmm0, %xmm0	 #, vect__37.173, tmp246
	addss	%xmm2, %xmm1	 # stmp_res_39.174, stmp_res_39.174
	addss	%xmm0, %xmm1	 # stmp_res_39.174, res
	jne	.L72	 #,
	leaq	(%rax,%r12), %rdx	 #, tmp.179
	cmpl	%r15d, %r11d	 # niters_vector_mult_vf.163, k
	je	.L78	 #,
	movq	16(%rsp), %r8	 # %sfp, tmp.178
	movl	%r15d, %r9d	 # niters_vector_mult_vf.163,
.L71:
	movl	%r11d, %esi	 # k, niters.175
	subl	%r9d, %esi	 # _210, niters.175
	cmpl	$1, %esi	 #, niters.175
	je	.L74	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movq	(%rcx,%r9,4), %xmm2	 # MEM <vector(2) float> [(f32 *)vectp_data.182_261], vect__186.183
	movl	%esi, %edi	 # niters.175, niters_vector_mult_vf.177
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movq	(%rax,%r9,4), %xmm0	 # MEM <vector(2) float> [(f32 *)vectp_p.185_267], vect__187.186
	andl	$-2, %edi	 #, niters_vector_mult_vf.177
	movl	%edi, %r9d	 # niters_vector_mult_vf.177, niters_vector_mult_vf.177
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	mulps	%xmm2, %xmm0	 # vect__186.183, vect__188.187
	salq	$2, %r9	 #, _254
	addq	%r9, %r8	 # _254, tmp.178
	addq	%r9, %rdx	 # _254, tmp.179
	cmpl	%edi, %esi	 # niters_vector_mult_vf.177, niters.175
	addss	%xmm0, %xmm1	 # stmp_res_189.188, stmp_res_189.188
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movshdup	%xmm0, %xmm0	 # vect__188.187, stmp_res_189.188
	addss	%xmm0, %xmm1	 # stmp_res_189.188, res
	je	.L78	 #,
.L74:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movss	(%r8), %xmm0	 # *data_246, *data_246
	mulss	(%rdx), %xmm0	 # *weight_247, tmp252
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	addss	%xmm0, %xmm1	 # tmp252, res
.L78:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:122: 				*out = f32_f32_vec_in_prod(data, p, k);  // 计算内积并存储到输出矩阵
	movss	%xmm1, (%r10)	 # res, MEM[(f32 *)out_94]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:121: 			for (int j = 0; j < n; j++, p += k, out++) // 遍历矩阵B的行
	addq	$4, %r10	 #, out
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:121: 			for (int j = 0; j < n; j++, p += k, out++) // 遍历矩阵B的行
	addq	%r14, %rax	 # _3, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:121: 			for (int j = 0; j < n; j++, p += k, out++) // 遍历矩阵B的行
	cmpq	%r13, %r10	 # ivtmp.240, out
	jne	.L79	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:119: 		for (int i = 0; i < m; i++, data += k) {		 // 遍历矩阵A的行
	addl	$1, 24(%rsp)	 #, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:119: 		for (int i = 0; i < m; i++, data += k) {		 // 遍历矩阵A的行
	addq	%r14, %rcx	 # _3, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:119: 		for (int i = 0; i < m; i++, data += k) {		 // 遍历矩阵A的行
	movl	24(%rsp), %eax	 # %sfp, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:119: 		for (int i = 0; i < m; i++, data += k) {		 // 遍历矩阵A的行
	addq	8(%rsp), %r13	 # %sfp, ivtmp.240
	cmpl	%eax, 40(%rbp)	 # i, m
	jne	.L59	 #,
.L55:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:125: }
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
.L56:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:113: 		for (int j = 0; j < n; j++, weight += k) { // 遍历矩阵B的行
	movl	48(%rbp), %r9d	 # n,
	testl	%r9d, %r9d	 #
	jle	.L55	 #,
	movl	40(%rbp), %eax	 # m,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:115: 			for (int i = 0; i < m; i++, p += k, out++)  // 遍历矩阵A的行
	movslq	%r11d, %r14	 # k, k
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:115: 			for (int i = 0; i < m; i++, p += k, out++)  // 遍历矩阵A的行
	salq	$2, %r14	 #, _62
	testl	%eax, %eax	 #
	jle	.L55	 #,
	movl	40(%rbp), %eax	 # m, _144
	movl	%r11d, %r15d	 # k, niters_vector_mult_vf.135
	movq	%rcx, 16(%rbp)	 # data, data
	pxor	%xmm3, %xmm3	 # res
	andl	$-4, %r15d	 #, niters_vector_mult_vf.135
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:113: 		for (int j = 0; j < n; j++, weight += k) { // 遍历矩阵B的行
	movl	$0, 24(%rsp)	 #, %sfp
	leaq	0(,%rax,4), %rdi	 #, _134
	negq	%rax	 # tmp238
	leaq	(%r8,%rdi), %r13	 #, ivtmp.213
	movl	%r11d, %r8d	 # k, bnd.134
	movq	%rdi, 8(%rsp)	 # _134, %sfp
	salq	$2, %rax	 #, tmp239
	leal	-1(%r11), %edi	 #, _5
	shrl	$2, %r8d	 #, bnd.134
	movq	%rax, (%rsp)	 # tmp239, %sfp
	leal	-1(%r8), %ebx	 #, tmp219
	movl	%r15d, %r8d	 # niters_vector_mult_vf.135, niters_vector_mult_vf.135
	movl	%edi, 28(%rsp)	 # _5, %sfp
	leaq	0(,%r8,4), %r12	 #, _53
	addq	$1, %rbx	 #, tmp220
	salq	$4, %rbx	 #, _145
	.p2align 4,,10
	.p2align 3
.L61:
	movq	(%rsp), %rax	 # %sfp, tmp239
	leaq	(%rax,%r13), %r10	 #, out
	leaq	(%rdx,%r12), %rax	 #, tmp.137
	movq	%rax, 16(%rsp)	 # tmp.137, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:114: 			f32 * p = data;
	movq	16(%rbp), %rax	 # data, p
	.p2align 4,,10
	.p2align 3
.L70:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:105: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	testl	%r11d, %r11d	 # k
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104: 	f32 res = 0;  // 初始化结果为0
	movaps	%xmm3, %xmm1	 # res, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:105: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	jle	.L69	 #,
	cmpl	$2, 28(%rsp)	 #, %sfp
	jbe	.L80	 #,
	xorl	%ecx, %ecx	 # ivtmp.194
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104: 	f32 res = 0;  // 初始化结果为0
	movaps	%xmm3, %xmm1	 # res, res
	.p2align 4,,10
	.p2align 3
.L63:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movq	(%rdx,%rcx), %xmm0	 # MEM <vector(4) float> [(f32 *)weight_117 + ivtmp.194_4 * 1], vect__49.144
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movq	(%rax,%rcx), %xmm2	 # MEM <vector(4) float> [(f32 *)p_115 + ivtmp.194_4 * 1], vect__47.141
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movhps	8(%rdx,%rcx), %xmm0	 # MEM <vector(4) float> [(f32 *)weight_117 + ivtmp.194_4 * 1], vect__49.144
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movhps	8(%rax,%rcx), %xmm2	 # MEM <vector(4) float> [(f32 *)p_115 + ivtmp.194_4 * 1], vect__47.141
	addq	$16, %rcx	 #, ivtmp.194
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	mulps	%xmm2, %xmm0	 # vect__47.141, vect__50.145
	cmpq	%rbx, %rcx	 # _145, ivtmp.194
	addss	%xmm0, %xmm1	 # stmp_res_52.146, stmp_res_52.146
	movaps	%xmm0, %xmm2	 # vect__50.145, tmp224
	shufps	$85, %xmm0, %xmm2	 #, vect__50.145, tmp224
	addss	%xmm2, %xmm1	 # stmp_res_52.146, stmp_res_52.146
	movaps	%xmm0, %xmm2	 # vect__50.145, tmp225
	unpckhps	%xmm0, %xmm2	 # vect__50.145, tmp225
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	shufps	$255, %xmm0, %xmm0	 #, vect__50.145, tmp228
	addss	%xmm2, %xmm1	 # stmp_res_52.146, stmp_res_52.146
	addss	%xmm0, %xmm1	 # stmp_res_52.146, res
	jne	.L63	 #,
	leaq	(%rax,%r12), %rcx	 #, tmp.150
	cmpl	%r11d, %r15d	 # k, niters_vector_mult_vf.135
	je	.L69	 #,
	movq	16(%rsp), %r8	 # %sfp, tmp.151
	movl	%r15d, %r9d	 # niters_vector_mult_vf.135,
.L62:
	movl	%r11d, %esi	 # k, niters.147
	subl	%r9d, %esi	 # _86, niters.147
	cmpl	$1, %esi	 #, niters.147
	je	.L65	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movq	(%rax,%r9,4), %xmm2	 # MEM <vector(2) float> [(f32 *)vectp_p.154_162], vect__14.155
	movl	%esi, %edi	 # niters.147, niters_vector_mult_vf.149
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movq	(%rdx,%r9,4), %xmm0	 # MEM <vector(2) float> [(f32 *)vectp_weight.157_168], vect__13.158
	andl	$-2, %edi	 #, niters_vector_mult_vf.149
	movl	%edi, %r9d	 # niters_vector_mult_vf.149, niters_vector_mult_vf.149
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	mulps	%xmm2, %xmm0	 # vect__14.155, vect__12.159
	salq	$2, %r9	 #, _155
	addq	%r9, %rcx	 # _155, tmp.150
	addq	%r9, %r8	 # _155, tmp.151
	cmpl	%edi, %esi	 # niters_vector_mult_vf.149, niters.147
	addss	%xmm0, %xmm1	 # stmp_res_10.160, stmp_res_10.160
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movshdup	%xmm0, %xmm0	 # vect__12.159, stmp_res_10.160
	addss	%xmm0, %xmm1	 # stmp_res_10.160, res
	je	.L69	 #,
.L65:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movss	(%rcx), %xmm0	 # *data_147, *data_147
	mulss	(%r8), %xmm0	 # *weight_148, tmp234
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	addss	%xmm0, %xmm1	 # tmp234, res
.L69:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:116: 				*out = f32_f32_vec_in_prod(p, weight, k);  // 计算内积并存储到输出矩阵
	movss	%xmm1, (%r10)	 # res, MEM[(f32 *)out_113]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:115: 			for (int i = 0; i < m; i++, p += k, out++)  // 遍历矩阵A的行
	addq	$4, %r10	 #, out
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:115: 			for (int i = 0; i < m; i++, p += k, out++)  // 遍历矩阵A的行
	addq	%r14, %rax	 # _62, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:115: 			for (int i = 0; i < m; i++, p += k, out++)  // 遍历矩阵A的行
	cmpq	%r13, %r10	 # ivtmp.213, out
	jne	.L70	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:113: 		for (int j = 0; j < n; j++, weight += k) { // 遍历矩阵B的行
	addl	$1, 24(%rsp)	 #, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:113: 		for (int j = 0; j < n; j++, weight += k) { // 遍历矩阵B的行
	addq	%r14, %rdx	 # _62, weight
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:113: 		for (int j = 0; j < n; j++, weight += k) { // 遍历矩阵B的行
	movl	24(%rsp), %eax	 # %sfp, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:113: 		for (int j = 0; j < n; j++, weight += k) { // 遍历矩阵B的行
	addq	8(%rsp), %r13	 # %sfp, ivtmp.213
	cmpl	%eax, 48(%rbp)	 # j, n
	jne	.L61	 #,
	jmp	.L55	 #
.L81:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:105: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	movq	%rax, %rdx	 # p, tmp.179
	movq	%rcx, %r8	 # data, tmp.178
	xorl	%r9d, %r9d	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104: 	f32 res = 0;  // 初始化结果为0
	movaps	%xmm3, %xmm1	 # res, res
	jmp	.L71	 #
.L80:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:105: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	movq	%rdx, %r8	 # weight, tmp.151
	movq	%rax, %rcx	 # p, tmp.150
	xorl	%r9d, %r9d	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104: 	f32 res = 0;  // 初始化结果为0
	movaps	%xmm3, %xmm1	 # res, res
	jmp	.L62	 #
	.p2align 4
	.globl	_Z20f32_u8_1_vec_in_prodPfPhii
	.def	_Z20f32_u8_1_vec_in_prodPfPhii;	.scl	2;	.type	32;	.endef
_Z20f32_u8_1_vec_in_prodPfPhii:
	pushq	%rbp	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:132: 	int n0 = (8 - b_start) < n ? (8 - b_start) : n;  // 计算开头不满足8位的部分长度
	movl	$8, %r11d	 #, tmp160
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:128: f32 f32_u8_1_vec_in_prod(f32 *data, b8 *weight, int n, int b_start) {
	movq	%rcx, %r10	 # tmp222, data
	movl	%r9d, %ecx	 # tmp225, b_start
	movq	%rsp, %rbp	 #,
	pushq	%rsi	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:132: 	int n0 = (8 - b_start) < n ? (8 - b_start) : n;  // 计算开头不满足8位的部分长度
	subl	%r9d, %r11d	 # b_start, tmp159
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:128: f32 f32_u8_1_vec_in_prod(f32 *data, b8 *weight, int n, int b_start) {
	pushq	%rbx	 #
	movq	%rdx, %rbx	 # tmp223, weight
	subq	$64, %rsp	 #,
	andq	$-16, %rsp	 #,
	movaps	%xmm6, (%rsp)	 #,
	movaps	%xmm7, 16(%rsp)	 #,
	movaps	%xmm8, 32(%rsp)	 #,
	movaps	%xmm9, 48(%rsp)	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:131: 	b8 p = (*weight) << b_start;  // 将bit向量左移b_start位
	movzbl	(%rdx), %edx	 # *weight_35(D), *weight_35(D)
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:131: 	b8 p = (*weight) << b_start;  // 将bit向量左移b_start位
	sall	%cl, %edx	 # b_start, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:132: 	int n0 = (8 - b_start) < n ? (8 - b_start) : n;  // 计算开头不满足8位的部分长度
	cmpl	%r8d, %r11d	 # n, tmp159
	cmovg	%r8d, %r11d	 # tmp159,, n, n0
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:135: 	for (; i < n0; i++, data++, p <<= 1)
	testl	%r11d, %r11d	 # n0
	jle	.L111	 #,
	movslq	%r11d, %rax	 # n0, n0
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:136: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movss	.LC8(%rip), %xmm2	 #, tmp221
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:130: 	f32 res = 0;  // 初始化结果为0
	pxor	%xmm0, %xmm0	 # <retval>
	leaq	(%r10,%rax,4), %rax	 #, data
	.p2align 4,,10
	.p2align 3
.L96:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:136: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	testb	%dl, %dl	 # p
	movss	(%r10), %xmm1	 # MEM[(f32 *)data_207], pretmp_212
	js	.L95	 #,
	xorps	%xmm2, %xmm1	 # tmp221, pretmp_212
.L95:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:135: 	for (; i < n0; i++, data++, p <<= 1)
	addq	$4, %r10	 #, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:136: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	addss	%xmm1, %xmm0	 # pretmp_212, <retval>
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:135: 	for (; i < n0; i++, data++, p <<= 1)
	addl	%edx, %edx	 # p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:135: 	for (; i < n0; i++, data++, p <<= 1)
	cmpq	%r10, %rax	 # data, data
	jne	.L96	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:139: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	leal	-7(%r8), %edx	 #, tmp166
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:139: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	leaq	1(%rbx), %r10	 #, weight
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:139: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	cmpl	%r11d, %edx	 # n0, tmp166
	jle	.L112	 #,
.L118:
	leal	-8(%r8), %ebx	 #, tmp167
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movss	.LC8(%rip), %xmm1	 #, tmp213
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:139: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	movq	%r10, %r9	 # weight, weight
	subl	%r11d, %ebx	 # n0, tmp168
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movaps	%xmm1, %xmm8	 # tmp213, tmp214
	movaps	%xmm1, %xmm7	 # tmp213, tmp215
	shrl	$3, %ebx	 #, _47
	movaps	%xmm1, %xmm6	 # tmp213, tmp216
	movaps	%xmm1, %xmm5	 # tmp213, tmp217
	leal	1(%rbx), %esi	 #,
	movaps	%xmm1, %xmm4	 # tmp213, tmp218
	movaps	%xmm1, %xmm3	 # tmp213, tmp219
	movq	%rsi, %rcx	 # _49, tmp170
	movaps	%xmm1, %xmm2	 # tmp213, tmp220
	salq	$5, %rcx	 #, tmp170
	addq	%rax, %rcx	 # data, data
	.p2align 4,,10
	.p2align 3
.L98:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:140: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	movzbl	(%r9), %edx	 # MEM[(b8 *)weight_121], p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movss	(%rax), %xmm9	 # MEM[(f32 *)data_118], pretmp_134
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	testb	%dl, %dl	 # p
	js	.L99	 #,
	xorps	%xmm1, %xmm9	 # tmp213, pretmp_134
.L99:
	testb	$64, %dl	 #, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addss	%xmm9, %xmm0	 # pretmp_134, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movss	4(%rax), %xmm9	 # MEM[(f32 *)data_118 + 4B], pretmp_148
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	jne	.L100	 #,
	xorps	%xmm8, %xmm9	 # tmp214, pretmp_148
.L100:
	testb	$32, %dl	 #, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addss	%xmm9, %xmm0	 # pretmp_148, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movss	8(%rax), %xmm9	 # MEM[(f32 *)data_118 + 8B], pretmp_162
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	jne	.L101	 #,
	xorps	%xmm7, %xmm9	 # tmp215, pretmp_162
.L101:
	testb	$16, %dl	 #, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addss	%xmm9, %xmm0	 # pretmp_162, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movss	12(%rax), %xmm9	 # MEM[(f32 *)data_118 + 12B], pretmp_175
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	jne	.L102	 #,
	xorps	%xmm6, %xmm9	 # tmp216, pretmp_175
.L102:
	testb	$8, %dl	 #, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addss	%xmm9, %xmm0	 # pretmp_175, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movss	16(%rax), %xmm9	 # MEM[(f32 *)data_118 + 16B], pretmp_181
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	jne	.L103	 #,
	xorps	%xmm5, %xmm9	 # tmp217, pretmp_181
.L103:
	testb	$4, %dl	 #, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addss	%xmm9, %xmm0	 # pretmp_181, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movss	20(%rax), %xmm9	 # MEM[(f32 *)data_118 + 20B], pretmp_187
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	jne	.L104	 #,
	xorps	%xmm4, %xmm9	 # tmp218, pretmp_187
.L104:
	testb	$2, %dl	 #, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addss	%xmm9, %xmm0	 # pretmp_187, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movss	24(%rax), %xmm9	 # MEM[(f32 *)data_118 + 24B], pretmp_193
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	jne	.L105	 #,
	xorps	%xmm3, %xmm9	 # tmp219, pretmp_193
.L105:
	andl	$1, %edx	 #, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addss	%xmm9, %xmm0	 # pretmp_193, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movss	28(%rax), %xmm9	 # MEM[(f32 *)data_118 + 28B], pretmp_199
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	jne	.L106	 #,
	xorps	%xmm2, %xmm9	 # tmp220, pretmp_199
.L106:
	addq	$32, %rax	 #, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:141: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addss	%xmm9, %xmm0	 # pretmp_199, <retval>
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:139: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	addq	$1, %r9	 #, weight
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:139: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	cmpq	%rcx, %rax	 # data, data
	jne	.L98	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:139: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	leal	8(%r11,%rbx,8), %r11d	 #, n0
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:139: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	addq	%rsi, %r10	 # _49, weight
.L97:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:144: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	cmpl	%r11d, %r8d	 # n0, n
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:144: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	movzbl	(%r10), %eax	 # *weight_56, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:144: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	jle	.L93	 #,
	subl	$1, %r8d	 #, tmp203
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:145: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movss	.LC8(%rip), %xmm2	 #, tmp212
	subl	%r11d, %r8d	 # n0, tmp205
	leaq	4(%rcx,%r8,4), %rdx	 #, _10
	.p2align 4,,10
	.p2align 3
.L110:
	testb	%al, %al	 # p
	movss	(%rcx), %xmm1	 # MEM[(f32 *)data_102], pretmp_139
	js	.L109	 #,
	xorps	%xmm2, %xmm1	 # tmp212, pretmp_139
.L109:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:144: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	addq	$4, %rcx	 #, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:145: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	addss	%xmm1, %xmm0	 # pretmp_139, <retval>
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:144: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	addl	%eax, %eax	 # p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:144: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	cmpq	%rcx, %rdx	 # data, _10
	jne	.L110	 #,
.L93:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:148: }
	movaps	(%rsp), %xmm6	 #,
	movaps	16(%rsp), %xmm7	 #,
	movaps	32(%rsp), %xmm8	 #,
	movaps	48(%rsp), %xmm9	 #,
	leaq	-16(%rbp), %rsp	 #,
	popq	%rbx	 #
	popq	%rsi	 #
	popq	%rbp	 #
	ret	
.L111:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:139: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	leal	-7(%r8), %edx	 #, tmp166
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:129: 	int i = 0, j = 0;
	xorl	%r11d, %r11d	 # n0
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:135: 	for (; i < n0; i++, data++, p <<= 1)
	movq	%r10, %rax	 # data, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:130: 	f32 res = 0;  // 初始化结果为0
	pxor	%xmm0, %xmm0	 # <retval>
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:139: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	leaq	1(%rbx), %r10	 #, weight
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:139: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	cmpl	%r11d, %edx	 # n0, tmp166
	jg	.L118	 #,
.L112:
	movq	%rax, %rcx	 # data, data
	jmp	.L97	 #
	.p2align 4
	.globl	_Z16f32_u8_1_mat_mulPfPhS_iiib
	.def	_Z16f32_u8_1_mat_mulPfPhS_iiib;	.scl	2;	.type	32;	.endef
_Z16f32_u8_1_mat_mulPfPhS_iiib:
	pushq	%r15	 #
	movq	%r8, %rax	 # tmp155, out
	pushq	%r14	 #
	pushq	%r13	 #
	pushq	%r12	 #
	movq	%rdx, %r12	 # tmp154, weight
	pushq	%rbp	 #
	pushq	%rdi	 #
	movq	%rcx, %rdi	 # tmp153, data
	pushq	%rsi	 #
	pushq	%rbx	 #
	subq	$72, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:152: 	if (Transpose) {
	cmpb	$0, 192(%rsp)	 #, Transpose
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:151: void f32_u8_1_mat_mul(f32 *data, b8 *weight, f32 *out, int m, int n, int k, bool Transpose) {
	movl	%r9d, 168(%rsp)	 # tmp156, m
	movl	184(%rsp), %r15d	 # k, k
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:152: 	if (Transpose) {
	jne	.L120	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:164: 		for (int i = 0; i < m; i++, data += k)
	testl	%r9d, %r9d	 #
	jle	.L119	 #,
	movl	176(%rsp), %r8d	 # n,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:164: 		for (int i = 0; i < m; i++, data += k)
	movslq	%r15d, %r13	 # k, k
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:164: 		for (int i = 0; i < m; i++, data += k)
	leaq	0(,%r13,4), %rsi	 #, _16
	movq	%rsi, 40(%rsp)	 # _16, %sfp
	testl	%r8d, %r8d	 #
	jle	.L119	 #,
	movslq	176(%rsp), %rdx	 # n, n
	leaq	0(,%rdx,4), %rsi	 #, _101
	leaq	(%rax,%rsi), %rbp	 #, out
	movq	%rsi, 48(%rsp)	 # _101, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:164: 		for (int i = 0; i < m; i++, data += k)
	xorl	%esi, %esi	 # i
	movq	%rbp, %rbx	 # out, ivtmp.316
	.p2align 4,,10
	.p2align 3
.L122:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:156: 			int start_q = (j * k) % 8; //
	movq	%rax, %r13	 # out, out
	xorl	%r14d, %r14d	 # ivtmp.309
	.p2align 4,,10
	.p2align 3
.L125:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:166: 				*out = f32_u8_1_vec_in_prod(data,  weight + (j * k) / 8,  k,  (j * k) % 8);
	leal	7(%r14), %edx	 #, tmp147
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:166: 				*out = f32_u8_1_vec_in_prod(data,  weight + (j * k) / 8,  k,  (j * k) % 8);
	movl	%r14d, %ecx	 # ivtmp.309, tmp140
	movl	%r15d, %r8d	 # k,
	sarl	$31, %ecx	 #, tmp140
	shrl	$29, %ecx	 #, tmp141
	leal	(%r14,%rcx), %r9d	 #, tmp142
	andl	$7, %r9d	 #, tmp143
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:166: 				*out = f32_u8_1_vec_in_prod(data,  weight + (j * k) / 8,  k,  (j * k) % 8);
	testl	%r14d, %r14d	 # ivtmp.309
	cmovns	%r14d, %edx	 # tmp147,, ivtmp.309, _8
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:166: 				*out = f32_u8_1_vec_in_prod(data,  weight + (j * k) / 8,  k,  (j * k) % 8);
	subl	%ecx, %r9d	 # tmp141,
	movq	%rdi, %rcx	 # data,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:165: 			for (int j = 0; j < n; j++,  out++)
	addq	$4, %r13	 #, out
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:165: 			for (int j = 0; j < n; j++,  out++)
	addl	%r15d, %r14d	 # k, ivtmp.309
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:166: 				*out = f32_u8_1_vec_in_prod(data,  weight + (j * k) / 8,  k,  (j * k) % 8);
	sarl	$3, %edx	 #, tmp148
	movslq	%edx, %rdx	 # tmp148, tmp149
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:166: 				*out = f32_u8_1_vec_in_prod(data,  weight + (j * k) / 8,  k,  (j * k) % 8);
	addq	%r12, %rdx	 # weight, tmp150
	call	_Z20f32_u8_1_vec_in_prodPfPhii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:166: 				*out = f32_u8_1_vec_in_prod(data,  weight + (j * k) / 8,  k,  (j * k) % 8);
	movss	%xmm0, -4(%r13)	 # tmp158, MEM[(f32 *)out_68]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:165: 			for (int j = 0; j < n; j++,  out++)
	cmpq	%rbx, %r13	 # ivtmp.316, out
	jne	.L125	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:164: 		for (int i = 0; i < m; i++, data += k)
	movq	48(%rsp), %rcx	 # %sfp, _101
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:164: 		for (int i = 0; i < m; i++, data += k)
	addl	$1, %esi	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:165: 			for (int j = 0; j < n; j++,  out++)
	movq	%rbp, %rax	 # out, out
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:164: 		for (int i = 0; i < m; i++, data += k)
	addq	40(%rsp), %rdi	 # %sfp, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:164: 		for (int i = 0; i < m; i++, data += k)
	addq	%rcx, %rbx	 # _101, ivtmp.316
	cmpl	%esi, 168(%rsp)	 # i, m
	je	.L119	 #,
	addq	%rcx, %rbp	 # _101, out
	jmp	.L122	 #
	.p2align 4,,10
	.p2align 3
.L120:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:153: 		for (int j = 0; j < n; j++) {
	movl	176(%rsp), %ecx	 # n,
	testl	%ecx, %ecx	 #
	jg	.L130	 #,
.L119:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:168: }
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
.L130:
	movl	168(%rsp), %edx	 # m,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:158: 			for (int i = 0; i < m; i++, p += k, out++) {
	movslq	%r15d, %r13	 # k, k
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:158: 			for (int i = 0; i < m; i++, p += k, out++) {
	salq	$2, %r13	 #, _7
	testl	%edx, %edx	 #
	jle	.L119	 #,
	movslq	168(%rsp), %r14	 # m, m
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:153: 		for (int j = 0; j < n; j++) {
	movq	%rdi, 144(%rsp)	 # data, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:158: 			for (int i = 0; i < m; i++, p += k, out++) {
	xorl	%ebp, %ebp	 # ivtmp.301
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:153: 		for (int j = 0; j < n; j++) {
	movl	$0, 40(%rsp)	 #, %sfp
	movq	%r12, 152(%rsp)	 # weight, weight
	leaq	0(,%r14,4), %rsi	 #, _62
	movq	%rsi, 56(%rsp)	 # _62, %sfp
	addq	%r8, %rsi	 # out, out
	movq	%rsi, 48(%rsp)	 # out, %sfp
	movq	%rsi, %rdi	 # ivtmp.298, ivtmp.298
	.p2align 4,,10
	.p2align 3
.L124:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:155: 			b8  *q = weight + (j * k) / 8; // 移动 weight 的索引
	testl	%ebp, %ebp	 # ivtmp.301
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:156: 			int start_q = (j * k) % 8; //
	movl	%ebp, %edx	 # ivtmp.301, tmp134
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:154: 			float *p = data; //
	movq	144(%rsp), %r14	 # data, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:156: 			int start_q = (j * k) % 8; //
	movq	%rax, %r12	 # out, out
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:155: 			b8  *q = weight + (j * k) / 8; // 移动 weight 的索引
	leal	7(%rbp), %ebx	 #, tmp131
	cmovns	%ebp, %ebx	 # tmp131,, ivtmp.301, _1
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:156: 			int start_q = (j * k) % 8; //
	sarl	$31, %edx	 #, tmp134
	shrl	$29, %edx	 #, tmp135
	leal	0(%rbp,%rdx), %esi	 #, tmp136
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:155: 			b8  *q = weight + (j * k) / 8; // 移动 weight 的索引
	sarl	$3, %ebx	 #, tmp132
	movslq	%ebx, %rbx	 # tmp132, tmp133
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:155: 			b8  *q = weight + (j * k) / 8; // 移动 weight 的索引
	addq	152(%rsp), %rbx	 # weight, q
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:156: 			int start_q = (j * k) % 8; //
	andl	$7, %esi	 #, tmp137
	subl	%edx, %esi	 # tmp135, tmp138
	.p2align 4,,10
	.p2align 3
.L123:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:159: 				*out = f32_u8_1_vec_in_prod(p, q, k, start_q);
	movq	%r14, %rcx	 # p,
	movl	%esi, %r9d	 # tmp138,
	movl	%r15d, %r8d	 # k,
	movq	%rbx, %rdx	 # q,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:158: 			for (int i = 0; i < m; i++, p += k, out++) {
	addq	$4, %r12	 #, out
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:158: 			for (int i = 0; i < m; i++, p += k, out++) {
	addq	%r13, %r14	 # _7, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:159: 				*out = f32_u8_1_vec_in_prod(p, q, k, start_q);
	call	_Z20f32_u8_1_vec_in_prodPfPhii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:159: 				*out = f32_u8_1_vec_in_prod(p, q, k, start_q);
	movss	%xmm0, -4(%r12)	 # tmp157, MEM[(f32 *)out_65]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:158: 			for (int i = 0; i < m; i++, p += k, out++) {
	cmpq	%rdi, %r12	 # ivtmp.298, out
	jne	.L123	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:153: 		for (int j = 0; j < n; j++) {
	movq	56(%rsp), %rcx	 # %sfp, _62
	addl	%r15d, %ebp	 # k, ivtmp.301
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:153: 		for (int j = 0; j < n; j++) {
	addl	$1, 40(%rsp)	 #, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:158: 			for (int i = 0; i < m; i++, p += k, out++) {
	movq	48(%rsp), %rbx	 # %sfp, out
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:153: 		for (int j = 0; j < n; j++) {
	movl	40(%rsp), %esi	 # %sfp, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:153: 		for (int j = 0; j < n; j++) {
	addq	%rcx, %rdi	 # _62, ivtmp.298
	cmpl	%esi, 176(%rsp)	 # j, n
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:158: 			for (int i = 0; i < m; i++, p += k, out++) {
	movq	%rbx, %rax	 # out, out
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:153: 		for (int j = 0; j < n; j++) {
	je	.L119	 #,
	leaq	(%rbx,%rcx), %rsi	 #, out
	movq	%rsi, 48(%rsp)	 # out, %sfp
	jmp	.L124	 #
	.p2align 4
	.globl	_Z17i8_i8_vec_in_prodPcS_i
	.def	_Z17i8_i8_vec_in_prodPcS_i;	.scl	2;	.type	32;	.endef
_Z17i8_i8_vec_in_prodPcS_i:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	testl	%r8d, %r8d	 # n
	jle	.L136	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:174: i32 i8_i8_vec_in_prod(i8 *data, i8 *weight, int n) {
	pushq	%rbp	 #
	leal	-1(%r8), %eax	 #, tmp201
	movq	%rcx, %r9	 # tmp317, data
	movq	%rdx, %rcx	 # tmp318, weight
	movq	%rsp, %rbp	 #,
	subq	$48, %rsp	 #,
	andq	$-16, %rsp	 #,
	cmpl	$14, %eax	 #, tmp201
	movaps	%xmm6, (%rsp)	 #,
	movaps	%xmm7, 16(%rsp)	 #,
	movaps	%xmm8, 32(%rsp)	 #,
	jbe	.L137	 #,
	movl	%r8d, %edx	 # n, bnd.325
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	xorl	%eax, %eax	 # ivtmp.347
	pxor	%xmm1, %xmm1	 # vect_res_26.330
	pxor	%xmm5, %xmm5	 # tmp209
	shrl	$4, %edx	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	pxor	%xmm4, %xmm4	 # tmp228
	salq	$4, %rdx	 #, _27
	.p2align 4,,10
	.p2align 3
.L134:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movdqu	(%r9,%rax), %xmm3	 # MEM <vector(16) char> [(i8 *)data_11(D) + ivtmp.347_56 * 1], MEM <vector(16) char> [(i8 *)data_11(D) + ivtmp.347_56 * 1]
	movdqa	%xmm5, %xmm7	 # tmp209, tmp210
	movdqa	%xmm5, %xmm6	 # tmp209, tmp214
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movdqu	(%rcx,%rax), %xmm0	 # MEM <vector(16) char> [(i8 *)weight_12(D) + ivtmp.347_56 * 1], MEM <vector(16) char> [(i8 *)weight_12(D) + ivtmp.347_56 * 1]
	addq	$16, %rax	 #, ivtmp.347
	pcmpgtb	%xmm3, %xmm7	 # MEM <vector(16) char> [(i8 *)data_11(D) + ivtmp.347_56 * 1], tmp210
	movdqa	%xmm3, %xmm8	 # MEM <vector(16) char> [(i8 *)data_11(D) + ivtmp.347_56 * 1], tmp211
	cmpq	%rax, %rdx	 # ivtmp.347, _27
	pcmpgtb	%xmm0, %xmm6	 # MEM <vector(16) char> [(i8 *)weight_12(D) + ivtmp.347_56 * 1], tmp214
	movdqa	%xmm0, %xmm2	 # MEM <vector(16) char> [(i8 *)weight_12(D) + ivtmp.347_56 * 1], tmp215
	punpcklbw	%xmm7, %xmm8	 # tmp210, tmp211
	punpckhbw	%xmm7, %xmm3	 # tmp210, tmp221
	punpcklbw	%xmm6, %xmm2	 # tmp214, tmp215
	pmullw	%xmm8, %xmm2	 # tmp211, vect_patt_10.337
	punpckhbw	%xmm6, %xmm0	 # tmp214, tmp225
	pmullw	%xmm3, %xmm0	 # tmp221, vect_patt_10.337
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movdqa	%xmm4, %xmm3	 # tmp228, tmp229
	pcmpgtw	%xmm2, %xmm3	 # vect_patt_10.337, tmp229
	movdqa	%xmm2, %xmm6	 # vect_patt_10.337, tmp230
	punpcklwd	%xmm3, %xmm6	 # tmp229, tmp230
	paddd	%xmm6, %xmm1	 # tmp230, vect_res_16.341
	punpckhwd	%xmm3, %xmm2	 # tmp229, tmp235
	movdqa	%xmm0, %xmm3	 # vect_patt_10.337, tmp240
	paddd	%xmm2, %xmm1	 # tmp235, vect_res_16.341
	movdqa	%xmm4, %xmm2	 # tmp228, tmp239
	pcmpgtw	%xmm0, %xmm2	 # vect_patt_10.337, tmp239
	punpcklwd	%xmm2, %xmm3	 # tmp239, tmp240
	paddd	%xmm3, %xmm1	 # tmp240, vect_res_16.341
	punpckhwd	%xmm2, %xmm0	 # tmp239, tmp245
	paddd	%xmm0, %xmm1	 # tmp245, vect_res_26.330
	jne	.L134	 #,
	movdqa	%xmm1, %xmm0	 # vect_res_26.330, tmp247
	movl	%r8d, %eax	 # n, niters_vector_mult_vf.326
	psrldq	$8, %xmm0	 #, tmp247
	andl	$-16, %eax	 #, niters_vector_mult_vf.326
	paddd	%xmm0, %xmm1	 # tmp247, _93
	movl	%eax, %r10d	 # niters_vector_mult_vf.326, _65
	movdqa	%xmm1, %xmm0	 # _93, tmp249
	addq	%r10, %r9	 # _65, data
	addq	%r10, %rcx	 # _65, weight
	psrldq	$4, %xmm0	 #, tmp249
	cmpl	%eax, %r8d	 # tmp.329, n
	paddd	%xmm0, %xmm1	 # tmp249, tmp250
	movd	%xmm1, %edx	 # tmp250, stmp_res_16.342
	je	.L135	 #,
.L133:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	(%r9), %r10d	 # *data_58, *data_58
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	(%rcx), %r11d	 # *weight_59, *weight_59
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	imull	%r11d, %r10d	 # *weight_59, tmp253
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	addl	%r10d, %edx	 # tmp253, stmp_res_16.342
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	leal	1(%rax), %r10d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	cmpl	%r10d, %r8d	 # i, n
	jle	.L135	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	1(%r9), %r10d	 # MEM[(i8 *)data_58 + 1B], MEM[(i8 *)data_58 + 1B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	1(%rcx), %r11d	 # MEM[(i8 *)weight_59 + 1B], MEM[(i8 *)weight_59 + 1B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	imull	%r11d, %r10d	 # MEM[(i8 *)weight_59 + 1B], tmp257
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	addl	%r10d, %edx	 # tmp257, stmp_res_16.342
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	leal	2(%rax), %r10d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	cmpl	%r10d, %r8d	 # i, n
	jle	.L135	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	2(%rcx), %r10d	 # MEM[(i8 *)weight_59 + 2B], MEM[(i8 *)weight_59 + 2B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	2(%r9), %r11d	 # MEM[(i8 *)data_58 + 2B], MEM[(i8 *)data_58 + 2B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	imull	%r11d, %r10d	 # MEM[(i8 *)data_58 + 2B], tmp261
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	addl	%r10d, %edx	 # tmp261, stmp_res_16.342
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	leal	3(%rax), %r10d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	cmpl	%r10d, %r8d	 # i, n
	jle	.L135	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	3(%r9), %r10d	 # MEM[(i8 *)data_58 + 3B], MEM[(i8 *)data_58 + 3B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	3(%rcx), %r11d	 # MEM[(i8 *)weight_59 + 3B], MEM[(i8 *)weight_59 + 3B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	imull	%r11d, %r10d	 # MEM[(i8 *)weight_59 + 3B], tmp265
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	addl	%r10d, %edx	 # tmp265, stmp_res_16.342
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	leal	4(%rax), %r10d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	cmpl	%r10d, %r8d	 # i, n
	jle	.L135	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	4(%r9), %r10d	 # MEM[(i8 *)data_58 + 4B], MEM[(i8 *)data_58 + 4B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	4(%rcx), %r11d	 # MEM[(i8 *)weight_59 + 4B], MEM[(i8 *)weight_59 + 4B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	imull	%r11d, %r10d	 # MEM[(i8 *)weight_59 + 4B], tmp269
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	addl	%r10d, %edx	 # tmp269, stmp_res_16.342
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	leal	5(%rax), %r10d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	cmpl	%r10d, %r8d	 # i, n
	jle	.L135	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	5(%r9), %r10d	 # MEM[(i8 *)data_58 + 5B], MEM[(i8 *)data_58 + 5B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	5(%rcx), %r11d	 # MEM[(i8 *)weight_59 + 5B], MEM[(i8 *)weight_59 + 5B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	imull	%r11d, %r10d	 # MEM[(i8 *)weight_59 + 5B], tmp273
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	addl	%r10d, %edx	 # tmp273, stmp_res_16.342
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	leal	6(%rax), %r10d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	cmpl	%r10d, %r8d	 # i, n
	jle	.L135	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	6(%r9), %r10d	 # MEM[(i8 *)data_58 + 6B], MEM[(i8 *)data_58 + 6B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	6(%rcx), %r11d	 # MEM[(i8 *)weight_59 + 6B], MEM[(i8 *)weight_59 + 6B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	imull	%r11d, %r10d	 # MEM[(i8 *)weight_59 + 6B], tmp277
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	addl	%r10d, %edx	 # tmp277, stmp_res_16.342
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	leal	7(%rax), %r10d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	cmpl	%r10d, %r8d	 # i, n
	jle	.L135	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	7(%r9), %r10d	 # MEM[(i8 *)data_58 + 7B], MEM[(i8 *)data_58 + 7B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	7(%rcx), %r11d	 # MEM[(i8 *)weight_59 + 7B], MEM[(i8 *)weight_59 + 7B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	imull	%r11d, %r10d	 # MEM[(i8 *)weight_59 + 7B], tmp281
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	addl	%r10d, %edx	 # tmp281, stmp_res_16.342
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	leal	8(%rax), %r10d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	cmpl	%r10d, %r8d	 # i, n
	jle	.L135	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	8(%r9), %r10d	 # MEM[(i8 *)data_58 + 8B], MEM[(i8 *)data_58 + 8B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	8(%rcx), %r11d	 # MEM[(i8 *)weight_59 + 8B], MEM[(i8 *)weight_59 + 8B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	imull	%r11d, %r10d	 # MEM[(i8 *)weight_59 + 8B], tmp285
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	addl	%r10d, %edx	 # tmp285, stmp_res_16.342
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	leal	9(%rax), %r10d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	cmpl	%r10d, %r8d	 # i, n
	jle	.L135	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	9(%r9), %r10d	 # MEM[(i8 *)data_58 + 9B], MEM[(i8 *)data_58 + 9B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	9(%rcx), %r11d	 # MEM[(i8 *)weight_59 + 9B], MEM[(i8 *)weight_59 + 9B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	imull	%r11d, %r10d	 # MEM[(i8 *)weight_59 + 9B], tmp289
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	addl	%r10d, %edx	 # tmp289, stmp_res_16.342
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	leal	10(%rax), %r10d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	cmpl	%r10d, %r8d	 # i, n
	jle	.L135	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	10(%r9), %r10d	 # MEM[(i8 *)data_58 + 10B], MEM[(i8 *)data_58 + 10B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	10(%rcx), %r11d	 # MEM[(i8 *)weight_59 + 10B], MEM[(i8 *)weight_59 + 10B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	imull	%r11d, %r10d	 # MEM[(i8 *)weight_59 + 10B], tmp293
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	addl	%r10d, %edx	 # tmp293, stmp_res_16.342
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	leal	11(%rax), %r10d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	cmpl	%r10d, %r8d	 # i, n
	jle	.L135	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	11(%r9), %r10d	 # MEM[(i8 *)data_58 + 11B], MEM[(i8 *)data_58 + 11B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	11(%rcx), %r11d	 # MEM[(i8 *)weight_59 + 11B], MEM[(i8 *)weight_59 + 11B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	imull	%r11d, %r10d	 # MEM[(i8 *)weight_59 + 11B], tmp297
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	addl	%r10d, %edx	 # tmp297, stmp_res_16.342
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	leal	12(%rax), %r10d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	cmpl	%r10d, %r8d	 # i, n
	jle	.L135	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	12(%r9), %r10d	 # MEM[(i8 *)data_58 + 12B], MEM[(i8 *)data_58 + 12B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	12(%rcx), %r11d	 # MEM[(i8 *)weight_59 + 12B], MEM[(i8 *)weight_59 + 12B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	imull	%r11d, %r10d	 # MEM[(i8 *)weight_59 + 12B], tmp301
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	addl	%r10d, %edx	 # tmp301, stmp_res_16.342
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	leal	13(%rax), %r10d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	cmpl	%r10d, %r8d	 # i, n
	jle	.L135	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	13(%r9), %r10d	 # MEM[(i8 *)data_58 + 13B], MEM[(i8 *)data_58 + 13B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	addl	$14, %eax	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	13(%rcx), %r11d	 # MEM[(i8 *)weight_59 + 13B], MEM[(i8 *)weight_59 + 13B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	imull	%r11d, %r10d	 # MEM[(i8 *)weight_59 + 13B], tmp305
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	addl	%r10d, %edx	 # tmp305, stmp_res_16.342
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	cmpl	%eax, %r8d	 # i, n
	jle	.L135	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	14(%r9), %eax	 # MEM[(i8 *)data_58 + 14B], MEM[(i8 *)data_58 + 14B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	movsbl	14(%rcx), %ecx	 # MEM[(i8 *)weight_59 + 14B], MEM[(i8 *)weight_59 + 14B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	imull	%ecx, %eax	 # MEM[(i8 *)weight_59 + 14B], tmp309
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:178: 		res += (i16)(*data) * (i16)(*weight);  // 累加乘积
	addl	%eax, %edx	 # tmp309, stmp_res_16.342
.L135:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:179: 	return (res/127);  // 返回内积结果
	movslq	%edx, %rax	 # stmp_res_16.342, stmp_res_16.342
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:180: }
	movaps	(%rsp), %xmm6	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:179: 	return (res/127);  // 返回内积结果
	imulq	$-2130574327, %rax, %rax	 #, stmp_res_16.342, tmp311
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:180: }
	movaps	16(%rsp), %xmm7	 #,
	movaps	32(%rsp), %xmm8	 #,
	leave	
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:179: 	return (res/127);  // 返回内积结果
	shrq	$32, %rax	 #, tmp312
	addl	%edx, %eax	 # stmp_res_16.342, tmp313
	sarl	$31, %edx	 #, tmp315
	sarl	$6, %eax	 #, tmp314
	subl	%edx, %eax	 # tmp315, <retval>
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:180: }
	ret	
	.p2align 4,,10
	.p2align 3
.L136:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:177: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	xorl	%eax, %eax	 # <retval>
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:180: }
	ret	
.L137:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:176: 	i32 res = 0;  // 初始化结果为0, 结果要除以127,解析成(-1,1)的8字节数
	xorl	%edx, %edx	 # stmp_res_16.342
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:175: 	int i = 0;
	xorl	%eax, %eax	 # tmp.329
	jmp	.L133	 #
	.p2align 4
	.globl	_Z17i8_b8_vec_in_prodPcPhii
	.def	_Z17i8_b8_vec_in_prodPcPhii;	.scl	2;	.type	32;	.endef
_Z17i8_b8_vec_in_prodPcPhii:
	pushq	%rbp	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:190: 	int n0 = (8 - b_start) < n ? (8 - b_start) : n;  // 计算开头不满足8位的部分长度
	movl	$8, %r11d	 #, tmp449
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:185: i32 i8_b8_vec_in_prod(i8 *data, b8 *weight, int n, int b_start) {
	movq	%rdx, %r10	 # tmp1179, weight
	movq	%rsp, %rbp	 #,
	pushq	%r15	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:190: 	int n0 = (8 - b_start) < n ? (8 - b_start) : n;  // 计算开头不满足8位的部分长度
	subl	%r9d, %r11d	 # b_start, tmp448
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:185: i32 i8_b8_vec_in_prod(i8 *data, b8 *weight, int n, int b_start) {
	pushq	%r14	 #
	pushq	%r13	 #
	pushq	%r12	 #
	pushq	%rdi	 #
	pushq	%rsi	 #
	pushq	%rbx	 #
	movq	%rcx, %rbx	 # tmp1178, data
	movl	%r9d, %ecx	 # tmp1181, b_start
	subq	$160, %rsp	 #,
	andq	$-16, %rsp	 #,
	addq	$-128, %rsp	 #,
	movaps	%xmm6, 128(%rsp)	 #,
	movaps	%xmm7, 144(%rsp)	 #,
	movaps	%xmm8, 160(%rsp)	 #,
	movaps	%xmm9, 176(%rsp)	 #,
	movaps	%xmm10, 192(%rsp)	 #,
	movaps	%xmm11, 208(%rsp)	 #,
	movaps	%xmm12, 224(%rsp)	 #,
	movaps	%xmm13, 240(%rsp)	 #,
	movaps	%xmm14, 256(%rsp)	 #,
	movaps	%xmm15, 272(%rsp)	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:189: 	b8 p = (*weight) << b_start;  // 将bit向量左移b_start位
	movzbl	(%rdx), %esi	 # *weight_41(D), *weight_41(D)
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:189: 	b8 p = (*weight) << b_start;  // 将bit向量左移b_start位
	sall	%cl, %esi	 # b_start, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:190: 	int n0 = (8 - b_start) < n ? (8 - b_start) : n;  // 计算开头不满足8位的部分长度
	cmpl	%r8d, %r11d	 # n, tmp448
	cmovg	%r8d, %r11d	 # tmp448,, n, n0
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:193: 	for (; i < n0; i++, data++, p <<= 1)
	testl	%r11d, %r11d	 # n0
	jle	.L165	 #,
	movslq	%r11d, %r9	 # n0, n0
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:187: 	i32 res = 0;  // 初始化结果为0, 结果要除以127,解析成(-1,1)的8字节数
	xorl	%eax, %eax	 # <retval>
	addq	%rbx, %r9	 # data, data
	.p2align 4,,10
	.p2align 3
.L146:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:194: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movsbl	(%rbx), %edx	 # MEM[(i8 *)data_225], _231
	movl	%edx, %ecx	 # _231, tmp1158
	negl	%ecx	 # tmp1158
	testb	%sil, %sil	 # p
	cmovns	%ecx, %edx	 # tmp1158,, _231
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:193: 	for (; i < n0; i++, data++, p <<= 1)
	addq	$1, %rbx	 #, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:193: 	for (; i < n0; i++, data++, p <<= 1)
	addl	%esi, %esi	 # p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:194: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	addl	%edx, %eax	 # _231, <retval>
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:193: 	for (; i < n0; i++, data++, p <<= 1)
	cmpq	%rbx, %r9	 # data, data
	jne	.L146	 #,
.L144:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:197: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	leal	-7(%r8), %esi	 #, _10
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:197: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	leaq	1(%r10), %r12	 #, weight
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:197: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	cmpl	%r11d, %esi	 # n0, _10
	jle	.L147	 #,
	leal	-8(%r8), %edx	 #, tmp451
	subl	%r11d, %edx	 # n0, _295
	movl	%edx, %r13d	 # _295, _296
	shrl	$3, %r13d	 #, _296
	cmpl	$119, %edx	 #, _295
	leal	1(%r13), %edi	 #, tmp1156
	jbe	.L166	 #,
	movl	%edi, %ebx	 # tmp1156, bnd.363
	pxor	%xmm14, %xmm14	 # vect_res_103.368
	movq	%r12, %rcx	 # weight, ivtmp.480
	movq	%r9, %rdx	 # data, ivtmp.483
	shrl	$4, %ebx	 #, bnd.363
	movaps	%xmm14, 80(%rsp)	 # vect_res_103.368, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm14, %xmm15	 #, tmp568
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	pxor	%xmm0, %xmm0	 # tmp527
	subl	$1, %ebx	 #, tmp454
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:197: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	movaps	%xmm14, 96(%rsp)	 # vect_res_103.368, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	pxor	%xmm13, %xmm13	 # tmp535
	salq	$4, %rbx	 #, tmp455
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:197: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	movaps	%xmm14, 112(%rsp)	 # vect_res_191.460, %sfp
	leaq	17(%r10,%rbx), %r10	 #, _203
	.p2align 4,,10
	.p2align 3
.L149:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqu	(%rdx), %xmm2	 # MEM <vector(16) char> [(i8 *)_208], MEM <vector(16) char> [(i8 *)_208]
	addq	$16, %rcx	 #, ivtmp.480
	subq	$-128, %rdx	 #, ivtmp.483
	movdqu	-112(%rdx), %xmm6	 # MEM <vector(16) char> [(i8 *)_208 + 16B], MEM <vector(16) char> [(i8 *)_208 + 16B]
	movdqa	.LC9(%rip), %xmm5	 #, tmp466
	movdqa	.LC9(%rip), %xmm10	 #, tmp467
	pand	%xmm2, %xmm5	 # MEM <vector(16) char> [(i8 *)_208], tmp466
	psrlw	$8, %xmm2	 #, tmp469
	movdqu	-96(%rdx), %xmm7	 # MEM <vector(16) char> [(i8 *)_208 + 32B], MEM <vector(16) char> [(i8 *)_208 + 32B]
	pand	%xmm6, %xmm10	 # MEM <vector(16) char> [(i8 *)_208 + 16B], tmp467
	psrlw	$8, %xmm6	 #, tmp470
	movdqu	-80(%rdx), %xmm4	 # MEM <vector(16) char> [(i8 *)_208 + 48B], MEM <vector(16) char> [(i8 *)_208 + 48B]
	packuswb	%xmm10, %xmm5	 # tmp467, vect_perm_even_406
	packuswb	%xmm6, %xmm2	 # tmp470, vect_perm_odd_407
	movdqa	.LC9(%rip), %xmm6	 #, tmp471
	movdqa	.LC9(%rip), %xmm10	 #, tmp472
	pand	%xmm7, %xmm6	 # MEM <vector(16) char> [(i8 *)_208 + 32B], tmp471
	psrlw	$8, %xmm7	 #, tmp474
	movdqu	-64(%rdx), %xmm1	 # MEM <vector(16) char> [(i8 *)_208 + 64B], MEM <vector(16) char> [(i8 *)_208 + 64B]
	pand	%xmm4, %xmm10	 # MEM <vector(16) char> [(i8 *)_208 + 48B], tmp472
	psrlw	$8, %xmm4	 #, tmp475
	packuswb	%xmm10, %xmm6	 # tmp472, vect_perm_even_408
	movdqu	-48(%rdx), %xmm8	 # MEM <vector(16) char> [(i8 *)_208 + 80B], MEM <vector(16) char> [(i8 *)_208 + 80B]
	packuswb	%xmm4, %xmm7	 # tmp475, vect_perm_odd_409
	movdqa	.LC9(%rip), %xmm4	 #, tmp476
	movdqa	.LC9(%rip), %xmm10	 #, tmp477
	pand	%xmm1, %xmm4	 # MEM <vector(16) char> [(i8 *)_208 + 64B], tmp476
	psrlw	$8, %xmm1	 #, tmp479
	movdqu	-32(%rdx), %xmm3	 # MEM <vector(16) char> [(i8 *)_208 + 96B], MEM <vector(16) char> [(i8 *)_208 + 96B]
	pand	%xmm8, %xmm10	 # MEM <vector(16) char> [(i8 *)_208 + 80B], tmp477
	psrlw	$8, %xmm8	 #, tmp480
	movdqu	-16(%rdx), %xmm9	 # MEM <vector(16) char> [(i8 *)_208 + 112B], MEM <vector(16) char> [(i8 *)_208 + 112B]
	packuswb	%xmm10, %xmm4	 # tmp477, vect_perm_even_410
	movdqa	.LC9(%rip), %xmm10	 #, tmp482
	packuswb	%xmm8, %xmm1	 # tmp480, vect_perm_odd_411
	movdqa	.LC9(%rip), %xmm8	 #, tmp481
	pand	%xmm9, %xmm10	 # MEM <vector(16) char> [(i8 *)_208 + 112B], tmp482
	psrlw	$8, %xmm9	 #, tmp485
	pand	%xmm3, %xmm8	 # MEM <vector(16) char> [(i8 *)_208 + 96B], tmp481
	psrlw	$8, %xmm3	 #, tmp484
	packuswb	%xmm10, %xmm8	 # tmp482, vect_perm_even_412
	movdqa	.LC9(%rip), %xmm10	 #, tmp487
	packuswb	%xmm9, %xmm3	 # tmp485, vect_perm_odd_413
	movdqa	.LC9(%rip), %xmm9	 #, tmp486
	pand	%xmm6, %xmm10	 # vect_perm_even_408, tmp487
	psrlw	$8, %xmm6	 #, tmp490
	pand	%xmm5, %xmm9	 # vect_perm_even_406, tmp486
	psrlw	$8, %xmm5	 #, tmp489
	packuswb	%xmm10, %xmm9	 # tmp487, vect_perm_even_414
	movdqa	.LC9(%rip), %xmm10	 #, tmp492
	packuswb	%xmm6, %xmm5	 # tmp490, vect_perm_odd_415
	movdqa	.LC9(%rip), %xmm6	 #, tmp491
	pand	%xmm8, %xmm10	 # vect_perm_even_412, tmp492
	psrlw	$8, %xmm8	 #, tmp495
	pand	%xmm4, %xmm6	 # vect_perm_even_410, tmp491
	psrlw	$8, %xmm4	 #, tmp494
	packuswb	%xmm10, %xmm6	 # tmp492, vect_perm_even_416
	movdqa	.LC9(%rip), %xmm10	 #, tmp497
	packuswb	%xmm8, %xmm4	 # tmp495, vect_perm_odd_417
	movdqa	.LC9(%rip), %xmm8	 #, tmp496
	pand	%xmm7, %xmm10	 # vect_perm_odd_409, tmp497
	psrlw	$8, %xmm7	 #, tmp500
	pand	%xmm2, %xmm8	 # vect_perm_odd_407, tmp496
	psrlw	$8, %xmm2	 #, tmp499
	packuswb	%xmm10, %xmm8	 # tmp497, vect_perm_even_418
	movdqa	.LC9(%rip), %xmm10	 #, tmp502
	packuswb	%xmm7, %xmm2	 # tmp500, vect_perm_odd_419
	movdqa	.LC9(%rip), %xmm7	 #, tmp501
	pand	%xmm3, %xmm10	 # vect_perm_odd_413, tmp502
	psrlw	$8, %xmm3	 #, tmp505
	pand	%xmm1, %xmm7	 # vect_perm_odd_411, tmp501
	psrlw	$8, %xmm1	 #, tmp504
	packuswb	%xmm10, %xmm7	 # tmp502, vect_perm_even_420
	movdqa	.LC9(%rip), %xmm10	 #, tmp507
	packuswb	%xmm3, %xmm1	 # tmp505, vect_perm_odd_421
	movdqa	.LC9(%rip), %xmm3	 #, tmp506
	pand	%xmm6, %xmm10	 # vect_perm_even_416, tmp507
	psrlw	$8, %xmm6	 #, tmp510
	pand	%xmm9, %xmm3	 # vect_perm_even_414, tmp506
	psrlw	$8, %xmm9	 #, tmp509
	packuswb	%xmm10, %xmm3	 # tmp507, vect_perm_even_422
	movdqa	%xmm9, %xmm11	 # tmp509, vect_perm_odd_423
	movdqa	.LC9(%rip), %xmm9	 #, tmp512
	packuswb	%xmm6, %xmm11	 # tmp510, vect_perm_odd_423
	movdqa	.LC9(%rip), %xmm6	 #, tmp511
	movaps	%xmm11, 64(%rsp)	 # vect_perm_odd_423, %sfp
	pand	%xmm7, %xmm9	 # vect_perm_even_420, tmp512
	psrlw	$8, %xmm7	 #, tmp515
	pand	%xmm8, %xmm6	 # vect_perm_even_418, tmp511
	psrlw	$8, %xmm8	 #, tmp514
	packuswb	%xmm9, %xmm6	 # tmp512, vect_perm_even_424
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm9	 # tmp568, vect_iftmp.385
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm8, %xmm12	 # tmp514, vect_perm_odd_425
	movdqa	.LC9(%rip), %xmm8	 #, tmp516
	packuswb	%xmm7, %xmm12	 # tmp515, vect_perm_odd_425
	movaps	%xmm12, 48(%rsp)	 # vect_perm_odd_425, %sfp
	pand	%xmm5, %xmm8	 # vect_perm_odd_415, tmp516
	movdqa	%xmm8, %xmm7	 # tmp516, tmp516
	psrlw	$8, %xmm5	 #, tmp519
	movdqa	.LC9(%rip), %xmm8	 #, tmp517
	pand	%xmm4, %xmm8	 # vect_perm_odd_417, tmp517
	psrlw	$8, %xmm4	 #, tmp520
	packuswb	%xmm8, %xmm7	 # tmp517, vect_perm_even_426
	movdqa	%xmm5, %xmm8	 # tmp519, vect_perm_odd_427
	packuswb	%xmm4, %xmm8	 # tmp520, vect_perm_odd_427
	movaps	%xmm8, 32(%rsp)	 # vect_perm_odd_427, %sfp
	movdqa	.LC9(%rip), %xmm5	 #, tmp521
	movdqa	.LC9(%rip), %xmm4	 #, tmp522
	pand	%xmm2, %xmm5	 # vect_perm_odd_419, tmp521
	psrlw	$8, %xmm2	 #, tmp524
	movdqa	%xmm5, %xmm10	 # tmp521, vect_perm_even_428
	movdqa	%xmm13, %xmm5	 # tmp535, tmp565
	pand	%xmm1, %xmm4	 # vect_perm_odd_421, tmp522
	psrlw	$8, %xmm1	 #, tmp525
	packuswb	%xmm4, %xmm10	 # tmp522, vect_perm_even_428
	movdqa	%xmm2, %xmm4	 # tmp524, vect_perm_odd_429
	packuswb	%xmm1, %xmm4	 # tmp525, vect_perm_odd_429
	movdqa	%xmm0, %xmm1	 # tmp527, tmp528
	movdqa	%xmm3, %xmm2	 # vect_perm_even_422, tmp529
	movaps	%xmm4, (%rsp)	 # vect_perm_odd_429, %sfp
	pcmpgtb	%xmm3, %xmm1	 # vect_perm_even_422, tmp528
	movaps	%xmm10, 16(%rsp)	 # vect_perm_even_428, %sfp
	punpcklbw	%xmm1, %xmm2	 # tmp528, tmp529
	punpckhbw	%xmm1, %xmm3	 # tmp528, tmp533
	movdqa	%xmm13, %xmm1	 # tmp535, tmp536
	movdqa	%xmm2, %xmm8	 # tmp529, tmp529
	pcmpgtw	%xmm2, %xmm1	 # tmp529, tmp536
	punpckhwd	%xmm1, %xmm8	 # tmp536, tmp541
	punpcklwd	%xmm1, %xmm2	 # tmp536, tmp537
	movdqa	%xmm13, %xmm1	 # tmp535, tmp544
	movdqa	%xmm2, %xmm12	 # tmp537, tmp537
	pcmpgtw	%xmm3, %xmm1	 # tmp533, tmp544
	movdqa	%xmm3, %xmm2	 # tmp533, tmp545
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm12, %xmm9	 # tmp537, vect_iftmp.385
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	punpcklwd	%xmm1, %xmm2	 # tmp544, tmp545
	movdqa	%xmm2, %xmm10	 # tmp545, tmp545
	movdqu	-16(%rcx), %xmm2	 # MEM <vector(16) unsigned char> [(b8 *)_207], tmp1227
	punpckhwd	%xmm1, %xmm3	 # tmp544, tmp549
	movdqa	%xmm0, %xmm1	 # tmp527, tmp551
	pcmpgtb	%xmm2, %xmm1	 # tmp1227, tmp551
	movdqa	%xmm0, %xmm2	 # tmp527, tmp556
	pcmpeqb	%xmm0, %xmm1	 # tmp527, tmp553
	pcmpgtb	%xmm1, %xmm2	 # tmp553, tmp556
	movdqa	%xmm2, %xmm4	 # tmp556, tmp556
	movdqa	%xmm1, %xmm2	 # tmp553, tmp557
	punpcklbw	%xmm4, %xmm2	 # tmp556, tmp557
	pcmpgtw	%xmm2, %xmm5	 # tmp557, tmp565
	punpckhbw	%xmm4, %xmm1	 # tmp556, tmp561
	movdqa	%xmm5, %xmm11	 # tmp565, tmp565
	movdqa	%xmm2, %xmm5	 # tmp557, tmp566
	punpcklwd	%xmm11, %xmm5	 # tmp565, tmp566
	pand	%xmm5, %xmm9	 # tmp566, tmp569
	pandn	%xmm12, %xmm5	 # tmp537, tmp570
	movdqa	%xmm5, %xmm4	 # tmp570, tmp570
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm5	 # tmp568, vect_iftmp.385
	punpckhwd	%xmm11, %xmm2	 # tmp565, tmp575
	por	%xmm9, %xmm4	 # tmp569, vect_patt_243.389
	movdqa	%xmm15, %xmm9	 # tmp568, vect_iftmp.385
	psubd	%xmm8, %xmm5	 # tmp541, vect_iftmp.385
	psubd	%xmm10, %xmm9	 # tmp545, vect_iftmp.385
	movdqa	%xmm13, %xmm12	 # tmp535, tmp638
	pand	%xmm2, %xmm5	 # tmp575, tmp578
	pandn	%xmm8, %xmm2	 # tmp541, tmp579
	por	%xmm5, %xmm2	 # tmp578, vect_patt_243.389
	movdqa	%xmm13, %xmm5	 # tmp535, tmp583
	pcmpgtw	%xmm1, %xmm5	 # tmp561, tmp583
	movdqa	%xmm1, %xmm8	 # tmp561, tmp584
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	112(%rsp), %xmm4	 # %sfp, vect_res_86.390
	paddd	96(%rsp), %xmm2	 # %sfp, vect_res_86.390
	punpcklwd	%xmm5, %xmm8	 # tmp583, tmp584
	punpckhwd	%xmm5, %xmm1	 # tmp583, tmp593
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm5	 # tmp568, vect_iftmp.385
	pand	%xmm8, %xmm9	 # tmp584, tmp587
	psubd	%xmm3, %xmm5	 # tmp549, vect_iftmp.385
	pandn	%xmm10, %xmm8	 # tmp545, tmp588
	por	%xmm9, %xmm8	 # tmp587, vect_patt_243.389
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm6, %xmm9	 # vect_perm_even_424, tmp601
	pand	%xmm1, %xmm5	 # tmp593, tmp596
	pandn	%xmm3, %xmm1	 # tmp549, tmp597
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm14, %xmm3	 # vect_res_103.368, vect_res_103.368
	por	%xmm5, %xmm1	 # tmp596, vect_patt_243.389
	paddd	%xmm1, %xmm3	 # vect_patt_243.389, vect_res_103.368
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm0, %xmm1	 # tmp527, tmp600
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	80(%rsp), %xmm5	 # %sfp, vect_patt_243.389
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm10	 # tmp568, vect_iftmp.395
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	pcmpgtb	%xmm6, %xmm1	 # vect_perm_even_424, tmp600
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movaps	%xmm3, 96(%rsp)	 # vect_res_103.368, %sfp
	movdqa	%xmm0, %xmm3	 # tmp527, tmp624
	paddd	%xmm8, %xmm5	 # vect_patt_243.389, vect_patt_243.389
	movdqa	%xmm0, %xmm8	 # tmp527, tmp629
	movaps	%xmm5, 112(%rsp)	 # vect_patt_243.389, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	punpcklbw	%xmm1, %xmm9	 # tmp600, tmp601
	punpckhbw	%xmm1, %xmm6	 # tmp600, tmp605
	movdqa	%xmm13, %xmm1	 # tmp535, tmp608
	movdqa	%xmm9, %xmm5	 # tmp601, tmp609
	pcmpgtw	%xmm9, %xmm1	 # tmp601, tmp608
	movdqa	%xmm6, %xmm11	 # tmp605, tmp617
	punpckhwd	%xmm1, %xmm9	 # tmp608, tmp613
	punpcklwd	%xmm1, %xmm5	 # tmp608, tmp609
	movdqa	%xmm13, %xmm1	 # tmp535, tmp616
	movdqa	%xmm5, %xmm14	 # tmp609, tmp609
	pcmpgtw	%xmm6, %xmm1	 # tmp605, tmp616
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm14, %xmm10	 # tmp609, vect_iftmp.395
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	punpcklwd	%xmm1, %xmm11	 # tmp616, tmp617
	punpckhwd	%xmm1, %xmm6	 # tmp616, tmp621
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:198: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	movdqu	-16(%rcx), %xmm1	 # MEM <vector(16) unsigned char> [(b8 *)_207], vect_p_89.391
	cmpq	%rcx, %r10	 # ivtmp.480, _203
	paddb	%xmm1, %xmm1	 # tmp1251, vect_p_89.391
	pcmpgtb	%xmm1, %xmm3	 # vect_p_89.391, tmp624
	paddb	%xmm1, %xmm1	 # vect_p_89.391, vect_p_104.401
	pcmpeqb	%xmm0, %xmm3	 # tmp527, tmp626
	pcmpgtb	%xmm3, %xmm8	 # tmp626, tmp629
	movdqa	%xmm3, %xmm5	 # tmp626, tmp630
	punpcklbw	%xmm8, %xmm5	 # tmp629, tmp630
	pcmpgtw	%xmm5, %xmm12	 # tmp630, tmp638
	punpckhbw	%xmm8, %xmm3	 # tmp629, tmp634
	movdqa	%xmm5, %xmm8	 # tmp630, tmp639
	punpcklwd	%xmm12, %xmm8	 # tmp638, tmp639
	punpckhwd	%xmm12, %xmm5	 # tmp638, tmp648
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm12	 # tmp568, vect_iftmp.395
	pand	%xmm8, %xmm10	 # tmp639, tmp642
	psubd	%xmm9, %xmm12	 # tmp613, vect_iftmp.395
	pandn	%xmm14, %xmm8	 # tmp609, tmp643
	por	%xmm10, %xmm8	 # tmp642, vect_patt_246.399
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm8, %xmm4	 # vect_patt_246.399, vect_res_101.400
	pand	%xmm5, %xmm12	 # tmp648, tmp651
	pandn	%xmm9, %xmm5	 # tmp613, tmp652
	por	%xmm12, %xmm5	 # tmp651, tmp652
	movdqa	%xmm13, %xmm12	 # tmp535, tmp656
	pcmpgtw	%xmm3, %xmm12	 # tmp634, tmp656
	movdqa	%xmm5, %xmm10	 # tmp652, vect_patt_246.399
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm5	 # tmp568, vect_iftmp.395
	psubd	%xmm11, %xmm5	 # tmp617, vect_iftmp.395
	movdqa	%xmm3, %xmm9	 # tmp634, tmp657
	movdqa	%xmm0, %xmm8	 # tmp527, tmp703
	movdqa	%xmm5, %xmm14	 # vect_iftmp.395, vect_iftmp.395
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm10, %xmm2	 # vect_patt_246.399, vect_res_101.400
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm10	 # tmp568, vect_iftmp.405
	punpcklwd	%xmm12, %xmm9	 # tmp656, tmp657
	movdqa	%xmm9, %xmm5	 # tmp657, tmp661
	pand	%xmm9, %xmm14	 # tmp657, tmp660
	movdqa	%xmm15, %xmm9	 # tmp568, vect_iftmp.395
	pandn	%xmm11, %xmm5	 # tmp617, tmp661
	por	%xmm14, %xmm5	 # tmp660, vect_patt_246.399
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm0, %xmm14	 # tmp527, tmp673
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm6, %xmm9	 # tmp621, vect_iftmp.395
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	pcmpgtb	%xmm7, %xmm14	 # vect_perm_even_426, tmp673
	punpckhwd	%xmm12, %xmm3	 # tmp656, tmp666
	pand	%xmm3, %xmm9	 # tmp666, tmp669
	pandn	%xmm6, %xmm3	 # tmp621, tmp670
	por	%xmm9, %xmm3	 # tmp669, vect_patt_246.399
	movdqa	%xmm7, %xmm9	 # vect_perm_even_426, tmp674
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	96(%rsp), %xmm3	 # %sfp, vect_patt_246.399
	paddd	112(%rsp), %xmm5	 # %sfp, vect_res_101.400
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	punpckhbw	%xmm14, %xmm7	 # tmp673, tmp678
	punpcklbw	%xmm14, %xmm9	 # tmp673, tmp674
	movdqa	%xmm13, %xmm14	 # tmp535, tmp681
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movaps	%xmm3, 112(%rsp)	 # vect_patt_246.399, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	pcmpgtw	%xmm9, %xmm14	 # tmp674, tmp681
	movdqa	%xmm7, %xmm11	 # tmp678, tmp690
	movdqa	%xmm7, %xmm3	 # tmp678, tmp694
	movdqa	%xmm14, %xmm6	 # tmp681, tmp681
	movdqa	%xmm9, %xmm14	 # tmp674, tmp682
	punpcklwd	%xmm6, %xmm14	 # tmp681, tmp682
	punpckhwd	%xmm6, %xmm9	 # tmp681, tmp686
	movdqa	%xmm13, %xmm6	 # tmp535, tmp689
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm14, %xmm10	 # tmp682, vect_iftmp.405
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	pcmpgtw	%xmm7, %xmm6	 # tmp678, tmp689
	punpcklwd	%xmm6, %xmm11	 # tmp689, tmp690
	punpckhwd	%xmm6, %xmm3	 # tmp689, tmp694
	movdqa	%xmm0, %xmm6	 # tmp527, tmp698
	pcmpgtb	%xmm1, %xmm6	 # vect_p_104.401, tmp698
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:198: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	paddb	%xmm1, %xmm1	 # vect_p_104.401, vect_p_119.411
	pcmpeqb	%xmm0, %xmm6	 # tmp527, tmp700
	pcmpgtb	%xmm6, %xmm8	 # tmp700, tmp703
	movdqa	%xmm6, %xmm12	 # tmp700, tmp704
	punpcklbw	%xmm8, %xmm12	 # tmp703, tmp704
	movdqa	%xmm12, %xmm7	 # tmp704, tmp704
	movdqa	%xmm13, %xmm12	 # tmp535, tmp712
	punpckhbw	%xmm8, %xmm6	 # tmp703, tmp708
	pcmpgtw	%xmm7, %xmm12	 # tmp704, tmp712
	movdqa	%xmm7, %xmm8	 # tmp704, tmp713
	punpcklwd	%xmm12, %xmm8	 # tmp712, tmp713
	pand	%xmm8, %xmm10	 # tmp713, tmp716
	pandn	%xmm14, %xmm8	 # tmp682, tmp717
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm14	 # tmp568, vect_iftmp.405
	psubd	%xmm9, %xmm14	 # tmp686, vect_iftmp.405
	por	%xmm10, %xmm8	 # tmp716, vect_patt_249.409
	punpckhwd	%xmm12, %xmm7	 # tmp712, tmp722
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm8, %xmm4	 # vect_patt_249.409, vect_res_116.410
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm14, %xmm10	 # vect_iftmp.405, vect_iftmp.405
	movdqa	%xmm13, %xmm14	 # tmp535, tmp730
	pcmpgtw	%xmm6, %xmm14	 # tmp708, tmp730
	pand	%xmm7, %xmm10	 # tmp722, tmp725
	pandn	%xmm9, %xmm7	 # tmp686, tmp726
	movdqa	%xmm6, %xmm9	 # tmp708, tmp731
	por	%xmm10, %xmm7	 # tmp725, vect_patt_249.409
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm7, %xmm2	 # vect_patt_249.409, vect_res_116.410
	movdqa	%xmm14, %xmm10	 # tmp730, tmp730
	punpcklwd	%xmm14, %xmm9	 # tmp730, tmp731
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm14	 # tmp568, vect_iftmp.405
	psubd	%xmm11, %xmm14	 # tmp690, vect_iftmp.405
	punpckhwd	%xmm10, %xmm6	 # tmp730, tmp740
	movdqa	%xmm14, %xmm12	 # vect_iftmp.405, vect_iftmp.405
	movdqa	%xmm15, %xmm14	 # tmp568, vect_iftmp.405
	psubd	%xmm3, %xmm14	 # tmp694, vect_iftmp.405
	pand	%xmm9, %xmm12	 # tmp731, tmp734
	pandn	%xmm11, %xmm9	 # tmp690, tmp735
	por	%xmm12, %xmm9	 # tmp734, vect_patt_249.409
	movdqa	%xmm14, %xmm10	 # vect_iftmp.405, vect_iftmp.405
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm9, %xmm5	 # vect_patt_249.409, vect_res_116.410
	movdqa	%xmm13, %xmm12	 # tmp535, tmp787
	movdqa	112(%rsp), %xmm14	 # %sfp, vect_patt_249.409
	pand	%xmm6, %xmm10	 # tmp740, tmp743
	pandn	%xmm3, %xmm6	 # tmp694, tmp744
	por	%xmm10, %xmm6	 # tmp743, vect_patt_249.409
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm0, %xmm3	 # tmp527, tmp747
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm6, %xmm14	 # vect_patt_249.409, vect_patt_249.409
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	16(%rsp), %xmm6	 # %sfp, vect_perm_even_428
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movaps	%xmm14, 112(%rsp)	 # vect_patt_249.409, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	pcmpgtb	%xmm6, %xmm3	 # vect_perm_even_428, tmp747
	movdqa	%xmm6, %xmm9	 # vect_perm_even_428, tmp748
	punpcklbw	%xmm3, %xmm9	 # tmp747, tmp748
	punpckhbw	%xmm3, %xmm6	 # tmp747, vect_perm_even_428
	movdqa	%xmm13, %xmm3	 # tmp535, tmp755
	movdqa	%xmm6, %xmm8	 # vect_perm_even_428, tmp752
	pcmpgtw	%xmm9, %xmm3	 # tmp748, tmp755
	movdqa	%xmm9, %xmm6	 # tmp748, tmp756
	punpcklwd	%xmm3, %xmm6	 # tmp755, tmp756
	punpckhwd	%xmm3, %xmm9	 # tmp755, tmp760
	movdqa	%xmm13, %xmm3	 # tmp535, tmp763
	movdqa	%xmm6, %xmm14	 # tmp756, tmp756
	pcmpgtw	%xmm8, %xmm3	 # tmp752, tmp763
	movdqa	%xmm8, %xmm6	 # tmp752, tmp764
	punpcklwd	%xmm3, %xmm6	 # tmp763, tmp764
	punpckhwd	%xmm3, %xmm8	 # tmp763, tmp768
	movdqa	%xmm0, %xmm3	 # tmp527, tmp773
	movdqa	%xmm6, %xmm11	 # tmp764, tmp764
	pcmpgtb	%xmm1, %xmm3	 # vect_p_119.411, tmp773
	movdqa	%xmm0, %xmm6	 # tmp527, tmp778
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:198: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	paddb	%xmm1, %xmm1	 # vect_p_119.411, vect_p_134.421
	pcmpeqb	%xmm0, %xmm3	 # tmp527, tmp775
	pcmpgtb	%xmm3, %xmm6	 # tmp775, tmp778
	movdqa	%xmm6, %xmm7	 # tmp778, tmp778
	movdqa	%xmm3, %xmm6	 # tmp775, tmp779
	punpcklbw	%xmm7, %xmm6	 # tmp778, tmp779
	pcmpgtw	%xmm6, %xmm12	 # tmp779, tmp787
	movdqa	%xmm6, %xmm10	 # tmp779, tmp788
	punpckhbw	%xmm7, %xmm3	 # tmp778, tmp783
	punpcklwd	%xmm12, %xmm10	 # tmp787, tmp788
	movdqa	%xmm10, %xmm7	 # tmp788, tmp788
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm10	 # tmp568, vect_iftmp.415
	punpckhwd	%xmm12, %xmm6	 # tmp787, tmp797
	psubd	%xmm14, %xmm10	 # tmp756, vect_iftmp.415
	movdqa	%xmm15, %xmm12	 # tmp568, vect_iftmp.415
	pand	%xmm7, %xmm10	 # tmp788, tmp791
	pandn	%xmm14, %xmm7	 # tmp756, tmp792
	por	%xmm10, %xmm7	 # tmp791, vect_patt_252.419
	movdqa	%xmm15, %xmm10	 # tmp568, vect_iftmp.415
	psubd	%xmm9, %xmm10	 # tmp760, vect_iftmp.415
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm7, %xmm4	 # vect_patt_252.419, vect_res_131.420
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	64(%rsp), %xmm7	 # %sfp, vect_perm_odd_423
	pand	%xmm6, %xmm10	 # tmp797, tmp800
	pandn	%xmm9, %xmm6	 # tmp760, tmp801
	por	%xmm10, %xmm6	 # tmp800, vect_patt_252.419
	movdqa	%xmm13, %xmm10	 # tmp535, tmp805
	pcmpgtw	%xmm3, %xmm10	 # tmp783, tmp805
	movdqa	%xmm3, %xmm9	 # tmp783, tmp806
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm11, %xmm12	 # tmp764, vect_iftmp.415
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm6, %xmm2	 # vect_patt_252.419, vect_res_131.420
	punpcklwd	%xmm10, %xmm9	 # tmp805, tmp806
	punpckhwd	%xmm10, %xmm3	 # tmp805, tmp815
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm10	 # tmp568, vect_iftmp.415
	pand	%xmm9, %xmm12	 # tmp806, tmp809
	psubd	%xmm8, %xmm10	 # tmp768, vect_iftmp.415
	pandn	%xmm11, %xmm9	 # tmp764, tmp810
	por	%xmm12, %xmm9	 # tmp809, vect_patt_252.419
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm9, %xmm5	 # vect_patt_252.419, vect_res_131.420
	pand	%xmm3, %xmm10	 # tmp815, tmp818
	pandn	%xmm8, %xmm3	 # tmp768, tmp819
	por	%xmm10, %xmm3	 # tmp818, vect_patt_252.419
	paddd	112(%rsp), %xmm3	 # %sfp, vect_patt_252.419
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm7, %xmm9	 # vect_perm_odd_423, tmp823
	movdqa	%xmm0, %xmm10	 # tmp527, tmp854
	movdqa	%xmm13, %xmm12	 # tmp535, tmp863
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movaps	%xmm3, 112(%rsp)	 # vect_patt_252.419, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm0, %xmm3	 # tmp527, tmp822
	pcmpgtb	%xmm7, %xmm3	 # vect_perm_odd_423, tmp822
	punpckhbw	%xmm3, %xmm7	 # tmp822, vect_perm_odd_423
	punpcklbw	%xmm3, %xmm9	 # tmp822, tmp823
	movdqa	%xmm13, %xmm3	 # tmp535, tmp830
	movdqa	%xmm9, %xmm14	 # tmp823, tmp831
	pcmpgtw	%xmm9, %xmm3	 # tmp823, tmp830
	movdqa	%xmm7, %xmm8	 # vect_perm_odd_423, tmp827
	movdqa	%xmm7, %xmm11	 # tmp827, tmp839
	punpcklwd	%xmm3, %xmm14	 # tmp830, tmp831
	punpckhwd	%xmm3, %xmm9	 # tmp830, tmp835
	movdqa	%xmm13, %xmm3	 # tmp535, tmp838
	pcmpgtw	%xmm7, %xmm3	 # tmp827, tmp838
	punpcklwd	%xmm3, %xmm11	 # tmp838, tmp839
	punpckhwd	%xmm3, %xmm8	 # tmp838, tmp843
	movdqa	%xmm0, %xmm3	 # tmp527, tmp849
	pcmpgtb	%xmm1, %xmm3	 # vect_p_134.421, tmp849
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:198: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	paddb	%xmm1, %xmm1	 # vect_p_134.421, vect_p_149.431
	pcmpeqb	%xmm0, %xmm3	 # tmp527, tmp851
	pcmpgtb	%xmm3, %xmm10	 # tmp851, tmp854
	movdqa	%xmm3, %xmm6	 # tmp851, tmp855
	punpcklbw	%xmm10, %xmm6	 # tmp854, tmp855
	pcmpgtw	%xmm6, %xmm12	 # tmp855, tmp863
	punpckhbw	%xmm10, %xmm3	 # tmp854, tmp859
	movdqa	%xmm6, %xmm10	 # tmp855, tmp864
	punpcklwd	%xmm12, %xmm10	 # tmp863, tmp864
	movdqa	%xmm10, %xmm7	 # tmp864, tmp864
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm10	 # tmp568, vect_iftmp.425
	punpckhwd	%xmm12, %xmm6	 # tmp863, tmp873
	psubd	%xmm14, %xmm10	 # tmp831, vect_iftmp.425
	pand	%xmm7, %xmm10	 # tmp864, tmp867
	pandn	%xmm14, %xmm7	 # tmp831, tmp868
	por	%xmm10, %xmm7	 # tmp867, vect_patt_255.429
	movdqa	%xmm15, %xmm10	 # tmp568, vect_iftmp.425
	psubd	%xmm9, %xmm10	 # tmp835, vect_iftmp.425
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm7, %xmm4	 # vect_patt_255.429, vect_res_146.430
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	48(%rsp), %xmm7	 # %sfp, vect_perm_odd_425
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm14	 # tmp568, vect_iftmp.425
	pand	%xmm6, %xmm10	 # tmp873, tmp876
	pandn	%xmm9, %xmm6	 # tmp835, tmp877
	por	%xmm10, %xmm6	 # tmp876, vect_patt_255.429
	movdqa	%xmm13, %xmm10	 # tmp535, tmp881
	pcmpgtw	%xmm3, %xmm10	 # tmp859, tmp881
	movdqa	%xmm3, %xmm9	 # tmp859, tmp882
	psubd	%xmm11, %xmm14	 # tmp839, vect_iftmp.425
	movdqa	%xmm14, %xmm12	 # vect_iftmp.425, vect_iftmp.425
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm6, %xmm2	 # vect_patt_255.429, vect_res_146.430
	punpcklwd	%xmm10, %xmm9	 # tmp881, tmp882
	punpckhwd	%xmm10, %xmm3	 # tmp881, tmp891
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm10	 # tmp568, vect_iftmp.425
	pand	%xmm9, %xmm12	 # tmp882, tmp885
	psubd	%xmm8, %xmm10	 # tmp843, vect_iftmp.425
	pandn	%xmm11, %xmm9	 # tmp839, tmp886
	por	%xmm12, %xmm9	 # tmp885, vect_patt_255.429
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm9, %xmm5	 # vect_patt_255.429, vect_res_146.430
	pand	%xmm3, %xmm10	 # tmp891, tmp894
	pandn	%xmm8, %xmm3	 # tmp843, tmp895
	por	%xmm10, %xmm3	 # tmp894, vect_patt_255.429
	paddd	112(%rsp), %xmm3	 # %sfp, vect_patt_255.429
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm7, %xmm9	 # vect_perm_odd_425, tmp899
	movdqa	%xmm0, %xmm12	 # tmp527, tmp931
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movaps	%xmm3, 112(%rsp)	 # vect_patt_255.429, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm0, %xmm3	 # tmp527, tmp898
	pcmpgtb	%xmm7, %xmm3	 # vect_perm_odd_425, tmp898
	punpckhbw	%xmm3, %xmm7	 # tmp898, vect_perm_odd_425
	punpcklbw	%xmm3, %xmm9	 # tmp898, tmp899
	movdqa	%xmm13, %xmm3	 # tmp535, tmp906
	movdqa	%xmm9, %xmm14	 # tmp899, tmp907
	pcmpgtw	%xmm9, %xmm3	 # tmp899, tmp906
	movdqa	%xmm7, %xmm8	 # vect_perm_odd_425, tmp903
	movdqa	%xmm7, %xmm11	 # tmp903, tmp915
	punpcklwd	%xmm3, %xmm14	 # tmp906, tmp907
	punpckhwd	%xmm3, %xmm9	 # tmp906, tmp911
	movdqa	%xmm13, %xmm3	 # tmp535, tmp914
	pcmpgtw	%xmm7, %xmm3	 # tmp903, tmp914
	punpcklwd	%xmm3, %xmm11	 # tmp914, tmp915
	punpckhwd	%xmm3, %xmm8	 # tmp914, tmp919
	movdqa	%xmm0, %xmm3	 # tmp527, tmp926
	pcmpgtb	%xmm1, %xmm3	 # vect_p_149.431, tmp926
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:198: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	paddb	%xmm1, %xmm1	 # vect_p_149.431, vect_p_164.441
	pcmpeqb	%xmm0, %xmm3	 # tmp527, tmp928
	pcmpgtb	%xmm3, %xmm12	 # tmp928, tmp931
	movdqa	%xmm3, %xmm6	 # tmp928, tmp932
	punpcklbw	%xmm12, %xmm6	 # tmp931, tmp932
	punpckhbw	%xmm12, %xmm3	 # tmp931, tmp936
	movdqa	%xmm13, %xmm12	 # tmp535, tmp940
	movdqa	%xmm6, %xmm10	 # tmp932, tmp941
	pcmpgtw	%xmm6, %xmm12	 # tmp932, tmp940
	punpcklwd	%xmm12, %xmm10	 # tmp940, tmp941
	movdqa	%xmm10, %xmm7	 # tmp941, tmp941
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm10	 # tmp568, vect_iftmp.435
	punpckhwd	%xmm12, %xmm6	 # tmp940, tmp950
	psubd	%xmm14, %xmm10	 # tmp907, vect_iftmp.435
	movdqa	%xmm15, %xmm12	 # tmp568, vect_iftmp.435
	pand	%xmm7, %xmm10	 # tmp941, tmp944
	pandn	%xmm14, %xmm7	 # tmp907, tmp945
	por	%xmm10, %xmm7	 # tmp944, vect_patt_258.439
	movdqa	%xmm15, %xmm10	 # tmp568, vect_iftmp.435
	psubd	%xmm9, %xmm10	 # tmp911, vect_iftmp.435
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm7, %xmm4	 # vect_patt_258.439, vect_res_161.440
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	32(%rsp), %xmm7	 # %sfp, vect_perm_odd_427
	pand	%xmm6, %xmm10	 # tmp950, tmp953
	pandn	%xmm9, %xmm6	 # tmp911, tmp954
	por	%xmm10, %xmm6	 # tmp953, vect_patt_258.439
	movdqa	%xmm13, %xmm10	 # tmp535, tmp958
	pcmpgtw	%xmm3, %xmm10	 # tmp936, tmp958
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm6, %xmm2	 # vect_patt_258.439, vect_res_161.440
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm0, %xmm6	 # tmp527, tmp975
	pcmpgtb	%xmm7, %xmm6	 # vect_perm_odd_427, tmp975
	movdqa	%xmm3, %xmm9	 # tmp936, tmp959
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm11, %xmm12	 # tmp915, vect_iftmp.435
	punpcklwd	%xmm10, %xmm9	 # tmp958, tmp959
	punpckhwd	%xmm10, %xmm3	 # tmp958, tmp968
	movdqa	%xmm15, %xmm10	 # tmp568, vect_iftmp.435
	pand	%xmm9, %xmm12	 # tmp959, tmp962
	psubd	%xmm8, %xmm10	 # tmp919, vect_iftmp.435
	pandn	%xmm11, %xmm9	 # tmp915, tmp963
	por	%xmm12, %xmm9	 # tmp962, vect_patt_258.439
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm9, %xmm5	 # vect_patt_258.439, vect_res_161.440
	pand	%xmm3, %xmm10	 # tmp968, tmp971
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm7, %xmm9	 # vect_perm_odd_427, tmp976
	pandn	%xmm8, %xmm3	 # tmp919, tmp972
	punpckhbw	%xmm6, %xmm7	 # tmp975, vect_perm_odd_427
	movdqa	%xmm7, %xmm8	 # vect_perm_odd_427, tmp980
	movdqa	%xmm13, %xmm7	 # tmp535, tmp983
	punpcklbw	%xmm6, %xmm9	 # tmp975, tmp976
	movdqa	%xmm9, %xmm14	 # tmp976, tmp984
	pcmpgtw	%xmm9, %xmm7	 # tmp976, tmp983
	movdqa	%xmm8, %xmm11	 # tmp980, tmp992
	por	%xmm10, %xmm3	 # tmp971, vect_patt_258.439
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	112(%rsp), %xmm3	 # %sfp, vect_patt_258.439
	movdqa	%xmm13, %xmm12	 # tmp535, tmp1018
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm10	 # tmp568, vect_iftmp.445
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movaps	%xmm3, 112(%rsp)	 # vect_patt_258.439, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	punpcklwd	%xmm7, %xmm14	 # tmp983, tmp984
	punpckhwd	%xmm7, %xmm9	 # tmp983, tmp988
	movdqa	%xmm13, %xmm7	 # tmp535, tmp991
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm14, %xmm10	 # tmp984, vect_iftmp.445
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	pcmpgtw	%xmm8, %xmm7	 # tmp980, tmp991
	punpcklwd	%xmm7, %xmm11	 # tmp991, tmp992
	punpckhwd	%xmm7, %xmm8	 # tmp991, tmp996
	movdqa	%xmm0, %xmm7	 # tmp527, tmp1004
	pcmpgtb	%xmm1, %xmm7	 # vect_p_164.441, tmp1004
	movdqa	%xmm7, %xmm3	 # tmp1004, tmp1004
	movdqa	%xmm0, %xmm7	 # tmp527, tmp1009
	pcmpeqb	%xmm0, %xmm3	 # tmp527, tmp1006
	pcmpgtb	%xmm3, %xmm7	 # tmp1006, tmp1009
	movdqa	%xmm3, %xmm6	 # tmp1006, tmp1010
	punpcklbw	%xmm7, %xmm6	 # tmp1009, tmp1010
	pcmpgtw	%xmm6, %xmm12	 # tmp1010, tmp1018
	punpckhbw	%xmm7, %xmm3	 # tmp1009, tmp1014
	movdqa	%xmm6, %xmm7	 # tmp1010, tmp1019
	punpcklwd	%xmm12, %xmm7	 # tmp1018, tmp1019
	pand	%xmm7, %xmm10	 # tmp1019, tmp1022
	pandn	%xmm14, %xmm7	 # tmp984, tmp1023
	por	%xmm10, %xmm7	 # tmp1022, vect_patt_261.449
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm10	 # tmp568, vect_iftmp.445
	punpckhwd	%xmm12, %xmm6	 # tmp1018, tmp1028
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm4, %xmm7	 # vect_res_161.440, vect_res_176.450
	movdqa	%xmm13, %xmm12	 # tmp535, tmp1036
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	(%rsp), %xmm4	 # %sfp, vect_perm_odd_429
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm9, %xmm10	 # tmp988, vect_iftmp.445
	pcmpgtw	%xmm3, %xmm12	 # tmp1014, tmp1036
	movdqa	%xmm15, %xmm14	 # tmp568, vect_iftmp.455
	pand	%xmm6, %xmm10	 # tmp1028, tmp1031
	pandn	%xmm9, %xmm6	 # tmp988, tmp1032
	por	%xmm10, %xmm6	 # tmp1031, vect_patt_261.449
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm2, %xmm6	 # vect_res_161.440, vect_res_176.450
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm0, %xmm2	 # tmp527, tmp1053
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm10	 # tmp568, vect_iftmp.445
	movdqa	%xmm3, %xmm9	 # tmp1014, tmp1037
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	pcmpgtb	%xmm4, %xmm2	 # vect_perm_odd_429, tmp1053
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm11, %xmm10	 # tmp992, vect_iftmp.445
	punpcklwd	%xmm12, %xmm9	 # tmp1036, tmp1037
	punpckhwd	%xmm12, %xmm3	 # tmp1036, tmp1046
	pand	%xmm9, %xmm10	 # tmp1037, tmp1040
	pandn	%xmm11, %xmm9	 # tmp992, tmp1041
	por	%xmm10, %xmm9	 # tmp1040, vect_patt_261.449
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm5, %xmm9	 # vect_res_161.440, vect_res_176.450
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm4, %xmm5	 # vect_perm_odd_429, tmp1054
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm10	 # tmp568, vect_iftmp.445
	movdqa	%xmm13, %xmm12	 # tmp535, tmp1097
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	punpcklbw	%xmm2, %xmm5	 # tmp1053, tmp1054
	punpckhbw	%xmm2, %xmm4	 # tmp1053, vect_perm_odd_429
	movdqa	%xmm13, %xmm2	 # tmp535, tmp1061
	movdqa	%xmm5, %xmm11	 # tmp1054, tmp1062
	pcmpgtw	%xmm5, %xmm2	 # tmp1054, tmp1061
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm8, %xmm10	 # tmp996, vect_iftmp.445
	pand	%xmm3, %xmm10	 # tmp1046, tmp1049
	pandn	%xmm8, %xmm3	 # tmp996, tmp1050
	por	%xmm10, %xmm3	 # tmp1049, vect_patt_261.449
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm4, %xmm10	 # tmp1058, tmp1070
	movdqa	%xmm0, %xmm8	 # tmp527, tmp1088
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	112(%rsp), %xmm3	 # %sfp, vect_res_176.450
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	punpcklwd	%xmm2, %xmm11	 # tmp1061, tmp1062
	punpckhwd	%xmm2, %xmm5	 # tmp1061, tmp1066
	movdqa	%xmm13, %xmm2	 # tmp535, tmp1069
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	psubd	%xmm11, %xmm14	 # tmp1062, vect_iftmp.455
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	pcmpgtw	%xmm4, %xmm2	 # tmp1058, tmp1069
	punpcklwd	%xmm2, %xmm10	 # tmp1069, tmp1070
	punpckhwd	%xmm2, %xmm4	 # tmp1069, tmp1074
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:198: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	movdqa	%xmm1, %xmm2	 # vect_p_164.441, vect_p_164.441
	paddb	%xmm1, %xmm2	 # vect_p_164.441, vect_p_164.441
	movdqa	%xmm0, %xmm1	 # tmp527, tmp1083
	pcmpgtb	%xmm2, %xmm1	 # vect_p_179.451, tmp1083
	pcmpeqb	%xmm0, %xmm1	 # tmp527, tmp1085
	pcmpgtb	%xmm1, %xmm8	 # tmp1085, tmp1088
	movdqa	%xmm1, %xmm2	 # tmp1085, tmp1089
	punpcklbw	%xmm8, %xmm2	 # tmp1088, tmp1089
	pcmpgtw	%xmm2, %xmm12	 # tmp1089, tmp1097
	punpckhbw	%xmm8, %xmm1	 # tmp1088, tmp1093
	movdqa	%xmm2, %xmm8	 # tmp1089, tmp1098
	punpcklwd	%xmm12, %xmm8	 # tmp1097, tmp1098
	pand	%xmm8, %xmm14	 # tmp1098, tmp1101
	pandn	%xmm11, %xmm8	 # tmp1062, tmp1102
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm11	 # tmp568, vect_iftmp.455
	psubd	%xmm5, %xmm11	 # tmp1066, vect_iftmp.455
	punpckhwd	%xmm12, %xmm2	 # tmp1097, tmp1107
	movdqa	%xmm15, %xmm12	 # tmp568, vect_iftmp.455
	por	%xmm14, %xmm8	 # tmp1101, vect_patt_264.459
	pand	%xmm2, %xmm11	 # tmp1107, tmp1110
	pandn	%xmm5, %xmm2	 # tmp1066, tmp1111
	por	%xmm11, %xmm2	 # tmp1110, vect_patt_264.459
	movdqa	%xmm13, %xmm11	 # tmp535, tmp1115
	pcmpgtw	%xmm1, %xmm11	 # tmp1093, tmp1115
	movdqa	%xmm1, %xmm5	 # tmp1093, tmp1116
	psubd	%xmm10, %xmm12	 # tmp1070, vect_iftmp.455
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm8, %xmm7	 # vect_patt_264.459, vect_res_176.450
	paddd	%xmm2, %xmm6	 # vect_patt_264.459, vect_res_176.450
	movaps	%xmm7, 112(%rsp)	 # vect_res_176.450, %sfp
	punpcklwd	%xmm11, %xmm5	 # tmp1115, tmp1116
	pand	%xmm5, %xmm12	 # tmp1116, tmp1119
	pandn	%xmm10, %xmm5	 # tmp1070, tmp1120
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movdqa	%xmm15, %xmm10	 # tmp568, vect_iftmp.455
	psubd	%xmm4, %xmm10	 # tmp1074, vect_iftmp.455
	punpckhwd	%xmm11, %xmm1	 # tmp1115, tmp1125
	por	%xmm12, %xmm5	 # tmp1119, vect_patt_264.459
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	paddd	%xmm5, %xmm9	 # vect_patt_264.459, vect_res_176.450
	pand	%xmm1, %xmm10	 # tmp1125, tmp1128
	pandn	%xmm4, %xmm1	 # tmp1074, tmp1129
	por	%xmm10, %xmm1	 # tmp1128, vect_patt_264.459
	paddd	%xmm1, %xmm3	 # vect_patt_264.459, vect_res_176.450
	movaps	%xmm6, 96(%rsp)	 # vect_res_176.450, %sfp
	movdqa	%xmm3, %xmm14	 # vect_res_176.450, vect_res_103.368
	movaps	%xmm9, 80(%rsp)	 # vect_res_176.450, %sfp
	jne	.L149	 #,
	movdqa	%xmm7, %xmm0	 # vect_res_176.450, vect_res_191.460
	paddd	96(%rsp), %xmm0	 # %sfp, vect_res_191.460
	movl	%edi, %ecx	 # tmp1156, niters_vector_mult_vf.364
	paddd	80(%rsp), %xmm0	 # %sfp, tmp1134
	andl	$-16, %ecx	 #, niters_vector_mult_vf.364
	leal	(%r11,%rcx,8), %ebx	 #, tmp.367
	movl	%ecx, %r10d	 # niters_vector_mult_vf.364, _372
	paddd	%xmm3, %xmm0	 # vect_res_103.368, _658
	movdqa	%xmm0, %xmm1	 # _658, tmp1136
	psrldq	$8, %xmm1	 #, tmp1136
	paddd	%xmm1, %xmm0	 # tmp1136, _660
	movdqa	%xmm0, %xmm1	 # _660, tmp1138
	psrldq	$4, %xmm1	 #, tmp1138
	paddd	%xmm1, %xmm0	 # tmp1138, tmp1139
	movd	%xmm0, %edx	 # tmp1139, stmp_res_191.461
	addl	%edx, %eax	 # stmp_res_191.461, <retval>
	leaq	(%r9,%r10,8), %rdx	 #, tmp.365
	addq	%r12, %r10	 # weight, tmp.366
	cmpl	%ecx, %edi	 # niters_vector_mult_vf.364, tmp1156
	je	.L151	 #,
.L148:
	subl	%edx, %ebx	 # _94, tmp1143
	.p2align 4,,10
	.p2align 3
.L154:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movsbl	(%rdx), %r14d	 # MEM[(i8 *)data_297], _304
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:198: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	movzbl	(%r10), %ecx	 # MEM[(b8 *)weight_298], p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	%r14d, %r15d	 # _304, tmp1160
	negl	%r15d	 # tmp1160
	testb	%cl, %cl	 # p
	cmovns	%r15d, %r14d	 # tmp1160,, _304
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addl	%r14d, %eax	 # _304, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movsbl	1(%rdx), %r14d	 # MEM[(i8 *)data_297 + 1B], iftmp.9_345
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	%r14d, %r15d	 # iftmp.9_345, tmp1170
	negl	%r15d	 # tmp1170
	testb	$64, %cl	 #, p
	cmove	%r15d, %r14d	 # tmp1170,, iftmp.9_345
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addl	%eax, %r14d	 # res, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movsbl	2(%rdx), %eax	 # MEM[(i8 *)data_297 + 2B], iftmp.9_338
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	%eax, %r15d	 # iftmp.9_338, tmp1174
	negl	%r15d	 # tmp1174
	testb	$32, %cl	 #, p
	cmove	%r15d, %eax	 # tmp1174,, iftmp.9_338
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addl	%eax, %r14d	 # iftmp.9_338, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movsbl	3(%rdx), %eax	 # MEM[(i8 *)data_297 + 3B], iftmp.9_331
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	%eax, %r15d	 # iftmp.9_331, tmp1166
	negl	%r15d	 # tmp1166
	testb	$16, %cl	 #, p
	cmove	%r15d, %eax	 # tmp1166,, iftmp.9_331
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addl	%eax, %r14d	 # iftmp.9_331, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movsbl	4(%rdx), %eax	 # MEM[(i8 *)data_297 + 4B], iftmp.9_324
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	%eax, %r15d	 # iftmp.9_324, tmp1176
	negl	%r15d	 # tmp1176
	testb	$8, %cl	 #, p
	cmove	%r15d, %eax	 # tmp1176,, iftmp.9_324
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addl	%eax, %r14d	 # iftmp.9_324, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movsbl	5(%rdx), %eax	 # MEM[(i8 *)data_297 + 5B], iftmp.9_317
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	%eax, %r15d	 # iftmp.9_317, tmp1164
	negl	%r15d	 # tmp1164
	testb	$4, %cl	 #, p
	cmove	%r15d, %eax	 # tmp1164,, iftmp.9_317
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addl	%eax, %r14d	 # iftmp.9_317, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movsbl	6(%rdx), %eax	 # MEM[(i8 *)data_297 + 6B], iftmp.9_310
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	%eax, %r15d	 # iftmp.9_310, tmp1172
	negl	%r15d	 # tmp1172
	testb	$2, %cl	 #, p
	cmove	%r15d, %eax	 # tmp1172,, iftmp.9_310
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addl	%r14d, %eax	 # res, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movsbl	7(%rdx), %r14d	 # MEM[(i8 *)data_297 + 7B], iftmp.9_305
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	movl	%r14d, %r15d	 # iftmp.9_305, tmp1162
	negl	%r15d	 # tmp1162
	andl	$1, %ecx	 #, p
	cmove	%r15d, %r14d	 # tmp1162,, iftmp.9_305
	addq	$8, %rdx	 #, _94
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:197: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	addq	$1, %r10	 #, tmp.366
	leal	(%rbx,%rdx), %ecx	 #, _85
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:199: 			res += (p & 128) ? *data : -*data;   // 如果bit为1，则累加对应的浮点数
	addl	%r14d, %eax	 # iftmp.9_305, <retval>
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:197: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	cmpl	%ecx, %esi	 # _85, _10
	jg	.L154	 #,
.L151:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:197: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	leal	8(%r11,%r13,8), %r11d	 #, n0
	movl	%edi, %edi	 # tmp1156, _123
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:198: 		for (j = 0, p = *weight; j < 8; j++, data++, p <<= 1)  // 遍历8位
	leaq	(%r9,%rdi,8), %r9	 #, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:197: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	addq	%rdi, %r12	 # _123, weight
.L147:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:202: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	cmpl	%r11d, %r8d	 # n0, n
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:202: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	movzbl	(%r12), %ecx	 # *weight_62, p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:202: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	jle	.L143	 #,
	subl	$1, %r8d	 #, tmp1151
	subl	%r11d, %r8d	 # n0, tmp1153
	leaq	1(%r9,%r8), %r10	 #, _11
	.p2align 4,,10
	.p2align 3
.L164:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:203: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	movsbl	(%r9), %edx	 # MEM[(i8 *)data_78], iftmp.11_49
	movl	%edx, %r8d	 # iftmp.11_49, tmp1168
	negl	%r8d	 # tmp1168
	testb	%cl, %cl	 # p
	cmovns	%r8d, %edx	 # tmp1168,, iftmp.11_49
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:202: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	addq	$1, %r9	 #, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:202: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	addl	%ecx, %ecx	 # p
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:203: 		res += (p & 128) ? *data : -*data;  // 如果bit为1，则累加对应的浮点数
	addl	%edx, %eax	 # iftmp.11_49, <retval>
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:202: 	for (p = *weight; i < n; i++, data++, p <<= 1)
	cmpq	%r9, %r10	 # data, _11
	jne	.L164	 #,
.L143:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:206: }
	movaps	128(%rsp), %xmm6	 #,
	movaps	144(%rsp), %xmm7	 #,
	movaps	160(%rsp), %xmm8	 #,
	movaps	176(%rsp), %xmm9	 #,
	movaps	192(%rsp), %xmm10	 #,
	movaps	208(%rsp), %xmm11	 #,
	movaps	224(%rsp), %xmm12	 #,
	movaps	240(%rsp), %xmm13	 #,
	movaps	256(%rsp), %xmm14	 #,
	movaps	272(%rsp), %xmm15	 #,
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
.L165:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:193: 	for (; i < n0; i++, data++, p <<= 1)
	movq	%rbx, %r9	 # data, data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:187: 	i32 res = 0;  // 初始化结果为0, 结果要除以127,解析成(-1,1)的8字节数
	xorl	%eax, %eax	 # <retval>
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:186: 	int i = 0, j=0;
	xorl	%r11d, %r11d	 # n0
	jmp	.L144	 #
.L166:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:197: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	movl	%r11d, %ebx	 # n0, tmp.367
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:197: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	movq	%r12, %r10	 # weight, tmp.366
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:197: 	for (weight += 1; i < n - 7; i += 8, weight++)  // 每次处理1字节（8位）
	movq	%r9, %rdx	 # data, tmp.365
	jmp	.L148	 #
	.p2align 4
	.globl	_Z7glinearPfS_S_S_iiii
	.def	_Z7glinearPfS_S_S_iiii;	.scl	2;	.type	32;	.endef
_Z7glinearPfS_S_S_iiii:
	pushq	%rbp	 #
	movq	%rdx, %r11	 # tmp190, out_data
	movq	%rsp, %rbp	 #,
	pushq	%r15	 #
	pushq	%r14	 #
	pushq	%r13	 #
	pushq	%r12	 #
	pushq	%rdi	 #
	pushq	%rsi	 #
	pushq	%rbx	 #
	andq	$-16, %rsp	 #,
	subq	$64, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:8:              int length,  int c_in, int c_out,  int group ) {
	movl	72(%rbp), %esi	 # group, group
	movq	%r8, 40(%rsp)	 # tmp191, %sfp
	movslq	56(%rbp), %r8	 # c_in,
	movl	48(%rbp), %ebx	 # length, length
	movl	%esi, 32(%rsp)	 # group, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:15: 	int g_in = c_in / group;
	movl	%r8d, %eax	 # c_in, tmp151
	cltd
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:8:              int length,  int c_in, int c_out,  int group ) {
	movl	%ebx, 8(%rsp)	 # length, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:15: 	int g_in = c_in / group;
	idivl	%esi	 # group
	movl	%eax, %edi	 # tmp151, tmp151
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:16: 	int g_out = c_out / group;
	movl	64(%rbp), %eax	 # c_out, tmp153
	cltd
	idivl	%esi	 # group
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:17: 	if (group == 1) {
	cmpl	$1, %esi	 #, group
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:16: 	int g_out = c_out / group;
	movl	%eax, 36(%rsp)	 # tmp153, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:17: 	if (group == 1) {
	je	.L175	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:20: 		for (i = 0; i < length; i++, in_data += c_in) // 遍历length
	testl	%ebx, %ebx	 # length
	jle	.L174	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:20: 		for (i = 0; i < length; i++, in_data += c_in) // 遍历length
	leaq	0(,%r8,4), %rbx	 #, _6
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:24: 				for (j = 0; j < g_out; j++, weight += g_in, out_data++) { // 组内遍历
	movslq	%edi, %rax	 # tmp151, g_in
	testl	%esi, %esi	 # group
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:24: 				for (j = 0; j < g_out; j++, weight += g_in, out_data++) { // 组内遍历
	leaq	0(,%rax,4), %r15	 #, _4
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:20: 		for (i = 0; i < length; i++, in_data += c_in) // 遍历length
	movq	%rbx, (%rsp)	 # _6, %sfp
	jle	.L174	 #,
	movl	36(%rsp), %eax	 # %sfp, tmp153
	movl	%edi, %edx	 # tmp151, bnd.501
	movl	%edi, %r12d	 # tmp151, niters_vector_mult_vf.502
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:20: 		for (i = 0; i < length; i++, in_data += c_in) // 遍历length
	movl	$0, 12(%rsp)	 #, %sfp
	shrl	$2, %edx	 #, bnd.501
	andl	$-4, %r12d	 #, niters_vector_mult_vf.502
	pxor	%xmm3, %xmm3	 # res
	leal	-1(%rdx), %r10d	 #, tmp161
	movl	%r12d, %edx	 # niters_vector_mult_vf.502, niters_vector_mult_vf.502
	leaq	0(,%rdx,4), %r14	 #, _92
	addq	$1, %r10	 #, tmp162
	subl	$1, %eax	 #, tmp158
	salq	$4, %r10	 #, _132
	addq	$1, %rax	 #, _9
	leaq	0(,%rax,4), %rsi	 #, _152
	imulq	%r15, %rax	 # _4, _9
	movq	%rsi, 16(%rsp)	 # _152, %sfp
	leal	-1(%rdi), %esi	 #, _59
	movl	%esi, 60(%rsp)	 # _59, %sfp
	movq	%rax, 24(%rsp)	 # _9, %sfp
.L178:
	leaq	(%rcx,%r14), %rax	 #, tmp.503
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:21: 			for (g = 0; g < group; g++ ) { // 遍历分组数量
	movl	$0, 56(%rsp)	 #, %sfp
	movq	%rax, 48(%rsp)	 # tmp.503, %sfp
	.p2align 4,,10
	.p2align 3
.L190:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:24: 				for (j = 0; j < g_out; j++, weight += g_in, out_data++) { // 组内遍历
	movl	36(%rsp), %eax	 # %sfp,
	testl	%eax, %eax	 #
	jle	.L188	 #,
	movq	16(%rsp), %rax	 # %sfp, _152
	leaq	(%r11,%rax), %r13	 #, _158
	movq	40(%rsp), %rax	 # %sfp, weight
	.p2align 4,,10
	.p2align 3
.L189:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:105: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	testl	%edi, %edi	 # tmp151
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104: 	f32 res = 0;  // 初始化结果为0
	movaps	%xmm3, %xmm1	 # res, res
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:105: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	jle	.L186	 #,
	cmpl	$2, 60(%rsp)	 #, %sfp
	jbe	.L191	 #,
	xorl	%edx, %edx	 # ivtmp.532
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104: 	f32 res = 0;  // 初始化结果为0
	movaps	%xmm3, %xmm1	 # res, res
	.p2align 4,,10
	.p2align 3
.L180:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movq	(%rcx,%rdx), %xmm0	 # MEM <vector(4) float> [(f32 *)in_data_65 + ivtmp.532_64 * 1], vect__41.508
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movq	(%rax,%rdx), %xmm2	 # MEM <vector(4) float> [(f32 *)weight_72 + ivtmp.532_64 * 1], vect__43.511
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movhps	8(%rcx,%rdx), %xmm0	 # MEM <vector(4) float> [(f32 *)in_data_65 + ivtmp.532_64 * 1], vect__41.508
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movhps	8(%rax,%rdx), %xmm2	 # MEM <vector(4) float> [(f32 *)weight_72 + ivtmp.532_64 * 1], vect__43.511
	addq	$16, %rdx	 #, ivtmp.532
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	mulps	%xmm2, %xmm0	 # vect__43.511, vect__44.512
	cmpq	%r10, %rdx	 # _132, ivtmp.532
	addss	%xmm0, %xmm1	 # stmp_res_46.513, stmp_res_46.513
	movaps	%xmm0, %xmm2	 # vect__44.512, tmp166
	shufps	$85, %xmm0, %xmm2	 #, vect__44.512, tmp166
	addss	%xmm2, %xmm1	 # stmp_res_46.513, stmp_res_46.513
	movaps	%xmm0, %xmm2	 # vect__44.512, tmp167
	unpckhps	%xmm0, %xmm2	 # vect__44.512, tmp167
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	shufps	$255, %xmm0, %xmm0	 #, vect__44.512, tmp170
	addss	%xmm2, %xmm1	 # stmp_res_46.513, stmp_res_46.513
	addss	%xmm0, %xmm1	 # stmp_res_46.513, res
	jne	.L180	 #,
	leaq	(%rax,%r14), %rdx	 #, tmp.518
	cmpl	%r12d, %edi	 # niters_vector_mult_vf.502, tmp151
	je	.L186	 #,
	movq	48(%rsp), %r8	 # %sfp, tmp.517
	movl	%r12d, %r9d	 # niters_vector_mult_vf.502,
.L179:
	movl	%edi, %ebx	 # tmp151, niters.514
	subl	%r9d, %ebx	 # _98, niters.514
	cmpl	$1, %ebx	 #, niters.514
	je	.L182	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movq	(%rcx,%r9,4), %xmm2	 # MEM <vector(2) float> [(f32 *)vectp_in_data.521_149], vect__15.522
	movl	%ebx, %esi	 # niters.514, niters_vector_mult_vf.516
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movq	(%rax,%r9,4), %xmm0	 # MEM <vector(2) float> [(f32 *)vectp_weight.524_155], vect__11.525
	andl	$-2, %esi	 #, niters_vector_mult_vf.516
	movl	%esi, %r9d	 # niters_vector_mult_vf.516, niters_vector_mult_vf.516
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	mulps	%xmm2, %xmm0	 # vect__15.522, vect__8.526
	salq	$2, %r9	 #, _142
	addq	%r9, %r8	 # _142, tmp.517
	addq	%r9, %rdx	 # _142, tmp.518
	cmpl	%esi, %ebx	 # niters_vector_mult_vf.516, niters.514
	addss	%xmm0, %xmm1	 # stmp_res_45.527, stmp_res_45.527
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movshdup	%xmm0, %xmm0	 # vect__8.526, stmp_res_45.527
	addss	%xmm0, %xmm1	 # stmp_res_45.527, res
	je	.L186	 #,
.L182:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	movss	(%r8), %xmm0	 # *data_134, *data_134
	mulss	(%rdx), %xmm0	 # *weight_135, tmp176
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:106: 		res += (*data) * (*weight);  // 累加乘积
	addss	%xmm0, %xmm1	 # tmp176, res
.L186:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:25: 					*out_data = f32_f32_vec_in_prod(in_data, weight, g_in);
	movss	%xmm1, (%r11)	 # res, MEM[(f32 *)out_data_67]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:24: 				for (j = 0; j < g_out; j++, weight += g_in, out_data++) { // 组内遍历
	addq	$4, %r11	 #, out_data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:24: 				for (j = 0; j < g_out; j++, weight += g_in, out_data++) { // 组内遍历
	addq	%r15, %rax	 # _4, weight
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:24: 				for (j = 0; j < g_out; j++, weight += g_in, out_data++) { // 组内遍历
	cmpq	%r13, %r11	 # _158, out_data
	jne	.L189	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:24: 				for (j = 0; j < g_out; j++, weight += g_in, out_data++) { // 组内遍历
	movq	24(%rsp), %rbx	 # %sfp, _19
	addq	%rbx, 40(%rsp)	 # _19, %sfp
.L188:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:21: 			for (g = 0; g < group; g++ ) { // 遍历分组数量
	addl	$1, 56(%rsp)	 #, %sfp
	movl	56(%rsp), %eax	 # %sfp, g
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:21: 			for (g = 0; g < group; g++ ) { // 遍历分组数量
	cmpl	%eax, 32(%rsp)	 # g, %sfp
	jne	.L190	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:20: 		for (i = 0; i < length; i++, in_data += c_in) // 遍历length
	addl	$1, 12(%rsp)	 #, %sfp
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:20: 		for (i = 0; i < length; i++, in_data += c_in) // 遍历length
	addq	(%rsp), %rcx	 # %sfp, in_data
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:20: 		for (i = 0; i < length; i++, in_data += c_in) // 遍历length
	movl	12(%rsp), %eax	 # %sfp, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:20: 		for (i = 0; i < length; i++, in_data += c_in) // 遍历length
	cmpl	%eax, 8(%rsp)	 # i, %sfp
	jne	.L178	 #,
.L174:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:30: }
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
.L191:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:105: 	for (; i < n; i++, data++, weight++)  // 遍历向量元素
	movq	%rax, %rdx	 # weight, tmp.518
	movq	%rcx, %r8	 # in_data, tmp.517
	xorl	%r9d, %r9d	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32matrix.h:104: 	f32 res = 0;  // 初始化结果为0
	movaps	%xmm3, %xmm1	 # res, res
	jmp	.L179	 #
.L175:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:18: 		f32_f32_mat_mul(in_data, weight, out_data, length, c_out, c_in);
	movl	%edi, 56(%rbp)	 # tmp151,
	movq	40(%rsp), %rdx	 # %sfp,
	movl	%ebx, %r9d	 # length,
	movq	%r11, %r8	 # out_data,
	movl	$0, 64(%rbp)	 #,
	movl	%eax, 48(%rbp)	 # tmp153,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:30: }
	leaq	-56(%rbp), %rsp	 #,
	popq	%rbx	 #
	popq	%rsi	 #
	popq	%rdi	 #
	popq	%r12	 #
	popq	%r13	 #
	popq	%r14	 #
	popq	%r15	 #
	popq	%rbp	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:18: 		f32_f32_mat_mul(in_data, weight, out_data, length, c_out, c_in);
	jmp	_Z15f32_f32_mat_mulPfS_S_iiib	 #
	.p2align 4
	.globl	_Z12f32topdinstyPfii
	.def	_Z12f32topdinstyPfii;	.scl	2;	.type	32;	.endef
_Z12f32topdinstyPfii:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:33: 	for (int i = 0; i < length; i++) {
	testl	%edx, %edx	 # length
	jle	.L218	 #,
	movl	%r8d, %eax	 # channel, bnd.573
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:32: void f32topdinsty(f32 *in_data, int length,  int channel) {
	pushq	%rbp	 #
	movq	%rcx, %r9	 # tmp228, in_data
	movl	%r8d, %r11d	 # channel, niters_vector_mult_vf.574
	shrl	$2, %eax	 #, bnd.573
	movq	%rsp, %rbp	 #,
	pushq	%r14	 #
	andl	$-4, %r11d	 #, niters_vector_mult_vf.574
	pushq	%r13	 #
	leal	-1(%r8), %r13d	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:33: 	for (int i = 0; i < length; i++) {
	xorl	%ecx, %ecx	 # ivtmp.601
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:33: 	for (int i = 0; i < length; i++) {
	xorl	%r10d, %r10d	 # i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:32: void f32topdinsty(f32 *in_data, int length,  int channel) {
	pushq	%r12	 #
	leaq	4(%r9), %r12	 #, tmp222
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:34: 		f32 e = 0;
	pxor	%xmm3, %xmm3	 # s
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:32: void f32topdinsty(f32 *in_data, int length,  int channel) {
	pushq	%rdi	 #
	movl	%edx, %edi	 # tmp229, length
	pushq	%rsi	 #
	movq	%r13, %rsi	 #,
	pushq	%rbx	 #
	leal	-1(%rax), %ebx	 #, tmp177
	salq	$4, %rbx	 #, _156
	andq	$-16, %rsp	 #,
	addq	$16, %rbx	 #, tmp224
	.p2align 4,,10
	.p2align 3
.L200:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:38: 		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
	testl	%r8d, %r8d	 # channel
	jle	.L211	 #,
	cmpl	$2, %esi	 #, _101
	jbe	.L213	 #,
	movslq	%ecx, %r14	 # ivtmp.601, _61
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:34: 		f32 e = 0;
	movaps	%xmm3, %xmm4	 # s, s
	movaps	%xmm3, %xmm1	 # s, e
	leaq	(%r9,%r14,4), %rax	 #, ivtmp.594
	leaq	(%rbx,%rax), %rdx	 #, _159
	.p2align 4,,10
	.p2align 3
.L202:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:38: 		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
	movq	(%rax), %xmm0	 # MEM <vector(4) float> [(f32 *)_153], tmp181
	addq	$16, %rax	 #, ivtmp.594
	movhps	-8(%rax), %xmm0	 # MEM <vector(4) float> [(f32 *)_153], tmp181
	cmpq	%rdx, %rax	 # _159, ivtmp.594
	addss	%xmm0, %xmm1	 # stmp_e_47.579, stmp_e_47.579
	movaps	%xmm0, %xmm2	 # tmp181, tmp182
	shufps	$85, %xmm0, %xmm2	 #, tmp181, tmp182
	addss	%xmm2, %xmm1	 # stmp_e_47.579, stmp_e_47.579
	movaps	%xmm0, %xmm2	 # tmp181, tmp183
	unpckhps	%xmm0, %xmm2	 # tmp181, tmp183
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:38: 		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
	shufps	$255, %xmm0, %xmm0	 #, tmp181, tmp186
	addss	%xmm2, %xmm1	 # stmp_e_47.579, stmp_e_47.579
	addss	%xmm0, %xmm1	 # stmp_e_47.579, e
	jne	.L202	 #,
	cmpl	%r8d, %r11d	 # channel, niters_vector_mult_vf.574
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:38: 		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
	movl	%r11d, %eax	 # niters_vector_mult_vf.574, j
	je	.L203	 #,
.L201:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:38: 		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
	leal	(%rcx,%rax), %edx	 #, tmp187
	movslq	%edx, %rdx	 # tmp187, tmp188
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:38: 		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
	addss	(%r9,%rdx,4), %xmm1	 # *_126, e
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:38: 		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
	leal	1(%rax), %edx	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:38: 		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
	cmpl	%edx, %r8d	 # j, channel
	jle	.L203	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:38: 		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
	addl	%ecx, %edx	 # ivtmp.601, tmp189
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:38: 		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
	addl	$2, %eax	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:38: 		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
	movslq	%edx, %rdx	 # tmp189, tmp190
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:38: 		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
	cmpl	%eax, %r8d	 # j, channel
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:38: 		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
	addss	(%r9,%rdx,4), %xmm1	 # *_135, e
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:38: 		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
	jle	.L203	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:38: 		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
	addl	%ecx, %eax	 # ivtmp.601, tmp191
	cltq
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:38: 		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
	addss	(%r9,%rax,4), %xmm1	 # *_95, e
.L203:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:39: 		e  = e / channel;
	pxor	%xmm0, %xmm0	 # tmp213
	cvtsi2ssl	%r8d, %xmm0	 # channel, tmp213
	leaq	(%r9,%r14,4), %rdx	 #, ivtmp.586
	addq	%r13, %r14	 # _101, tmp216
	leaq	(%r12,%r14,4), %r14	 #, _149
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:39: 		e  = e / channel;
	movq	%rdx, %rax	 # ivtmp.586, ivtmp.590
	movaps	%xmm4, %xmm2	 # s, s
	divss	%xmm0, %xmm1	 # tmp213, e
	.p2align 4,,10
	.p2align 3
.L212:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:43: 			if (in_data[i * channel + j] <= e) {
	movss	(%rax), %xmm0	 # MEM[(f32 *)_139], _13
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:43: 			if (in_data[i * channel + j] <= e) {
	comiss	%xmm0, %xmm1	 # _13, e
	jnb	.L204	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:46: 				in_data[i * channel + j] -= e;
	subss	%xmm1, %xmm0	 # e, _14
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:42: 		for (int j = 0; j < channel; j++) {
	addq	$4, %rax	 #, ivtmp.590
	movss	%xmm0, -4(%rax)	 # _14, MEM[(f32 *)_139]
	cmpq	%r14, %rax	 # _149, ivtmp.590
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:47: 				s += in_data[i * channel + j];
	addss	%xmm0, %xmm2	 # _14, s
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:42: 		for (int j = 0; j < channel; j++) {
	jne	.L212	 #,
	cmpl	$2, %esi	 #, _101
	jbe	.L214	 #,
.L221:
	leaq	(%rbx,%rdx), %rax	 #, _65
	movaps	%xmm2, %xmm1	 # s, vect_cst__31
	shufps	$0, %xmm1, %xmm1	 # vect_cst__31
	.p2align 4,,10
	.p2align 3
.L208:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:53: 			in_data[i * channel + j] /= s;
	movq	(%rdx), %xmm0	 # MEM <vector(4) float> [(f32 *)_15], vect__20.568
	addq	$16, %rdx	 #, ivtmp.586
	movhps	-8(%rdx), %xmm0	 # MEM <vector(4) float> [(f32 *)_15], vect__20.568
	divps	%xmm1, %xmm0	 # vect_cst__31, vect__23.569
	movlps	%xmm0, -16(%rdx)	 # vect__23.569, MEM <vector(4) float> [(f32 *)_15]
	movhps	%xmm0, -8(%rdx)	 # vect__23.569, MEM <vector(4) float> [(f32 *)_15]
	cmpq	%rdx, %rax	 # ivtmp.586, _65
	jne	.L208	 #,
	cmpl	%r8d, %r11d	 # channel, niters_vector_mult_vf.574
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:53: 			in_data[i * channel + j] /= s;
	movl	%r11d, %eax	 # niters_vector_mult_vf.574, j
	je	.L211	 #,
.L207:
	leal	(%rcx,%rax), %edx	 #, tmp198
	movslq	%edx, %rdx	 # tmp198, tmp199
	leaq	(%r9,%rdx,4), %rdx	 #, _6
	movss	(%rdx), %xmm0	 # *_6, *_6
	divss	%xmm2, %xmm0	 # s, tmp201
	movss	%xmm0, (%rdx)	 # tmp201, *_6
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:52: 		for (int j = 0; j < channel; j++) {
	leal	1(%rax), %edx	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:52: 		for (int j = 0; j < channel; j++) {
	cmpl	%r8d, %edx	 # channel, j
	jge	.L211	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:53: 			in_data[i * channel + j] /= s;
	addl	%ecx, %edx	 # ivtmp.601, tmp203
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:52: 		for (int j = 0; j < channel; j++) {
	addl	$2, %eax	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:53: 			in_data[i * channel + j] /= s;
	movslq	%edx, %rdx	 # tmp203, tmp204
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:52: 		for (int j = 0; j < channel; j++) {
	cmpl	%eax, %r8d	 # j, channel
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:53: 			in_data[i * channel + j] /= s;
	leaq	(%r9,%rdx,4), %rdx	 #, _71
	movss	(%rdx), %xmm0	 # *_71, *_71
	divss	%xmm2, %xmm0	 # s, tmp206
	movss	%xmm0, (%rdx)	 # tmp206, *_71
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:52: 		for (int j = 0; j < channel; j++) {
	jle	.L211	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:53: 			in_data[i * channel + j] /= s;
	addl	%ecx, %eax	 # ivtmp.601, tmp208
	cltq
	leaq	(%r9,%rax,4), %rax	 #, _76
	movss	(%rax), %xmm0	 # *_76, *_76
	divss	%xmm2, %xmm0	 # s, tmp211
	movss	%xmm0, (%rax)	 # tmp211, *_76
.L211:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:33: 	for (int i = 0; i < length; i++) {
	addl	$1, %r10d	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:33: 	for (int i = 0; i < length; i++) {
	addl	%r8d, %ecx	 # channel, ivtmp.601
	cmpl	%r10d, %edi	 # i, length
	jne	.L200	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:56: }
	leaq	-48(%rbp), %rsp	 #,
	popq	%rbx	 #
	popq	%rsi	 #
	popq	%rdi	 #
	popq	%r12	 #
	popq	%r13	 #
	popq	%r14	 #
	popq	%rbp	 #
	ret	
	.p2align 4,,10
	.p2align 3
.L204:
	movl	$0x00000000, (%rax)	 #, MEM[(f32 *)_139]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:42: 		for (int j = 0; j < channel; j++) {
	addq	$4, %rax	 #, ivtmp.590
	cmpq	%r14, %rax	 # _149, ivtmp.590
	jne	.L212	 #,
	cmpl	$2, %esi	 #, _101
	ja	.L221	 #,
.L214:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:53: 			in_data[i * channel + j] /= s;
	xorl	%eax, %eax	 # j
	jmp	.L207	 #
.L213:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:38: 		for (int j = 0; j < channel; j++)			e += in_data[i * channel + j];
	xorl	%eax, %eax	 # j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:34: 		f32 e = 0;
	movaps	%xmm3, %xmm4	 # s, s
	movaps	%xmm3, %xmm1	 # s, e
	movslq	%ecx, %r14	 # ivtmp.601, _61
	jmp	.L201	 #
.L218:
	ret	
	.p2align 4
	.globl	_Z12i32topdinstyPiii
	.def	_Z12i32topdinstyPiii;	.scl	2;	.type	32;	.endef
_Z12i32topdinstyPiii:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	pushq	%r15	 #
	pushq	%r14	 #
	pushq	%r13	 #
	pushq	%r12	 #
	pushq	%rdi	 #
	pushq	%rsi	 #
	pushq	%rbx	 #
	andq	$-16, %rsp	 #,
	subq	$16, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:60: 	for (int i = 0; i < length; i++) {
	testl	%edx, %edx	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:58: void i32topdinsty(i32 *io_data, int length,  int channel) {
	movl	%edx, 24(%rbp)	 # tmp250, length
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:60: 	for (int i = 0; i < length; i++) {
	jle	.L222	 #,
	movl	%r8d, %eax	 # channel, bnd.626
	movq	%rcx, %r9	 # tmp249, io_data
	movl	%r8d, %ebx	 # channel, niters_vector_mult_vf.627
	shrl	$2, %eax	 #, bnd.626
	andl	$-4, %ebx	 #, niters_vector_mult_vf.627
	xorl	%r10d, %r10d	 # ivtmp.656
	leal	-1(%r8), %ecx	 #,
	subl	$1, %eax	 #, tmp185
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:60: 	for (int i = 0; i < length; i++) {
	xorl	%esi, %esi	 # i
	movq	%rcx, %r12	 #,
	salq	$4, %rax	 #, _171
	movq	%rcx, 8(%rsp)	 # _148, %sfp
	leaq	4(%r9), %rcx	 #, tmp245
	movq	%rcx, (%rsp)	 # tmp245, %sfp
	leaq	16(%rax), %r13	 #, tmp248
	.p2align 4,,10
	.p2align 3
.L224:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:65: 		for (int j = 0; j < channel; j++) e += io_data[i * channel + j];
	testl	%r8d, %r8d	 # channel
	jle	.L238	 #,
	cmpl	$2, %r12d	 #, _148
	jbe	.L241	 #,
	movslq	%r10d, %r11	 # ivtmp.656, _50
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:65: 		for (int j = 0; j < channel; j++) e += io_data[i * channel + j];
	pxor	%xmm0, %xmm0	 # vect_e_47.633
	leaq	(%r9,%r11,4), %rax	 #, ivtmp.649
	leaq	0(%r13,%rax), %rdx	 #, _174
	.p2align 4,,10
	.p2align 3
.L226:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:65: 		for (int j = 0; j < channel; j++) e += io_data[i * channel + j];
	movdqu	(%rax), %xmm4	 # MEM <vector(4) int> [(i32 *)_60], tmp295
	addq	$16, %rax	 #, ivtmp.649
	cmpq	%rdx, %rax	 # _174, ivtmp.649
	paddd	%xmm4, %xmm0	 # tmp295, vect_e_47.633
	jne	.L226	 #,
	movdqa	%xmm0, %xmm1	 # vect_e_47.633, tmp191
	cmpl	%r8d, %ebx	 # channel, niters_vector_mult_vf.627
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:65: 		for (int j = 0; j < channel; j++) e += io_data[i * channel + j];
	movl	%ebx, %edx	 # niters_vector_mult_vf.627, j
	psrldq	$8, %xmm1	 #, tmp191
	paddd	%xmm1, %xmm0	 # tmp191, _165
	movdqa	%xmm0, %xmm1	 # _165, tmp193
	psrldq	$4, %xmm1	 #, tmp193
	paddd	%xmm1, %xmm0	 # tmp193, tmp194
	movd	%xmm0, %eax	 # tmp194, stmp_e_47.634
	je	.L250	 #,
.L225:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:65: 		for (int j = 0; j < channel; j++) e += io_data[i * channel + j];
	leal	(%r10,%rdx), %ecx	 #, tmp195
	movslq	%ecx, %rcx	 # tmp195, tmp196
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:65: 		for (int j = 0; j < channel; j++) e += io_data[i * channel + j];
	addl	(%r9,%rcx,4), %eax	 # *_74, e
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:65: 		for (int j = 0; j < channel; j++) e += io_data[i * channel + j];
	leal	1(%rdx), %ecx	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:65: 		for (int j = 0; j < channel; j++) e += io_data[i * channel + j];
	cmpl	%ecx, %r8d	 # j, channel
	jle	.L228	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:65: 		for (int j = 0; j < channel; j++) e += io_data[i * channel + j];
	addl	%r10d, %ecx	 # ivtmp.656, tmp198
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:65: 		for (int j = 0; j < channel; j++) e += io_data[i * channel + j];
	addl	$2, %edx	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:65: 		for (int j = 0; j < channel; j++) e += io_data[i * channel + j];
	movslq	%ecx, %rcx	 # tmp198, tmp199
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:65: 		for (int j = 0; j < channel; j++) e += io_data[i * channel + j];
	addl	(%r9,%rcx,4), %eax	 # *_80, e
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:65: 		for (int j = 0; j < channel; j++) e += io_data[i * channel + j];
	cmpl	%edx, %r8d	 # j, channel
	jle	.L228	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:65: 		for (int j = 0; j < channel; j++) e += io_data[i * channel + j];
	addl	%r10d, %edx	 # ivtmp.656, tmp203
	movslq	%edx, %rdx	 # tmp203, tmp204
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:65: 		for (int j = 0; j < channel; j++) e += io_data[i * channel + j];
	addl	(%r9,%rdx,4), %eax	 # *_142, e
.L228:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:66: 		e  = e / channel;
	cltd
	idivl	%r8d	 # channel
	cmpl	$2, %r12d	 #, _148
	jbe	.L251	 #,
.L240:
	movd	%eax, %xmm5	 # e, e
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:71: 				io_data[i * channel + j] = 0;
	pxor	%xmm1, %xmm1	 # vect_s_26.620
	pshufd	$0, %xmm5, %xmm2	 # e, vect_cst__109
	leaq	(%r9,%r11,4), %rdx	 #, ivtmp.643
	leaq	0(%r13,%rdx), %rcx	 #, _134
	.p2align 4,,10
	.p2align 3
.L234:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:70: 			if (io_data[i * channel + j] <= e) {
	movdqu	(%rdx), %xmm0	 # MEM <vector(4) int> [(i32 *)_10], MEM <vector(4) int> [(i32 *)_10]
	addq	$16, %rdx	 #, ivtmp.643
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:73: 				io_data[i * channel + j] -= e;
	movdqa	%xmm0, %xmm3	 # MEM <vector(4) int> [(i32 *)_10], vect__13.618
	pcmpgtd	%xmm2, %xmm0	 # vect_cst__109, tmp227
	psubd	%xmm2, %xmm3	 # vect_cst__109, vect__13.618
	pand	%xmm3, %xmm0	 # vect__13.618, vect__ifc__65.619
	movups	%xmm0, -16(%rdx)	 # vect__ifc__65.619, MEM <vector(4) int> [(i32 *)_10]
	cmpq	%rcx, %rdx	 # _134, ivtmp.643
	paddd	%xmm0, %xmm1	 # vect__ifc__65.619, vect_s_26.620
	jne	.L234	 #,
	movdqa	%xmm1, %xmm0	 # vect_s_26.620, tmp209
	cmpl	%r8d, %ebx	 # channel, niters_vector_mult_vf.627
	movl	%ebx, %edx	 # niters_vector_mult_vf.627, j
	psrldq	$8, %xmm0	 #, tmp209
	paddd	%xmm0, %xmm1	 # tmp209, _118
	movdqa	%xmm1, %xmm0	 # _118, tmp211
	psrldq	$4, %xmm0	 #, tmp211
	paddd	%xmm0, %xmm1	 # tmp211, tmp212
	movd	%xmm1, %ecx	 # tmp212, stmp_s_26.621
	je	.L229	 #,
.L239:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:70: 			if (io_data[i * channel + j] <= e) {
	leal	(%r10,%rdx), %edi	 #, tmp213
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:71: 				io_data[i * channel + j] = 0;
	xorl	%r14d, %r14d	 # _47
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:70: 			if (io_data[i * channel + j] <= e) {
	movslq	%edi, %rdi	 # tmp213, tmp214
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:70: 			if (io_data[i * channel + j] <= e) {
	leaq	(%r9,%rdi,4), %rdi	 #, _5
	movl	(%rdi), %r15d	 # *_5, _6
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:70: 			if (io_data[i * channel + j] <= e) {
	cmpl	%eax, %r15d	 # e, _6
	jle	.L230	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:73: 				io_data[i * channel + j] -= e;
	subl	%eax, %r15d	 # e, _6
	movl	%r15d, %r14d	 # _6, _47
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:74: 				s += io_data[i * channel + j];
	addl	%r15d, %ecx	 # _47, stmp_s_26.621
.L230:
	movl	%r14d, (%rdi)	 # _47, *_5
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:69: 		for (int j = 0; j < channel; j++) {
	leal	1(%rdx), %edi	 #, j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:69: 		for (int j = 0; j < channel; j++) {
	cmpl	%r8d, %edi	 # channel, j
	jge	.L229	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:70: 			if (io_data[i * channel + j] <= e) {
	addl	%r10d, %edi	 # ivtmp.656, tmp216
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:71: 				io_data[i * channel + j] = 0;
	xorl	%r14d, %r14d	 # cstore_70
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:70: 			if (io_data[i * channel + j] <= e) {
	movslq	%edi, %rdi	 # tmp216, tmp217
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:70: 			if (io_data[i * channel + j] <= e) {
	leaq	(%r9,%rdi,4), %rdi	 #, _52
	movl	(%rdi), %r15d	 # *_52, _46
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:70: 			if (io_data[i * channel + j] <= e) {
	cmpl	%r15d, %eax	 # _46, e
	jge	.L231	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:73: 				io_data[i * channel + j] -= e;
	movl	%r15d, %r14d	 # _46, _46
	subl	%eax, %r14d	 # e, _46
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:74: 				s += io_data[i * channel + j];
	addl	%r14d, %ecx	 # cstore_70, stmp_s_26.621
.L231:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:69: 		for (int j = 0; j < channel; j++) {
	addl	$2, %edx	 #, j
	movl	%r14d, (%rdi)	 # cstore_70, *_52
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:69: 		for (int j = 0; j < channel; j++) {
	cmpl	%edx, %r8d	 # j, channel
	jle	.L229	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:70: 			if (io_data[i * channel + j] <= e) {
	addl	%r10d, %edx	 # ivtmp.656, tmp219
	movslq	%edx, %rdx	 # tmp219, tmp220
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:70: 			if (io_data[i * channel + j] <= e) {
	leaq	(%r9,%rdx,4), %rdi	 #, _25
	movl	(%rdi), %edx	 # *_25, _24
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:70: 			if (io_data[i * channel + j] <= e) {
	cmpl	%eax, %edx	 # e, _24
	jg	.L232	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:71: 				io_data[i * channel + j] = 0;
	xorl	%edx, %edx	 # cstore_21
.L233:
	movl	%edx, (%rdi)	 # cstore_21, *_25
.L229:
	movq	(%rsp), %rax	 # %sfp, tmp245
	leaq	(%r9,%r11,4), %rdi	 #, ivtmp.639
	addq	8(%rsp), %r11	 # %sfp, tmp231
	leaq	(%rax,%r11,4), %r11	 #, _128
	.p2align 4,,10
	.p2align 3
.L236:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:80: 			io_data[i * channel + j] = io_data[i * channel + j] *128/ s;
	movl	(%rdi), %eax	 # MEM[(i32 *)_137], tmp234
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:79: 		for (int j = 0; j < channel; j++) {
	addq	$4, %rdi	 #, ivtmp.639
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:80: 			io_data[i * channel + j] = io_data[i * channel + j] *128/ s;
	sall	$7, %eax	 #, tmp234
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:80: 			io_data[i * channel + j] = io_data[i * channel + j] *128/ s;
	cltd
	idivl	%ecx	 # stmp_s_26.621
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:80: 			io_data[i * channel + j] = io_data[i * channel + j] *128/ s;
	movl	%eax, -4(%rdi)	 # tmp236, MEM[(i32 *)_137]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:79: 		for (int j = 0; j < channel; j++) {
	cmpq	%rdi, %r11	 # ivtmp.639, _128
	jne	.L236	 #,
.L238:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:60: 	for (int i = 0; i < length; i++) {
	addl	$1, %esi	 #, i
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:60: 	for (int i = 0; i < length; i++) {
	addl	%r8d, %r10d	 # channel, ivtmp.656
	cmpl	%esi, 24(%rbp)	 # i, length
	jne	.L224	 #,
.L222:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:83: }
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
	.p2align 4,,10
	.p2align 3
.L232:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:73: 				io_data[i * channel + j] -= e;
	subl	%eax, %edx	 # e, cstore_21
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:74: 				s += io_data[i * channel + j];
	addl	%edx, %ecx	 # cstore_21, stmp_s_26.621
	jmp	.L233	 #
	.p2align 4,,10
	.p2align 3
.L250:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:66: 		e  = e / channel;
	cltd
	idivl	%ebx	 # niters_vector_mult_vf.627
	jmp	.L240	 #
.L251:
	xorl	%edx, %edx	 # j
	xorl	%ecx, %ecx	 # stmp_s_26.621
	jmp	.L239	 #
.L241:
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:65: 		for (int j = 0; j < channel; j++) e += io_data[i * channel + j];
	xorl	%edx, %edx	 # j
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/f32layer.h:61: 		i32 e = 0;
	xorl	%eax, %eax	 # stmp_e_47.634
	movslq	%r10d, %r11	 # ivtmp.656, _50
	jmp	.L225	 #
	.def	__main;	.scl	2;	.type	32;	.endef
	.section	.text.startup,"x"
	.p2align 4
	.globl	main
	.def	main;	.scl	2;	.type	32;	.endef
main:
	pushq	%rbp	 #
	movq	%rsp, %rbp	 #,
	pushq	%r12	 #
	andq	$-16, %rsp	 #,
	addq	$-128, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:3: int main(){
	call	__main	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:5: 	f32topdinsty(a , 1, 10);
	leaq	32(%rsp), %r12	 #, tmp92
	movl	$1, %edx	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:4: 	f32 a[10] = {0,1,2,3,4,5,6,7,8,9};
	movq	.LC12(%rip), %rax	 #, tmp91
	movaps	.LC10(%rip), %xmm0	 #, tmp89
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:5: 	f32topdinsty(a , 1, 10);
	movl	$10, %r8d	 #,
	movq	%r12, %rcx	 # tmp92,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:4: 	f32 a[10] = {0,1,2,3,4,5,6,7,8,9};
	movaps	%xmm0, 32(%rsp)	 # tmp89, MEM <vector(4) float> [(float *)&a]
	movaps	.LC11(%rip), %xmm0	 #, tmp90
	movq	%rax, 64(%rsp)	 # tmp91, MEM <vector(2) float> [(float *)&a + 32B]
	movaps	%xmm0, 48(%rsp)	 # tmp90, MEM <vector(4) float> [(float *)&a + 16B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:5: 	f32topdinsty(a , 1, 10);
	call	_Z12f32topdinstyPfii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:7: 		a[i]*= 127;
	movaps	32(%rsp), %xmm1	 # MEM <vector(4) float> [(float *)&a], vect__2.665
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:9: 	printmat(a, 1, 10);
	movq	%r12, %rcx	 # tmp92,
	movl	$1, %edx	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:7: 		a[i]*= 127;
	movaps	.LC13(%rip), %xmm0	 #, tmp94
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:11: 	i32topdinsty(b , 1, 10); 
	leaq	80(%rsp), %r12	 #, tmp104
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:7: 		a[i]*= 127;
	mulps	%xmm0, %xmm1	 # tmp94, vect__2.665
	mulps	48(%rsp), %xmm0	 # MEM <vector(4) float> [(float *)&a + 16B], vect__2.665
	movaps	%xmm1, 32(%rsp)	 # vect__2.665, MEM <vector(4) float> [(float *)&a]
	movq	.LC14(%rip), %xmm1	 #, tmp99
	movaps	%xmm0, 48(%rsp)	 # vect__2.665, MEM <vector(4) float> [(float *)&a + 16B]
	movq	64(%rsp), %xmm0	 # MEM <vector(2) float> [(float *)&a + 32B], MEM <vector(2) float> [(float *)&a + 32B]
	mulps	%xmm1, %xmm0	 # tmp99, vect__41.671
	movlps	%xmm0, 64(%rsp)	 # vect__41.671, MEM <vector(2) float> [(float *)&a + 32B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:9: 	printmat(a, 1, 10);
	call	_Z8printmatPfii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:10: 	i32 b[10] = {0,1,2,3,4,5,6,7,8,9};
	movq	.LC17(%rip), %rax	 #, tmp103
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:11: 	i32topdinsty(b , 1, 10); 
	movq	%r12, %rcx	 # tmp104,
	movl	$10, %r8d	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:10: 	i32 b[10] = {0,1,2,3,4,5,6,7,8,9};
	movdqa	.LC15(%rip), %xmm0	 #, tmp101
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:11: 	i32topdinsty(b , 1, 10); 
	movl	$1, %edx	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:10: 	i32 b[10] = {0,1,2,3,4,5,6,7,8,9};
	movaps	%xmm0, 80(%rsp)	 # tmp101, MEM <vector(4) int> [(int *)&b]
	movdqa	.LC16(%rip), %xmm0	 #, tmp102
	movq	%rax, 112(%rsp)	 # tmp103, MEM <vector(2) int> [(int *)&b + 32B]
	movaps	%xmm0, 96(%rsp)	 # tmp102, MEM <vector(4) int> [(int *)&b + 16B]
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:11: 	i32topdinsty(b , 1, 10); 
	call	_Z12i32topdinstyPiii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:12: 	printmat(b, 1, 10);
	movq	%r12, %rcx	 # tmp104,
	movl	$1, %edx	 #,
	call	_Z8printmatPiii	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test_layer.cpp:14: }
	movq	-8(%rbp), %r12	 #,
	xorl	%eax, %eax	 #
	leave	
	ret	
	.section .rdata,"dr"
	.align 16
.LC8:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
	.align 16
.LC9:
	.word	255
	.word	255
	.word	255
	.word	255
	.word	255
	.word	255
	.word	255
	.word	255
	.align 16
.LC10:
	.long	0
	.long	1065353216
	.long	1073741824
	.long	1077936128
	.align 16
.LC11:
	.long	1082130432
	.long	1084227584
	.long	1086324736
	.long	1088421888
	.align 8
.LC12:
	.long	1090519040
	.long	1091567616
	.align 16
.LC13:
	.long	1123942400
	.long	1123942400
	.long	1123942400
	.long	1123942400
	.set	.LC14,.LC13
	.align 16
.LC15:
	.long	0
	.long	1
	.long	2
	.long	3
	.align 16
.LC16:
	.long	4
	.long	5
	.long	6
	.long	7
	.align 8
.LC17:
	.long	8
	.long	9
	.ident	"GCC: (x86_64-posix-seh, Built by MinGW-Builds project) 11.4.0"
	.def	__mingw_vfprintf;	.scl	2;	.type	32;	.endef
