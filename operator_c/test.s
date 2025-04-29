	.file	"test.cpp"
 # GNU C++17 (x86_64-posix-seh, Built by MinGW-Builds project) version 11.4.0 (x86_64-w64-mingw32)
 #	compiled by GNU C version 11.4.0, GMP version 6.2.1, MPFR version 4.1.0, MPC version 1.2.1, isl version isl-0.25-GMP

 # GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
 # options passed: -mtune=core2 -march=nocona -O3 -fno-asynchronous-unwind-tables
	.text
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
	.def	__main;	.scl	2;	.type	32;	.endef
	.section .rdata,"dr"
.LC0:
	.ascii "%d%d%d\0"
	.section	.text.startup,"x"
	.p2align 4
	.globl	main
	.def	main;	.scl	2;	.type	32;	.endef
main:
	subq	$40, %rsp	 #,
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test.cpp:9: int main() { 
	call	__main	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test.cpp:55: 	printf("%d%d%d",b,n,m);
	movl	$548, %r9d	 #,
	movl	$548, %r8d	 #,
	movl	$548, %edx	 #,
	leaq	.LC0(%rip), %rcx	 #, tmp83
	call	_Z6printfPKcz	 #
 # C:/Users/Administrator/Documents/GitHub/InnovativeStructure/operator_c/test.cpp:56: }
	xorl	%eax, %eax	 #
	addq	$40, %rsp	 #,
	ret	
	.ident	"GCC: (x86_64-posix-seh, Built by MinGW-Builds project) 11.4.0"
	.def	__mingw_vfprintf;	.scl	2;	.type	32;	.endef
