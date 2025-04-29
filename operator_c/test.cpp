#include<stdio.h>
#include<time.h>
#include<inttypes.h>

union F32{
	float f;
	uint32_t u;
};
int main() { 
	float a;
 
	long long t0, t1, t2, t3;
	unsigned int *p = (unsigned int *)&a;

	float res = 0;
	
	
	int m,n,b;
 
	m = 0x223;
	for (int i = 0; i < 1000000000; i++)
		res += (i % 2) ? a : -a;

//	i%2
//	jump   21 22;
//	add res a;
//	jump
//	xor a temp;
//	add res temp;

	
//	i%2
//	<<31
//	^
//	add  
	
	n = 0x223;
	for (int i = 0; i < 1000000000; i++){
		F32 temp;
		temp.u = (i % 2) ^ (*p);
		res += temp.f;
	}
	
	 
	b = 0x223;
	for (int i = 0; i < 1000000000; i++)
		if (i % 2) {
			res +=  *p;
		} else {
			res -=  *p;
		}
	n+=1;
	m+=1;
	b+=1; 
	printf("%d%d%d",b,n,m);
}
