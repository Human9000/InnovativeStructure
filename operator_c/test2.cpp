#include<stdio.h>
#include<uchar.h>
union F32{
	float f;
	unsigned u;
};

int main(){
	int a = 128;
	float _b[10] = {-1,1,1,1,1,1,1,1,1,1};
	F32 *b = (F32 *)_b;
//	float c;
//	float d;
//	F32 *_d = (F32*)&d;
//	_d->u = (0x80000000)^b[2].u;
//	
	float res=0;
//	res += (0x80000000)^b[2].u;
	for(int i=0; i<1; i++){
		F32 temp;
		temp.u = (unsigned)(0x80000000)^b[2].u;
		res += temp.f;
	}
//	printf("%x \n", b[0].bytes);
//	printf("%f \n", b[0].f32);
//	printf("%x \n", b[1].bytes);
//	printf("%x \n", d); 
//	printf("%f \n", d); 
	printf("%f \n", res); 
}
