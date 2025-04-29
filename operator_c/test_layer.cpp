#include "f32layer.h"

int main(){
	f32 a[10] = {0,1,2,3,4,5,6,7,8,9};
	f32topdinsty(a , 1, 10);
	for(int i=0;i<10;i++){
		a[i]*= 127;
	}
	printmat(a, 1, 10);
	i32 b[10] = {0,1,2,3,4,5,6,7,8,9};
	i32topdinsty(b , 1, 10); 
	printmat(b, 1, 10);
	
}
