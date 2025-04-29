#include "opt.h"


void test_i32_i8_prod() {
	i32 a[3] = {0, 1, 2}; // 0, 1/64 , 2/64
	i8 b[3] = {64, 64, 64};
	printmat("a", a, 1, 3);
	printmat("b", b, 3, 1);
	i32 res = i32_i8_vec_in_prod(a, b, 3);
	printmat("res", &res, 1, 1); 
}
void test_i32_i8_mul() {
	i32 a[6] = {1, 1, 2,
	            1, 1, 4
	           }; // 0, 1/64 , 2/64
	i8 b[6] = {64, 64, 32,
	           0, 64, 64
	          };
	i32 c[4];
	printmat("a", a, 2, 3);
	printmat("b", b, 2, 3);
	i32_i8_mul(a, b, c, 2, 3, 2, false);
	printmat("c", c, 2, 2);

	printmat("a", a, 2, 3);
	printmat("b", b, 2, 3);
	i32_i8_mul(a, b, c, 2, 3, 2, false);
	printmat("c", c, 2, 2); 
}

void test_i32_b8_mul() {
	i32 a[6] = {1, 1, 2,
		1, 1, 4
	}; // 0, 1/64 , 2/64
	b8 b[1] = {0xF0};
	i32 c[4];
	printmat("a", a, 2, 3);
	printmat("b", b, 2, 3);
	i32_b8_mul(a, b, c, 2, 3, 2, false);
	printmat("c", c, 2, 2);
	
	
	printmat("a", a, 2, 3);
	printmat("b", b, 2, 3);
	i32_b8_mul(a, b, c, 2, 3, 2, true);
	printmat("c", c, 2, 2); 
}

void test_i32_b8_vec_in_prod() {
	i32 a[6] = {1, 1, 2,
	 
	}; // 0, 1/64 , 2/64
	b8 b[1] = {0xF0};
	i32 res;
	printmat("a", a, 1, 3);
	printmat("b", b, 1, 3);
	res = i32_b8_vec_in_prod(a, b, 3, 0);
	printmat("res", &res, 1, 1);
	
}

int main() { 
	test_i32_b8_mul();
//	test_i32_b8_vec_in_prod();
}
