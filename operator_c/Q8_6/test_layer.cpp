#include "h/layer.h"
#include<iostream>
using namespace std;
void test_avgpool1d_i32(){
	i32 data[4] = {64,32,0,32}; // 1, 0.5, 0 
	printmat("data", data, 4, 1);
	i32 o[3];
	i32 l = avg_pool1d(data, o, 4, 1, 3, 1);
	printmat("o", o, l, 1);
}

void test_avgpool1d_f32(){
	f32 data[8] = {64,32,
		0,32,
		32,0,
		15,1,
	}; // 1, 0.5, 0 
	printmat("data", data, 4, 2);
	f32 o[8];
	f32 l = avg_pool1d(data, o, 4, 2, 3, 1);
	printmat("o", o, l, 2);
}

void test_conv1d_f32(){
	f32 data[8] = {
		64,32,
		0,32,
		32,0,
		15,1,
	}; // 1, 0.5, 0 
	f32 weight[3*3*2] = {
		1,1,
		1,1,
		1,1,
		
		0.5,1,
		0.5,1,
		0.5,1,
		
		
		1,0.5,
		1,0.5,
		1,0.5,
	
	
	}; // 输出1通道，卷积核大小3, 输入2通道
	f32 bias[10] = { 0, }; // 输出1通道，卷积核大小3, 输入2通道
	f32 o[80]={0,0,0,0}; 
	
	int res = conv1d(data, weight, bias, o, 4, 2, 3, 3, 1);
	
	printmat("data", data, 4, 2);
	printmat("w", weight, 3,3, 2); 
	printmat("o", o, res, 3);
}

void test_conv1d_i32(){
	i32 data[4*2] = {
		64,32,
		0,32,
		32,0,
		15,1,
	}; // 1, 0.5, 0 
	i8 weight[3*3*2] = {
		64,64,
		64,64,
		64,64,
		
		32,64,
		32,64,
		32,64,
		
		
		64,32,
		64,32,
		64,32, 
		
	}; // 输出1通道，卷积核大小3, 输入2通道
	i32 bias1[3] = { 0,0,0}; // 输出1通道，卷积核大小3, 输入2通道
	i32 bias2[3] = { 1,1,1 }; // 输出1通道，卷积核大小3, 输入2通道
	i32 o[2*3]={0,0,0,0,0,0}; 
	
	int res1 = conv1d(data, weight, bias1, o, 4, 2, 3, 3, 1);
	
	printmat("data", data, 4, 2);
	printmat("w", weight, 3,3, 2); 
	printmat("b1", bias1, 1, 3); 
	printmat("o1", o, res1, 3);
	
	int res2 = conv1d(data, weight, bias2, o, 4, 2, 3, 3, 1);
	
	printmat("data", data, 4, 2);
	printmat("w", weight, 3, 3, 2); 
	printmat("b2", bias2, 1, 3); 
	printmat("o2", o, res2, 3); 
}

void test_upsample_linear(){
	i32 data[3] = {0,1*16,3*16};
	i32 o[20] = {0};
	int res =upsample_linear(data, o, 3, 1, 5 );
	printmat("o",  o, res,1);
}
void test_upsample_linearf32(){
	f32 data[3] = {0,0.25,0.75};
	f32 o[20] = {0};
	int res =upsample_linear(data, o, 3, 1, 5 );
	
	printmat("o",o, res,1);
	f32_2_i32(o, (i32 *)o, res);
	printmat("o",(i32 *)o, res,1);
}


void test_conv1d_i32_b8(){
	i32 data[4*2] = {
		64,32,
		0,32,
		32,0,
		15,1,
	}; // 1, 0.5, 0 
	b8 weight[3*3*2 /8 + 1] = {
		0xff,0xff,0xff 
	}; // 输出1通道，卷积核大小3, 输入2通道
	i32 bias1[3] = { 0,0,0}; // 输出1通道，卷积核大小3, 输入2通道
	i32 bias2[3] = { 1,1,1 }; // 输出1通道，卷积核大小3, 输入2通道
	i32 o[2*3]={0,0,0,0,0,0}; 
	
	int res1 = conv1d(data, weight, bias1, o, 4, 2, 3, 3, 1);
	
	printmat("data", data, 4, 2); 
	printmat("b1", bias1, 1, 3); 
	printmat("o1", o, res1, 3);
	
	int res2 = conv1d(data, weight, bias2, o, 4, 2, 3, 3, 1); 
	printmat("data", data, 4, 2); 
	printmat("b2", bias2, 1, 3); 
	printmat("o2", o, res2, 3); 
}

int main(){
//	test_upsample_linear();
//	test_upsample_linearf32();
	test_conv1d_i32_b8();
	cout<<endl;
}
