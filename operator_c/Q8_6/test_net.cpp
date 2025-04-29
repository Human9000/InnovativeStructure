#include "h/net.h"

int main(){
	f32 data[1500*12];
	MatI32 y = net(data, 1500, 12);
	printmat("y", y.data, y.length, y.channels);	
	free(y.data);
}
