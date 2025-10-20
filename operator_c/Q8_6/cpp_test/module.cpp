//
// Created by Administrator on 25-9-11.
//
#include <iostream>
using namespace std;


class Tensor {
private :
    Mat *data;
public:
    Tensor() {
        cout << "Tensor()" << endl;
    }
    ~Tensor() {
        cout << "~Tensor()" << endl;
    }
};


class Module {
public:
    Module() {
        cout << "Module()" << endl;
    }
    void forward(Tensor in, Tensor out){

    }
    ~Module() {
        cout << "~Module()" << endl;
    }

}


class Conv1d_I32_B8 : public Module {
private:
    MatI32 weights;
    MatI32 mem;
public:
    Conv1d_I32_B8(b8* w_data, i32* mem_data, int l, int c, int n, int m_l, int m_c);
    MatI32 forward(MatI32 input) override;
};