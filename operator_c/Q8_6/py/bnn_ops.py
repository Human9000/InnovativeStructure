"""
    BNN operation
    File    :bnn_ops.py
    Author  :JiaLi.Ou   <109553196@qq.com>
    Note    :Define backpropagation in a static manner,
             implement binarization, and implement hard-tanh functions
"""
import torch
import torch.nn.functional as F
from torch.autograd import Function  

Q_factor = 2 ** 6 #- 1
H_factor = 10**6
class Binarize(Function):
    """
    实现二值化Sign函数以及直通估计器件STE（Straight Through Estimator）的效果
    """
    @staticmethod
    def forward(ctx, input):
        """
        静态方式自定义前向传播，以避免自动求导
        实现功能:   y = Sign(x)
        当 x ≥ 0 时       y = 1
        当 x < 0 时       y = -1
        :param ctx:         用于给反向传播传递前向的信息
        :param input:       前向传播输入
        :return:            二值化输出
        """
        ctx.save_for_backward(input)
        output = torch.where(input >= 0, torch.tensor(1.0, dtype=input.dtype),
                             torch.tensor(-1.0 ,dtype=input.dtype),)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        静态方式自定义反向传播，以避免自动求导
        :param ctx:         用于给反向传播传递前向的信息
        :param grad_output: 反向梯度输入
        :return:            经过裁剪的梯度
        """
        input, = ctx.saved_tensors
        grad_Htanh = grad_output.clone()
        grad_Htanh[input.abs() > 1] = 0
        return grad_Htanh

class Hardtanh(Function):
    """
    实现Hard-tanh函数以及直通估计器件STE（Straight Through Estimator）的效果
    """
    @staticmethod
    def forward(ctx, input):
        """
        静态方式自定义前向传播，以避免自动求导
        实现功能:   y = Sign(x)
        当  x  ≥ 1 时     y =  1
        当  x  < 1 时     y = -1
        当 |x| < 1 时     y =  x

        ！！！以下量化操作占用大量运算空间，建议用C++底层优化！！！
        量化操作    torch.round(output*10**6)/10**6          只保留小数点后7位
        量化操作    flt_to_qtf_tensor   将浮点数压缩为int8     符号位bit0 阶码bit1-4 尾码bit5-7
        量化操作    qtf_to_flt_tensor   将int8解压为浮点数     解压范围为-0.937~0.937

        :param ctx:         用于给反向传播传递前向的信息
        :param input:       前向传播输入
        :return:            硬双曲正切输出
        """
        ctx.save_for_backward(input)
        output = F.hardtanh(input, min_val= -1, max_val=1)
        # output = torch.round(output*H_factor)/H_factor
        # binary_tensor = flt_to_qtf_tensor(output)
        # output = qtf_to_flt_tensor(binary_tensor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        静态方式自定义反向传播，以避免自动求导
        :param ctx:         用于给反向传播传递前向的信息
        :param grad_output: 反向梯度输入
        :return:            经过裁剪的梯度
        """
        input, = ctx.saved_tensors
        grad_Htanh = grad_output.clone()
        grad_Htanh[input.abs() > 1] = 0
        return grad_Htanh


class LBitTanh(Function):
    """
    实现低位宽Hard-tanh函数以及直通估计器件STE（Straight Through Estimator）的效果
    """
    @staticmethod
    def forward(ctx, input):
        """
        静态方式自定义前向传播，以避免自动求导
        实现功能:   y = Sign(x)
        当  x  ≥ 1 时     y =  1
        当  x  < 1 时     y = -1
        当 |x| < 1 时     y = Q(x)量化为int8类型

        :param ctx:         用于给反向传播传递前向的信息
        :param input:       前向传播输入
        :return:            硬双曲正切输出
        """
        ctx.save_for_backward(input)
        if input == None:
            return None
        output = F.hardtanh(input, min_val=-1, max_val=1)  
        # output = torch.round(output*Q_factor) / Q_factor
        output = torch.trunc(output*Q_factor) / Q_factor 
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        静态方式自定义反向传播，以避免自动求导
        :param ctx:         用于给反向传播传递前向的信息
        :param grad_output: 反向梯度输入
        :return:            经过裁剪的梯度
        """
        
        input, = ctx.saved_tensors
        if input == None:
            return None
        grad_Htanh = grad_output.clone()
        grad_Htanh[input.abs() > 1] = 0
        return grad_Htanh

def lbit(x, factor=64 ):
    return torch.trunc(x * factor) / factor

class LBit(Function):
    """
    实现低位宽Hard-tanh函数以及直通估计器件STE（Straight Through Estimator）的效果

    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if input == None:
            return None
        # 向 0 取整
        output = torch.trunc(input*Q_factor) / Q_factor
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        if input == None:
            return None
        grad_Htanh = grad_output.clone()
        return grad_Htanh

class LInt(Function):
    """
    实现低位宽Hard-tanh函数以及直通估计器件STE（Straight Through Estimator）的效果
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if input == None:
            return None
        # 向下取整
        output = torch.floor(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        if input == None:
            return None
        grad_Htanh = grad_output.clone()
        return grad_Htanh

def test():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用gpu时，无节点张量不会返回到cpu，无法查看中间张量；测试请使用cpu，gpu训练时无异常
    device = torch.device("cpu")  # 使用CPU
    input = torch.tensor([0.5, -0.1, -2.0], requires_grad=True).to(device)  # 将输入张量移动到GPU上

    # output = LBitTanh.apply(input)
    output = Hardtanh.apply(input)

    loss = torch.ones_like(input)
    output.backward(loss)
    print("Hardtanh_ops")
    print("Input:", input.detach().numpy())
    print("Output:", output.detach().numpy())
    print("loss:", loss)
    print("Irad:", input.grad)


if __name__ == "__main__":
    test()