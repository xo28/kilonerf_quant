from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function
import numpy as np

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)

scale_bits = 8


def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0,  reduce_type='mean', keepdim=False, true_zero=False):
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)
        if x_flat.dim() == 1:
            min_values = _deflatten_as(x_flat.min(), x)
            max_values = _deflatten_as(x_flat.max(), x)
        else:
            min_values = _deflatten_as(x_flat.min(-1)[0], x)
            max_values = _deflatten_as(x_flat.max(-1)[0], x)
        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]
        # TODO: re-add true zero computation
        range_values = max_values - min_values
        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)


def calculate_qparams_dws(x, num_bits):
    with torch.no_grad():
        min_values = x.min(-1)[0].min(-1)[0].min(0)[0].view(1, -1, 1, 1)
        max_values = x.max(-1)[0].max(-1)[0].max(0)[0].view(1, -1, 1, 1)

        range_values = max_values - min_values
        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=True, stochastic=False, inplace=False):

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if qparams is None:
            assert num_bits is not None, "either provide qparams of num_bits to quantize"
            qparams = calculate_qparams(
                input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        min_val = qparams.zero_point
        max_val = qparams.range + qparams.zero_point
        num_bits = qparams.num_bits

        # symmetric quantization
        qrange = 2.**(num_bits - 1) - 1
        scale = torch.max(max_val, torch.abs(min_val)) / qrange
        # print("lalala", scale)
        # print(scale.shape)

        # print(scale.shape)
        # print(zero_point.shape)

        with torch.no_grad():
            # print("output scale", output.unique())
            output.div_(scale).round_()
            # print("div scale", scale, "ouput scale", output.unique())
            M0, n = quantize_scale(scale, bit=8)

            if dequantize:
                output.mul_(scale)

        # return output
        return (output, torch.tensor(M0).cuda(), torch.tensor(n).cuda(), scale)


    @staticmethod
    def backward(ctx, *grad_output):
        # straight-through estimator
        # print(grad_output)
        # print(grad_output[0].shape)
        grad_input = grad_output[0]
        return grad_input, None, None, None, None, None, None, None, None



def quantize_scale(scale, bit=8):
    # print(scale)
    # n = np.floor(np.log2((2**bit-1)/scale.item()))
    # M0 = np.floor(2**n * scale.item())

    n = np.floor(np.log2((2**bit-1)/scale.view(-1)[0].item()))
    M0 = np.floor(2**n * scale.view(-1)[0].item())

    # print('n', n)
    # print('new_scale', np.floor(2**n * scale.item()))

    return torch.tensor(M0).cuda().view(-1), n



def Quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=True, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic, inplace)



class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False, num_bits=8, num_bits_weight=8, dws=False):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits

        self.dws = dws

        self.momentum = 0.1

        if self.dws:
            shape_measure = (1, in_channels, 1, 1)
        else:
            shape_measure = (1, 1, 1, 1)

        self.register_buffer('running_zero_point', torch.zeros(*shape_measure).cuda())
        self.register_buffer('running_range', torch.zeros(*shape_measure).cuda())

        self.register_buffer('M0_input', torch.zeros([1]).cuda())
        self.register_buffer('n_input', torch.zeros(()).cuda())
        self.register_buffer('M0_weight', torch.zeros([1]).cuda())
        self.register_buffer('n_weight', torch.zeros(()).cuda())
        self.register_buffer('scale_weight', torch.zeros([1, 1, 1]))
        self.register_buffer('scale_input', torch.zeros([1, 1, 1]))        


    def forward(self, input, num_bits=8):
        if self.bias is not None:
            self.qbias = self.bias.div(self.scale_weight).round().clamp(-128,127).view(-1)

        if num_bits > 0:
            # num_bits = num_bits + 1 -> only applicable for conv-relu where all the activations are > 0 so that unsighed data type can be used 

            if self.training:
                qparams = calculate_qparams(input, num_bits=num_bits+1, flatten_dims=(1, -1), reduce_dim=0, reduce_type='extreme')
                with torch.no_grad():
                    self.running_zero_point.mul_(self.momentum).add_(
                        qparams.zero_point.cuda() * (1 - self.momentum))
                    self.running_range.mul_(self.momentum).add_(
                        qparams.range.cuda() * (1 - self.momentum))
            else:
                qparams = QParams(range=self.running_range,
                  zero_point=self.running_zero_point, num_bits=num_bits+1)

            # print('quantized act:')

            qinput, M0_input, n_input, scale_input = Quantize(input, qparams=qparams, dequantize=True,
                               stochastic=False, inplace=False)

            # qinput = Quantize(input, qparams=qparams, dequantize=True,
            #                    stochastic=False, inplace=False)

            self.M0_input.data = M0_input.data
            self.n_input.data = n_input.data
            self.scale_input.data = scale_input.data 

            # print('============')

            weight_qparams = calculate_qparams(
                self.weight, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=0, reduce_type='extreme')

            qweight, M0_weight, n_weight, scale_weight = Quantize(self.weight, qparams=weight_qparams)

            # qweight = Quantize(self.weight, qparams=weight_qparams)

            self.M0_weight.data = M0_weight.data
            self.n_weight.data = n_weight.data
            self.scale_weight.data = scale_weight.data 

            # print('============')

            # print('quantized weight:')
            # min_val = weight_qparams.zero_point
            # max_val = weight_qparams.range + weight_qparams.zero_point
            # weight_scale = (2**(num_bits-1)-1) / (2**scale_bits-1) * 8 / torch.max(max_val, torch.abs(min_val))
            # quantize_scale(weight_scale)

            if self.bias is not None:
                self.qbias = self.bias.div(scale_weight).round().clamp(-128,127).view(-1)
                qbias = self.qbias.mul(scale_weight).view(-1)
                # qbias, _, _, _ = Quantize(
                #     self.bias, num_bits=num_bits,
                #     flatten_dims=(0, -1))
            else:
                qbias = None

            # qbias = None

            
            output = F.conv2d(qinput, qweight, qbias, self.stride,
                                  self.padding, self.dilation, self.groups)

        else:
            output = F.conv2d(input, self.weight, self.qbias, self.stride,
                                  self.padding, self.dilation, self.groups)
        return output



class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=False, num_bits=8, num_bits_weight=8, num_bits_grad=8, biprecision=False):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision
        self.momentum = 0.1

        shape_measure = (1,)
        self.register_buffer('running_zero_point', torch.zeros(*shape_measure).cuda())
        self.register_buffer('running_range', torch.zeros(*shape_measure).cuda())

        self.register_buffer('M0_input', torch.zeros(()))
        self.register_buffer('n_input', torch.zeros(()))
        self.register_buffer('M0_weight', torch.zeros(()))
        self.register_buffer('n_weight', torch.zeros(()))
        self.register_buffer('scale_weight', torch.zeros(()))
        self.register_buffer('scale_input', torch.zeros(()))        


    def forward(self, input, num_bits=8):
        if num_bits > 0:
            # num_bits = num_bits + 1 -> only applicable for conv-relu where all the activations are > 0 so that unsighed data type can be used 

            if self.training:
                qparams = calculate_qparams(
                        input, num_bits=num_bits+1, flatten_dims=(1, -1), reduce_dim=0, reduce_type='extreme')
                with torch.no_grad():
                    self.running_zero_point.mul_(self.momentum).add_(
                        qparams.zero_point.cuda() * (1 - self.momentum))
                    self.running_range.mul_(self.momentum).add_(
                        qparams.range.cuda() * (1 - self.momentum))
            else:
                qparams = QParams(range=self.running_range,
                  zero_point=self.running_zero_point, num_bits=num_bits+1)

            qinput, M0_input, n_input, scale_input = Quantize(input, qparams=qparams, dequantize=True,
                               stochastic=False, inplace=False)

            # qinput = Quantize(input, qparams=qparams, dequantize=True,
            #                    stochastic=False, inplace=False)

            self.M0_input.data = M0_input.data
            self.n_input.data = n_input.data
            self.scale_input.data = scale_input.data 

            # print('============')

            weight_qparams = calculate_qparams(
                self.weight, num_bits=num_bits, flatten_dims=(1, -1), reduce_dim=0, reduce_type='extreme')

            qweight, M0_weight, n_weight, scale_weight = Quantize(self.weight, qparams=weight_qparams)

            # qweight = Quantize(self.weight, qparams=weight_qparams)

            self.M0_weight.data = M0_weight.data
            self.n_weight.data = n_weight.data
            self.scale_weight.data = scale_weight.data 

            if self.bias is not None:
                qbias, _, _, _ = Quantize(
                    self.bias, num_bits=num_bits,
                    flatten_dims=(0, -1))
            else:
                qbias = None

            output = F.linear(qinput, qweight, qbias)

        else:
            output = F.linear(input, self.weight, self.bias)

        return output


if __name__ == '__main__':
    x = torch.rand(3, 5)

    weight_qparams = calculate_qparams(
        x, num_bits=8, flatten_dims=(1, -1), reduce_dim=0, reduce_type='extreme')
    x_q = Quantize(x, qparams=weight_qparams)

    print(x)
    print(x_q)
