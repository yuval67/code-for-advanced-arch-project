import numpy
import math
from torch import nn
import Config as cfg
from custom_matmul import custom_matmul


class customConv2d(nn.Conv2d):
    def __init__(self, epoch, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', compute_flavour=0, device='cuda', verbose=0):
        super(customConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) == 1 else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) == 1 else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) == 1 else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) == 1 else (dilation, dilation)
        self.groups = groups

        self.compute_flavour = compute_flavour
        self.device = device
        self.verbose = verbose


    def print_verbose(self, msg, v):
        if self.verbose >= v:
            cfg.LOG.write(msg)


    def forward(self, inputs, epoch):
        if self.compute_flavour == 0:
            #run legacy Conv2D
            return super().forward(inputs)


        out_X_dim = math.floor((inputs.size(2) + 2 * self.padding[0] - 1 * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        out_Y_dim = math.floor((inputs.size(3) + 2 * self.padding[1] - 1 * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        x_unfolded_orig = nn.functional.unfold(inputs, kernel_size=self.kernel_size, padding=self.padding,
                                               stride=self.stride, dilation=self.dilation)

        w_unfolded_orig = self.weight.view(self.weight.size(0), -1)
        x_unfolded_orig = x_unfolded_orig.transpose(1, 2).contiguous()
        outputs = custom_matmul(x_unfolded_orig, w_unfolded_orig, self.compute_flavour, epoch)
        outputs = outputs.transpose(1, 2).contiguous()
        output_folded = outputs.reshape(outputs.size(0), outputs.size(1), out_X_dim, out_Y_dim)


        if self.bias is not None:
            bias_unsqueeze = self.bias.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
            output_folded += bias_unsqueeze

        return output_folded