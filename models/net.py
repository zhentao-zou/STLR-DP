import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import models.basicblock as B


class DPMOD(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64):
        super(DPMOD, self).__init__()
        L =[]
        L.append(nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        L.append(nn.PReLU(num_parameters=nc))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.BatchNorm2d(nc, momentum=0.1))
        L.append(nn.PReLU(num_parameters=nc))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.BatchNorm2d(nc, momentum=0.1))
        L.append(nn.PReLU(num_parameters=nc))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=4, dilation=4, bias=True))
        L.append(nn.BatchNorm2d(nc, momentum=0.1))
        L.append(nn.PReLU(num_parameters=nc))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.BatchNorm2d(nc, momentum=0.1))
        L.append(nn.PReLU(num_parameters=nc))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.BatchNorm2d(nc, momentum=0.1))
        L.append(nn.PReLU(num_parameters=nc))
        L.append(nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        self.model = B.sequential(*L)

    def forward(self, x):
        n = self.model(x)
        return x+n


class DSPMOD(nn.Module):
    def __init__(self, img_channels, in_channels, out_channels):
        super(DSPMOD, self).__init__()
        self.SPU_layer=DictConv2d(img_channels, in_channels, out_channels, kernel_size=3)

    def forward(self,data):
        code=data
        data,code = self.SPU_layer(data, code)
        data,code = self.SPU_layer(data, code)
        data,code = self.SPU_layer(data, code)
        data,code = self.SPU_layer(data, code)
        data,code = self.SPU_layer(data, code)
        data,code = self.SPU_layer(data, code)
        data,code = self.SPU_layer(data, code)
        return data,code



'''
class DictConv2dBlock(nn.Module):
    def __int__(self, img_channels,in_channels,out_channels,kernel_size,stride=1,dilation=1,groups=1,bias=False,num_convs=2):
        super(DictConv2dBlock, self).__init__()
        self.conv = DictConv2d(img_channels, in_channels, out_channels, kernel_size, stride,
                               dilation, groups, bias, num_convs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = SST(out_channels,init=None)
        self.recon_layer=
    def foward(self, data,code):
        code=self.conv(data,code)
        code=self.norm(code)
        code=self.activation(code)
        data=self.
        return data,code
'''


class DictConv2d(nn.Module):
    def __init__(self, img_channels,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1, # only support stride=1
                 dilation=1,
                 groups=1,
                 bias=False,
                 num_convs=2,
                 padding_mode='reflection',
                 padding='same',
                 value=None):
        super(DictConv2d, self).__init__()
        self.in_channels = in_channels
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = SST(out_channels, init=1e-3)
        self.conv_encoder = nn.Sequential()
        for i in range(num_convs - 1):
            self.conv_encoder.add_module('en_conv' + str(i),
                                         nn.Conv2d(img_channels, img_channels, kernel_size=kernel_size,
                                                   padding=int((kernel_size - 1) / 2),
                                                   stride=1, dilation=1, bias=True))
        self.conv_encoder.add_module('en_conv' + str(num_convs - 1),
                                     nn.Conv2d(img_channels, in_channels, kernel_size=kernel_size,
                                               padding=int((kernel_size - 1) / 2),
                                               stride=1, dilation=1, bias=True))
        self.shift_flag = out_channels != in_channels
        self.conv_channel_shift = nn.Conv2d(in_channels, out_channels, 1) if self.shift_flag else None

    def forward(self,data,code):
        dcode=self.conv_encoder(code)
        res=data-dcode
        dres=self.conv_encoder(res)
        code=code+dres
        code=self.norm(code)
        code=self.activation(code)
        data=self.conv_encoder(code)
        data=self.activation(data)
        return data,code




class SST(nn.Module):
    def __init__(self, num_parameters=1,
                 init=1e-2):
        super(SST, self).__init__()
        self.theta = nn.Parameter(torch.full(size=(1, num_parameters, 1, 1),
                                             fill_value=init))
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return x.sign() * self.relu(x.abs() - torch.clamp(self.theta, min=0.))



class StdLoss(nn.Module):
    def __init__(self):
        """
        Loss on the variance of the image.
        Works in the grayscale.
        If the image is smooth, gets zero
        """
        super(StdLoss, self).__init__()
        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)
        self.gray_scale = GrayscaleLayer()

    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(F.conv2d(x, self.image), F.conv2d(x, self.blur))



class GrayscaleLayer(nn.Module):
    def __init__(self):
        super(GrayscaleLayer, self).__init__()

    def forward(self, x):
        return torch.mean(x, 1, keepdim=True)