# encoder-decoder architecture with skip connections as defined in the supplementary material
# of the paper Deep Image Prior
import torch.nn as nn
import torch as torch

class DownsampleModule(nn.Module):
    def __init__(self, input_depth, num_filters, kernel_size, pad_size):
        super(DownsampleModule, self).__init__()
        self.activation_function = nn.LeakyReLU(0.2, inplace=True)
        self.batch_norm = nn.BatchNorm2d(num_filters)
        self.cnn_1 = nn.Conv2d(input_depth, num_filters, kernel_size, stride=2)
        self.cnn_2 = nn.Conv2d(num_filters, num_filters, kernel_size, stride=1)
        self.padder_layer = nn.ReflectionPad2d(pad_size)

    def forward(self, x):
        x = self.padder_layer(x)
        x = self.cnn_1(x)
        x = self.batch_norm(x)
        x = self.activation_function(x)

        x = self.padder_layer(x)
        x = self.cnn_2(x)
        x = self.batch_norm(x)
        x = self.activation_function(x)

        return x

class UpsampleModule(nn.Module):
    def __init__(self, input_depth, num_filters, kernel_size, pad_size, upsample_mode):
        super(UpsampleModule, self).__init__()
        self.activation_function = nn.LeakyReLU(0.2, inplace=True)
        self.batch_norm = nn.BatchNorm2d(num_filters)
        self.batch_norm_fixed = nn.BatchNorm2d(input_depth)
        self.cnn_1 = nn.Conv2d(input_depth, num_filters, kernel_size, stride=1)
        self.cnn_2 = nn.Conv2d(num_filters, num_filters, kernel_size=1, stride=1)
        self.padder_layer = nn.ReflectionPad2d(pad_size)
        self.upsample_layer = nn.Upsample(scale_factor=2, mode=upsample_mode,)

    def forward(self, x):
        x = self.batch_norm_fixed(x)
        x = self.padder_layer(x)
        x = self.cnn_1(x)
        x = self.batch_norm(x)
        x = self.activation_function(x)
        #x = self.padder_layer(x)
        x = self.cnn_2(x)
        x = self.batch_norm(x)
        x = self.activation_function(x)

        x = self.upsample_layer(x)
        return x

class SkipConnection(nn.Module):
    def __init__(self, input_depth, num_filters, kernel_size, pad_size):
        super(SkipConnection, self).__init__()
        self.activation_function = nn.LeakyReLU(0.2, inplace=True)
        self.batch_norm = nn.BatchNorm2d(num_filters)
        self.cnn = nn.Conv2d(input_depth, num_filters, kernel_size, stride=1)
        self.padder_layer = nn.ReflectionPad2d(pad_size)

    def forward(self, x):
        #x = self.padder_layer(x)
        x = self.cnn(x)
        x = self.batch_norm(x)
        x = self.activation_function(x)

        return x

class SkipArchitecture(nn.Module):
    def __init__(self, input_channels, output_channels, filters_down, filters_up, filters_skip,
                 kernel_size_down, kernel_size_up, kernel_size_skip, upsample_mode):
        super(SkipArchitecture, self).__init__()
        self.down_modules = nn.ModuleList([DownsampleModule(input_depth = input_channels if i == 0 
                                                                    else filters_down[i-1], 
                                                      num_filters = filters_down[i], 
                                                      kernel_size = kernel_size_down[i],
                                                      pad_size = int((kernel_size_down[i] - 1) / 2)) 
                                          for i in range(len(filters_down))])
        self.up_modules = nn.ModuleList([UpsampleModule(input_depth = filters_skip[i] + filters_up[i + 1] if i != len(filters_down) - 1
                                                                    else filters_up[i],
                                                  num_filters = filters_up[i],
                                                  kernel_size = kernel_size_up[i],
                                                  pad_size = int((kernel_size_up[i] - 1) / 2),
                                                  upsample_mode = upsample_mode)
                                          for i in range(len(filters_up)-1, -1, -1)]) ##change later

        self.skip_connections = nn.ModuleList([SkipConnection(input_depth = filters_down[i], 
                                                              num_filters = filters_skip[i], 
                                                              kernel_size = kernel_size_skip[i],
                                                              pad_size = int((kernel_size_skip[i] - 1) / 2))
                                               for i in range(len(filters_up))])
        self.cnn_last = nn.Conv2d(filters_up[0], output_channels, 1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.number_of_channels = len(filters_down)

    def forward(self, x):
        s = []

        for i in range(self.number_of_channels):
            x = self.down_modules[i].forward(x)
            s.append(self.skip_connections[i].forward(x))

        for i in range(self.number_of_channels): 
            if i == 0:
              x = self.up_modules[i].forward(s[-1])
            else:
                x = self.up_modules[i].forward(torch.cat([x,s[self.number_of_channels-i-1]], 1))

        x = self.cnn_last(x)
        x = self.sigmoid(x)
        return x