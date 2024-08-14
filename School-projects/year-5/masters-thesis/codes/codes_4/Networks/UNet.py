import torch
import torch.nn as  nn
import torch.nn.functional as F

from Networks.ResNet import build_ResNet


class UNetFeatures(nn.Module):
    def __init__(self, channels, resnet_layers):
        super().__init__()
        self.encoder = build_ResNet(channels, resnet_layers)

        out_channel = 8 * (64 if resnet_layers in (18, 34) else 256)
        self.up_conv_1 = nn.ConvTranspose2d(out_channel,
                                            out_channel // 2,
                                            2, 2, 0)

        self.double_conv_1 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, 3, 1, 1),
                                           nn.BatchNorm2d(out_channel // 2),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(out_channel // 2, out_channel // 2, 3, 1, 1),
                                           nn.BatchNorm2d(out_channel // 2),
                                           nn.ReLU(inplace=True),)                                          
        out_channel //= 2

        self.up_conv_2 = nn.ConvTranspose2d(out_channel,
                                            out_channel // 2,
                                            2, 2, 0)

        self.double_conv_2 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, 3, 1, 1),
                                           nn.BatchNorm2d(out_channel // 2),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(out_channel // 2, out_channel // 2, 3, 1, 1),
                                           nn.BatchNorm2d(out_channel // 2),
                                           nn.ReLU(inplace=True),)                                          
        out_channel //= 2

        self.up_conv_3 = nn.ConvTranspose2d(out_channel,
                                            out_channel // 2,
                                            2, 2, 0)

        self.double_conv_3 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, 3, 1, 1),
                                           nn.BatchNorm2d(out_channel // 2),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(out_channel // 2, out_channel // 2, 3, 1, 1),
                                           nn.BatchNorm2d(out_channel // 2),
                                           nn.ReLU(inplace=True),)      
        """self.double_conv_3 = nn.Sequential(nn.Conv2d(out_channel, 256, 3, 1, 1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(256, 256, 3, 1, 1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(inplace=True),)  """

        

        """self.up_conv_1 = nn.ConvTranspose2d(out_channel,
                                            256,
                                            2, 2, 0)

        self.double_conv_1 = nn.Sequential(#nn.Conv2d((out_channel // 2) + 256, 256, 3, 1, 1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(inplace=True),)
                                           #nn.Conv2d(256, 256, 3, 1, 1),
                                           #nn.BatchNorm2d(256),
                                           #nn.ReLU(inplace=True),)                                         
        out_channel //= 2

        self.up_conv_2 = nn.ConvTranspose2d(256,
                                            256,
                                            2, 2, 0)

        self.double_conv_2 = nn.Sequential(#nn.Conv2d((out_channel // 2) + 256, 256, 3, 1, 1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(inplace=True),)
                                           #nn.Conv2d(256, 256, 3, 1, 1),
                                           #nn.BatchNorm2d(256),
                                           #nn.ReLU(inplace=True),)                                       
        out_channel //= 2

        self.up_conv_3 = nn.ConvTranspose2d(256,
                                            256,
                                            2, 2, 0)

        self.double_conv_3 = nn.Sequential(#nn.Conv2d((out_channel // 2) + 256, 256, 3, 1, 1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(inplace=True),)
                                           #nn.Conv2d(256, 256, 3, 1, 1),
                                           #nn.BatchNorm2d(256),
                                           #nn.ReLU(inplace=True),)   
        
        out_channel = 256 #//= 2"""
        out_channel //= 2
        self.out_channels = out_channel


    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)

        #print(x.shape)
        x = self.up_conv_1(x4)
        del x4
        torch.cuda.empty_cache()
        x = torch.cat((x3, x), 1)   
        del x3
        torch.cuda.empty_cache()
        x = self.double_conv_1(x)
        
        #print(x.shape)
        x = self.up_conv_2(x)
        x = torch.cat((x2, x), 1)
        del x2
        torch.cuda.empty_cache()
        x = self.double_conv_2(x)

        #print(x.shape)
        x = self.up_conv_3(x)
        x = torch.cat((x1, x), 1)
        del x1
        torch.cuda.empty_cache()
        x = self.double_conv_3(x)
        #print(x.shape)
        return x
