import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                               kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.LeakyReLU(), 
                                     nn.Conv2d(out_channels, out_channels,
                                               kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.LeakyReLU())
        
    def forward(self, input_):
        return self.network(input_)

class DownConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.network = nn.Sequential(nn.MaxPool2d(2, 2, 0),
                                     DoubleConvolution(in_channels,
                                                       out_channels))
        
    def forward(self, input_):
        return self.network(input_)

class UpConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_convolution = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                                 kernel_size=2, stride=2)
        self.convolution = DoubleConvolution(in_channels, out_channels)
                                     
    def forward(self, input_1, input_2):
        input_1 = self.up_convolution(input_1)

        diff_y = input_2.size(dim=2) - input_1.size(dim=2)
        diff_x = input_2.size(dim=3) - input_1.size(dim=3)

        input_1 = F.pad(input_1, [diff_x // 2, diff_x - (diff_x // 2),
                                  diff_y // 2, diff_y - (diff_y // 2)])
        
        input_ = torch.cat((input_2, input_1), dim=1)
        return self.convolution(input_)

class OutConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                     nn.Softmax(1))
        
        
    def forward(self, input_):
        return self.network(input_)


        
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        ...
    
class UNetEncoder(nn.Module):
    def __init__(self, n_channels):
        super(UNetEncoder, self).__init__()
        self.n_channels = n_channels
        self.conv = DoubleConvolution(n_channels, 64)
        self.down_conv_1 = DownConvolution(64, 128)
        self.down_conv_2 = DownConvolution(128, 256)
        self.down_conv_3 = DownConvolution(256, 512)
        self.down_conv_4 = DownConvolution(512, 1024)

    def forward(self, input_):
        input_1 = self.conv(input_)
        input_2 = self.down_conv_1(input_1)
        input_3 = self.down_conv_2(input_2)
        input_4 = self.down_conv_3(input_3)
        input_5 = self.down_conv_4(input_4)
        return input_1, input_2, input_3, input_4, input_5

class UNetDecoder(nn.Module):
    def __init__(self, n_classes):
        super(UNetDecoder, self).__init__()
        self.n_classes = n_classes

        self.up_conv_1 = UpConvolution(1024, 512)
        self.up_conv_2 = UpConvolution(512, 256)
        self.up_conv_3 = UpConvolution(256, 128)
        self.up_conv_4 = UpConvolution(128, 64)
        self.conv = OutConvolution(64, n_classes)

    def forward(self, input_1, input_2, input_3, input_4, input_5):
        input_ = self.up_conv_1(input_5, input_4)
        input_ = self.up_conv_2(input_, input_3)
        input_ = self.up_conv_3(input_, input_2)
        input_ = self.up_conv_4(input_, input_1)
        input_ = self.conv(input_)

        return input_

class UNetRegression(nn.Module):
    def __init__(self, number_of_joints):
        super(UNetRegression, self).__init__()
        self.number_of_joints = number_of_joints
        self.network = nn.Sequential(
            nn.Linear(14 * 14 * 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(2048, number_of_joints * 3),
            nn.BatchNorm1d(number_of_joints * 3),
            nn.Tanh()
            )
        
    def forward(self, input_):
        return self.network(input_)

class UNetRegression2(nn.Module):
    def __init__(self, number_of_joints):
        super(UNetRegression2, self).__init__()
        self.number_of_joints = number_of_joints
        self.network = nn.Sequential(
            nn.Linear(14 * 14 * 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.25),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Dropout(0.25),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.25),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, number_of_joints * 3),
            nn.BatchNorm1d(number_of_joints * 3),
            nn.Tanh()
            )
        
    def forward(self, input_):
        return self.network(input_)

class UNetMultitask(nn.Module):
    def __init__(self, n_channels, n_classes, n_joints):
        super(UNetMultitask, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_joints = n_joints

        self.encoder = UNetEncoder(n_channels)
        self.decoder = UNetDecoder(n_classes)
        self.regression = UNetRegression(n_joints)

    def forward(self, input_):
        input_1, input_2, input_3, input_4, input_5 = self.encoder(input_)
        segmentations = self.decoder(input_1, input_2, input_3, input_4, input_5)
        #segmentations = torch.argmax(segmentations, 1).to(torch.uint8)
        input_ = torch.flatten(input_5, 1)
        poses = self.regression(input_)
        samples = input_.shape[0]
        poses = torch.reshape(poses, (samples, 3, self.n_joints))
        return poses, segmentations


class UNetMultitask2(nn.Module):
    def __init__(self, n_channels, n_classes, n_joints):
        super(UNetMultitask2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_joints = n_joints

        self.encoder = UNetEncoder(n_channels)
        self.decoder = UNetDecoder(n_classes)
        self.regression = UNetRegression2(n_joints)

        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        self.regression.apply(init_weights)

    def forward(self, input_):
        input_1, input_2, input_3, input_4, input_5 = self.encoder(input_)
        segmentations = self.decoder(input_1, input_2, input_3, input_4, input_5)
        #segmentations = torch.argmax(segmentations, 1).to(torch.uint8)
        input_ = torch.flatten(input_5, 1)
        poses = self.regression(input_)
        samples = input_.shape[0]
        poses = torch.reshape(poses, (samples, 3, self.n_joints))
        return poses, segmentations  

class UNetPoseEstimation(nn.Module):
    def __init__(self, n_channels, n_joints):
        super(UNetPoseEstimation, self).__init__()
        self.channels = self.n_channels = n_channels
        self.joints = self.n_joints = n_joints
        self.features_network = nn.Sequential(DoubleConvolution(n_channels, 32),
                                              DownConvolution(32, 64),
                                              DownConvolution(64, 128),
                                              DownConvolution(128, 256),
                                              DownConvolution(256, 512))
        
        self.regression_network = nn.Sequential(nn.Linear(14 * 14 * 512, 1024),
                                                nn.BatchNorm1d(1024),
                                                nn.LeakyReLU(),
                                                nn.Dropout(0.25),

                                                nn.Linear(1024, 2048),
                                                nn.BatchNorm1d(2048),
                                                nn.LeakyReLU(),
                                                nn.Dropout(0.25),

                                                nn.Linear(2048, 1024),
                                                nn.BatchNorm1d(1024),
                                                nn.LeakyReLU(),
                                                nn.Dropout(0.5),

                                                nn.Linear(1024, 512),
                                                nn.BatchNorm1d(512),
                                                nn.LeakyReLU(),
                                                nn.Dropout(0.25),

                                                nn.Linear(512, n_joints * 3),
                                                nn.BatchNorm1d(n_joints * 3),
                                                nn.Tanh())
        
        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.regression_network.apply(init_weights)
        self.features_network.apply(init_weights)

    def forward(self, input_):
        features = self.features_network(input_)
        features = torch.flatten(features, 1)
        poses = self.regression_network(features)
        samples = input_.shape[0]
        poses = torch.reshape(poses, (samples, 3, self.n_joints))
        return poses
    
class UnetPoseEstimation2(nn.Module):
    def __init__(self, n_channels, n_joints):
        super().__init__()
        self.channels = self.n_channels = n_channels
        self.joints = self.n_joints = n_joints
        self.features_network = UNetEncoder(n_channels)
        self.regression_network = UNetRegression2(n_joints)

        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.regression_network.apply(init_weights)
        self.features_network.apply(init_weights)

    def forward(self, input_):
        _, _, _, _, features = self.features_network(input_)
        features = torch.flatten(features, 1)
        poses = self.regression_network(features)
        samples = input_.shape[0]
        poses = torch.reshape(poses, (samples, 3, self.n_joints))
        return poses

class UNetEncoderHalf(nn.Module):
    def __init__(self, n_channels):
        super(UNetEncoderHalf, self).__init__()
        self.n_channels = n_channels
        self.conv = DoubleConvolution(n_channels, 32)
        self.down_conv_1 = DownConvolution(32, 64)
        self.down_conv_2 = DownConvolution(64, 128)
        self.down_conv_3 = DownConvolution(128, 256)
        self.down_conv_4 = DownConvolution(256, 512)

    def forward(self, input_):
        input_1 = self.conv(input_)
        input_2 = self.down_conv_1(input_1)
        input_3 = self.down_conv_2(input_2)
        input_4 = self.down_conv_3(input_3)
        input_5 = self.down_conv_4(input_4)
        return input_1, input_2, input_3, input_4, input_5

class UNetDecoderHalf(nn.Module):
    def __init__(self, n_classes):
        super(UNetDecoderHalf, self).__init__()
        self.n_classes = n_classes

        self.up_conv_1 = UpConvolution(512, 256)
        self.up_conv_2 = UpConvolution(256, 128)
        self.up_conv_3 = UpConvolution(128, 64)
        self.up_conv_4 = UpConvolution(64, 32)
        self.conv = OutConvolution(32, n_classes)

    def forward(self, input_1, input_2, input_3, input_4, input_5):
        input_ = self.up_conv_1(input_5, input_4)
        input_ = self.up_conv_2(input_, input_3)
        input_ = self.up_conv_3(input_, input_2)
        input_ = self.up_conv_4(input_, input_1)
        input_ = self.conv(input_)

        return input_

class UNetRegressionHalf(nn.Module):
    def __init__(self, number_of_joints):
        super(UNetRegressionHalf, self).__init__()
        self.number_of_joints = number_of_joints
        self.network = nn.Sequential(
            nn.Linear(14 * 14 * 512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(2048, number_of_joints * 3),
            nn.BatchNorm1d(number_of_joints * 3),
            nn.Tanh()
            )
        
    def forward(self, input_):
        return self.network(input_)

class UNetMultitaskHalf(nn.Module):
    def __init__(self, n_channels, n_classes, n_joints):
        super(UNetMultitaskHalf, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_joints = n_joints

        self.encoder = UNetEncoderHalf(n_channels)
        self.decoder = UNetDecoderHalf(n_classes)
        self.regression = UNetRegressionHalf(n_joints)

        def init_weights(m):
            if type(m) in [nn.Linear, nn.Conv2d]:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        self.regression.apply(init_weights)

    def forward(self, input_):
        input_1, input_2, input_3, input_4, input_5 = self.encoder(input_)
        segmentations = self.decoder(input_1, input_2, input_3, input_4, input_5)
        #segmentations = torch.argmax(segmentations, 1).to(torch.uint8)
        input_ = torch.flatten(input_5, 1)
        poses = self.regression(input_)
        samples = input_.shape[0]
        poses = torch.reshape(poses, (samples, 3, self.n_joints))
        return poses, segmentations
    
class UNet3D(nn.Module): #  TODO
    def __init__(self, n_channels, n_joints):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_joints = self.joints = n_joints

        self.conv = nn.Sequential(nn.Conv2d(n_channels, 32, 5),
                                  nn.BatchNorm2d(32),
                                  nn.LeakyReLU())
        self.encoder = UNetEncoder(32)
        self.decoder = UNetDecoder(300)
        self.heat_map_net = nn.Conv2d(300, 115 * n_joints, 3, 3)

    def soft_argmax(self, voxels):
        """
        Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
        Return: 3D coordinates in shape (batch_size, channel, 3)
        """
        assert voxels.dim()==5
        # alpha is here to make the largest element really big, so it
        # would become very close to 1 after softmax
        alpha = 1#1000.0 
        N,C,W,D,H = voxels.shape  
        soft_max = nn.functional.softmax(voxels.view(N,C,-1)*alpha,dim=2)
        soft_max = soft_max.view(voxels.shape)
        indices_kernel = torch.arange(start=0,end=H*W*D,device=voxels.device).unsqueeze(0)
        indices_kernel = indices_kernel.view((H,W,D))
        conv = soft_max*indices_kernel
        indices = conv.sum(2).sum(2).sum(2)
        z = indices%D
        y = (indices/D).floor()%W
        x = (((indices/D).floor())/W).floor()%H
        coords = torch.stack([x,y,z],dim=2)
        return coords

    def forward(self, input_):
        input_ = self.conv(input_)
        input_1, input_2, input_3, input_4, input_5 = self.encoder(input_)
        input_ = self.decoder(input_1, input_2, input_3, input_4, input_5)
        heat_map = self.heat_map_net(input_)
        samples = input_.shape[0]
        heat_map = torch.reshape(heat_map, (samples, self.n_joints, 115, 115, 115))

        coords = self.soft_argmax(heat_map)
        coords /= torch.tensor((114, 114, 114), device=input_.device)
        coords *= 2
        coords -= 1
        
        return torch.swapaxes(coords, 1, 2)