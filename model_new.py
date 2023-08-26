import torch.nn as nn
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_filters, out_filters, kernelsize=(3, 3), n=2):
        super(Encoder, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(n):
            self.convs.append(nn.Conv2d(in_filters, out_filters, kernel_size=kernelsize, padding='same'))
            in_filters = out_filters
        self.bn = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU()
        self.n = n
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        for i in range(self.n):
            x = self.relu(self.bn(self.convs[i](x)))
        x, indices = self.pool(x)

        return x, indices


class Decoder(nn.Module):
    def __init__(self, in_filters, out_filters, kernelsize=(3, 3), n=2):
        super(Decoder, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(n):
            self.convs.append(nn.Conv2d(in_filters, out_filters, kernel_size=kernelsize, padding='same'))
            in_filters = out_filters
        self.bn = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU()
        self.unpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.n = n
        self.out_filter = out_filters

    def forward(self, x, mask):

        x = self.unpooling(x, mask)
        for i in range(self.n):
            x = self.relu(self.bn(self.convs[i](x)))
        return x


class SegNet(nn.Module):
    def __init__(self, out_channels=19):
        super(SegNet, self).__init__()
        self.encoder_1 = Encoder(in_filters=3, out_filters=64, n=2)
        self.encoder_2 = Encoder(in_filters=64, out_filters=64, n=2)
        self.encoder_3 = Encoder(in_filters=64, out_filters=128, n=2)
        self.encoder_4 = Encoder(in_filters=128, out_filters=256, n=2)
        self.encoder_5 = Encoder(in_filters=256, out_filters=512, n=2)
        self.decoder_1 = Decoder(in_filters=512, out_filters=256, n=2)
        self.decoder_2 = Decoder(in_filters=256, out_filters=128, n=2)
        self.decoder_3 = Decoder(in_filters=128, out_filters=64, n=2)
        self.decoder_4 = Decoder(in_filters=64, out_filters=64, n=2)
        self.decoder_5 = Decoder(in_filters=64, out_filters=64, n=2)
        self.conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=(3, 3), padding='same')
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        mask0 = x
        x, mask1 = self.encoder_1(x)
        x, mask2 = self.encoder_2(x)
        x, mask3 = self.encoder_3(x)
        x, mask4 = self.encoder_4(x)
        x, mask5 = self.encoder_5(x)
        x = self.decoder_1(x, mask5)  # Pass mask5[0] instead of mask5
        x = self.decoder_2(x, mask4)  # Pass mask4[0] instead of mask4
        x = self.decoder_3(x, mask3)  # Pass mask3[0] instead of mask3
        x = self.decoder_4(x, mask2)  # Pass mask2[0] instead of mask2
        x = self.decoder_5(x, mask1)  # Pass mask1[0] instead of mask1
        x = self.bn(self.conv(x))

        return x
    
    


import torch
import torch.nn as nn
import torch.nn.functional as F



class ASPP_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dialation):
        super(ASPP_Block, self).__init__()
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), dilation=dialation, padding=dialation, stride=(1, 1))

    def forward(self, x):
        #print(x.shape)
        x = self.conv(self.relu(self.batchnorm(x)))

        return x


class DenseASPP(nn.Module):
    def __init__(self, in_channels, num_classes, depth=16):
        super(DenseASPP, self).__init__()

        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channels, depth, (1, 1), (1, 1))

        self.Blocks = nn.ModuleList()
        self.Block_1 = ASPP_Block(in_channels, depth, 3)
        self.Block_2 = ASPP_Block(in_channels, depth*2, 6)
        self.Block_3 = ASPP_Block(in_channels, depth*4, 12)
        self.Block_4 = ASPP_Block(in_channels, depth*8, 18)
        self.Block_5 = ASPP_Block(in_channels, depth*16, 24)
        self.Blocks.extend([self.Block_1, self.Block_2, self.Block_3, self.Block_4, self.Block_5])
        self.conv_out = nn.Conv2d(in_channels=depth*32, out_channels=num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):

        #image_feature = self.mean(x)
        x = self.conv(x)
        output = x

        for i in range(5):
            #print(i)
            out = self.Blocks[i](x)
            #print(x.shape, out.shape)
            output = torch.cat([output, out], dim=1)
            x = torch.cat([x, out], dim=1)


        x = self.conv_out(output)

        return x

