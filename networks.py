import os
import numpy as np
import torch.nn as nn
from torchvision.models import vgg19
import torch.nn.functional as F
import torch
from math import sqrt
import math
from torch.nn import init
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class ConvBlock(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size,stride,padding,tdim):
        super(ConvBlock, self).__init__()
        self.conv1=nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,padding=padding),

        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),)

        self.conv2=nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Conv2d(out_ch, out_ch, kernel_size, stride=stride,padding=padding),
        )
        self.conv3=nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Conv2d(out_ch, out_ch, kernel_size, stride=stride,padding=padding),

        )
        self.temb_proj2 = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),)

        self.conv4=nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Conv2d(out_ch, out_ch, kernel_size, stride=stride,padding=padding),
        )

    def forward(self, x, temb):
        h = self.conv1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.conv2(h)
        h = self.conv3(h)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.conv4(h)
        return h

class Upconv(nn.Module):

    def __init__(self,in_ch,out_ch):
        super(Upconv, self).__init__()
        self.up=nn.Sequential(
        # nn.UpsamplingBilinear2d(scale_factor=2),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.up(x)
        return x

class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.main(x)
        return x

class Generator(nn.Module):
    def __init__(self,T,tdim=512):
        super(Generator, self).__init__()
        self.time_embedding = TimeEmbedding(T, 64, tdim)
        self.head = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.conv1 = ConvBlock(in_ch=64,out_ch=64,kernel_size=3,stride=1,padding=1,tdim=tdim)
        self.down1 =DownSample(64)
        self.conv2 = ConvBlock(in_ch=64, out_ch=128, kernel_size=3, stride=1, padding=1,tdim=tdim)
        self.down2 = DownSample(128)
        self.conv3 = ConvBlock(in_ch=128, out_ch=256, kernel_size=3, stride=1, padding=1,tdim=tdim)
        self.down3 = DownSample(256)
        self.conv4 = ConvBlock(in_ch=256, out_ch=512, kernel_size=3, stride=1, padding=1,tdim=tdim)
        self.down4 = DownSample(512)
        self.conv5 = ConvBlock(in_ch=512, out_ch=1024, kernel_size=3, stride=1, padding=1,tdim=tdim)

        self.up5 =Upconv(in_ch=512, out_ch=512)
        self.up_conv5=ConvBlock(in_ch=1024,out_ch=512,kernel_size=3,stride=1,padding=1,tdim=tdim)
        self.up4 =Upconv(in_ch=512, out_ch=256)
        self.up_conv4=ConvBlock(in_ch=512,out_ch=256,kernel_size=3,stride=1,padding=1,tdim=tdim)
        self.up3=Upconv(in_ch=256, out_ch=128)
        self.up_conv3=ConvBlock(in_ch=256,out_ch=128,kernel_size=3,stride=1,padding=1,tdim=tdim)
        self.up2=Upconv(in_ch=128, out_ch=64)
        self.up_conv2=ConvBlock(in_ch=128,out_ch=64,kernel_size=3,stride=1,padding=1,tdim=tdim)
        self.transformer = Transformer()
        self.last_conv = nn.Sequential(
            nn.GroupNorm(32, 64),
            Swish(),
            nn.Conv2d(64, 1, 3, stride=1, padding=1)
        )
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
    def forward(self,y,x,t):
        temb = self.time_embedding(t)
        x0=torch.cat((y, x), dim=1)
        e1=self.head(x0)
        e1 = self.conv1(e1,temb)
        e2 = self.down1(e1)

        e2 = self.conv2(e2,temb)
        e3 = self.down2(e2)

        e3 = self.conv3(e3,temb)
        e4 = self.down3(e3)

        e4 = self.conv4(e4,temb)
        e5 = self.down4(e4)
        # e5 = self.conv5(e5,temb)
        e5 = self.transformer(e5)
        d5 = self.up5(e5)
        d5 = torch.cat((e4, d5), dim=1)  # 将e4特征图与d5特征图横向拼接
        d5 = self.up_conv5(d5,temb)

        d4 = self.up4(d5)
        d4 = torch.cat((e3, d4), dim=1)  # 将e3特征图与d4特征图横向拼接
        d4 = self.up_conv4(d4,temb)

        d3 = self.up3(d4)
        d3 = torch.cat((e2, d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        d3 = self.up_conv3(d3,temb)

        d2 = self.up2(d3)
        d2 = torch.cat((e1, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        d2 = self.up_conv2(d2,temb)
        out = self.last_conv(d2)
        return out



class GeneratorTransUNet(nn.Module):
    def __init__(self):
        super(GeneratorTransUNet, self).__init__()


        # for content learning
        # encoder
        self.conv1_1_01 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1_1_01 = nn.BatchNorm2d(64)
        self.conv1_2_01 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2_01 = nn.BatchNorm2d(64)

        self.conv2_1_01 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1_01 = nn.BatchNorm2d(128)
        self.conv2_2_01 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2_01 = nn.BatchNorm2d(128)

        self.conv3_1_01 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1_01 = nn.BatchNorm2d(256)
        self.conv3_2_01 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2_01 = nn.BatchNorm2d(256)

        self.conv4_1_01 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1_01 = nn.BatchNorm2d(512)
        self.conv4_2_01 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2_01 = nn.BatchNorm2d(512)

        # self.conv4_3_01 = nn.Conv2d(512, 256, 3, padding=1)
        # self.bn4_3_01 = nn.BatchNorm2d(256)
        self.upconv4_1_01 = nn.Conv2d(1024, 512, 3, padding=1)
        self.upbn4_1_01 = nn.BatchNorm2d(512)
        self.upconv4_2_01 = nn.Conv2d(512, 256, 3, padding=1)
        self.upbn4_2_01 = nn.BatchNorm2d(256)

        self.upconv3_1_01 = nn.Conv2d(512, 256, 3, padding=1)
        self.upbn3_1_01 = nn.BatchNorm2d(256)
        self.upconv3_2_01 = nn.Conv2d(256, 128, 3, padding=1)
        self.upbn3_2_01 = nn.BatchNorm2d(128)

        self.upconv2_1_01 = nn.Conv2d(256, 128, 3, padding=1)
        self.upbn2_1_01 = nn.BatchNorm2d(128)
        self.upconv2_2_01 = nn.Conv2d(128, 64, 3, padding=1)
        self.upbn2_2_01 = nn.BatchNorm2d(64)

        self.upconv1_1_01 = nn.Conv2d(128, 32, 3, padding=1)
        self.upbn1_1_01 = nn.BatchNorm2d(32)
        self.upconv1_2_01 = nn.Conv2d(32, 1, 3, padding=1)

        self.transformer=Transformer()

        # ************************************************************
        # for noise learning
        # encoder
        # ************************************************************
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # ************************************************************
        # fusion mechanism
        # ************************************************************


    def forward(self, x0):

        # encoder for content learning
        x1_1_01 = F.relu(self.bn1_1_01(self.conv1_1_01(x0)))
        x1_2_01 = F.relu(self.bn1_2_01(self.conv1_2_01(x1_1_01)))

        x2_0_01 = self.maxpool(x1_2_01)
        x2_1_01 = F.relu(self.bn2_1_01(self.conv2_1_01(x2_0_01)))
        x2_2_01 = F.relu(self.bn2_2_01(self.conv2_2_01(x2_1_01)))

        x3_0_01 = self.maxpool(x2_2_01)
        x3_1_01 = F.relu(self.bn3_1_01(self.conv3_1_01(x3_0_01)))
        x3_2_01 = F.relu(self.bn3_2_01(self.conv3_2_01(x3_1_01)))

        x4_0_01 = self.maxpool(x3_2_01)
        x4_1_01 = F.relu(self.bn4_1_01(self.conv4_1_01(x4_0_01)))
        x4_2_01 = F.relu(self.bn4_2_01(self.conv4_2_01(x4_1_01)))#512,64,64
        x5_0_01 = self.maxpool(x4_2_01)  #512,32,32
        x5_1_01=self.transformer(x5_0_01)#512,32,32
        # x5_2_01=self.upsample(x5_1_01)

        upx4_1_01 = self.upsample(x5_1_01)#512,64,64
        upx4_2_01 = F.relu(self.upbn4_1_01(self.upconv4_1_01(torch.cat((upx4_1_01, x4_2_01), 1))))
        upx4_3_01 = F.relu(self.upbn4_2_01(self.upconv4_2_01(upx4_2_01)))

        upx3_1_01 = self.upsample(upx4_3_01)
        upx3_2_01 = F.relu(self.upbn3_1_01(self.upconv3_1_01(torch.cat((upx3_1_01, x3_2_01), 1))))
        upx3_3_01 = F.relu(self.upbn3_2_01(self.upconv3_2_01(upx3_2_01)))

        upx2_1_01 = self.upsample(upx3_3_01)
        upx2_2_01 = F.relu(self.upbn2_1_01(self.upconv2_1_01(torch.cat((upx2_1_01, x2_2_01), 1))))
        upx2_3_01 = F.relu(self.upbn2_2_01(self.upconv2_2_01(upx2_2_01)))

        upx1_1_01 = self.upsample(upx2_3_01)
        upx1_2_01 = self.upconv1_1_01(torch.cat((upx1_1_01, x1_2_01), 1))
        content_1 = F.relu(self.upconv1_2_01(upx1_2_01))

        # ************************************************************
        # encoder for noise learning

        # fusion mechanism
        return content_1



class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self):
        super(Embeddings, self).__init__()
        self.patch_embeddings = nn.Conv2d(in_channels=512,
                                       out_channels=256,
                                       kernel_size=32,
                                       stride=32)
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1024, 256))
    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        return embeddings

class attention(nn.Module):
    # 初始化
    def __init__(self, dim=256, num_heads=4, qkv_bias=False, atten_drop_ratio=0., proj_drop_ratio=0.):
        super(attention, self).__init__()

        # 多头注意力的数量
        self.num_heads = num_heads
        # 将生成的qkv均分成num_heads个。得到每个head的qkv对应的通道数。
        head_dim = dim // num_heads
        # 公式中的分母
        self.scale = head_dim ** -0.5

        # 通过一个全连接层计算qkv
        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=qkv_bias)
        # dropout层
        self.atten_drop = nn.Dropout(atten_drop_ratio)

        # 再qkv计算完之后通过一个全连接提取特征
        self.proj = nn.Linear(in_features=dim, out_features=dim)
        # dropout层
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    # 前向传播
    def forward(self, inputs):
        # 获取输入图像的shape=[b,1024,256]
        B, N, C = inputs.shape

        # 将输入特征图经过全连接层生成qkv [b,1024,256]==>[b,1024,256*3]
        qkv = self.qkv(inputs)

        # 维度调整 [b,1024,256*3]==>[b, 1024, 3, 4, 256//4]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # 维度重排==> [3, B, 4, 1024, 256//4]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # 切片提取q、k、v的值，单个的shape=[B, 4, 1024, 256//4]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 针对每个head计算 ==> [B, 4, 1024, 1024]
        atten = (q @ k.transpose(-2, -1)) * self.scale  # @ 代表在多维tensor的最后两个维度矩阵相乘
        # 对计算结果的每一行经过softmax
        atten = atten.softmax(dim=-1)
        # dropout层
        atten = self.atten_drop(atten)

        # softmax后的结果和v加权 ==> [B, 4, 1024, 256//4]
        x = atten @ v
        # 通道重排 ==> [B, 1024, 4, 256//4]
        x = x.transpose(1, 2)
        # 维度调整 ==> [B, 1024, 256]
        x = x.reshape(B, N, C)

        # 通过全连接层融合特征 ==> [B, 1024, 256]
        x = self.proj(x)
        # dropout层
        x = self.proj_drop(x)

        return x


# --------------------------------------- #
# （4）MLP多层感知器
'''
in_features : 输入特征图的通道数
hidden_features : 第一个全连接层上升通道数
out_features : 第二个全连接层的下降的通道数
drop : 全连接层后面的dropout层的杀死神经元的概率
'''


# --------------------------------------- #
class MLP(nn.Module):
    # 初始化
    def __init__(self, in_features, hidden_features, out_features=None, drop=0.):
        super(MLP, self).__init__()

        # MLP的输出通道数默认等于输入通道数
        out_features = out_features or in_features
        # 第一个全连接层上升通道数
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        # GeLU激活函数
        self.act = nn.GELU()
        # 第二个全连接下降通道数
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        # dropout层
        self.drop = nn.Dropout(drop)

    # 前向传播
    def forward(self, inputs):
        # [b,1024,256]==>[b,1024,1024]
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop(x)

        # [b,1024,1024]==>[b,1024,256]
        x = self.fc2(x)
        x = self.drop(x)

        return x


# --------------------------------------- #
# （5）Encoder Block
'''
dim : 该模块的输入特征图个数
mlp_ratio ： MLP中第一个全连接层上升的通道数
drop_ratio : 该模块的dropout层的杀死神经元的概率
'''


# --------------------------------------- #
class encoder_block(nn.Module):
    # 初始化
    def __init__(self, dim=256, mlp_ratio=4., drop_ratio=0.):
        super(encoder_block, self).__init__()

        # LayerNormalization层
        self.norm1 = nn.LayerNorm(dim)
        # 实例化多头注意力
        self.atten = attention(dim)
        # dropout
        self.drop = nn.Dropout()

        # LayerNormalization层
        self.norm2 = nn.LayerNorm(dim)
        # MLP中第一个全连接层上升的通道数
        hidden_features = int(dim * mlp_ratio)
        # MLP多层感知器
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features)

    # 前向传播
    def forward(self, inputs):
        # [b,1024,256]==>[b,1024,256]
        x = self.norm1(inputs)
        x = self.atten(x)
        x = self.drop(x)
        feat1 = x + inputs  # 残差连接

        # [b,1024,256]==>[b,1024,256]
        x = self.norm2(feat1)
        x = self.mlp(x)
        x = self.drop(x)
        feat2 = x + feat1  # 残差连接

        return feat2
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.embeddings=Embeddings()
        self.blocks = nn.Sequential(*[encoder_block() for _ in range(12)])
        self.conv_more = Conv2dReLU(256,512,kernel_size=3,padding=1,use_batchnorm=True,)
    def forward(self, inputs):
        inputs=self.embeddings(inputs)
        inputs = self.blocks(inputs)
        B, n_patch, hidden = inputs.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = inputs.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        return self.conv_more(x)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Generator(T=1000).to(device)

    tensor1 = torch.randn([2,1,512,512]).to(device)
    t = torch.randint(1000, size=(tensor1 .shape[0],), device=device)
    # tensor2=torch.tensor(np.ones((2, 1, 64, 64)))
    # print(tensor2.shape)
    # B, n_patch, hidden = tensor1.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
    # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
    # x = tensor1.permute(0, 2, 1)
    # x = x.contiguous().view(B, hidden, h, w)
    out=model(tensor1,tensor1,t)
    print(out.shape)