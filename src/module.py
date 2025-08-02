import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
    
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerNormCompatible(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return norm_x * self.weight + self.bias

class MLP_base(nn.Module):
    def __init__(self,inp,oup=None,hidden=None,drop=0.0):
        super(MLP_base, self).__init__()
        oup = oup or inp
        hidden = hidden or inp
        self.fc1 = nn.Linear(inp,hidden)
        self.fc2 = nn.Sequential(
            LayerNormCompatible(hidden),
            nn.GELU(),
            nn.Linear(hidden,oup),
        )
        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 norm_layer=LayerNormCompatible):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP_base(dim, hidden=mlp_hidden_dim, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, tgt, memory):  # tgt = query, memory = key/value
        B, N, C = tgt.shape
        _, M, _ = memory.shape

        q = self.q(tgt).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)  # B, H, N, D
        kv = self.kv(memory).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # B, H, M, D

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, H, N, M
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.0,
                 attn_drop_ratio=0.0,
                 drop_path_ratio=0.0,
                 norm_layer=LayerNormCompatible):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop_ratio, drop_ratio)
        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop_ratio, drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP_base(dim, hidden=mlp_hidden_dim, drop=drop_ratio)

    def forward(self, query, memory):
        query = query + self.drop_path(self.self_attn(self.norm1(query)))              # Self-Attention
        query = query + self.drop_path(self.cross_attn(self.norm2(query), memory))     # Cross-Attention
        query = query + self.drop_path(self.mlp(self.norm3(query)))                    # FFN
        return query

class ConvDw(nn.Module):
    # Serapable convolution module consisting of
    # 1. Depthwise convolution (3x3)
    # 2. pointwise convolution (1x1)
    # Reference:
    # Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang,
    # Tobias Weyand, Marco Andreetto, and Hartwig Adam. MobileNets: Efficient
    # convolutional neural neworks for mobile vision applications. CoRR, abs/1704.04861, 2017.

    def __init__(self, inp, oup, stride):
        super(ConvDw, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, stride=stride, padding=1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.LeakyReLU(inplace=True),
            #nn.Dropout2d(p=0.1,inplace=False),
            # pw
            nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(inplace=True),
        )
        # self.depth = oup
    def forward(self, x):
        return self.conv(x)

class ConvBasic(nn.Module):
    def __init__(self, inp, oup, size, stride, padding):
        super(ConvBasic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, size, stride=stride, padding=padding),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class SE(nn.Module):
    def __init__(self, c1, ratio=8):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // ratio, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(c1 // ratio, c1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)
        
class UpConv(nn.Module):
    def __init__(self, inp, oup=None):
        super(UpConv, self).__init__()
        oup = oup or inp // 2
        self.conv1 = nn.Sequential(
            ConvDw(inp, 2*oup, 1),
            ConvDw(2*oup, oup, 1),
        )
        self.conv2 = ConvDw(oup, oup, 1)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = F.interpolate(x1, scale_factor=2.0, mode='bilinear', align_corners=False)
        x3 = self.conv2(x2)
        return x3
