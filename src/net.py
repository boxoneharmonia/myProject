import torch
import torch.nn as nn
import torch.nn.functional as F
from src.module import *
from random import randint

class EventImg2Token(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Sequential(
            ConvDw(2, config.conv_params[0][0], 1),
            *[ConvDw(config.conv_params[i][0], config.conv_params[i][1], config.conv_params[i][2]) for i in range(len(config.conv_params))],
        )
        self.fcOut = MLP_base(config.conv_params[-1][1], config.embed_dim, int(config.mlp_ratio * config.embed_dim))

        self.patch_size = config.patch_size
        self.patches = config.patches

    def forward(self, x_seq:torch.Tensor):
        """
        Forward pass of the model.
        Args:
            x_seq (torch.Tensor): image input tensor. (batch_size, seq_len, C, H, W).
        """
        B, S, C, H, W = x_seq.shape

        x_seq = x_seq.view(B * S, C, H, W)
        x_seq = self.conv(x_seq)  # conv to (B*S, config.conv_params[-1][1], H', W')
        x_seq = nn.functional.adaptive_max_pool2d(x_seq, self.patch_size) # (B*S, config.conv_params[-1][1], 3, 3)
        x_seq = x_seq.flatten(start_dim=-2).view(B, S, -1, self.patches).permute(0, 1, 3, 2).contiguous().view(B, self.patches * S, -1) # (B, 9*S, config.conv_params[-1][1])

        x = self.fcOut(x_seq)  # (B, 9*S, embed_dim)
        return x
    
class Token2EventImg(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_in = MLP_base(config.embed_dim, config.deconv_params[0][0])
        self.upconv = nn.Sequential(
            *[UpConv(config.deconv_params[i][0], config.deconv_params[i][1]) for i in range(len(config.deconv_params))],
            nn.Conv2d(config.deconv_params[-1][1], 2, kernel_size=3, padding=1),
            nn.Tanh()
        ) # H & W x 2 every deconv
        self.patch_size = config.patch_size[0]
        self.patches = config.patches

    def forward(self, token:torch.Tensor):
        token = self.fc_in(token)
        B, _, C = token.shape
        token = token.view(B, -1, self.patches, C).permute(0, 1, 3, 2).contiguous()
        token = token.view(B, -1, C, self.patch_size, self.patch_size).view(-1, C, self.patch_size, self.patch_size)  # (B*S, embed_dim, 3, 3)
        x = self.upconv(token)
        _, _, H, W = x.shape
        x_out = x.view(B, -1, 2, H, W)
        return x_out

# class Token2Traj(nn.Module):
#     def __init__(self, config) -> None:
#         super().__init__()
#         self.head = 
#         self.patches = config.patches

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential(*[
            Block(dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                  drop_ratio=config.drop_ratio, attn_drop_ratio=config.attn_drop_ratio, drop_path_ratio=config.drop_path_ratio)
            for i in range(config.depth)
        ])
        self.fcOut = nn.Sequential(
            MLP_base(config.embed_dim),
            LayerNormCompatible(config.embed_dim)
        )
        self.posEmbed = nn.Parameter(torch.zeros(1, config.max_seq_len * config.patches, config.embed_dim), requires_grad=True)
        self.clsToken = nn.Parameter(torch.zeros(1, 1, config.embed_dim), requires_grad=True)

    def forward(self, x:torch.Tensor):
        """ 
        Forward pass of the transformer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
        """
        B = x.size(0)
        cls_token = self.clsToken.expand(B, -1, -1) 
        x = x + self.posEmbed
        x = torch.cat((cls_token, x), dim=1)  # (B, seq_len+1, embed_dim)
        x = self.encoder(x)
        x = self.fcOut(x)
        return x[:, 1:, :]
    
class EventBERTBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eventImg2Token = EventImg2Token(config)
        self.transformer = Transformer(config)

    def forward(self, x_seq):
        """
        Forward pass of the model.
        Args:
            x_seq (torch.Tensor): image input tensor. (batch_size, seq_len, C, H, W).
        """
        tokens = self.eventImg2Token(x_seq)  # (B, 9*S, embed_dim)
        features = self.transformer(tokens)  # (B, 9*S, embed_dim)
        return features
    
class EventBERTMLM(EventBERTBackbone):
    def __init__(self, config):
        super().__init__(config)
        self.MLMHead = Token2EventImg(config)
        self.mask_token = nn.Parameter(torch.zeros(1, config.embed_dim), requires_grad=True)  # Mask token for MLM
        self.patches = self.eventImg2Token.patches

    def forward(self, x_seq, mask_probability=0.25):
        """
        Forward pass of the model with Mask Language Modeling (MLM).
        Args:
            x_seq (torch.Tensor): image input tensor. (batch_size, seq_len, C, H, W).
            mask_probability (float): Probability of masking tokens for MLM.
        """
        token = self.eventImg2Token(x_seq)  # (B, 9*S, embed_dim)
        B, _, T = token.shape
        S = int(token.size(1) / self.patches)  # Sequence length 
        # Create a mask for MLM
        mask_num = int(mask_probability * S)
        mask_indices = torch.randperm(S)[:mask_num]
        token = token.view(B, S, self.patches, T)
        mask_token = self.mask_token.to(token.dtype).expand(self.patches, -1)
        token[:, mask_indices] = mask_token  # Replace selected token with mask token
        token = token.view(B, -1, T)
        features = self.transformer(token)  # (B, 9*S, embed_dim)
        features = features.view(B, -1, self.patches, T)
        features = features[:, mask_indices]
        x_out = self.MLMHead(features.view(B, -1, T))  # (B, S, 2, H'', W'')
        return (x_seq[:, mask_indices], x_out)  # Return tuple of MLM output

class EventBERT(EventBERTBackbone):
    def __init__(self, config):
        super().__init__(config)
        self.patches = self.eventImg2Token.patches
        self.fcIn = MLP_base(7, config.embed_dim, config.embed_dim)
        self.fusionEncoder =  nn.Sequential(*[
            Block(dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                  drop_ratio=config.drop_ratio, attn_drop_ratio=config.attn_drop_ratio, drop_path_ratio=config.drop_path_ratio)
            for i in range(config.depth_head)
        ])
        self.fusionPosEmbed = nn.Parameter(torch.zeros(1, config.max_seq_len * (config.patches+1), config.embed_dim), requires_grad=True)
        self.head = nn.Linear(config.embed_dim, 3)

    def forward(self, x_seq:torch.Tensor, traj_seq:torch.Tensor):
        tokens = self.eventImg2Token(x_seq)  # (B, 9*S, embed_dim)
        features = self.transformer(tokens)  # (B, 9*S, embed_dim)
        traj = self.fcIn(traj_seq)  # (B, S, embed_dim)
        S = traj.size(1)
        fusion = torch.cat((traj, features), dim=1) + self.fusionPosEmbed
        fusion = self.fusionEncoder(fusion)
        traj_pr  = self.head(fusion[:, :S, :])
        return traj_pr

class EventImg2TokenV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Sequential(
            ConvDw(2, config.conv_params[0][0], 2),
            *[InvertedBottleneck(config.conv_params[i][0], config.conv_params[i][1], config.conv_params[i][2], config.conv_params[i][3]) for i in range(len(config.conv_params))],
        )
        self.fcOut = nn.Sequential(
            MLPSwiGLU(config.conv_params[-1][1], config.embed_dim, int(config.mlp_ratio * config.embed_dim)),
            LayerNormCompatible(config.embed_dim)
        )

        self.patch_size = config.patch_size
        self.patches = config.patches

    def forward(self, x_seq:torch.Tensor):
        """
        Forward pass of the model.
        Args:
            x_seq (torch.Tensor): image input tensor. (batch_size, seq_len, C, H, W).
        """
        B, S, C, H, W = x_seq.shape

        x_seq = x_seq.view(B * S, C, H, W)
        x_seq = self.conv(x_seq)  # conv to (B*S, config.conv_params[-1][1], H', W')
        x_seq = nn.functional.adaptive_max_pool2d(x_seq, self.patch_size) # (B*S, config.conv_params[-1][1], H, W)
        x_seq = x_seq.flatten(start_dim=-2).view(B, S, -1, self.patches).permute(0, 1, 3, 2).contiguous().view(B, self.patches * S, -1) # (B, num_patches*S, config.conv_params[-1][1])
        x = self.fcOut(x_seq)  # (B, num_patches*S, embed_dim)
        return x

class TransformerV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_registers = 2 # 4 from Vit need registers
        self.num_patches = config.patches 
        self.frame_len_with_registers = self.num_patches + self.num_registers
        encoder_layers = []
        for i in range(config.depth // 2):
            encoder_layers.append(BlockFT(dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                  drop_ratio=config.drop_ratio, attn_drop_ratio=config.attn_drop_ratio, drop_path_ratio=config.drop_path_ratio, frame_len=self.frame_len_with_registers, seq_len=config.max_seq_len))
            encoder_layers.append(BlockV2(dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                  drop_ratio=config.drop_ratio, attn_drop_ratio=config.attn_drop_ratio, drop_path_ratio=config.drop_path_ratio, max_seq_len=config.max_seq_len*self.frame_len_with_registers))
        self.encoder = nn.Sequential(*encoder_layers)
        self.fcOut = nn.Sequential(
            # MLPSwiGLU(config.embed_dim),
            LayerNormCompatible(config.embed_dim)
        )
        self.register = nn.Parameter(torch.zeros(1, 1, self.num_registers, config.embed_dim), requires_grad=True)
        self.seq_len = config.max_seq_len   

    def forward(self, x:torch.Tensor):
        """ 
        Forward pass of the transformer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len * num_patches, embed_dim).
        """
        B, _, C = x.shape
        register = self.register.expand(B, self.seq_len, -1, -1).flatten(0, 1)# (B*S, 4, embed_dim) 
        x = torch.cat((register, x.view(-1, self.num_patches, C)), dim=1)   # (B*S, num_patches + 4, embed_dim)
        x = x.view(B, -1, C) # (B, S*(num_patches+4), embed_dim)
        x = self.encoder(x)
        x = self.fcOut(x)
        x = x.view(-1, self.frame_len_with_registers, C)[:, self.num_registers:, :].reshape(B, -1, C) # (B, S*num_patches, embed_dim)
        return x
    
class EventBERTBackboneV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eventImg2Token = EventImg2TokenV2(config)
        self.transformer = TransformerV2(config)

    def forward(self, x_seq):
        """
        Forward pass of the model.
        Args:
            x_seq (torch.Tensor): image input tensor. (batch_size, seq_len, C, H, W).
        """
        tokens = self.eventImg2Token(x_seq)  # (B, num_patches*S, embed_dim)
        features = self.transformer(tokens)  # (B, num_patches*S, embed_dim)
        return features
    
class Token2EventImgV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_in = nn.Sequential(
            MLPSwiGLU(config.embed_dim, config.deconv_params[0][0]),
            LayerNormCompatible(config.deconv_params[0][0])
        )
        self.upconv = nn.Sequential(
            *[UpConv(config.deconv_params[i][0], config.deconv_params[i][1]) for i in range(len(config.deconv_params))],
            nn.Conv2d(config.deconv_params[-1][1], 4, kernel_size=1, padding=0),
            # nn.BatchNorm2d(4)
        ) # H & W x 2 every deconv
        self.patch_size = int(math.sqrt(config.patches))
        self.patches = config.patches

    def forward(self, token:torch.Tensor):
        token = self.fc_in(token)
        B, _, C = token.shape
        token = token.view(B, -1, self.patches, C).permute(0, 1, 3, 2).contiguous()
        token = token.view(B, -1, C, self.patch_size, self.patch_size).view(-1, C, self.patch_size, self.patch_size)  # (B*S, embed_dim, H', W')
        x = self.upconv(token)
        _, _, H, W = x.shape
        x = x.view(-1, 2, 2, H, W)
        x = F.gumbel_softmax(x, tau=1.0, dim=2, hard=False)
        x = x[:,:,0,:,:]
        x_out = x.view(B, -1, 2, H, W)
        return x_out

class EventBERTMLMV2(EventBERTBackboneV2):
    def __init__(self, config):
        super().__init__(config)
        self.MLMHead = Token2EventImgV2(config)
        self.mask_token = nn.Parameter(torch.zeros(1, config.embed_dim), requires_grad=True)  # Mask token for MLM
        self.patches = self.eventImg2Token.patches

    def forward(self, x_seq, mask_probability=0.5):
        """
        Forward pass of the model with Mask Language Modeling (MLM).
        Args:
            x_seq (torch.Tensor): image input tensor. (batch_size, seq_len, C, H, W).
            mask_probability (float): Probability of masking tokens for MLM.
        """
        token = self.eventImg2Token(x_seq)  # (B, num_patches*S, embed_dim)
        B, _, C = token.shape
        S = int(token.size(1) / self.patches)  # Sequence length 
        # Create a mask for MLM
        max_mask_num = int(mask_probability * S)
        min_mask_num = 1
        mask_num = randint(min_mask_num, max_mask_num)
        mask_indices = torch.randperm(S)[:mask_num]
        token = token.view(B, S, self.patches, C)
        mask_token = self.mask_token.to(token.dtype).expand(self.patches, -1)
        token[:, mask_indices] = mask_token  # Replace selected token with mask token
        token = token.view(B, -1, C)
        features = self.transformer(token)  # (B, num_patches*S, embed_dim)
        features = features.view(B, -1, self.patches, C)
        features = features[:, mask_indices]
        x_out = self.MLMHead(features.view(B, -1, C))  # (B, S, 2, H'', W'')
        return (x_seq[:, mask_indices], x_out)  # Return tuple of MLM output

class Token2TrajV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_registers = 1 # 4 from Vit need registers
        self.num_patches = config.patches 
        self.frame_len_with_registers = self.num_patches + self.num_registers + 1 # 1 for traj
        self.fcIn = MLPSwiGLU(8, config.embed_dim, config.embed_dim * 2)
        self.fusionEncoder =  nn.Sequential(*[
            BlockV2(dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                  drop_ratio=config.drop_ratio, attn_drop_ratio=config.attn_drop_ratio, drop_path_ratio=config.drop_path_ratio, max_seq_len=config.max_seq_len*self.frame_len_with_registers)
            for i in range(config.depth_head)
        ])
        self.head = nn.Linear(config.embed_dim, 6)
        self.register = nn.Parameter(torch.zeros(1, 1, self.num_registers, config.embed_dim), requires_grad=True)
        self.seq_len = config.max_seq_len  

    def forward(self, features:torch.Tensor, traj_seq:torch.Tensor):
        traj = self.fcIn(traj_seq)  # (B, S, embed_dim)
        B, S, C = traj.shape
        traj = traj.view(-1, C).unsqueeze(1)  # (B*S, 1, embed_dim)
        register = self.register.expand(B, self.seq_len, -1, -1).flatten(0, 1)    # (B*S, 4, embed_dim) 
        features = features.view(-1, self.num_patches, C) # (B*S, num_patches, embed_dim)
        fusion = torch.cat((traj, register, features), dim=1).view(B, -1, C) # (B, S*(1+4+num_patches), embed_dim)
        fusion = self.fusionEncoder(fusion).view(B*S, -1, C)
        fusion = fusion[:, 0, :].view(B, S, C)  # (B, S, embed_dim)
        traj_pr  = self.head(fusion)
        return traj_pr

class EventBERTV2(EventBERTBackboneV2):
    def __init__(self, config):
        super().__init__(config)
        self.trajHead = Token2TrajV2(config)

    def forward(self, x_seq:torch.Tensor, traj_seq:torch.Tensor):
        tokens = self.eventImg2Token(x_seq)  # (B, num_patches*S, embed_dim)
        features = self.transformer(tokens)  # (B, num_patches*S, embed_dim)
        traj_pr = self.trajHead(features, traj_seq)
        return traj_pr

def build_model(config):
    """
    Build the EventBERT model based on the configuration.
    Args:
        config (Config): Configuration object containing model parameters.
    Returns:
        nn.Module: The EventBERT model instance.
    """
    if config.task == 'mlm':
        model = EventBERTMLM(config)
    elif config.task == 'traj':
        model = EventBERT(config)
    elif config.task == 'mlm_v2':
        model = EventBERTMLMV2(config)
    elif config.task == 'traj_v2':
        model = EventBERTV2(config)
    else:
        raise ValueError(f"Unsupported task: {config.task}")
    
    return model

class MLMLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.bce_fn = nn.BCELoss(reduction='mean')

    def forward(self, output):
        x_gt, x_pr = output
        target_size = x_pr.shape[-2:]

        x_gt_flat = x_gt.flatten(0, 1).contiguous()  # (B*S, 2, H, W)
        x_pr_flat = x_pr.flatten(0, 1).contiguous()

        x_gt_resized = (F.interpolate(x_gt_flat, size=target_size, mode='bilinear') > 0.05).float()

        x_pr_bin = x_pr_flat.view(-1)
        x_gt_bin = x_gt_resized.view(-1)
        intersection = (x_pr_bin * x_gt_bin).sum()
        cardinality = x_pr_bin.sum() + x_gt_bin.sum()
        dice_coeff = (2. * intersection + self.eps) / (cardinality + self.eps)
        loss_dice = 1 - dice_coeff

        loss_bce = self.bce_fn(x_pr_flat, x_gt_resized)

        return loss_dice, loss_bce
    
class TrajLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output):
        traj_gt, traj_pr = output   # (B, S, 14) ['x', 'y', 'z','vx', 'vy', 'vz', 'roll', 'pitch', 'yaw','wr', 'wp', 'wy', 'rangermeter', 'dt'] (B, S, 6)
        pos_pr = traj_pr[:, :, 0:3]  
        pos_gt = traj_gt[:, :, 0:3]  
        vel_pr = traj_pr[:, :, 3:6] 
        vel_gt = traj_gt[:, :, 3:6]
        z_gt = pos_gt[:, :, 2]
        dt = traj_gt[:, 0, 13].view(-1, 1, 1)
        factor_z = torch.abs(z_gt)
        dpos_gt = (pos_gt[:, 2:, :] - pos_gt[:, :-2, :]) / (2*dt)
        dpos_pr = (pos_pr[:, 2:, :] - pos_pr[:, :-2, :]) / (2*dt)
        dvel_gt = (vel_gt[:, 2:, :] - vel_gt[:, :-2, :]) / (2*dt)
        dvel_pr = (vel_pr[:, 2:, :] - vel_pr[:, :-2, :]) / (2*dt)

        loss_position = (((pos_pr - pos_gt)**2).sum(dim=-1)**0.5 / factor_z).mean()
        loss_velocity = (((vel_pr - vel_gt)**2).sum(dim=-1)**0.5 / factor_z).mean()
        loss_dposition = (((dpos_pr - dpos_gt)**2).sum(dim=-1)**0.5 / factor_z[:, 1:-1]).mean() \
            + (((vel_pr[:, 1:-1, :] - dpos_gt)**2).sum(dim=-1)**0.5 / factor_z[:, 1:-1]).mean()
        loss_dvelocity = (((dvel_pr - dvel_gt)**2).sum(dim=-1)**0.5 / factor_z[:, 1:-1]).mean() 
        return loss_position, loss_velocity, loss_dposition, loss_dvelocity

def build_criterion(config):
    """
    Build the loss function based on the configuration.
    Args:
        config (Config): Configuration object containing task parameters.
    Returns:
        nn.Module: The loss function instance.
    """
    if config.task in ['mlm','mlm_v2']:
        criterion = MLMLoss()
    elif config.task in ['traj','traj_v2']:
        criterion = TrajLoss()
    else:
        raise ValueError(f"Unsupported task: {config.task}")
    
    return criterion
