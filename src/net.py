import torch
import torch.nn as nn
import torch.nn.functional as F
from src.module import *

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
            *[ConvDw(config.conv_params[i][0], config.conv_params[i][1], config.conv_params[i][2]) for i in range(len(config.conv_params))],
        )
        self.fcOut = MLPSwiGLU(config.conv_params[-1][1], config.embed_dim, int(config.mlp_ratio * config.embed_dim))

        # self.patch_size = config.patch_size
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
        # x_seq = nn.functional.adaptive_max_pool2d(x_seq, self.patch_size) # (B*S, config.conv_params[-1][1], 3, 3)
        x_seq = x_seq.flatten(start_dim=-2).view(B, S, -1, self.patches).permute(0, 1, 3, 2).contiguous().view(B, self.patches * S, -1) # (B, num_patches*S, config.conv_params[-1][1])
        x = self.fcOut(x_seq)  # (B, num_patches*S, embed_dim)
        return x

class TransformerV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_registers = 4 # 4 from Vit need registers
        self.num_patches = config.patches 
        self.frame_len_with_registers = self.num_patches + self.num_registers
        self.encoder = nn.Sequential(*[
            BlockFT(dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                  drop_ratio=config.drop_ratio, attn_drop_ratio=config.attn_drop_ratio, drop_path_ratio=config.drop_path_ratio, frame_len=self.frame_len_with_registers, seq_len=config.max_seq_len)
            for i in range(config.depth)
        ])
        self.fcOut = nn.Sequential(
            MLPSwiGLU(config.embed_dim),
            LayerNormCompatible(config.embed_dim)
        )
        self.register = nn.Parameter(torch.zeros(1, config.max_seq_len, self.num_registers, config.embed_dim), requires_grad=True)   

    def forward(self, x:torch.Tensor):
        """ 
        Forward pass of the transformer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len * num_patches, embed_dim).
        """
        B, _, C = x.shape
        register = self.register.expand(B, -1, -1, -1).flatten(0, 1)# (B*S, 4, embed_dim) 
        x = torch.cat((register, x.view(-1, self.num_patches, C)), dim=1)   # (B*S, num_patches + 4, embed_dim)
        x = x.view(B, -1, C) # (B, S*(num_patches+4), embed_dim)
        x = self.encoder(x)
        x = self.fcOut(x)
        x = x.view(-1, self.frame_len_with_registers, C)[:, 4:, :].reshape(B, -1, C) # (B, S*num_patches, embed_dim)
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
        self.fc_in = MLP_base(config.embed_dim, config.deconv_params[0][0])
        self.upconv = nn.Sequential(
            *[UpConv(config.deconv_params[i][0], config.deconv_params[i][1]) for i in range(len(config.deconv_params))],
            nn.Conv2d(config.deconv_params[-1][1], 2, kernel_size=3, padding=1),
            nn.Tanh()
        ) # H & W x 2 every deconv
        self.patch_size = int(math.sqrt(config.patches))
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
        mask_num = int(mask_probability * S)
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
        self.num_registers = 4 # 4 from Vit need registers
        self.num_patches = config.patches 
        self.frame_len_with_registers = self.num_patches + self.num_registers + 1 # 1 for traj
        self.fcIn = MLPSwiGLU(7, config.embed_dim, config.embed_dim * 2)
        self.fusionEncoder =  nn.Sequential(*[
            BlockV2(dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, qkv_bias=config.qkv_bias, qk_scale=config.qk_scale,
                  drop_ratio=config.drop_ratio, attn_drop_ratio=config.attn_drop_ratio, drop_path_ratio=config.drop_path_ratio, max_seq_len=config.max_seq_len*self.frame_len_with_registers)
            for i in range(config.depth_head)
        ])
        self.head = nn.Linear(config.embed_dim, 3)
        self.register = nn.Parameter(torch.zeros(1, config.max_seq_len, self.num_registers, config.embed_dim), requires_grad=True) 

    def forward(self, features:torch.Tensor, traj_seq:torch.Tensor):
        traj = self.fcIn(traj_seq)  # (B, S, embed_dim)
        B, S, C = traj.shape
        traj = traj.view(-1, C).unsqueeze(1)  # (B*S, 1, embed_dim)
        register = self.register.expand(B, -1, -1, -1).flatten(0, 1)    # (B*S, 4, embed_dim) 
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
    def __init__(self):
        super().__init__()

    def _get_weights(self, true_tensor):
        is_event_pixel  = (true_tensor > -0.95).float()
        num_pixels = true_tensor.shape[1] * true_tensor.shape[2] * true_tensor.shape[3]
        num_events = torch.sum(is_event_pixel, dim=(1, 2, 3), keepdim=True)
        num_background = num_pixels - num_events
        weight_event = torch.clamp(num_background / (num_events + 1), max=1000.0)
        # print(f"num events {num_events.mean().item()}, num_pixels {num_pixels}, weight {weight_event.mean().item()}")
        weights = 1.0 + (weight_event - 1.0) * is_event_pixel 
        return weights

    def weighed_mse(self, pred, true, weights):
        return (((pred - true) ** 2) * weights).mean()

    def weighed_mae(self, pred, true, weights):
        return (torch.abs(pred - true) * weights).mean()

    def forward(self, output):
        x_gt, x_pred = output
        target_size = x_pred.shape[-2:]

        x_gt_flat = x_gt.flatten(0, 1)  # (B*S, 2, H, W)
        x_pred_flat = x_pred.flatten(0, 1)

        x_gt_resized = F.interpolate(x_gt_flat, size=target_size, mode='bilinear')
        weights = self._get_weights(x_gt_resized)
        
        loss_mse = self.weighed_mse(x_pred_flat, x_gt_resized, weights)
        # loss_mae = self.weighed_mae(x_pred_flat, x_gt_resized, weights)
        
        # total_loss = (loss_mse + loss_mae) / 2
        return loss_mse
    
class TrajLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output):
        traj_gt, vel_pr = output   # (B, S, 4) (B, S, 3) 
        vel_gt = traj_gt[:, :, 1:]
        z_gt = traj_gt[:, :, 0]
        factor_z = torch.abs(z_gt)

        loss_velocity = (((vel_pr - vel_gt)**2).sum(dim=-1)**0.5 / factor_z).mean()

        return loss_velocity

def build_criterion(config):
    """
    Build the loss function based on the configuration.
    Args:
        config (Config): Configuration object containing task parameters.
    Returns:
        nn.Module: The loss function instance.
    """
    if config.task == 'mlm' or config.task == 'mlm_v2':
        criterion = MLMLoss()
    elif config.task == 'traj' or config.task == 'traj_v2':
        criterion = TrajLoss()
    else:
        raise ValueError(f"Unsupported task: {config.task}")
    
    return criterion
