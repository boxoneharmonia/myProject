import torch
import torch.nn as nn
import torch.nn.functional as F
from src.module import *

class EventImg2Token(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.convPos = nn.Sequential(
            *[ConvDw(config.conv_params[i][0], config.conv_params[i][1], config.conv_params[i][2]) for i in range(len(config.conv_params))],
        )
        self.convNeg = nn.Sequential(
            *[ConvDw(config.conv_params[i][0], config.conv_params[i][1], config.conv_params[i][2]) for i in range(len(config.conv_params))],
        )
        self.fcOut = nn.Sequential(
            MLP_base(config.conv_params[-1][1] * 2, config.token_len, int(config.mlp_ratio * config.token_len)),
            LayerNormCompatible(config.token_len)
        )
        self.patch_size = config.patch_size
        self.patches = config.patches

    def forward(self, x_pos:torch.Tensor, x_neg:torch.Tensor):
        """
        Forward pass of the model.
        Args:
            x_pos (torch.Tensor): Positive image input tensor. (batch_size, seq_len, C, H, W).
            x_neg (torch.Tensor): Negative image input tensor. (batch_size, seq_len, C, H, W).
        """
        B, S, C, H, W = x_pos.shape

        x_pos = x_pos.view(B * S, C, H, W)
        x_pos = self.convPos(x_pos)  # conv to (B*S, config.conv_params[-1][1], H', W')
        x_pos = nn.functional.adaptive_max_pool2d(x_pos, self.patch_size) # (B*S, config.conv_params[-1][1], 3, 3)
        x_pos = x_pos.flatten(start_dim=-2).view(B, S, -1, self.patches).permute(0, 1, 3, 2).reshape(B, self.patches * S, -1) # (B, 9*S, config.conv_params[-1][1])
        
        x_neg = x_neg.view(B * S, C, H, W)
        x_neg = self.convNeg(x_neg)
        x_neg = nn.functional.adaptive_max_pool2d(x_neg, self.patch_size)
        x_neg = x_neg.flatten(start_dim=-2).view(B, S, -1, self.patches).permute(0, 1, 3, 2).reshape(B, self.patches * S, -1)

        x = torch.cat((x_pos, x_neg), dim=-1)  # (B, 9*S, config.conv_params[-1][1]*2)
        x = self.fcOut(x)  # (B, 9*S, token_len)
        return x
    
class Token2EventImg(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_in = nn.Sequential(
            MLP_base(config.embed_dim, config.deconv_params[0][0]),
            LayerNormCompatible(config.deconv_params[0][0])
        )
        self.deconv = nn.Sequential(
            *[deConv(config.deconv_params[i][0], config.deconv_params[i][1]) for i in range(len(config.deconv_params))],
            nn.Conv2d(config.deconv_params[-1][1], 6, kernel_size=3, padding=1),
            nn.Tanh()
        ) # H & W x 2 every deconv
        self.patch_size = config.patch_size[0]
        self.patches = config.patches

    def forward(self, token:torch.Tensor):
        token = self.fc_in(token)
        B, _, T = token.shape
        token = token.view(B, -1, self.patches, T).permute(0, 1, 3, 2).contiguous()
        token = token.view(B, -1, T, self.patch_size, self.patch_size).view(-1, T, self.patch_size, self.patch_size)  # (B*S, token_len, 3, 3)
        x = self.deconv(token)
        x_pos = x[:,:3,:,:]
        x_neg = x[:,3:,:,:]
        _, _, H, W = x_pos.shape
        x_pos = x_pos.view(B, -1, 3, H, W)
        x_neg = x_neg.view(B, -1, 3, H, W)
        return x_pos, x_neg

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
        x = x + self.posEmbed[:, :x.size(1), :]
        x = torch.cat((cls_token, x), dim=1)  # (B, seq_len+1, embed_dim)
        x = self.encoder(x)
        x = self.fcOut(x)
        return x[:, 1:, :]
    
class EventBERTBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eventImg2Token = EventImg2Token(config)
        self.transformer = Transformer(config)

    def forward(self, x_pos, x_neg):
        """
        Forward pass of the model.
        Args:
            x_pos (torch.Tensor): Positive image input tensor. (batch_size, seq_len, C, H, W).
            x_neg (torch.Tensor): Negative image input tensor. (batch_size, seq_len, C, H, W).
        """
        tokens = self.eventImg2Token(x_pos, x_neg)  # (B, 9*S, token_len)
        features = self.transformer(tokens)  # (B, 9*S, embed_dim)
        return features
    
class EventBERTMLM(EventBERTBackbone):
    def __init__(self, config):
        super().__init__(config)
        self.MLMHead = Token2EventImg(config)
        self.mask_token = nn.Parameter(torch.zeros(1, config.token_len), requires_grad=True)  # Mask token for MLM
        self.patches = self.eventImg2Token.patches

    def forward(self, x_pos, x_neg, mask_probability=0.25):
        """
        Forward pass of the model with Mask Language Modeling (MLM).
        Args:
            x_pos (torch.Tensor): Positive image input tensor. (batch_size, seq_len, C, H, W).
            x_neg (torch.Tensor): Negative image input tensor. (batch_size, seq_len, C, H, W).
            mask_probability (float): Probability of masking tokens for MLM.
        """
        token = self.eventImg2Token(x_pos, x_neg)  # (B, 9*S, token_len)
        B, _, T = token.shape
        S = int(token.size(1) / self.patches)  # Sequence length 
        # Create a mask for MLM
        mask_num = int(mask_probability * S)
        mask_indices = torch.randperm(S)[:mask_num]
        token = token.view(B, -1, self.patches, T)
        mask_token = self.mask_token.to(token.dtype).expand(self.patches, -1)
        token[:, mask_indices] = mask_token  # Replace selected token with mask token
        token = token.view(B, -1, T)
        features = self.transformer(token)  # (B, 9*S, embed_dim)
        features = features.view(B, -1, self.patches, T)
        features = features[:, mask_indices]
        y_pos, y_neg = self.MLMHead(features.view(B, -1, T))  # (B, S, 3, H'', W'')
        return (x_pos[:, mask_indices], x_neg[:, mask_indices], y_pos, y_neg)  # Return tuple of MLM output

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
    else:
        raise ValueError(f"Unsupported task: {config.task}")
    
    return model

class MLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.non_black_weight = 50.0

    def forward(self, output):
        """
        Compute the MLM loss.
        Args:
            output (tuple): Tuple containing x_pos[:, mask_indices], x_neg[:, mask_indices], y_pos[:, mask_indices], y_neg[:, mask_indices]
        Returns:
            torch.Tensor: Computed loss value.
        """
        true_pos, true_neg, pred_pos, pred_neg = output
        target_size = pred_pos.shape[-2:]
        true_pos_resized = F.interpolate(true_pos.flatten(0,1), size=target_size, mode='bilinear')
        true_neg_resized = F.interpolate(true_neg.flatten(0,1), size=target_size, mode='bilinear')
        loss_pos = self.weighed_mse(pred_pos.flatten(0,1), true_pos_resized)
        loss_neg = self.weighed_mse(pred_neg.flatten(0,1), true_neg_resized)
        total_loss = loss_pos + loss_neg
        return total_loss
    
    def weighed_mse(self, pred, true):
        is_non_black = (true.mean(dim=1, keepdim=True) > -0.95).float()
        weights = 1.0 + (self.non_black_weight - 1.0) * is_non_black
        weighted_mse_loss = ((pred - true) ** 2) * weights
        return weighted_mse_loss.mean()
    
def build_criterion(config):
    """
    Build the loss function based on the configuration.
    Args:
        config (Config): Configuration object containing task parameters.
    Returns:
        nn.Module: The loss function instance.
    """
    if config.task == 'mlm':
        criterion = MLMLoss()
    else:
        raise ValueError(f"Unsupported task: {config.task}")
    
    return criterion
