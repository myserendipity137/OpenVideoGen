import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False, down=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # 3D Convolution
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.relu = nn.SiLU()

        self.up = up
        self.down = down
        
        # Upsample / Downsample layers
        if up:
            self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        if down:
            # stride=(1,2,2) means we keep Time dim, reduce H/W
            self.downsample = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=(1, 2, 2), padding=1)

        # Shortcut connection
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        # 1. Handle Sampling (Up/Down) implies spatial change
        input_x = x
        if self.down:
            input_x = self.downsample(input_x)
        elif self.up:
            input_x = self.upsample(input_x)

        # 2. Main Branch
        # Fix: Ensure conv1 is applied to the resized input_x
        h = self.conv1(input_x) 
        h = self.norm1(h) # Norm expects out_channels. h has out_channels. OK.
        
        # 3. Time Embedding Injection
        time_emb = self.relu(self.time_mlp(t))
        # Broadcast time_emb to match h shape: [B, C, 1, 1, 1]
        h = h + time_emb[(..., ) + (None, ) * 3]
        
        h = self.relu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        
        # 4. Residual Connection
        # We must transform the original x to match h's shape (spatial and channel)
        return h + self.shortcut(input_x)

class SimpleDiffusion3D(nn.Module):
    def __init__(self, in_channels=16, base_dim=64):
        super().__init__()
        input_channels = in_channels * 2 
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, base_dim * 4),
            nn.SiLU(),
            nn.Linear(base_dim * 4, base_dim * 4),
        )
        time_dim = base_dim * 4

        # Initial Conv
        self.init_conv = nn.Conv3d(input_channels, base_dim, kernel_size=3, padding=1)

        # Encoder
        self.down1 = Block3D(base_dim, base_dim * 2, time_dim, down=True) 
        self.down2 = Block3D(base_dim * 2, base_dim * 4, time_dim, down=True) 

        # Bottleneck
        self.mid1 = Block3D(base_dim * 4, base_dim * 4, time_dim)
        self.mid2 = Block3D(base_dim * 4, base_dim * 4, time_dim)

        # Decoder
        # up1 Input: Concat(x_mid, x3) = 256 + 256 = 512
        self.up1 = Block3D(base_dim * 8, base_dim * 2, time_dim, up=True)
        
        # up2 Input: Concat(up1_out, x2) = 128 + 128 = 256
        self.up2 = Block3D(base_dim * 4, base_dim, time_dim, up=True)

        # Output
        self.out_conv = nn.Conv3d(base_dim, in_channels, kernel_size=1)

    def forward(self, x, t, first_frame_latent):
        """
        x: [B, C, T, H, W]
        """
        # 1. 自动对齐处理 (Auto-Padding)
        # 记录原始尺寸
        h_orig, w_orig = x.shape[-2], x.shape[-1]
        
        # 计算需要填充的像素数 (确保是 4 的倍数，因为网络有 2 次下采样 2^2=4)
        h_pad = (4 - h_orig % 4) % 4
        w_pad = (4 - w_orig % 4) % 4
        
        # F.pad 参数顺序: (Left, Right, Top, Bottom, ...)
        if h_pad > 0 or w_pad > 0:
            x = F.pad(x, (0, w_pad, 0, h_pad))
            first_frame_latent = F.pad(first_frame_latent, (0, w_pad, 0, h_pad))

        # 2. 网络前向传播
        B, C, T, H, W = x.shape
        cond_expanded = first_frame_latent.repeat(1, 1, T, 1, 1)
        x_in = torch.cat([x, cond_expanded], dim=1)
        
        t_emb = self.time_mlp(t)

        x1 = self.init_conv(x_in)           
        
        x2 = self.down1(x1, t_emb)          
        x3 = self.down2(x2, t_emb)          
        
        x_mid = self.mid1(x3, t_emb)
        x_mid = self.mid2(x_mid, t_emb)     
        
        x_up1 = self.up1(torch.cat([x_mid, x3], dim=1), t_emb) 
        x_up2 = self.up2(torch.cat([x_up1, x2], dim=1), t_emb) 
        
        out = self.out_conv(x_up2)

        # 3. 自动裁剪 (Auto-Cropping)
        # 把刚才补进去的边切掉，恢复成 75x75
        if h_pad > 0 or w_pad > 0:
            out = out[..., :h_orig, :w_orig]

        return out

class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))