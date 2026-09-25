import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from diffusers import DDPMScheduler
import os
import math
import sys


CONFIG = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'image_size': 64,
    'num_timesteps': 1000,
    'time_embed_dim': 256,
    'checkpoint_path': "checkpoint.pth",
    'output_dir': 'generated',
    'num_images': 1,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)



def get_timestep_embedding(timesteps, embedding_dim, max_period=10000):
    half = embedding_dim // 2
    freqs = torch.exp(
        -math.log(max_period) * 
        torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if embedding_dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1) 
            if in_channels != out_channels 
            else nn.Identity()
        )
    
    def forward(self, x, temb):
        h = self.act(self.norm1(x))
        h = self.conv1(h) + self.temb_proj(self.act(temb))[:, :, None, None]
        h = self.act(self.norm2(h))
        return self.conv2(h) + self.conv_shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(
            B, 3, self.num_heads, C // self.num_heads, H * W
        ).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        h = F.scaled_dot_product_attention(q, k, v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj_out(h)


class SimpleUNet(nn.Module):
    def __init__(self, time_embed_dim=CONFIG['time_embed_dim']):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        self.conv_in = nn.Conv2d(3, 64, 3, padding=1)
        self.down1_b1 = ResnetBlock(64, 64, time_embed_dim)
        self.down1_b2 = ResnetBlock(64, 64, time_embed_dim)
        self.downsample1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down2_b1 = ResnetBlock(128, 128, time_embed_dim)
        self.down2_b2 = ResnetBlock(128, 128, time_embed_dim)
        self.downsample2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.down3_b1 = ResnetBlock(256, 256, time_embed_dim)
        self.down3_b2 = ResnetBlock(256, 256, time_embed_dim)
        self.downsample3 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.mid_block1 = ResnetBlock(512, 512, time_embed_dim)
        self.mid_attn = AttentionBlock(512)
        self.mid_block2 = ResnetBlock(512, 512, time_embed_dim)
        self.upsample3 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.up3_b1 = ResnetBlock(512, 256, time_embed_dim)
        self.up3_b2 = ResnetBlock(256, 256, time_embed_dim)
        self.upsample2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.up2_b1 = ResnetBlock(256, 128, time_embed_dim)
        self.up2_b2 = ResnetBlock(128, 128, time_embed_dim)
        self.upsample1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.up1_b1 = ResnetBlock(128, 64, time_embed_dim)
        self.up1_b2 = ResnetBlock(64, 64, time_embed_dim)
        self.norm_out = nn.GroupNorm(8, 64)
        self.conv_out = nn.Conv2d(64, 3, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t):
        temb = self.time_proj(get_timestep_embedding(t, self.time_embed_dim))
        
        h = self.conv_in(x)
        h = self.down1_b1(h, temb)
        h = self.down1_b2(h, temb)
        d1 = h
        h = self.downsample1(h)
        h = self.down2_b1(h, temb)
        h = self.down2_b2(h, temb)
        d2 = h
        h = self.downsample2(h)
        h = self.down3_b1(h, temb)
        h = self.down3_b2(h, temb)
        d3 = h
        h = self.downsample3(h)
        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)
        h = self.upsample3(h)
        h = torch.cat([h, d3], dim=1)
        h = self.up3_b1(h, temb)
        h = self.up3_b2(h, temb)
        h = self.upsample2(h)
        h = torch.cat([h, d2], dim=1)
        h = self.up2_b1(h, temb)
        h = self.up2_b2(h, temb)
        h = self.upsample1(h)
        h = torch.cat([h, d1], dim=1)
        h = self.up1_b1(h, temb)
        h = self.up1_b2(h, temb)
        
        return self.conv_out(self.act(self.norm_out(h)))


def generate_images(model, scheduler_ddpm, num_images=None, seed=None):
    if num_images is None:
        num_images = CONFIG['num_images']
    
    if seed is not None:
        torch.manual_seed(seed)
    
    model.eval()
    print(f"Generating {num_images} images...")
    
    with torch.no_grad():
        x = torch.randn(
            num_images, 3, CONFIG['image_size'], CONFIG['image_size'],
            device=CONFIG['device']
        )
        
        for i in range(CONFIG['num_timesteps'] - 1, -1, -1):
            t = torch.full((num_images,), i, device=CONFIG['device'], dtype=torch.long)
            x = scheduler_ddpm.step(model(x, t), i, x).prev_sample
            
            if (i + 1) % 100 == 0:
                print(f"  Step {i+1}/{CONFIG['num_timesteps']}")
    
    x = (x + 1) / 2
    x = x.clamp(0, 1).cpu()
    return x


def save_generated_images(images, prefix="generated"):
    saved_paths = []
    
    for i in range(images.shape[0]):
        filename = f"{prefix}_img{i+1}.png"
        filepath = os.path.join(CONFIG['output_dir'], filename)
        utils.save_image(images[i], filepath)
        saved_paths.append(filepath)
        print(f"Saved: {filepath}")
    
    return saved_paths


def open_with_windows_photos(filepath):
    if sys.platform == "win32":
        try:
            os.startfile(filepath)
            print(f"Opened with Windows Photos: {filepath}")
        except Exception as e:
            print(f"Could not open image automatically: {e}")
    else:
        print(f"Auto-open only available on Windows. Image saved at: {filepath}")


def main():
    if not os.path.exists(CONFIG['checkpoint_path']):
        raise FileNotFoundError(
            f"Checkpoint not found: {CONFIG['checkpoint_path']}\n"
            "Make sure you've trained the model first."
        )
    
    print(f"Loading checkpoint from: {CONFIG['checkpoint_path']}")
    
    model = SimpleUNet().to(CONFIG['device'])
    scheduler_ddpm = DDPMScheduler(
        num_train_timesteps=CONFIG['num_timesteps'],
        beta_schedule="linear"
    )
    
    checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=CONFIG['device'])
    
    if 'ema' in checkpoint:
        model.load_state_dict(checkpoint['ema'])
        print("Loaded EMA weights")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print("Loaded model weights")
    else:
        raise ValueError("Checkpoint doesn't contain valid weights")
    
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch'] + 1}")
    if 'loss' in checkpoint:
        print(f"  Loss: {checkpoint['loss']:.4f}")
    
    images = generate_images(model, scheduler_ddpm)
    saved_paths = save_generated_images(images, prefix="faces")
    
    if saved_paths:
        open_with_windows_photos(saved_paths[0])
    
    print("\nGeneration complete!")
    print(f"Images saved to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()