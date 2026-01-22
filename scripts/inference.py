import os
import sys
import torch
import cv2
import numpy as np
from tqdm import tqdm
import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.vae import WanVAE
from modules.model import SimpleDiffusion3D, GaussianDiffusion

def load_first_frame(video_path, device):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame from {video_path}")
        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    frame_tensor = torch.from_numpy(frame).float() / 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    
    return frame_tensor.to(device)

@torch.no_grad()
def ddpm_sample(model, diffusion, condition_latent, shape, device):
    img = torch.randn(shape, device=device)
    
    betas = diffusion.betas
    alphas = diffusion.alphas
    alphas_cumprod = diffusion.alphas_cumprod
    sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod
    
    for i in tqdm(reversed(range(0, diffusion.timesteps)), desc="Sampling", total=diffusion.timesteps):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        
        pred_noise = model(img, t, condition_latent)
        
        beta_t = diffusion.extract(betas, t, img.shape)
        alpha_t = diffusion.extract(alphas, t, img.shape)
        sqrt_one_minus_alpha_cumprod_t = diffusion.extract(sqrt_one_minus_alphas_cumprod, t, img.shape)
        
        posterior_mean = (1 / torch.sqrt(alpha_t)) * (
            img - (beta_t / sqrt_one_minus_alpha_cumprod_t) * pred_noise
        )
        
        if i > 0:
            noise = torch.randn_like(img)
            posterior_variance = torch.sqrt(beta_t) * noise
            img = posterior_mean + posterior_variance
        else:
            img = posterior_mean
            
    return img

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(project_root, "checkpoints/diffusion_final.pth")
    val_videos = glob.glob(os.path.join(project_root, "data/val_videos/*.mp4"))
    
    if not val_videos:
        print("No validation videos found. Please run datagen.py first.")
        return
    
    source_video_path = val_videos[0]
    output_path = os.path.join(project_root, "generated_result.mp4")
    
    print(f"Device: {device}")
    print(f"Source Condition: {source_video_path}")
    print(f"Loading Models...")

    vae = WanVAE(vae_pth=os.path.join(project_root, "pretrained/Wan2.1_VAE.pth"), device=device)
    
    diffusion_model = SimpleDiffusion3D(in_channels=16, base_dim=64).to(device)
    diffusion_model.load_state_dict(torch.load(model_path, map_location=device))
    diffusion_model.eval()
    
    diffusion_utils = GaussianDiffusion(timesteps=1000, device=device)

    print("Encoding condition frame...")
    first_frame_raw = load_first_frame(source_video_path, device)
    
    with torch.no_grad():
        encoder_input = first_frame_raw.squeeze(0)
        condition_latent = vae.encode([encoder_input])[0]
        condition_latent = condition_latent.unsqueeze(0)

    print(f"Condition Shape: {condition_latent.shape}")

    target_shape = (1, 16, 15, 75, 75)
    
    print("Starting Diffusion Sampling...")
    generated_latent = ddpm_sample(
        diffusion_model, 
        diffusion_utils, 
        condition_latent, 
        target_shape, 
        device
    )
    
    print("Decoding latent to video...")
    with torch.no_grad():
        recon_video_tensor = vae.decode([generated_latent.squeeze(0)])[0]
        
    recon_video = recon_video_tensor.permute(1, 2, 3, 0).cpu().numpy()
    recon_video = (np.clip(recon_video, 0, 1) * 255).astype(np.uint8)

    print(f"Saving to {output_path}...")
    T, H, W, C = recon_video.shape
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    for frame in recon_video:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        
    out.release()
    print("Done!")

if __name__ == "__main__":
    main()