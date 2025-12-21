import os
import sys
import glob
import torch
import cv2
import numpy as np
from einops import rearrange
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.vae import WanVAE as VAE

def reconstruct_videos(
    vae_model,
    source_dir: str,
    cache_dir: str,
    device: torch.device
):
    """
    Reconstructs videos from a source directory using a VAE model and saves them to a cache directory.
    """
    video_files = glob.glob(os.path.join(source_dir, "**", "*.mp4"), recursive=True)
    
    if not video_files:
        print(f"No .mp4 files found in {source_dir}")
        return

    print(f"Found {len(video_files)} videos to reconstruct.")

    for video_path in tqdm(video_files, desc="Reconstructing videos"):
        # Create corresponding output path
        relative_path = os.path.relpath(video_path, source_dir)
        output_path = os.path.join(cache_dir, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        if frames:
            # Preprocess frames
            frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
            frames_tensor = torch.from_numpy(np.array(frames_rgb)).float() / 255.0
            frames_tensor = frames_tensor.to(device)
            frames_tensor = rearrange(frames_tensor, 't h w c -> 1 c t h w') # (1, C, T, H, W)

            # Reconstruct frames
            with torch.no_grad():
                latent=vae_model.encode(frames_tensor)
                print(f"latent shape: LEN :{len(latent)} {latent[0].shape}")
                recon_frames_tensor = vae_model.decode(latent)[0]

            # Postprocess frames
            recon_frames = recon_frames_tensor.squeeze(0)
            recon_frames = rearrange(recon_frames, 'c t h w -> t h w c')
            recon_frames = (recon_frames.clamp(0, 1) * 255).byte().cpu().numpy()
            
            for recon_frame in recon_frames:
                recon_frame_bgr = cv2.cvtColor(recon_frame, cv2.COLOR_RGB2BGR)
                out.write(recon_frame_bgr)

        cap.release()
        out.release()

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading VAE model...")
    vae = VAE(vae_pth=os.path.join(project_root, "pretrained/Wan2.1_VAE.pth"), device=device)
    print("VAE model loaded.")

    VIDEO_SOURCE_DIR = os.path.join(project_root, "data/val_videos")
    CACHE_DIR = os.path.join(project_root, "cache")

    reconstruct_videos(
        vae_model=vae,
        source_dir=VIDEO_SOURCE_DIR,
        cache_dir=CACHE_DIR,
        device=device
    )

    print("All videos have been reconstructed and saved to the cache directory.")