import os
import sys
import glob
import torch
import cv2
import numpy as np
from tqdm import tqdm

# 1. 环境路径修复
# 这是一个项目，因此我们需要将上一级目录和项目根目录强行加到Python的搜索路径中
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

# 导入所需的编码器WanVAE
from modules.vae import WanVAE

# 2. 配置参数
source_dir = os.path.join(project_root, "data/train_videos")
output_dir = os.path.join(project_root, "data/latents")
vae_path = os.path.join(project_root, "pretrained/Wan2.1_VAE.pth")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 3. Single Video Process
def process_video(video_path, vae_model):
    """
    Input: video path(MP4)
    Operate: Read -> RGB -> Normalization -> Dimension Adjust -> VAE Encode
    Output: Latent Tensor
    """
    # 使用OPENCV打开视频，cap可以看作是被解包的视频
    cap = cv2.VideoCapture(video_path)
    frames = []     # 这里的话，frame就是帧的意思呢，我们先是用的list，后面会转成数组的

    # 3.1 逐帧读取
    while True:
        # ret: bool,表示是否读到了帧
        # frame: 具体读到的帧
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if not frames:
        return None
    
    # 3.2 数据形态转化
    frames_np = np.array(frames)
    frames_tensor = torch.from_numpy(frames_np).float() / 255.0
    # 当前维度(T,H,W,C): time,height,width,channel
    # VAE input: (channel,time,height,width)
    frames_tensor = frames_tensor.permute(3, 0, 1, 2)

    # 3.3 VAE Encode
    with torch.no_grad():
        frames_tensor = frames_tensor.to(device)
        latent = vae_model.encode([frames_tensor])[0]

    return latent.cpu()

def main():

    # Prepare output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    vae = WanVAE(vae_pth=vae_path, device=device)
    print("VAE has been loaded.")

    video_files = glob.glob(os.path.join(source_dir, "*.mp4"))
    video_files.sort()

    first_shape = None

    for i, video_path in enumerate(tqdm(video_files)):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_path = os.path.join(output_dir,f"{video_name}.pt")

        if os.path.exists(save_path):
            if i == 0:
                first_shape = torch.load(save_path).shape
            continue

        try:
            latent = process_video(video_path, vae)
            if latent is not None:
                torch.save(latent, save_path)

                if first_shape is None:
                    first_shape = latent.shape

        except Exception as e:
            print("Error has occurred.")

    print("Process Completed.")
    print(f"{first_shape}")
    
if __name__ == "__main__":
    main()
