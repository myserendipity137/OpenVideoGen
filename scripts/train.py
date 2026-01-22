import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.model import SimpleDiffusion3D, GaussianDiffusion

# 1. 配置参数 (Configuration)
CONFIG = {
    "data_dir": os.path.join(project_root, "data/latents"),
    "save_dir": os.path.join(project_root, "checkpoints"),
    "batch_size": 4,          # 如果显存不够，改小这个 (例如 1 或 2)
    "num_epochs": 25,        # 训练轮数
    "learning_rate": 1e-4,
    "save_interval": 10,      # 每多少个 epoch 保存一次
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,         # 数据加载线程数
    "latent_shape": (16, 15, 75, 75) # (C, T, H, W)
}

# 2. 数据集定义 (Dataset)
class VideoLatentDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {data_dir}. Did you run preprocess.py?")
        print(f"Found {len(self.files)} latent files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 加载 .pt 文件
        # Shape: [16, 15, 75, 75]
        latent = torch.load(self.files[idx], map_location="cpu")
        return latent

# 3. 训练流程 (Training Loop)
def train():
    # 创建保存目录
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    
    # 初始化
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")

    # 1. 准备数据
    dataset = VideoLatentDataset(CONFIG["data_dir"])
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    # 2. 准备模型
    # 输入通道 16，内部基础维度 64
    model = SimpleDiffusion3D(in_channels=16, base_dim=64).to(device)
    
    # 扩散过程工具类
    diffusion = GaussianDiffusion(timesteps=1000, device=device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    
    # 损失函数 (预测噪声和真实噪声的均方误差)
    criterion = nn.MSELoss()

    # 记录 Loss
    loss_history = []

    print("Starting training...")
    
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        
        for latents in progress_bar:
            # latents: [B, 16, 15, 75, 75]
            latents = latents.to(device)
            batch_size = latents.shape[0]
            
            first_frame = latents[:, :, 0:1, :, :]
            
            # B. 加噪过程 (Forward Process)
            # 1. 随机采样时间步 t
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
            
            # 2. 生成高斯噪声 epsilon
            noise = torch.randn_like(latents)
            
            # 3. 计算加噪后的图 x_t
            x_noisy = diffusion.q_sample(latents, t, noise)
            
            # C. 预测噪声 (Reverse Process Prediction)
            # 模型输入: 加噪图 + 时间步 + 第一帧条件
            # 模型输出: 预测的噪声
            predicted_noise = model(x_noisy, t, first_frame)
            
            # D. 计算 Loss 并更新
            loss = criterion(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

        # 保存模型权重
        if (epoch + 1) % CONFIG["save_interval"] == 0:
            save_path = os.path.join(CONFIG["save_dir"], f"diffusion_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

    # 训练结束，保存最终模型
    torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "diffusion_final.pth"))
    print("Training finished.")

    # 绘制 Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig(os.path.join(CONFIG["save_dir"], "loss_curve.png"))
    print("Loss curve saved.")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")