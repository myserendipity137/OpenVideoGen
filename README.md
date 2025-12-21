# OpenVideoGen
OpenVideoGen将是一个非常有趣的项目，它将用成本最小的方式帮助大家了解视频生成的原理，我们将构建分布差异最小最紧凑的视频数据集帮助大家实现无痛的训练和部署。

## Introduction
在此之前，相信大家已经对*vae*和*diffusion model*有了一定程度的了解。大家需要了解当前video generation的主流范式，也就是 [Latent Diffusion Model](https://arxiv.org/abs/2112.10752)。具体来说，我们需要一个预训练好的冻结的VAE将视频压缩到低位的潜空间，然后用扩散模型在潜空间完成生成，再解码到视频空间去。本项目推荐使用 [Wan](https://github.com/Wan-Video/Wan2.2) 的预训练VAE。因此大家可以参考该项目的算法实现。

## Prepare
1. 我们采用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 进行项目管理，大家可以首先安装该工具。
完成安装后，初始化所需的环境：
```bash
cd OpenVideoGen
uv sync
```
这会帮助你自动配置该项目所需的依赖，一旦你需要安装其它依赖：
```bash
uv pip install <package-name>
```
激活环境：
```bash
source .venv/bin/activate
```


2. 我们将使用 [huggingface](https://huggingface.co/) 上的模型，因此大家需要先在huggingface上注册一个账号，然后获取到自己的token。
接下来下载 [wan 2.1 vae ckpt](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B/blob/main/Wan2.1_VAE.pth) :
```bash
cd pretrained
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B/resolve/main/Wan2.1_VAE.pth
cd ..
```

我们已经在[modules/vae.py](modules/vae.py)中实现了wan 2.1 vae的加载，大家可以直接使用。

## Reconstruction
1. 首先，大家可以用下面的程序构建我们的dataset，这会生成大量紧凑的简单的视频样本，以构建我们在有限算力下可以生成的数据集：
```bash
python scripts/datagen.py
```
可以通过调节参数实现更大的训练集构建。

2. 然后，大家可以用下面的程序来重建视频，以检查VAE的重建效果：
```bash
python scripts/recon.py
```

## Assignment
在此基础上，大家需要实现一个简单的扩散模型，在vae的编码空间下完成根据视频的第一帧来生成完整视频的任务，本次任务没有codebase，大家可以根据自己的理解来实现。