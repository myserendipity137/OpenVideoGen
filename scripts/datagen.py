import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle, Rectangle
import os
from tqdm import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
output_dirs = [os.path.join(project_root, 'data/train_videos'), os.path.join(project_root, 'data/val_videos')]
for dir_path in output_dirs:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
FPS = 30
DURATION = 2
TOTAL_FRAMES = FPS * DURATION
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
def create_water_drop_animation(drop_start_pos, save_path):
    fig = plt.figure(figsize=(6, 6), dpi=100, facecolor='#0a1a2e')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#0a1a2e')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    background = Rectangle((-2, -2), 4, 4,
                          color='#0a1a2e',
                          zorder=0)
    ax.add_patch(background)
    drop_radius = 0.06
    ripple_speeds = [0.08, 0.065, 0.05]
    ripple_amplitudes = [0.3, 0.2, 0.1]
    ripple_alphas = [0.7, 0.5, 0.3]
    ripple_colors = ['#6bb5ff', '#88c6ff', '#a5d7ff']
    ripple_linewidths = [2.0, 1.5, 1.0]
    drop = Circle((drop_start_pos[0], drop_start_pos[1]),
                  drop_radius,
                  color='#ffffff',
                  alpha=0.95,
                  zorder=10)
    ax.add_patch(drop)
    underwater = Rectangle((-2, -2), 4, 2,
                          color='#0a2a4a',
                          alpha=0.7,
                          zorder=1)
    ax.add_patch(underwater)
    water_surface = plt.Rectangle((-2, -0.02), 4, 0.04,
                                 color='#2a5a8a',
                                 alpha=0.8,
                                 zorder=2)
    ax.add_patch(water_surface)
    ripples = []
    for i in range(len(ripple_speeds)):
        ripple = Circle((drop_start_pos[0], 0),
                      0.01,
                      fill=False,
                      linewidth=ripple_linewidths[i],
                      edgecolor=ripple_colors[i],
                      alpha=0,
                      zorder=5-i)
        ax.add_patch(ripple)
        ripples.append(ripple)
    surface_highlights = []
    for _ in range(20):
        x = np.random.uniform(-1.8, 1.8)
        y = np.random.uniform(-0.02, 0.02)
        size = np.random.uniform(0.003, 0.015)
        highlight = Circle((x, y), size,
                          color='#ffffff',
                          alpha=np.random.uniform(0.2, 0.5),
                          zorder=3)
        ax.add_patch(highlight)
        surface_highlights.append(highlight)
    underwater_bubbles = []
    for _ in range(10):
        x = np.random.uniform(-1.5, 1.5)
        y = np.random.uniform(-1.5, -0.1)
        size = np.random.uniform(0.005, 0.02)
        bubble = Circle((x, y), size,
                       color='#a0c8ff',
                       alpha=np.random.uniform(0.1, 0.3),
                       zorder=1)
        ax.add_patch(bubble)
        underwater_bubbles.append(bubble)
    drop_y = drop_start_pos[1]
    drop_speed = 0.0
    gravity = 0.0015
    has_hit = False
    hit_frame = 0
    splash_drops = []
    splash_params = []
    def init():
        return [background, underwater, water_surface, drop] + ripples + surface_highlights + underwater_bubbles
    def update(frame):
        nonlocal drop_y, drop_speed, has_hit, hit_frame
        if not has_hit and drop_y > drop_radius:
            drop_speed += gravity
            drop_y -= drop_speed
            drop.set_center((drop_start_pos[0], drop_y))
            if drop_y < drop_start_pos[1] * 0.7:
                stretch_factor = 1.0 + (drop_start_pos[1] - drop_y) * 0.02
                current_radius = drop_radius * (1 + 0.05 * np.sin(frame * 0.15))
                drop.set_radius(current_radius)
        elif not has_hit and drop_y <= drop_radius:
            has_hit = True
            hit_frame = frame
            drop.set_alpha(0)
            for _ in range(6):
                angle = np.random.uniform(0, 2 * np.pi)
                speed = np.random.uniform(0.02, 0.06)
                splash = Circle((drop_start_pos[0], 0), 0.012,
                               color='#ffffff',
                               alpha=0.9,
                               zorder=9)
                ax.add_patch(splash)
                splash_drops.append(splash)
                splash_params.append([
                    drop_start_pos[0], 0,
                    np.cos(angle) * speed,
                    np.sin(angle) * speed,
                    0.9,
                    0.012
                ])
        if has_hit:
            time_since_hit = frame - hit_frame
            for i, ripple in enumerate(ripples):
                if time_since_hit > i * 3:
                    age = time_since_hit - i * 3
                    if age < 60:
                        radius = ripple_speeds[i] * age * 0.4
                        ripple.set_radius(radius)
                        if age < 25:
                            alpha = (age / 25) * ripple_alphas[i]
                        else:
                            alpha = (1 - (age - 25) / 35) * ripple_alphas[i]
                        ripple.set_alpha(max(0, min(1, alpha)))
                        current_width = ripple_linewidths[i] * (1 - age / 60)
                        ripple.set_linewidth(max(0.3, current_width))
        for i in range(len(splash_drops) - 1, -1, -1):
            params = splash_params[i]
            params[0] += params[2]
            params[1] += params[3]
            params[3] -= 0.0015
            params[4] -= 0.012
            params[5] *= 0.985
            if params[4] > 0.1 and params[5] > 0.001:
                splash_drops[i].set_center((params[0], params[1]))
                splash_drops[i].set_alpha(params[4])
                splash_drops[i].set_radius(params[5])
            else:
                splash_drops[i].remove()
                splash_drops.pop(i)
                splash_params.pop(i)
        for bubble in underwater_bubbles:
            x, y = bubble.center
            bubble.set_center((x, y + 0.0005))
        for highlight in surface_highlights:
            current_alpha = highlight.get_alpha()
            new_alpha = current_alpha + np.sin(frame * 0.2 + hash(str(highlight)) % 10) * 0.05
            highlight.set_alpha(np.clip(new_alpha, 0.1, 0.6))
        return [background, underwater, water_surface, drop] + ripples + splash_drops + surface_highlights + underwater_bubbles
    ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES,
                       init_func=init, blit=True, interval=1000/FPS)
    try:
        writer = FFMpegWriter(fps=FPS, bitrate=5000,
                             metadata={'title': 'Water Drop Simulation',
                                      'artist': 'Python Matplotlib'})
        ani.save(save_path, writer=writer, dpi=100)
        print(f"已保存: {save_path}")
    except Exception as e:
        print(f"保存失败 {save_path}: {e}")
    plt.close(fig)
    return save_path
def generate_video_sets(num_train=100, num_val=20):
    print(f"开始生成 {num_train} 个训练视频和 {num_val} 个测试视频...")
    print(f"每个视频 {DURATION} 秒，{FPS} FPS，共 {TOTAL_FRAMES} 帧")
    print(f"分辨率: 600x600 像素")
    print("="*60)
    np.random.seed(42)
    train_positions = []
    for i in range(num_train):
        x = np.random.uniform(-1.5, 1.5)
        y = np.random.uniform(1.0, 1.8)
        train_positions.append((x, y))
    val_positions = []
    for i in range(num_val):
        x = np.random.uniform(-1.5, 1.5)
        y = np.random.uniform(1.0, 1.8)
        val_positions.append((x, y))
    print("\n生成训练集视频...")
    train_paths = []
    for i in tqdm(range(num_train), desc="训练集"):
        save_path = os.path.join(project_root, f"data/train_videos/train_{i:03d}.mp4")
        try:
            create_water_drop_animation(train_positions[i], save_path)
            train_paths.append(save_path)
        except Exception as e:
            print(f"训练视频 {i} 生成失败: {e}")
    print("\n生成测试集视频...")
    val_paths = []
    for i in tqdm(range(num_val), desc="测试集"):
        save_path = os.path.join(project_root, f"data/val_videos/val_{i:03d}.mp4")
        try:
            create_water_drop_animation(val_positions[i], save_path)
            val_paths.append(save_path)
        except Exception as e:
            print(f"测试视频 {i} 生成失败: {e}")
    print("\n生成元数据文件...")
    with open(os.path.join(project_root, 'data/train_videos/metadata.csv'), 'w') as f:
        f.write("video_name,start_x,start_y\n")
        for i, (x, y) in enumerate(train_positions):
            if i < len(train_paths):
                f.write(f"train_{i:03d}.mp4,{x:.4f},{y:.4f}\n")
    with open(os.path.join(project_root, 'data/val_videos/metadata.csv'), 'w') as f:
        f.write("video_name,start_x,start_y\n")
        for i, (x, y) in enumerate(val_positions):
            if i < len(val_paths):
                f.write(f"val_{i:03d}.mp4,{x:.4f},{y:.4f}\n")
    with open('./dataset_info.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("水滴数据集信息\n")
        f.write("="*60 + "\n\n")
        f.write(f"训练集: {len(train_paths)} 个视频 (目标: {num_train})\n")
        f.write(f"测试集: {len(val_paths)} 个视频 (目标: {num_val})\n")
        f.write(f"视频规格: {DURATION}秒, {FPS}FPS, {TOTAL_FRAMES}帧\n")
        f.write(f"分辨率: 600x600 像素 (6x6英寸, 100DPI)\n")
        f.write(f"水滴起始X范围: [-1.5, 1.5]\n")
        f.write(f"水滴起始Y范围: [1.0, 1.8]\n")
        f.write(f"视频格式: MP4 (H.264)\n")
        f.write(f"比特率: 5000 kbps\n")
        f.write(f"背景色: 深蓝色 (#0a1a2e)\n\n")
        f.write("视觉效果增强:\n")
        f.write("  ✓ 深色背景充满整个画面\n")
        f.write("  ✓ 水面/水下分层效果\n")
        f.write("  ✓ 水面闪光点\n")
        f.write("  ✓ 水下气泡\n")
        f.write("  ✓ 慢速波纹扩散\n")
        f.write("  ✓ 溅起效果\n\n")
        f.write("文件结构:\n")
        f.write("  train_videos/ - 训练集视频 (train_000.mp4 到 train_099.mp4)\n")
        f.write("  val_videos/   - 测试集视频 (val_000.mp4 到 val_019.mp4)\n")
        f.write("  metadata.csv  - 每个视频的水滴起始位置\n")
        f.write("  dataset_info.txt - 本文件\n\n")
        f.write("随机种子: 42 (可复现结果)\n")
    print("\n" + "="*60)
    print("完成！")
    print(f"- 成功生成训练集: {len(train_paths)}/{num_train} 个视频")
    print(f"- 成功生成测试集: {len(val_paths)}/{num_val} 个视频")
    print(f"- 视频保存到: ./train_videos/ 和 ./val_videos/")
    print(f"- 元数据: metadata.csv")
    print(f"- 详细信息: dataset_info.txt")
    print("="*60)
    print("\n建议查看样本视频:")
    if train_paths:
        print(f"  ffplay {train_paths[0]}")
    print("\n参数说明:")
    print("1. 画面完全被深蓝色背景填充")
    print("2. 水面线在 y=0 位置")
    print("3. 水滴从随机高度下落")
    print("4. 波纹速度缓慢而真实")
if __name__ == "__main__":
    try:
        from matplotlib.animation import FFMpegWriter
        print("FFmpeg检测成功，开始生成视频...")
        print("="*60)
    except ImportError:
        print("错误: 需要FFmpeg来生成MP4视频")
        print("请安装FFmpeg:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: 从 https://ffmpeg.org/download.html 下载")
        print("          或使用: choco install ffmpeg")
        exit(1)
    generate_video_sets(num_train=400, num_val=20)