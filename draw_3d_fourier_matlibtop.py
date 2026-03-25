import numpy as np
import matplotlib.pyplot as plt

# 设置
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 时间轴
t = np.linspace(0, 4 * np.pi, 500)

# 定义要叠加的谐波次数 (1, 3, 5, 7...)
harmonics = [1, 3, 5, 7, 9]
composite_wave = np.zeros_like(t)

# ---------- 关键：侧面“高度线”放置位置 ----------
# 放在时间轴最右侧那一面，并稍微往外挪一点点，避免和波形尾部重叠
x_side = t.max() + 0.35

# 竖线从基线 z=0 起（你也可以改成从 z_min 起）
z_base = 0.0
freq_scale = 2          # ⭐ 控制频率层间距
# 绘制每一个分量 (蓝色波浪)
for i, n in enumerate(harmonics):
    amplitude = 1 / n
    wave = amplitude * np.sin(n * t)

    # 累加到合成波
    composite_wave += wave
    y_pos = i * freq_scale
    # 在 y=n 的位置绘制波形（zdir='y' 表示 y 方向是“层”）
    ax.plot(t, wave, zs=n, zdir='y', color='cornflowerblue', alpha=0.6, linewidth=1.8)

    # ===============================
    # ✅ 每条波形一个“侧面高度竖线”
    # ===============================
    # 你想要“从正视图看到高低不同”，最合理的高度就是该谐波的幅值 amplitude=1/n
    h = amplitude

    # 画竖线：固定 x=x_side, 固定 y=n, z 从 0 到 h
    ax.plot([x_side, x_side],
            [n, n],
            [z_base, z_base + h],
            color='black', linewidth=2)

    # （可选）给每根竖线加一个小帽子，更像“高度标记”
    cap = 0.18
    ax.plot([x_side - cap, x_side + cap],
            [n, n],
            [z_base + h, z_base + h],
            color='black', linewidth=1.5)
# ===============================
# 在侧面把所有高度线的 0 轴用一条横线连起来
# ===============================

ax.plot(
    [x_side, x_side],                         # 固定在侧面
    [min(harmonics), max(harmonics)],          # 沿 frequency 方向
    [z_base, z_base],                          # z = 0 轴
    color='black',
    linewidth=2
)

# 绘制合成后的近似方波 (红色，放在最前面 y=0)
ax.plot(t, composite_wave, zs=0, zdir='y', color='red', linewidth=2.5)

# 设置轴标签
ax.set_xlabel('Time')
ax.set_ylabel('Frequency')
ax.set_zlabel('Amplitude')

# 调整视角（你现在这个角度 OK）
ax.view_init(elev=20, azim=-45)

# 让侧边高度线不被裁掉（因为 x_side 往外挪了）
ax.set_xlim(t.min(), x_side + 0.2)

# 让 z 轴范围更舒服（可按需调）
ax.set_zlim(-1.2, 1.2)

plt.tight_layout()
plt.show()
