"""
plot_enet_ca_comparison.py
绘制 ENet vs ENet+CA 对比图（验证Loss、验证mIoU、最优指标柱状图）。

数据来源：train_logs/ 目录下 ENet 和 ENet+CA 的 CSV。
输出：
  - train_logs/enet_ca_comparison.png  （高清对比图）
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 控制台输出：Windows 默认 GBK，打印 ✓ 等字符会抛 UnicodeEncodeError（历史运行失败的根因）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ===================== 字体设置（中文显示） =====================
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===================== 数据路径 =====================
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'train_logs')

# ENet vs ENet+CA 对比配置
models_config = [
    {'file': 'enet_metrics.csv',      'name': 'ENet',     'color': '#1f77b4', 'ls': '-',  'params': '0.26M'},
    {'file': 'enet_ca_metrics.csv',   'name': 'ENet+CA',  'color': '#ff7f0e', 'ls': '-',  'params': '0.27M'},
]

# ===================== 读取数据 =====================
data = {}
for cfg in models_config:
    fpath = os.path.join(LOG_DIR, cfg['file'])
    if not os.path.exists(fpath):
        print(f"⚠️ 文件不存在，跳过：{cfg['file']}")
        continue
    df = pd.read_csv(fpath)
    data[cfg['name']] = df
    print(f"✅ {cfg['name']}: {len(df)} epochs, 最优 mIoU = {df['val_mIoU'].max():.4f}")

if len(data) < 2:
    print("❌ 数据不足，需要 ENet 和 ENet+CA 两个模型的 CSV")
    exit(1)

# ===================== 创建图表 =====================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# ---- 子图1：训练 & 验证 Loss 对比 ----
ax1 = axes[0]
for cfg in models_config:
    if cfg['name'] not in data:
        continue
    df = data[cfg['name']]
    ax1.plot(df['epoch'], df['train_loss'], color=cfg['color'], ls='--',
             label=f"{cfg['name']} Train", linewidth=1.3, alpha=0.6)
    ax1.plot(df['epoch'], df['val_loss'], color=cfg['color'], ls='-',
             label=f"{cfg['name']} Val", linewidth=1.8, alpha=0.9)
ax1.set_xlabel('Epoch', fontsize=13)
ax1.set_ylabel('Loss', fontsize=13)
ax1.set_title('训练/验证 Loss 对比', fontsize=15, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, len(data['ENet']))

# ---- 子图2：验证 mIoU 曲线 ----
ax2 = axes[1]
for cfg in models_config:
    if cfg['name'] not in data:
        continue
    df = data[cfg['name']]
    best_miou = df['val_mIoU'].max()
    best_epoch = df.loc[df['val_mIoU'].idxmax(), 'epoch']
    ax2.plot(df['epoch'], df['val_mIoU'] * 100, color=cfg['color'], ls=cfg['ls'],
             label=f"{cfg['name']} ({cfg['params']})", linewidth=2.0, alpha=0.9)
    # 标注最优mIoU点
    ax2.scatter([best_epoch], [best_miou * 100], color=cfg['color'], s=80, zorder=5,
                edgecolors='black', linewidths=0.8)
    ax2.annotate(f'{best_miou*100:.2f}%\n(Epoch {int(best_epoch)})',
                 xy=(best_epoch, best_miou * 100),
                 xytext=(best_epoch + 8, best_miou * 100 - 1.5),
                 fontsize=9, color=cfg['color'], fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=cfg['color'], lw=1.2))
ax2.set_xlabel('Epoch', fontsize=13)
ax2.set_ylabel('Val mIoU (%)', fontsize=13)
ax2.set_title('验证集 mIoU 对比', fontsize=15, fontweight='bold')
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, len(data['ENet']))

# ---- 子图3：指标对比柱状图 ----
ax3 = axes[2]
metrics_names = ['最优 mIoU', 'Epoch-100 mIoU', '最优 Val Loss']
enet_vals = []
enet_ca_vals = []

df_enet = data['ENet']
df_enet_ca = data['ENet+CA']

enet_vals.append(df_enet['val_mIoU'].max() * 100)
enet_vals.append(df_enet['val_mIoU'].iloc[-1] * 100)
enet_vals.append(df_enet['val_loss'].min())

enet_ca_vals.append(df_enet_ca['val_mIoU'].max() * 100)
enet_ca_vals.append(df_enet_ca['val_mIoU'].iloc[-1] * 100)
enet_ca_vals.append(df_enet_ca['val_loss'].min())

x = np.arange(len(metrics_names))
width = 0.3

bars1 = ax3.bar(x - width/2, enet_vals, width, label='ENet', color='#1f77b4', alpha=0.85)
bars2 = ax3.bar(x + width/2, enet_ca_vals, width, label='ENet+CA', color='#ff7f0e', alpha=0.85)

# 在柱状图上标注数值
for bar, val in zip(bars1, enet_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1f77b4')
for bar, val in zip(bars2, enet_ca_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
             f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#ff7f0e')

ax3.set_xticks(x)
ax3.set_xticklabels(metrics_names, fontsize=11)
ax3.set_title('关键指标对比', fontsize=15, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')

# 计算提升幅度
miou_improve = (df_enet_ca['val_mIoU'].max() - df_enet['val_mIoU'].max()) * 100
miou_final_improve = (df_enet_ca['val_mIoU'].iloc[-1] - df_enet['val_mIoU'].iloc[-1]) * 100
fig.suptitle(f'ENet vs ENet+CA 坐标注意力改进对比  |  CA提升 mIoU: {miou_improve:+.2f}%',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

# ---- 保存 ----
save_path = os.path.join(LOG_DIR, 'enet_ca_comparison.png')
plt.savefig(save_path, dpi=200, bbox_inches='tight')
print(f"\n📊 对比图已保存：{save_path}")

# ===================== 汇总表输出 =====================
print("\n" + "=" * 65)
print("           ENet vs ENet+CA 对比汇总")
print("=" * 65)
print(f"{'指标':<20} {'ENet':<15} {'ENet+CA':<15} {'提升':<10}")
print("-" * 65)

best_miou_enet = df_enet['val_mIoU'].max() * 100
best_miou_enet_ca = df_enet_ca['val_mIoU'].max() * 100
final_miou_enet = df_enet['val_mIoU'].iloc[-1] * 100
final_miou_enet_ca = df_enet_ca['val_mIoU'].iloc[-1] * 100
best_epoch_enet = int(df_enet.loc[df_enet['val_mIoU'].idxmax(), 'epoch'])
best_epoch_enet_ca = int(df_enet_ca.loc[df_enet_ca['val_mIoU'].idxmax(), 'epoch'])
best_vloss_enet = df_enet['val_loss'].min()
best_vloss_enet_ca = df_enet_ca['val_loss'].min()

print(f"{'参数量':<20} {'0.26M':<15} {'0.27M':<15} {'+3.35K':<10}")
print(f"{'最优 mIoU (%)':<20} {best_miou_enet:<15.2f} {best_miou_enet_ca:<15.2f} {miou_improve:+.2f}%")
print(f"{'Epoch-100 mIoU (%)':<20} {final_miou_enet:<15.2f} {final_miou_enet_ca:<15.2f} {miou_final_improve:+.2f}%")
print(f"{'最优 Epoch':<20} {best_epoch_enet:<15} {best_epoch_enet_ca:<15} {'':<10}")
print(f"{'最优 Val Loss':<20} {best_vloss_enet:<15.4f} {best_vloss_enet_ca:<15.4f} {best_vloss_enet_ca - best_vloss_enet:+.4f}")
print("=" * 65)

# CA模块效果判定
if miou_improve > 0:
    print(f"✅ CA注意力在ENet上有效：mIoU 提升 {miou_improve:+.2f}%")
else:
    print(f"⚠️ CA注意力在ENet上未带来提升：mIoU 变化 {miou_improve:+.2f}%")
print("=" * 65)

plt.show()
