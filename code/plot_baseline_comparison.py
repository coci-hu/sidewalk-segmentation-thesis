"""
plot_baseline_comparison.py
绘制所有轻量化模型的基线对比图（训练/验证Loss曲线 + 验证mIoU曲线 + 最优指标柱状图）。

数据来源：train_logs/ 目录下各模型 100 epoch 训练日志 CSV。
输出：
  - train_logs/baseline_comparison.png  （高清对比图，3张子图）
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

# 各模型配置：CSV文件名、显示名称、颜色、线型
# 所有模型统一使用 CROP_SIZE=(768,1024)，对比公平
models_config = [
    {'file': 'train_log_100epoch_bin.csv',    'name': 'SBNet',      'color': '#1f77b4', 'ls': '-',  'params': '0.94M'},
    {'file': 'train_log_ca_100epoch_bin.csv',  'name': 'SBNet+CA',   'color': '#ff7f0e', 'ls': '-',  'params': '0.95M'},
    {'file': 'train_log_stdc_seg_100epoch_bin.csv', 'name': 'STDC-Seg',  'color': '#2ca02c', 'ls': '-',  'params': '2.25M'},
    {'file': 'train_log_ppliteseg_100epoch_bin.csv', 'name': 'PP-LiteSeg-T', 'color': '#d62728', 'ls': '-', 'params': '2.71M'},
    {'file': 'fastscnn_metrics.csv',           'name': 'Fast-SCNN',      'color': '#9467bd', 'ls': '--', 'params': '1.47M'},
    {'file': 'enet_metrics.csv',               'name': 'ENet',           'color': '#8c564b', 'ls': '--', 'params': '0.26M'},
    {'file': 'segformer_metrics.csv',          'name': 'SegFormer-B0',   'color': '#e377c2', 'ls': '--', 'params': '4.3M'},
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

# ===================== 创建图表 =====================
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# ---- 子图1：验证 Loss 曲线 ----
ax1 = axes[0]
for cfg in models_config:
    if cfg['name'] not in data:
        continue
    df = data[cfg['name']]
    ax1.plot(df['epoch'], df['val_loss'], color=cfg['color'], ls=cfg['ls'],
             label=f"{cfg['name']} ({cfg['params']})", linewidth=1.5, alpha=0.85)
ax1.set_xlabel('Epoch', fontsize=13)
ax1.set_ylabel('Val Loss', fontsize=13)
ax1.set_title('验证集 Loss 对比', fontsize=15, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, 100)

# ---- 子图2：验证 mIoU 曲线 ----
ax2 = axes[1]
for cfg in models_config:
    if cfg['name'] not in data:
        continue
    df = data[cfg['name']]
    ax2.plot(df['epoch'], df['val_mIoU'] * 100, color=cfg['color'], ls=cfg['ls'],
             label=f"{cfg['name']} ({cfg['params']})", linewidth=1.5, alpha=0.85)
ax2.set_xlabel('Epoch', fontsize=13)
ax2.set_ylabel('Val mIoU (%)', fontsize=13)
ax2.set_title('验证集 mIoU 对比', fontsize=15, fontweight='bold')
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 100)

# ---- 子图3：最优 mIoU 柱状图 ----
ax3 = axes[2]
names = []
best_mious = []
colors = []
for cfg in models_config:
    if cfg['name'] not in data:
        continue
    df = data[cfg['name']]
    best_miou = df['val_mIoU'].max() * 100
    names.append(cfg['name'])
    best_mious.append(best_miou)
    colors.append(cfg['color'])

# 按mIoU降序排列
sorted_idx = np.argsort(best_mious)[::-1]
names_sorted = [names[i] for i in sorted_idx]
best_mious_sorted = [best_mious[i] for i in sorted_idx]
colors_sorted = [colors[i] for i in sorted_idx]

bars = ax3.barh(range(len(names_sorted)), best_mious_sorted, color=colors_sorted, height=0.6, alpha=0.85)
ax3.set_yticks(range(len(names_sorted)))
ax3.set_yticklabels(names_sorted, fontsize=12)
ax3.set_xlabel('Best Val mIoU (%)', fontsize=13)
ax3.set_title('最优 mIoU 排名 (Epoch 1-100)', fontsize=15, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')
ax3.invert_yaxis()

# 在柱状图上标注数值
for i, (bar, val) in enumerate(zip(bars, best_mious_sorted)):
    ax3.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
             f'{val:.2f}%', va='center', fontsize=11, fontweight='bold')

ax3.set_xlim(0, max(best_mious_sorted) + 5)

plt.tight_layout(pad=2.0)

# ---- 保存 ----
save_path = os.path.join(LOG_DIR, 'baseline_comparison.png')
plt.savefig(save_path, dpi=200, bbox_inches='tight')
print(f"\n📊 对比图已保存：{save_path}")

# ===================== 汇总表输出 =====================
print("\n" + "=" * 75)
print("              轻量化可通行区域分割模型对比汇总 (100 Epoch)")
print("=" * 75)
print(f"{'模型':<18} {'参数量':<10} {'输入尺寸':<14} {'最优mIoU':<12} {'Epoch-100 mIoU':<14} {'最优Epoch':<10}")
print("-" * 75)

# 输入尺寸标注（所有模型统一 768×1024）
crop_sizes = {
    'SBNet': '768×1024',
    'SBNet+CA': '768×1024',
    'STDC-Seg': '768×1024',
    'PP-LiteSeg-T': '768×1024',
    'Fast-SCNN': '768×1024',
    'ENet': '768×1024',
    'SegFormer-B0': '768×1024',
}

for cfg in models_config:
    if cfg['name'] not in data:
        continue
    df = data[cfg['name']]
    best_miou = df['val_mIoU'].max()
    best_epoch = df.loc[df['val_mIoU'].idxmax(), 'epoch']
    final_miou = df['val_mIoU'].iloc[-1]
    crop = crop_sizes.get(cfg['name'], '-')
    marker = ' ★' if cfg['name'] == 'SBNet+CA' else ''
    print(f"{cfg['name']:<18} {cfg['params']:<10} {crop:<14} {best_miou*100:.2f}%{'':>5} {final_miou*100:.2f}%{'':>7} {int(best_epoch):<10}{marker}")

print("=" * 75)
print("所有模型统一 CROP_SIZE=(768,1024)，实验条件一致，对比公平")
print("★ 为本文改进方法（SBNet + 坐标注意力）")
print("=" * 75)

plt.show()
