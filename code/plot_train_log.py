import csv
import matplotlib.pyplot as plt
import os

# ============================================================
# 绘制 Baseline vs CA 改进版 对比图
# ============================================================

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'train_logs')
FIG_DIR = LOG_DIR   # 图表统一输出到 train_logs/，不再写到 figures/（避免同一论文的图分散两处）
os.makedirs(FIG_DIR, exist_ok=True)


def load_csv(path):
    with open(path, 'r') as f:
        rows = list(csv.DictReader(f))
    return (
        [int(r['epoch']) for r in rows],
        [float(r['train_loss']) for r in rows],
        [float(r['val_loss']) for r in rows],
        [float(r['val_mIoU']) for r in rows],
    )


# 读取两份日志
csv_bl = os.path.join(LOG_DIR, 'train_log_100epoch_bin.csv')
csv_ca = os.path.join(LOG_DIR, 'train_log_ca_100epoch_bin.csv')

epochs_bl, train_loss_bl, val_loss_bl, val_miou_bl = load_csv(csv_bl)
epochs_ca, train_loss_ca, val_loss_ca, val_miou_ca = load_csv(csv_ca)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# 子图1：Loss 对比
ax1.plot(epochs_bl, train_loss_bl, label='Baseline Train Loss', linewidth=1.5)
ax1.plot(epochs_bl, val_loss_bl,   label='Baseline Val Loss', linewidth=1.5)
ax1.plot(epochs_ca, train_loss_ca, label='CA Train Loss', linewidth=1.5, linestyle='--')
ax1.plot(epochs_ca, val_loss_ca,   label='CA Val Loss', linewidth=1.5, linestyle='--')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training & Validation Loss', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 子图2：mIoU 对比
ax2.plot(epochs_bl, val_miou_bl, label='Baseline mIoU', linewidth=1.5)
ax2.plot(epochs_ca, val_miou_ca, label='CA mIoU', linewidth=1.5, linestyle='--')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('mIoU', fontsize=12)
ax2.set_title('Validation mIoU', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(FIG_DIR, 'baseline_vs_ca_comparison.png')
fig.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'对比图已保存至: {os.path.abspath(save_path)}')
plt.close()
