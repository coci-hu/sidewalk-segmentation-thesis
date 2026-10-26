##训练每轮结束后，在验证集 val 调用这两个函数，输出本轮 mIoU、像素精度；
##保存每一轮指标到日志，画图对比「迁移学习训练」vs「从零训练」两组模型效果；
##论文实验章节表格直接填入 mIoU、Pixel Acc，证明迁移学习提升分割精度

import torch
import numpy as np

def compute_mIoU(pred_mask, true_mask, num_classes=2):
    """
    计算平均交并比 mIoU（语义分割核心评价指标）
    :param pred_mask: 模型预测掩码 [B, H, W]
    :param true_mask: 真实标签掩码 [B, H, W]
    :param num_classes: 类别数量，本任务固定2（可通行区域/背景）
    :return: 平均IoU数值
    """
    iou_list = []
    for cls in range(num_classes):
        # 当前类别的预测区域、真实区域
        pred_cls = (pred_mask == cls)
        label_cls = (true_mask == cls)
        # 交集：同时为当前类的像素
        intersection = torch.logical_and(pred_cls, label_cls).sum()
        # 并集：任意一个为当前类的像素
        union = torch.logical_or(pred_cls, label_cls).sum()

        if union == 0:
            iou = torch.tensor(0.0)
        else:
            iou = intersection / union
        iou_list.append(iou.item())
    return np.mean(iou_list)


def compute_pixel_acc(pred_mask, true_mask):
    """
    像素精度 Pixel Acc：预测正确像素 / 全部像素
    :param pred_mask: 模型输出预测掩码
    :param true_mask: 真值标签掩码
    :return: 像素准确率 0~1
    """
    correct_pixel = (pred_mask == true_mask).sum().item()
    total_pixel = torch.numel(pred_mask)
    return correct_pixel / total_pixel