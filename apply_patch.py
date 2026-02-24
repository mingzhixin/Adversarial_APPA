import os
import csv
import cv2
import tqdm
import torch
from torch import autograd
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import patch_config
from load_data import *


def xywhn_to_xyxy(xywhn, img_size: int):
    """
    YOLO normalized (x_c, y_c, w, h) -> pixel xyxy
    xywhn: (..., 4)
    """
    x_c = xywhn[..., 0] * img_size
    y_c = xywhn[..., 1] * img_size
    w = xywhn[..., 2] * img_size
    h = xywhn[..., 3] * img_size
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(box1, box2):
    """
    box1: (N,4) xyxy
    box2: (M,4) xyxy
    return: (N,M) IoU
    """
    if box1.numel() == 0 or box2.numel() == 0:
        return torch.zeros((box1.shape[0], box2.shape[0]), device=box1.device)

    # Intersection
    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    # Areas
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-9)


def extract_valid_labels(lab_one_img):
    """
    lab_one_img: (max_lab, 5)  [cls, x, y, w, h]  (通常是这种)
    返回: cls (K,), boxes_xyxy (K,4) 都在 GPU 上
    """
    # 过滤掉 padding（常见 padding 为全 0 或 w/h=0）
    cls = lab_one_img[:, 0]
    xywhn = lab_one_img[:, 1:5]
    valid = (xywhn[:, 2] > 0) & (xywhn[:, 3] > 0)
    return cls[valid], xywhn[valid]


if __name__ == '__main__':
    config = patch_config.patch_configs['yolov5']()
    model = config.model.eval().cuda()

    patch_path = "trained_patch/yolov5l_patch50_97_0.4.png"
    adv_patch_cpu = cv2.imread(patch_path)
    adv_patch_cpu = cv2.cvtColor(adv_patch_cpu, cv2.COLOR_BGR2RGB)
    adv_patch_cpu = torch.from_numpy(adv_patch_cpu)
    adv_patch_cpu = adv_patch_cpu.permute((2, 0, 1)) / 255

    max_lab = 60
    batch_size = 1
    img_size = 800
    conf_thres = 0.4  # 0.4
    iou_thres = 0.45  # 0.45
    max_det = 1000

    # 目标级别 ASR 的匹配 IoU 阈值（用来判断 GT 是否“仍被检测到”）
    iou_match = 0.3

    # 如果你只关心行人（Inria 通常是 person 类），设成 0；
    # 如果你想对所有 GT 类都算，就设成 None
    target_cls_id = 0

    # 结果输出
    os.makedirs("patch_applier", exist_ok=True)
    os.makedirs("patch_applier/images", exist_ok=True)
    asr_csv_path = "patch_applier/asr_target_level.csv"

    train_loader = torch.utils.data.DataLoader(
        InriaDataset(config.img_dir, config.lab_dir, max_lab, img_size, shuffle=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )
    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()

    # 累计统计
    total_targets = 0
    miss_targets = 0
    total_images = 0
    success_images_all_missed = 0

    with open(asr_csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "img_index",
            "gt_targets",
            "miss_targets",
            "target_asr_img",
            "all_gt_missed_img"
        ])

        for i_batch, (img_batch, lab_batch) in enumerate(train_loader):
            img_batch = img_batch.cuda()
            lab_batch = lab_batch.cuda()

            adv_patch = adv_patch_cpu.cuda()
            adv_batch_t = patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
            p_img_batch = patch_applier(img_batch, adv_batch_t)
            p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))

            # 保存可视化
            img = p_img_batch[0, :, :, ]
            img = transforms.ToPILImage()(img.detach().cpu())
            img.save(f"patch_applier/images/{i_batch + 1}.png")

            # 推理 + NMS
            yolov5_output = model(p_img_batch)
            output = non_max_suppression(
                yolov5_output,
                conf_thres,
                iou_thres,
                classes=None,
                agnostic=False,
                max_det=max_det
            )

            # -----------------------------
            # 目标级别 ASR 统计（按 GT 框）
            # -----------------------------
            total_images += 1

            # 取该图 GT
            cls_gt, xywhn_gt = extract_valid_labels(lab_batch[0])
            if target_cls_id is not None:
                keep = (cls_gt == float(target_cls_id)) | (cls_gt == target_cls_id)
                cls_gt = cls_gt[keep]
                xywhn_gt = xywhn_gt[keep]

            gt_count = int(xywhn_gt.shape[0])
            total_targets += gt_count

            if gt_count == 0:
                # 没 GT 就记一行，但不影响 ASR
                writer.writerow([i_batch + 1, 0, 0, 0.0, 0])
                continue

            gt_boxes = xywhn_to_xyxy(xywhn_gt, img_size)  # (K,4)

            # 取该图检测结果
            det = output[0]  # (N,6) xyxy conf cls 或者 None/empty
            if det is None or det.numel() == 0:
                # 全部 miss
                miss_k = gt_count
                miss_targets += miss_k
                success_images_all_missed += 1
                writer.writerow([i_batch + 1, gt_count, miss_k, 1.0, 1])
                continue

            det_boxes = det[:, 0:4]
            det_conf = det[:, 4]
            det_cls = det[:, 5]

            # 可选：只用某个类别的 detection 去匹配（比如 person=0）
            if target_cls_id is not None:
                det_keep = (det_cls == float(target_cls_id)) | (det_cls == target_cls_id)
                det_boxes = det_boxes[det_keep]
                det_conf = det_conf[det_keep]

            if det_boxes.numel() == 0:
                miss_k = gt_count
                miss_targets += miss_k
                success_images_all_missed += 1
                writer.writerow([i_batch + 1, gt_count, miss_k, 1.0, 1])
                continue

            # IoU 匹配：每个 GT 看是否存在任意 det 与它 IoU>=iou_match
            ious = box_iou(gt_boxes, det_boxes)  # (K,Nd)
            matched = (ious.max(dim=1).values >= iou_match)  # (K,)
            miss_k = int((~matched).sum().item())

            miss_targets += miss_k
            all_missed = 1 if miss_k == gt_count else 0
            if all_missed:
                success_images_all_missed += 1

            target_asr_img = miss_k / max(gt_count, 1)
            writer.writerow([i_batch + 1, gt_count, miss_k, target_asr_img, all_missed])

    # 总结打印
    overall_target_asr = miss_targets / max(total_targets, 1)
    overall_img_asr = success_images_all_missed / max(total_images, 1)

    print("========== ASR Summary ==========")
    print(f"Total images: {total_images}")
    print(f"Total GT targets: {total_targets}")
    print(f"Missed GT targets: {miss_targets}")
    print(f"[Target-level ASR] = {overall_target_asr:.4f}")
    print(f"[Image-level ASR (all missed)] = {overall_img_asr:.4f}")
    print(f"Per-image target ASR saved to: {asr_csv_path}")
