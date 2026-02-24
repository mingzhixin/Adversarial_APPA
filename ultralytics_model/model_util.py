import torch
import torch.nn.functional as F
import numpy as np
import cv2

def yolo_batch2image_and_annotation(all_batch):
    # 将yolo的dataset中的数据转成dpatch_attack输入的形式
    # inputs:
    # all_batch: list yolo的dataloader中的全部输出 图像部分是RGB
    # outputs:
    # image_numpy: np.ndarray [B,C,H,W] 所有的图像内容 BGR
    # annotations: list 所有图像的class和bboxes部分
    image_numpy = []
    annotations = []
    for batch in all_batch:
        batch_img = batch["img"].detach().numpy().astype(np.float32)  # RGB 变量
        batch_cls = batch["cls"].detach().numpy()  # Tensor
        batch_cls = batch_cls.flatten().astype(int).tolist()  # List int
        batch_bboxes = batch["bboxes"].detach().numpy().astype(float)  # float64

        image_numpy.append(batch_img)
        annotations.append(list(zip(batch_bboxes, batch_cls)))

    image_numpy = np.concatenate(image_numpy, axis=0)  # [B,C,H,W] RGB
    image_numpy_bgr = image_numpy[:, [2, 1, 0], :, :]  # BGR

    return image_numpy_bgr, annotations


def update_yolo_batch(batch, patched_image, ground_truth):
    # 将dpatch中的数据格式转化成yolo的batch形式，并对batch进行更新
    # inputs:
    # batch: dict yolo的原batch
    # patched_image: np.ndarray [C,H,W] BGR
    # ground_truth: list 包含bboxes和labels标签
    # outputs:
    # batch: dict 修改后的yolo_batch img为RGB
    patched_image = torch.from_numpy(patched_image)
    patched_image = patched_image.to(torch.uint8)
    patched_image = patched_image[[2, 1, 0], :, :]  # BGR2RGB
    patched_image = torch.unsqueeze(patched_image, 0)  # [1,C,H,W]
    batch["img"] = patched_image
    bboxes = []
    labels = []
    for box, label in ground_truth:
        bboxes.append(box)
        labels.append([label])
    bboxes = np.array(bboxes).astype(np.float32)
    bboxes = torch.from_numpy(bboxes)
    batch["bboxes"] = bboxes
    labels = torch.tensor(labels, dtype=torch.float)
    batch["cls"] = labels
    if batch["batch_idx"].numel() == 0:
        batch["batch_idx"] = torch.zeros(len(bboxes),dtype=torch.float)
    # 针对多batch这里需要增加对batch_ids的处理

    return batch


def pad_to_multiple_of_32(img):
    """
    将图像填充为最小的 32 倍数尺寸。用于YOLO攻击
    支持 NumPy (HWC 或 HW) 和 Tensor (CHW 或 BCHW) 格式。
    返回类型与输入一致。
    """
    is_tensor = isinstance(img, torch.Tensor)

    if is_tensor:
        # Tensor: shape [B,C,H,W] or [C,H,W]
        if img.dim() == 4:
            B, C, H, W = img.shape
        elif img.dim() == 3:
            C, H, W = img.shape
        else:
            raise ValueError("Only supports [C,H,W] or [H,W] tensor")
    else:
        # NumPy: shape [H,W,C] or [H,W]
        if img.ndim == 2:
            H, W = img.shape
            C = None
        elif img.ndim == 3:
            H, W, C = img.shape
        else:
            raise ValueError("Only supports [H,W,C] or [H,W] numpy array")

    # 计算目标大小（32 的倍数）
    new_H = ((H + 31) // 32) * 32
    new_W = ((W + 31) // 32) * 32
    pad_H = new_H - H
    pad_W = new_W - W

    if is_tensor:
        # Tensor padding: (left, right, top, bottom)
        padding = (0, pad_W, 0, pad_H)
        if C is None:
            img = img.unsqueeze(0)  # [1,H,W]
            padded = F.pad(img, padding, mode='constant', value=0)
            return padded.squeeze(0)  # [H',W']
        else:
            return F.pad(img, padding, mode='constant', value=0)  # [C,H',W']
    else:
        # NumPy padding: (top, bottom, left, right)
        if C is None:
            return cv2.copyMakeBorder(img, 0, pad_H, 0, pad_W, cv2.BORDER_CONSTANT, value=0)
        else:
            return cv2.copyMakeBorder(img, 0, pad_H, 0, pad_W, cv2.BORDER_CONSTANT, value=(0,0,0))

def unpad_from_multiple_of_32(padded_img, original_shape):
    """
    移除填充，恢复原始尺寸。用于YOLO攻击后的图像还原。
    支持 NumPy (HWC 或 HW) 和 Tensor (CHW 或 BCHW) 格式。
    返回类型与输入一致。
    
    参数:
        padded_img: 填充后的图像
        original_shape: 原始图像的形状 (H,W) 或 (H,W,C) 或 (B,C,H,W) 或 (C,H,W)
    """
    is_tensor = isinstance(padded_img, torch.Tensor)
    
    if is_tensor:
        raise NotImplementedError("Tensor input is not supported")
        # Tensor: shape [B,C,H,W] or [C,H,W]
        if padded_img.dim() == 4:
            B, C, H_pad, W_pad = padded_img.shape
            _, _, H_orig, W_orig = original_shape
        elif padded_img.dim() == 3:
            C, H_pad, W_pad = padded_img.shape
            _, H_orig, W_orig = original_shape
        else:
            raise ValueError("Only supports [C,H,W] or [H,W] tensor")
    else:
        # NumPy: shape [H,W,C] or [H,W]
        if padded_img.ndim == 2:
            H_pad, W_pad = padded_img.shape
            H_orig, W_orig = original_shape
            C = None
        elif padded_img.ndim == 3:
            H_pad, W_pad, C = padded_img.shape
            H_orig, W_orig = original_shape
        else:
            raise ValueError("Only supports [H,W,C] or [H,W] numpy array")

    # 计算裁剪区域
    if is_tensor:
        # Tensor: 直接切片
        if padded_img.dim() == 4:
            return padded_img[:, :, :H_orig, :W_orig]
        else:
            return padded_img[:, :H_orig, :W_orig]
    else:
        # NumPy: 直接切片
        if C is None:
            return padded_img[:H_orig, :W_orig]
        else:
            return padded_img[:H_orig, :W_orig, :]

def get_det_result(boxes, scores):
    # Tensor输入输出 把boxes和scores整合起来
    scores = scores.unsqueeze(1)  # [N,1]
    det_result = torch.cat([boxes, scores], dim=1)

    return det_result


def tensor_bbox_iou(box1, box2):
    """
    以Tensor为输入的iou计算
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    xmin1 = box1[:, 0].unsqueeze(-1)
    ymin1 = box1[:, 1].unsqueeze(-1)
    xmax1 = box1[:, 2].unsqueeze(-1)
    ymax1 = box1[:, 3].unsqueeze(-1)

    xmin2 = box2[:, 0].unsqueeze(-1)
    ymin2 = box2[:, 1].unsqueeze(-1)
    xmax2 = box2[:, 2].unsqueeze(-1)
    ymax2 = box2[:, 3].unsqueeze(-1)

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    ymin = torch.max(ymin1, torch.squeeze(ymin2, dim=-1))
    xmin = torch.max(xmin1, torch.squeeze(xmin2, dim=-1))
    ymax = torch.min(ymax1, torch.squeeze(ymax2, dim=-1))
    xmax = torch.min(xmax1, torch.squeeze(xmax2, dim=-1))

    h = torch.max(ymax - ymin, torch.zeros(ymax.shape, device=ymax.device))
    w = torch.max(xmax - xmin, torch.zeros(xmax.shape, device=xmax.device))
    intersect = h * w

    union = area1 + np.squeeze(area2, axis=-1) - intersect
    iou = intersect / union
    return iou


def numpy_bbox_iou(bbox1, bbox2):
    """
    以Numpy为输入的iou计算
    Returns the IoU of two bounding boxes
    """
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))

    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w

    union = area1 + np.squeeze(area2, axis=-1) - intersect
    return intersect / union

