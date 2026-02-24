import torch
from typing import Dict, Optional, Tuple
import os

from .base_mmdetection_detector import BaseMMDetectionDetector


class FasterRCNNDetector(BaseMMDetectionDetector):
    """
    Faster R-CNN检测器类，用于对抗攻击的检测器基础
    支持MMDetection框架下的Faster R-CNN模型
    """

    def __init__(self,
                 device: str = None,
                 config_path: str = None,
                 checkpoint_path: str = None,
                 input_size: Tuple[int, int] = (800, 800),
                 score_thr: float = 0.05):
        """
        初始化Faster R-CNN检测器
        
        Args:
            config_path: MMDetection配置文件路径，如果为None则使用默认路径
            checkpoint_path: 模型权重路径，如果为None则使用默认路径
            device: 设备 ('cuda:0', 'cpu'等)，如果为None则自动选择
            score_thr: 置信度阈值，用于过滤检测结果
        """
        super().__init__(
            model_name='fasterrcnn',
            device=device,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            input_size=input_size,
            score_thr=score_thr
        )

    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        return 'mmdetection/configs/faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py'

    def _get_default_checkpoint_path(self) -> str:
        """获取默认权重路径"""
        return 'mmdetection/weights/faster_rcnn_r101_fpn_1x_coco.pth'

    def _resize_image(self, img_np, target_size):
        """
        预处理函数，将图像调整到指定尺寸
        
        Args:
            img_np: 输入图像numpy数组
            target_size: 目标尺寸 (H, W)
        Returns:
            img_np: 调整尺寸后的图像numpy数组
        """
        import cv2
        # 注意：cv2.resize的size参数是(width, height)
        img_np = cv2.resize(img_np, (target_size[1], target_size[0]))
        return img_np

    def preprocess_img(self, img_np):
        # """
        # 预处理函数，将图像调整到指定尺寸
        # """
        # if img_np.shape[:2] != self.input_size:
        #     return self._resize_image(img_np, self.input_size)
        # else:
        #     return img_np
        return img_np

    def postprocess_img(self, img_np, original_shape: Tuple[int, int] = None):
        # """
        # 后处理函数，将图像缩放回原始尺寸

        # Args:
        #     img_np: 输入图像numpy数组，形状为 [H, W, 3]
        #     original_shape: 原始图像尺寸 (H, W)

        # Returns:
        #     img_np: 缩放回原始尺寸的图像numpy数组
        # """
        # if img_np.shape[:2] != original_shape:
        #     return self._resize_image(img_np, original_shape)
        # else:
        #     return img_np
        return img_np

    def preprocess_to_input_tensor4d(self, img_np):
        """获取网络输入图像张量）
        
        Args:
            img_np: numpy array of the image, [H,W,3], BGR
        
        Returns:
            img_tensor4D: torch tensor of the image, [1,3,H,W]
        """
        return super().preprocess_to_input_tensor4d(img_np)

    def get_predictions(self, image_tensor, original_shape: Optional[Tuple[int, int]] = None, score_thres: float =None):
        """
        获取可导预测（支持批量），并做阈值过滤

        Returns:
            单张：dict(bboxes=..., scores=..., labels=...)
            批量：List[dict(...)]，每张一份
        """
        preds = super().get_predictions(image_tensor, original_shape=original_shape, score_thres=score_thres)

        def _one(pred_dict):
            b, s, l = self._postprocess_predictions(
                pred_dict.get('bboxes'), pred_dict.get('scores'), pred_dict.get('labels'))
            return dict(bboxes=b, scores=s, labels=l)

        if isinstance(preds, list):
            return [_one(p) for p in preds]

        return _one(preds)

    def to_ultralytics(self, detections, img_shape: Tuple[int, int]):
        """
        将本检测器输出整理为 Ultralytics/YOLO 常用结果结构（每张一份）

        Args:
            detections: dict 或 List[dict]，来自 get_predictions()/detect()
            img_shape: (H, W)，用于归一化

        Returns:
            若输入为单张：dict，包含
              - boxes.xyxy: (N,4) float32
              - boxes.conf: (N,) float32
              - boxes.cls:  (N,) float32
              - yolo_txt:   (N,6) float32 -> [cls, x, y, w, h, conf] (归一化到0-1)
            若输入为批量：List[dict]，每张同结构
        """
        import numpy as np

        def _convert(one: dict):
            b = one['bboxes']
            s = one['scores']
            l = one['labels']

            # torch -> numpy
            if hasattr(b, 'detach'):
                b = b.detach().cpu().numpy()
            if hasattr(s, 'detach'):
                s = s.detach().cpu().numpy()
            if hasattr(l, 'detach'):
                l = l.detach().cpu().numpy()

            b = b.astype(np.float32) if b.size else b.reshape(0, 4).astype(np.float32)
            s = s.astype(np.float32) if s.size else s.reshape(0, ).astype(np.float32)
            l = l.astype(np.float32) if l.size else l.reshape(0, ).astype(np.float32)

            H, W = img_shape
            # xyxy -> xywh (pixel)
            x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            ww = (x2 - x1)
            hh = (y2 - y1)

            # normalize
            eps = 1e-6
            xc_n = xc / max(W, eps)
            yc_n = yc / max(H, eps)
            ww_n = ww / max(W, eps)
            hh_n = hh / max(H, eps)

            yolo_txt = np.stack([l, xc_n, yc_n, ww_n, hh_n, s], axis=1) if b.shape[0] else np.zeros((0, 6), np.float32)

            return {
                'boxes': {
                    'xyxy': b,
                    'conf': s,
                    'cls': l
                },
                'yolo_txt': yolo_txt
            }

        if isinstance(detections, list):
            return [_convert(d) for d in detections]
        return _convert(detections)

    def save_ultralytics_txt(self, ultra_results, out_dir: str, names: Optional[list] = None):
        """
        将 to_ultralytics() 的结果保存成 Ultralytics 常用的 txt 标签格式：
            cls x y w h conf   （均为归一化）

        Args:
            ultra_results: dict 或 List[dict]，来自 to_ultralytics()
            out_dir: 输出目录
            names: 可选，文件名列表（不含后缀）。若不传，默认用 0..B-1
        """
        import os
        import numpy as np

        os.makedirs(out_dir, exist_ok=True)

        def _save(one, stem: str):
            arr = one['yolo_txt']
            path = os.path.join(out_dir, f'{stem}.txt')
            with open(path, 'w') as f:
                for row in arr:
                    cls, x, y, w, h, conf = row.tolist()
                    f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

        if isinstance(ultra_results, list):
            if names is None:
                names = [str(i) for i in range(len(ultra_results))]
            for one, stem in zip(ultra_results, names):
                _save(one, stem)
        else:
            _save(ultra_results, names[0] if names else '0')


    def _postprocess_predictions(self, bboxes, scores, labels):
        """阈值过滤（保持梯度）"""
        if bboxes is None or scores is None or labels is None:
            # 兼容空输出
            device = bboxes.device if isinstance(bboxes, torch.Tensor) else self.model.device
            return (torch.empty(0, 4, device=device),
                    torch.empty(0, device=device),
                    torch.empty(0, dtype=torch.long, device=device))

        if self.score_thr > 0 and scores.numel() > 0:
            valid_mask = scores > self.score_thr
            if valid_mask.any():
                bboxes = bboxes[valid_mask]
                scores = scores[valid_mask]
                labels = labels[valid_mask]
            else:
                return (torch.empty(0, 4, device=bboxes.device),
                        torch.empty(0, device=bboxes.device),
                        torch.empty(0, dtype=torch.long, device=bboxes.device))
        return bboxes, scores, labels


    def get_model_info(self) -> Dict:
        """获取模型信息"""
        info = super().get_model_info()
        info['model_name'] = 'faster_rcnn'  # 确保返回正确的模型名称
        return info


def main():
    """测试函数"""
    # 创建检测器实例
    detector = FasterRCNNDetector()

    # 打印模型信息
    print("Model Info:", detector.get_model_info())

    # 测试单张图像检测
    img_path = 'dataset/vehicle_images_5/images/000000000471.jpg'
    if os.path.exists(img_path):
        result = detector.detect(img_path)
        print(f"Detection result: {result}")


if __name__ == '__main__':
    main()
