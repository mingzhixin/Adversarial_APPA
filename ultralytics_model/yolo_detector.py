import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from typing import Tuple, Dict, Optional, Union
import os
import warnings

from .model_util import pad_to_multiple_of_32, unpad_from_multiple_of_32
from .model_util import get_det_result, tensor_bbox_iou
from .yolo_loss import *
from .ops import non_max_suppression

class YOLODetector:
    """
    YOLO检测器类，用于对抗攻击的检测器基础
    支持YOLOv3、YOLOv5、YOLOv8等模型
    """

    def __init__(self,
                 model_name: str = 'yolov5',
                 checkpoint_path: str = None,
                 device: str = None,
                 conf_thr: float = 0.25,
                 iou_thr: float = 0.7,
                 input_size: int = 640
                 ):
        """
        初始化YOLO检测器

        Args:
            model_name: 模型名称 ('yolov3', 'yolov5', 'yolov8')
            device: 设备 ('cuda:0', 'cpu'等)，如果为None则自动选择
            checkpoint_path: 模型权重路径，如果为None则使用默认路径
            score_thr: 置信度阈值，用于过滤检测结果
        """
        self.model_name = model_name
        self.device = device if device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conf_thr = conf_thr  # 添加置信度阈值属性
        self.iou_thr = iou_thr

        # 设置默认权重路径
        if checkpoint_path is None:
            checkpoint_path = self._get_default_config_path(model_name)

        self.checkpoint_path = checkpoint_path
        self.model = None
        self.detection_model = None  # Ultralytics底层YOLO模型
        self.cfg = None

        # 初始化模型
        self._load_model()

    def _get_default_config_path(self, model_name: str) -> str:
        """获取默认权重路径"""
        default_paths = {
            'yolov3': 'weights/yolov3u_best.pt',  # 使用ultralytics自动下载
            'yolov5': 'weights/yolov5lu_best.pt',  # 使用ultralytics自动下载
            'yolov8': 'weights/yolov8l.pt'  # 使用ultralytics自动下载
        }
        return default_paths.get(model_name, 'yolo/weights/yolov8l.pt')

    def _load_model(self):
        """加载YOLO模型"""
        try:
            self.model = YOLO(self.checkpoint_path)
            print(f"we use {self.model_name}, {self.checkpoint_path}")
            self.detection_model = self.model.model
            self.cfg = get_cfg(DEFAULT_CFG)
            self.cfg.model = self.checkpoint_path
            self.cfg.device = self.device
            self.detection_model.args = self.cfg
            self.detection_model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect(self, source: Union[str, np.ndarray, torch.Tensor]) -> dict:
        """
        统一的检测函数，支持多种输入格式

        这个函数使用YOLO模型的高级接口，自动处理不同输入格式的预处理和后处理，
        返回统一的格式化检测结果。支持图像路径、numpy数组和PyTorch张量三种输入格式。

        Args:
            source: 输入源，支持以下格式：
                - str: 图像文件路径，如 "path/to/image.jpg"
                - np.ndarray: numpy数组格式的图像，形状为 [H, W, 3] 或 [3, H, W]，RGB格式
                - torch.Tensor: PyTorch张量格式的图像，形状为 [B, C, H, W] 或 [C, H, W]

        Returns:
            dict: 检测结果字典，包含以下键值对：
                - bboxes (torch.Tensor): 边界框坐标 [N, 4]，格式为 [x1, y1, x2, y2]，CPU张量
                - labels (torch.Tensor): 类别标签 [N]，CPU张量
                - scores (torch.Tensor): 置信度分数 [N]，CPU张量
                如果没有检测到目标，所有张量都是空的

        Raises:
            FileNotFoundError: 当输入是路径但文件不存在时
            ValueError: 当输入格式不支持或图像格式错误时
            TypeError: 当输入类型不是支持的三种类型时

        Examples:
            # 图像路径输入
            result = detector.detect("dataset/images/car.jpg")

            # numpy数组输入
            img_np = cv2.imread("image.jpg")
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            result = detector.detect(img_np)

            # PyTorch张量输入
            img_tensor = torch.randn(1, 3, 640, 640)
            result = detector.detect(img_tensor)

            # 访问检测结果
            bboxes = result['bboxes']
            labels = result['labels']
            scores = result['scores']

            print(f"检测到 {len(bboxes)} 个目标")
            for i in range(len(bboxes)):
                print(f"目标 {i+1}: 边界框={bboxes[i].tolist()}, 类别={labels[i].item()}, 置信度={scores[i].item():.3f}")

        Notes:
            - 对于numpy数组输入，如果形状是 [3, H, W]，会自动转换为 [H, W, 3]
            - 对于张量输入，建议使用 [B, C, H, W] 格式，其中B是批次大小
            - 所有输入都会自动进行预处理（归一化、尺寸调整等）
            - 检测结果会自动进行后处理（NMS、坐标缩放等）
            - 返回的字典包含CPU张量，便于后续处理
            - 如果没有检测到目标，返回空张量而不是None
        """
        # 输入类型检查和预处理
        if isinstance(source, str):
            # 图像路径输入
            if not os.path.exists(source):
                raise FileNotFoundError(f"Image path not found: {source}")
            # 直接使用YOLO模型处理路径，它会自动加载和预处理图像
            result = self.model(source)

        elif isinstance(source, np.ndarray):
            # numpy数组输入
            # 检查数组维度和格式
            if source.ndim != 3:
                raise ValueError(f"Expected 3D numpy array [H, W, 3] or [3, H, W], got shape {source.shape}")

            # 如果通道维度在前 [3, H, W]，转换为 [H, W, 3]
            if source.shape[0] == 3:
                source = source.transpose((1, 2, 0))

            # 检查转换后的形状
            if source.shape[2] != 3:
                raise ValueError(f"Expected 3 channels (RGB), got {source.shape[2]} channels")

            # 使用YOLO模型处理numpy数组
            result = self.model(source)

        elif isinstance(source, torch.Tensor):
            # PyTorch张量输入
            # 检查张量维度
            if source.dim() not in [3, 4]:
                raise ValueError(f"Expected 3D [C, H, W] or 4D [B, C, H, W] tensor, got shape {source.shape}")

            # 如果是3D张量 [C, H, W]，添加批次维度 [1, C, H, W]
            if source.dim() == 3:
                source = source.unsqueeze(0)

            # 检查通道数
            if source.shape[1] != 3:
                raise ValueError(f"Expected 3 channels (RGB), got {source.shape[1]} channels")

            # 使用YOLO模型处理张量
            result = self.model(source)

        else:
            # 不支持的输入类型
            raise TypeError(f"Unsupported input type: {type(source)}. "
                            f"Expected str (path), np.ndarray, or torch.Tensor")

        # 统一输出格式，与 get_predictions 保持一致
        if len(result[0].boxes) > 0:
            boxes = result[0].boxes
            bboxes = boxes.xyxy.detach().cpu()
            labels = boxes.cls.detach().cpu()
            scores = boxes.conf.cpu()

            # 根据置信度阈值过滤结果
            if self.conf_thr > 0:
                keep_mask = scores >= self.conf_thr
                bboxes = bboxes[keep_mask]
                labels = labels[keep_mask]
                scores = scores[keep_mask]
        else:
            # 如果没有检测到目标，返回空张量
            bboxes = torch.empty((0, 4), device='cpu')
            labels = torch.empty((0,), device='cpu')
            scores = torch.empty((0,), device='cpu')

        return dict(bboxes=bboxes, labels=labels, scores=scores)

    def get_predictions(self, image_tensor: torch.Tensor, original_shape: Optional[Tuple[int, int]] = None) -> dict:
        """
        获取模型预测结果

        这个函数使用YOLO模型的底层接口，直接处理张量输入并返回预测结果。
        与detect函数不同，这个函数需要手动预处理输入张量。

        Args:
            image_tensor: 图像张量 [B, C, H, W]，值范围应为 [0, 1]

        Returns:
            dict: 检测结果字典，包含以下键值对：
                - bboxes (torch.Tensor): 边界框坐标 [N, 4]，格式为 [x1, y1, x2, y2]，设备张量
                - labels (torch.Tensor): 类别标签 [N]，设备张量
                - scores (torch.Tensor): 置信度分数 [N]，设备张量
                如果没有检测到目标，所有张量都是空的

        Examples:
            # 准备输入张量
            img_tensor = torch.randn(1, 3, 640, 640) / 255.0
            result = detector.get_predictions(img_tensor)

            # 访问检测结果
            bboxes = result['bboxes']
            labels = result['labels']
            scores = result['scores']

            print(f"检测到 {len(bboxes)} 个目标")
            for i in range(len(bboxes)):
                print(f"目标 {i+1}: 边界框={bboxes[i].tolist()}, 类别={labels[i].item()}, 置信度={scores[i].item():.3f}")

        Notes:
            - 返回的张量保持在原始设备上（GPU/CPU）
            - 使用NMS后处理，置信度阈值0.25，IoU阈值0.7
            - 如果没有检测到目标，返回空张量而不是None
        """
        # image_tensor = image_tensor / 255.0  # TODO: no hardcoding 255
        unprocessed_prediction = self.detection_model(image_tensor)
        bboxes, labels, scores = self.post_process_predictions(unprocessed_prediction)

        return dict(bboxes=bboxes, labels=labels, scores=scores)

    def post_process_predictions(self, unprocessed_prediction):
        preds = non_max_suppression(
            unprocessed_prediction,
            conf_thres=self.conf_thr,
            iou_thres=self.iou_thr
        )

        if len(preds[0]) > 0:
            bboxes = preds[0][:, :4]
            labels = preds[0][:, 5]
            scores = preds[0][:, 4]
        else:
            bboxes = torch.empty((0, 4), device=self.device)
            labels = torch.empty((0,), device=self.device)
            scores = torch.empty((0,), device=self.device)

        # TODO recover padding

        return bboxes, labels, scores

    def postprocess_img(self, img_input, original_shape: Tuple[int, int] = None):
        if img_input.shape[:2] != original_shape:
            return unpad_from_multiple_of_32(img_input, original_shape)
        else:
            return img_input

    def preprocess_img(self, img_input):
        """
        预处理函数，兼容numpy数组和tensor输入

        Args:
            img_input: 输入图像，可以是numpy数组 [H,W,3] 或tensor [C,H,W]

        Returns:
            processed_img: 预处理后的图像，与输入类型相同
        """
        warnings.warn(
            "preprocess_img() is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2  # 确保警告指向调用者而非本文件
        )

        if isinstance(img_input, np.ndarray):
            # numpy数组输入
            processed_img = pad_to_multiple_of_32(img_input)
        elif isinstance(img_input, torch.Tensor):
            # tensor输入
            processed_img = pad_to_multiple_of_32(img_input)
        else:
            raise TypeError(f"Unsupported input type: {type(img_input)}. Expected np.ndarray or torch.Tensor")

        return processed_img

    def preprocess_to_input_tensor4d(self, img_np):
        """获取网络输入图像张量

        Args:
            img_np: numpy array of the image, [H,W,3], BGR

        Returns:
            img_tensor4D: torch tensor of the image, [1,3,H,W]
        """
        warnings.warn(
            "preprocess_to_input_tensor4d() is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2  # 确保警告指向调用者而非本文件
        )
        assert type(img_np) == np.ndarray, f'the img type is {type(img_np)}, but ndarray expected'
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_np).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC2CHW

        # 使用preprocess_img进行预处理
        img_tensor = self.preprocess_img(img_tensor)

        img_tensor4D = img_tensor.to(self.device).unsqueeze(0)
        return img_tensor4D

    def get_gradients(self, batch: Dict, config: object = None) -> Tuple[np.ndarray, torch.Tensor]:
        """
        获取模型梯度（DPatch格式接口）

        Args:
            batch: 包含图像信息的字典
            config: 配置对象

        Returns:
            grads_bgr: BGR格式的梯度图
            loss: YOLO损失值
        """
        if config:
            self.detection_model.args = config

        # 准备输入数据
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255  # TODO：no hardcoding 255
        batch["img"] = batch["img"].requires_grad_(True)

        # 前向传播
        loss, loss_items = self.detection_model(batch)

        # 确保loss是标量
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.sum()

        # 计算梯度
        grads = torch.autograd.grad(loss, batch["img"], retain_graph=True)[0]

        # 转换为numpy并调整通道顺序
        grads = grads.cpu().numpy()
        grads_bgr = grads[:, [2, 1, 0], :, :]  # RGB to BGR

        return grads_bgr, loss

    def get_loss(self, image_tensor, init_det):
        """

        Args:
            image_tensor: [0,255]
            init_det:

        Returns:

        """
        print("debug get_loss in yolo, image_tensor ", image_tensor)
        image_tensor = image_tensor / 255
        # Ultralytics官方的loss计算
        unprocessed_prediction = self.detection_model(image_tensor)
        self.detection_model.criterion = v8DetectionLossTensor(self.detection_model)
        loss, loss_item = self.detection_model.criterion(unprocessed_prediction, init_det)
        return loss, loss_item

    def get_loss_advpatch(self, image_tensor):
        """

        Args:
            image_tensor: [0,255]
            init_det:

        Returns:

        """
        image_tensor = image_tensor / 255
        # Ultralytics官方的loss计算
        unprocessed_prediction = self.detection_model(image_tensor)
        self.detection_model.criterion = v8DetectionLossAdvPatch(self.detection_model)
        loss, loss_item = self.detection_model.criterion(unprocessed_prediction)
        return loss, loss_item

    def get_loss_fca(self, image_tensor, init_det):
        """

        Args:
            image_tensor: [0,255]
            init_det:

        Returns:

        """
        image_tensor = image_tensor / 255
        # 修改过的Ultralytics官方的loss计算
        unprocessed_prediction = self.detection_model(image_tensor)
        self.detection_model.criterion = v8DetectionLossFCA(self.detection_model)
        loss, loss_item = self.detection_model.criterion(unprocessed_prediction, init_det)
        return loss, loss_item

    def detection_loss(self, det_bboxes, init_det, iou_thre):  # 原yolo_loss
        # 我看源代码是直接把三个尺度的特征图的每个cell的三个anchor的conf做mse损失
        # 我觉得这里不太合理，还是仿照着faster_loss的写法
        cls_score = det_bboxes[:, -1].to(self.device)
        mseloss = torch.nn.MSELoss()
        if cls_score[cls_score >= iou_thre].shape[0] == 0:
            class_loss = mseloss(cls_score * 0, torch.zeros_like(cls_score, device=self.device))  #########
            iou_loss = torch.zeros([1]).to(self.device)
        else:
            class_loss = mseloss(cls_score[cls_score >= iou_thre],
                                 torch.zeros(cls_score[cls_score >= iou_thre].shape, device=self.device))  #########
            box_pred = det_bboxes[:, 0:4]
            box_init = init_det[:, 0:4]
            pred_iou = tensor_bbox_iou(box_pred, box_init)
            iou_loss = torch.sum(pred_iou) / det_bboxes.shape[0]
        loss = class_loss + iou_loss

        return loss, class_loss.item(), iou_loss.item()

    def disappear_loss(self, cls_score: torch.Tensor) -> torch.Tensor:
        # 我这里直接让网络降低所有检测框中最高类别的置信度，用的是未处理的cls分数
        # disappera loss for general detection cls score
        mseloss = torch.nn.MSELoss()
        loss = mseloss(cls_score, torch.zeros(cls_score.shape).to(self.device))  #########
        # loss = 1. - torch.sum(cls_score[cls_score>=0.3]) / torch.numel(cls_score)
        # loss = 1. - torch.nn.BCELoss()(cls_score, torch.zeros(cls_score.shape).to(device))
        return loss

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'checkpoint_path': self.checkpoint_path,
            'device': self.device,
            'model_loaded': self.model is not None,
        }

    def to_device(self, device: str):
        """将模型移动到指定设备"""
        self.device = device
        if self.model:
            self.model.to(device)
        if self.detection_model:
            self.detection_model.to(device)
        if self.cfg:
            self.cfg.device = device

    def get_noise_and_loss_tpa(self, image_tensor, ori_image_tensor, init_det, iou_thre, mode):
        """自定义的YOLO模型的TPA损失计算
        mode: tpa, rpattack


        """

        # 用于TPA的自定义损失计算
        image_tensor = image_tensor / 255
        unprocessed_prediction = self.detection_model(image_tensor)

        class_tensor = unprocessed_prediction[0]  # [1,84,Features.size.sum()]
        class_tensor = class_tensor[:, 4:, :]
        max_cls_value, max_cls_index = torch.max(class_tensor, dim=1)

        preds = non_max_suppression(unprocessed_prediction, conf_thres=self.conf_thr, iou_thres=self.iou_thr)
        bboxes = preds[0][:, :4]
        labels = preds[0][:, 5]
        scores = preds[0][:, 4]

        det_bboxes = get_det_result(bboxes, scores)  # 带有scores的bboxes

        if mode == 'tpa':
            if len(scores) == 0:
                loss = self.disappear_loss(max_cls_value[0])
                class_loss = loss.item()
                iou_loss = 0
            else:
                loss, class_loss, iou_loss = self.detection_loss(det_bboxes, init_det, iou_thre)
        elif mode == 'rpattack':
            loss = self.disappear_loss(max_cls_value[0])
            class_loss = loss.item()
            iou_loss = 0
        else:
            raise ValueError(f"Unsupported mode {mode} for get_yolo_noise_and_loss")

        loss.backward()
        self.detection_model.zero_grad()
        noise = ori_image_tensor.grad.data.cpu().detach().clone().squeeze(0)
        # print(noise.shape)  # [3,480,640]

        if torch.norm(noise, p=1) != 0:
            noise = (noise / torch.norm(noise, p=1)).detach().numpy().transpose(1, 2, 0)  # [480,640,3]
        else:
            noise = noise.detach().cpu().numpy().transpose(1, 2, 0)
        del loss

        return noise, det_bboxes, labels, class_loss, iou_loss


def main():
    """测试函数"""
    # 创建检测器实例
    detector = YOLODetector(model_name='yolov3')

    # 打印模型信息
    print("Model Info:", detector.get_model_info())

    # 测试单张图像检测
    img_path = 'dataset/vehicle_images_5/images/000000000471.jpg'
    result = detector.detect(img_path)
    print(f"检测到 {len(result['bboxes'])} 个目标")

    # 测试梯度获取（需要数据集路径）
    # dataset_path = 'dataset/vehicle_images_5/images'
    # if os.path.exists(dataset_path):
    #     gradients = detector.get_gradients(dataset_path)
    #     print(f"Generated {len(gradients)} gradients")


if __name__ == '__main__':
    main()