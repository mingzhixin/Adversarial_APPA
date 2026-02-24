import time

import cv2
import torch
import numpy as np
from typing import Tuple, Dict, Optional, Union, Any, List
import os
import torch.nn.functional as F
from abc import ABC, abstractmethod

from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import InstanceData, PixelData
from mmdet.structures.bbox import BaseBoxes

from .utils.model_util import get_det_result, tensor_bbox_iou

# 注册所有模块
register_all_modules()


class PackDetInputsTensor(BaseTransform):
    """
    把输入打包成MMDetection的DetDataSample格式，img为Tensor类型
    Pack the inputs data for the detection / semantic segmentation /
    panoptic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction')``
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks'
    }

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
                img = to_tensor(img).permute(2, 0, 1).contiguous()  # HWC2CHW
            packed_results['inputs'] = img

        if 'instances' in results:
            # 这里对ignore进行处理
            gt_ignore_flags = []
            for instance in results['instances']:
                gt_ignore_flags.append(instance['ignore_flag'])
            results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=np.int64)

            if 'gt_ignore_flags' in results:
                valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
                ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

            # LoadAnnotation的功能,把instances的东西转成gt_bboxes和gt_bboxes_labels
            gt_bboxes = []
            for instance in results['instances']:
                gt_bboxes.append(instance['bbox'].cpu().numpy())
            results['gt_bboxes'] = np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4)
            gt_bboxes_labels = []
            for instance in results['instances']:
                gt_bboxes_labels.append(instance['bbox_label'].cpu().numpy())
            results['gt_bboxes_labels'] = np.array(gt_bboxes_labels, dtype=np.int64)

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            gt_sem_seg_data = PixelData(**gt_sem_seg_data)
            if 'ignore_index' in results:
                metainfo = dict(ignore_index=results['ignore_index'])
                gt_sem_seg_data.set_metainfo(metainfo)
            data_sample.gt_sem_seg = gt_sem_seg_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


class BaseMMDetectionDetector(ABC):
    """
    MMDetection检测器基类，提供通用的检测功能
    所有具体的检测器类都应该继承这个基类

    注意：这是一个抽象基类，派生类必须实现以下抽象方法：
    - preprocess_img: 图像预处理方法
    - postprocess_img: 图像后处理方法
    """

    def __init__(self,
                model_name: str,
                device: str = None,
                config_path: str = None,
                checkpoint_path: str = None,
                input_size: Tuple[int, int] = (800, 800),
                score_thr: float = 0.05):
        """
        初始化MMDetection检测器基类

        Args:
            model_name: 模型名称，用于标识具体的检测器类型
            device: 设备 ('cuda:0', 'cpu'等)，如果为None则自动选择
            config_path: MMDetection配置文件路径，如果为None则使用默认路径
            checkpoint_path: 模型权重路径，如果为None则使用默认路径  
            input_size: 模型输入尺寸 (height, width)
            score_thr: 置信度阈值，用于过滤检测结果，默认0.05
        """
        self.model_name = model_name
        self.device = device if device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.score_thr = score_thr  # 添加置信度阈值属性

        # 设置默认路径
        if config_path is None:
            config_path = self._get_default_config_path()
        if checkpoint_path is None:
            checkpoint_path = self._get_default_checkpoint_path()

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.cfg = None

        # 初始化模型
        self._load_model()

    @abstractmethod
    def preprocess_img(self, img_np):
        """
        图像预处理方法 - 抽象方法，派生类必须实现

        Args:
            img_np: 输入图像numpy数组，形状为 [H, W, 3]

        Returns:
            预处理后的图像numpy数组
        """
        pass

    @abstractmethod
    def postprocess_img(self, img_input, original_shape: Tuple[int, int] = None):
        """
        图像后处理方法 - 抽象方法，派生类必须实现

        Args:
            img_input: 输入图像（可能是numpy数组或张量）
            original_shape: 原始图像尺寸 (H, W)，可选参数

        Returns:
            后处理后的图像
        """
        pass

    def _resize_image_tensor(self, img_tensor, target_size) -> torch.Tensor:
        """
        图像缩放函数


        使用动态输入尺寸，支持上采样和下采样，不保持宽高比

        Args:
            img_tensor: 输入图像张量，格式：
                - torch.Tensor: [1, C, H, W]

        Returns:
            torch.Tensor: 预处理后的图像张量，形状为 [1, 3, H, W]
        """
        # 确保输入是4维张量 [B, C, H, W]
        if img_tensor.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {img_tensor.ndim}")
        if img_tensor.shape[0] != 1:
            raise ValueError(f"Expected 4D tensor with B=1, got {img_tensor.shape[0]}")

        # 使用动态输入尺寸
        current_h, current_w = img_tensor.shape[-2:]

        # 调整尺寸到目标尺寸
        if (current_h, current_w) != target_size:
            img_tensor = F.interpolate(img_tensor, size=target_size,
                                       mode='bilinear', align_corners=False)

        return img_tensor  # [1, 3, H, W]

    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径，子类需要重写此方法"""
        raise NotImplementedError("子类必须实现此方法")

    def _get_default_checkpoint_path(self) -> str:
        """获取默认权重路径，子类需要重写此方法"""
        raise NotImplementedError("子类必须实现此方法")

    def _load_model(self):
        """加载MMDetection模型"""
        try:
            # PyTorch 2.6+ 兼容性修复
            # 通过monkey patch torch.load来解决权重加载问题
            if not hasattr(self, '_torch_compatibility_fixed'):
                import torch
                original_torch_load = torch.load

                def patched_torch_load(f, *args, **kwargs):
                    # 确保weights_only=False以兼容旧版本权重文件
                    kwargs['weights_only'] = False
                    return original_torch_load(f, *args, **kwargs)

                torch.load = patched_torch_load
                self._torch_compatibility_fixed = True

            self.cfg = Config.fromfile(self.config_path)
            self.model = init_detector(
                config=self.config_path,
                checkpoint=self.checkpoint_path,
                device=self.device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name} model: {e}")

    def detect(self, source: Union[str, np.ndarray, torch.Tensor], score_thr: Optional[float] = None) -> object:
        """
        统一的检测函数，支持多种输入格式和阈值过滤

        这个函数使用MMDetection模型的高级接口，自动处理不同输入格式的预处理和后处理，
        返回统一的格式化检测结果。支持图像路径、numpy数组和PyTorch张量三种输入格式。

        Args:
            source: 输入源，支持以下格式：
                - str: 图像文件路径，如 "path/to/image.jpg"
                - np.ndarray: numpy数组格式的图像，形状为 [H, W, 3] 或 [3, H, W]，BGR格式
                - torch.Tensor: PyTorch张量格式的图像，形状为 [B, C, H, W] 或 [C, H, W]
            score_thr: 置信度阈值，用于过滤检测结果，如果为None则使用实例的默认阈值

        Returns:
            object: MMDetection检测结果对象，包含以下属性：
                - pred_instances: 检测到的实例信息
                    - bboxes: 边界框坐标 [x1, y1, x2, y2]
                    - scores: 置信度分数
                    - labels: 类别标签
                - metainfo: 元信息

        Raises:
            FileNotFoundError: 当输入是路径但文件不存在时
            ValueError: 当输入格式不支持或图像格式错误时
            TypeError: 当输入类型不是支持的三种类型时
        """
        # 使用指定的阈值或默认阈值
        threshold = score_thr if score_thr is not None else self.score_thr

        # 输入类型检查和预处理
        if isinstance(source, str):
            # 图像路径输入
            if not os.path.exists(source):
                raise FileNotFoundError(f"Image path not found: {source}")
            # 直接使用MMDetection模型处理路径，它会自动加载和预处理图像
            result = inference_detector(self.model, source)

        elif isinstance(source, np.ndarray):
            # numpy数组输入
            # 检查数组维度和格式
            if source.ndim != 3:
                raise ValueError(f"Expected 3D numpy array [H, W, 3] or [3, H, W], got shape {source.shape}")

            # 如果通道维度在前 [3, H, W]，转换为 [H, W, 3]
            if source.shape[0] == 3:
                source = source.transpose(1, 2, 0)

            # 检查转换后的形状
            if source.shape[2] != 3:
                raise ValueError(f"Expected 3 channels (BGR), got {source.shape[2]} channels")

            # 使用MMDetection模型处理numpy数组
            result = inference_detector(self.model, source)

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
                raise ValueError(f"Expected 3 channels (BGR), got {source.shape[1]} channels")

            # 转换为numpy数组进行处理
            source_np = source.squeeze(0).permute(1, 2, 0).cpu().numpy()
            result = inference_detector(self.model, source_np)

        else:
            # 不支持的输入类型
            raise TypeError(f"Unsupported input type: {type(source)}. "
                            f"Expected str (path), np.ndarray, or torch.Tensor")

        # 应用阈值过滤
        if hasattr(result, 'pred_instances') and len(result.pred_instances) > 0:
            # 使用InstanceData的索引功能来过滤结果
            # 这样可以确保所有字段的长度保持一致
            filtered_result = result.pred_instances[result.pred_instances.scores >= threshold]
            result.pred_instances = filtered_result

        # 从result对象中提取数据
        detections = {
            'bboxes': result.pred_instances.bboxes.detach().cpu().numpy(),
            'scores': result.pred_instances.scores.detach().cpu().numpy(),
            'labels': result.pred_instances.labels.detach().cpu().numpy()
        }

        return detections

    
    def get_predictions(self, image_tensor, original_shape: Optional[Tuple[int, int]] = None, score_thres: float = None) -> \
    List[Any]:
        """
        获取可导的预测结果，用于对抗攻击训练（支持批量）

        Args:
            image_tensor: torch.Tensor RGB（必须）
                - [C, H, W] or [B, C, H, W], 值域建议为[0,1]（会乘255送入模型）
            original_shape: 兼容旧接口，当前实现按 self.input_size 作为 img_shape/ori_shape

        Returns:
            若输入为单张：[dict]，包含 bboxes/scores/labels (torch.Tensor, 保留梯度)
            若输入为批量：[List[dict]]，每个元素对应一张图
        """
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError(f"image_tensor must be torch.Tensor, got {type(image_tensor)}")

        if score_thres is not None:
            if not isinstance(score_thres, (int, float)):
                raise TypeError(f"score_thres must be float or None, got {type(score_thres)}")
            score_thres = float(score_thres)

        # 统一到 [B, C, H, W]
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim != 4:
            raise ValueError(f"Expected [C,H,W] or [B,C,H,W], got shape {tuple(image_tensor.shape)}")

        B, C, H, W = image_tensor.shape
        if C != 3:
            raise ValueError(f"Expected 3 channels, got {C}")

        img_batch = image_tensor * 255  # MMDet 内部通常按 0-255 处理
        img_batch = torch.flip(img_batch, dims=[1])  # MMDetection内部以BGR处理

        # 构建 pipeline（只包含 PackDetInputsTensor）
        pipeline = Compose([PackDetInputsTensor()])

        inputs_list = []
        data_samples_list = []

        for i in range(B):
            img = img_batch[i]  # [C,H,W]
            data = {
                'img': img,
                'img_id': i,
                'img_path': None,
                'img_shape': self.input_size,
                'ori_shape': self.input_size,
                'scale_factor': (1.0, 1.0)
            }
            data_ = pipeline(data)

            # attack_*_step 期望 batch 输入：list[tensor], list[data_sample]
            inputs_list.append(data_['inputs'].float())
            data_samples_list.append(data_['data_samples'])

        batch_data = {
            'inputs': inputs_list,
            'data_samples': data_samples_list
        }

        # 前向：返回 List[DetDataSample]，长度=B
        predictions = self.model.attack_result_step(batch_data)

        results = []
        for pred in predictions:
            inst = pred.pred_instances
            bboxes = inst['bboxes']  # torch.Tensor, 可反传
            scores = inst['scores']  # torch.Tensor, 可反传
            labels = inst['labels']  # torch.Tensor

            if score_thres is not None:
                # 保留梯度：用 mask 过滤张量
                keep = scores >= score_thres
                bboxes = bboxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            results.append({
                'bboxes': bboxes,
                'scores': scores,
                'labels': labels
            })

        return results

    def preprocess_to_input_tensor4d(self, img_np):
        """获取YOLO格式的图像张量

        Args:
            img_np: numpy array of the image, [H,W,3], BGR

        Returns:
            img_tensor4D: torch tensor of the image, [1,3,H,W]
        """
        assert type(img_np) == np.ndarray, f'the img type is {type(img_np)}, but ndarray expected'
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_np).float()
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC2CHW
        img_tensor4D = img_tensor.to(self.device).unsqueeze(0)
        return img_tensor4D  # [C,H,W] RGB

    def get_loss(self, image_tensor, init_det):
        """
        走MMDetection官方途径得到的loss
        Args:
            image_tensor: tensor [B,C,H,W]/[C,H,W] [0,255]
            init_det: tensor [[label,bboxes]] 形状为[N,5]

        Returns:

        """
        if image_tensor.ndim == 4:  # 如果是4维变成3维
            image_tensor = image_tensor[0]

        # 手动构建dict
        data = {
            'img': image_tensor,
            'img_id': 0,
            'img_path': None,
            'img_shape': self.input_size,
            'ori_shape': self.input_size,
            'scale_factor': (1.0, 1.0),
            'instances': [
                {'bbox': det[1:],
                 'bbox_label': det[0],
                 'ignore_flag': 0
                 }
                for det in init_det  # ground_truth需要修改，对应dpatch里的调用是patch_target
            ]
        }
        train_pipeline = [PackDetInputsTensor()]
        train_pipeline = Compose(train_pipeline)

        data_ = train_pipeline(data)
        data_['inputs'] = data_['inputs'].float()
        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]
        # print(data_)

        loss, loss_item = self.model.attack_loss_step(data_)

        return loss, loss_item

    def get_loss_fca(self, image_tensor, init_det):
        """

        Args:
            image_tensor: tensor [B,C,H,W] [0,255]
            init_det: tensor [[label,bbox],...] [N,5]

        Returns:

        """
        if image_tensor.ndim == 4:  # 如果是4维变成3维
            image_tensor = image_tensor[0]

        # 手动构建dict
        data = {
            'img': image_tensor,
            'img_id': 0,
            'img_path': None,
            'img_shape': self.input_size,
            'ori_shape': self.input_size,
            'scale_factor': (1.0, 1.0)
        }

        train_pipeline = [PackDetInputsTensor()]
        train_pipeline = Compose(train_pipeline)
        data_ = train_pipeline(data)

        data_['inputs'] = data_['inputs'].float()
        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]

        # 拿到unprocessed数据用于计算loss
        # [1000,81] [1000,320]
        class_tensor, bbox_tensor = self.model.attack_det_step(data_)  # 输出未处理的labels和scores 可反向传播

        # 拿到processed预测结果
        prediction = self.model.attack_result_step(data_)
        bboxes = prediction[0].pred_instances['bboxes']  # 可反向传播
        scores = prediction[0].pred_instances['scores']  # 可反向传播
        labels = prediction[0].pred_instances['labels']

        det_bboxes = get_det_result(bboxes, scores)  # 带有scores的bboxes

        # 计算两个loss: lcls和liou
        loss, class_loss, iou_loss = self.detection_loss_fca(det_bboxes,class_tensor,init_det)
        loss_item = {
            'cls_loss': class_loss,
            'iou_loss': iou_loss
        }

        return loss, loss_item

    def detection_loss_fca(self, det_bboxes, class_tensor, init_det):
        """

        Args:
            det_bboxes: [[bbox,score],...] [N,5]
            class_tensor: tensor [1000,81]
            init_det: tensor [[label,bbox],...] [M,5]

        Returns:

        """
        det_boxes = det_bboxes[:, :4]
        det_scores = det_bboxes[:, 4]
        gt_boxes = init_det[:, 1:]
        gt_labels = init_det[:, 0]
        mseloss = torch.nn.MSELoss()

        # iou_loss为反向优化的损失
        iou = tensor_bbox_iou(det_boxes, gt_boxes)
        iou_loss = iou.sum()

        # cls_loss为反向优化的损失
        cls_loss = torch.max(class_tensor[:,7])  # 这里写死了是计算车这一类别的cls_loss

        loss = cls_loss + iou_loss

        return loss, cls_loss.item(), iou_loss.item()


    def get_gradients(self, batch: Dict, config: object = None) -> Tuple[np.ndarray, torch.Tensor]:
        """
        获取模型梯度（DPatch格式接口）

        Args:
            batch: 包含图像信息的字典
            config: 配置对象

        Returns:
            grads_bgr: BGR格式的梯度图
            loss: 模型损失值
        """
        if config:
            self.model.args = config

        # 准备输入数据
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float()
        batch["img"] = batch["img"].requires_grad_(True)

        # 前向传播
        loss, loss_items = self.model.attack_loss_step(batch)

        # 确保loss是标量
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.sum()

        # 计算梯度
        grads = torch.autograd.grad(loss, batch["img"], retain_graph=True)[0]

        # 转换为numpy并调整通道顺序
        grads = grads.cpu().numpy()
        grads_bgr = grads[:, [2, 1, 0], :, :]  # RGB to BGR

        return grads_bgr, loss

    def get_noise_and_loss_tpa(self, image_tensor, ori_image_tensor, init_det, score_thre, mode):
        """MMDetection模型的TPA损失计算
        mode: tpa, rpattack



        """
        if image_tensor.ndim == 4:  # 如果是4维变成3维
            image_tensor = image_tensor[0]

        # 手动构建dict
        data = {
            'img': image_tensor,
            'img_id': 0,
            'img_path': None,
            'img_shape': self.input_size,
            'ori_shape': self.input_size,
            'scale_factor': (1.0, 1.0)
        }

        train_pipeline = [PackDetInputsTensor()]
        train_pipeline = Compose(train_pipeline)
        data_ = train_pipeline(data)

        data_['inputs'] = data_['inputs'].float()
        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]

        # 拿到unprocessed数据用于计算loss
        class_tensor, bbox_tensor = self.model.attack_det_step(data_)  # 输出未处理的labels和scores 可反向传播

        # 拿到processed预测结果
        prediction = self.model.attack_result_step(data_)
        bboxes = prediction[0].pred_instances['bboxes']  # 可反向传播
        scores = prediction[0].pred_instances['scores']  # 可反向传播
        labels = prediction[0].pred_instances['labels']

        det_bboxes = get_det_result(bboxes, scores)  # 带有scores的bboxes

        if mode == 'tpa':
            if len(scores) == 0:
                loss = self.disappear_loss(class_tensor[:, -1])
                class_loss = loss.item()
                iou_loss = 0
            else:
                loss, class_loss, iou_loss = self.detection_loss(det_bboxes, init_det, score_thre)
        elif mode == 'rpattack':
            if len(scores) == 0:
                loss = self.disappear_loss(class_tensor[:, -1])
                class_loss = loss.item()
                iou_loss = 0
            else:
                loss = self.disappear_loss(det_bboxes[:, -1])
                class_loss = loss.item()
                iou_loss = 0
        else:
            raise ValueError(f"Unsupported mode {mode} for get_mmdetection_noise_and_loss")

        loss.backward()
        self.model.zero_grad()
        noise = ori_image_tensor.grad.data.cpu().detach().clone().squeeze(0)

        if torch.norm(noise, p=1) != 0:
            noise = (noise / torch.norm(noise, p=1)).numpy().transpose(1, 2, 0)  # [H,W,C]
        else:
            noise = noise.detach().cpu().numpy().transpose(1, 2, 0)
        del loss

        return noise, det_bboxes, labels, class_loss, iou_loss

    def disappear_loss(self, cls_score: torch.Tensor) -> torch.Tensor:
        # chat说是为了不让攻击早早停止，所以让背景类的分数下降前景类上升之后继续优化
        # disappera loss for general detection cls score
        cls_score = cls_score.to(self.device)
        mseloss = torch.nn.MSELoss()
        # loss = mseloss(cls_score[cls_score >= 0.3], torch.zeros(cls_score[cls_score >= 0.3].shape).to(self.device))  #########
        loss = mseloss(cls_score[cls_score >= 0.3], torch.zeros_like(cls_score[cls_score >= 0.3], device=self.device))
        # loss = 1. - torch.sum(cls_score[cls_score>=0.3]) / torch.numel(cls_score)
        # loss = 1. - torch.nn.BCELoss()(cls_score, torch.zeros(cls_score.shape).to(device))
        return loss

    def detection_loss(self, det_bboxes, init_det, score_thre):  # 原 faster_loss
        cls_score = det_bboxes[:, -1].to(self.device)
        mseloss = torch.nn.MSELoss()
        if cls_score[cls_score >= score_thre].shape[0] == 0:
            class_loss = mseloss(cls_score * 0, torch.zeros_like(cls_score, device=self.device))  #########
            iou_loss = torch.zeros([1]).to(self.device)
        else:
            class_loss = mseloss(cls_score[cls_score >= score_thre],
                                 torch.zeros(cls_score[cls_score >= score_thre].shape, device=self.device))  #########
            box_pred = det_bboxes[:, 0:4]
            box_init = init_det[:, 0:4]
            pred_iou = tensor_bbox_iou(box_pred, box_init)
            iou_loss = torch.sum(pred_iou) / det_bboxes.shape[0]
        loss = class_loss + iou_loss

        return loss, class_loss.item(), iou_loss.item()

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'config_path': self.config_path,
            'checkpoint_path': self.checkpoint_path,
            'device': self.device,
            'input_size': self.input_size,
            'model_loaded': self.model is not None,
            'score_thr': self.score_thr  # 添加置信度阈值信息
        }

    def set_score_threshold(self, score_thr: float):
        """
        设置置信度阈值

        Args:
            score_thr: 新的置信度阈值
        """
        self.score_thr = score_thr
        print(f"{self.model_name}检测器置信度阈值已设置为: {score_thr}")

    def to_device(self, device: str):
        """将模型移动到指定设备"""
        self.device = device
        if self.model:
            self.model.to(device)
        if self.cfg:
            self.cfg.device = device
