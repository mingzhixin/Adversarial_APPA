_base_ = './faster-rcnn_r50_fpn_1x_dior.py'

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=4,          # 每n个epoch保存一次
        save_last=True,      # 始终保存 last.pth
        save_best='auto'     # 保存最优模型
    )
)

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=None),
    roi_head=dict(
            bbox_head=dict(num_classes=20)
        )
)
