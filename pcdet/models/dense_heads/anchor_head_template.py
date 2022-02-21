import numpy as np
import torch
import torch.nn as nn

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner


class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        """
        AnchorHead模板
        Args:
            model_cfg: AnchorHeadSingle的配置
            num_class: 3
            class_names: ['Car','Pedestrian','Cyclist']
            grid_size: (432,493,1)
            point_cloud_range:(0, -39.68, -3, 69.12, 39.68, 1)
            predict_boxes_when_training:False
        """
        super().__init__()
        self.model_cfg = model_cfg  # AnchorHeadSingle
        self.num_class = num_class  # 3
        self.class_names = class_names  # ['Car','Pedestrian','Cyclist']
        self.predict_boxes_when_training = predict_boxes_when_training  # False
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)  # False
        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # 对生成的anchor和gt进行编码和解码
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )
        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG  # anchor生成的配置文件
        # 针对不同类别生成不同的anchor和每个位置生成anchor的数量
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        # 将anchor放到GPU上
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)
        # 前向传播结果字典初始化
        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        # 初始化AnchorGenerator类
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        # config['feature_map_stride'] == 8
        # e.g. size [[176,200],[176,200],[176,200]]
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in
                            anchor_generator_cfg]  # [[216,248],[216,248],[216,248]]
        # 计算所有3个类别的anchor和每个位置上的anchor数量
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        # 如果anchor的维度不等于7，则补0
        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors
        # list:3 [(1，248，216，1，2，7），(1，248，216，1，2，7），(1，248，216，1，2，7)], [2,2,2]
        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        # target_assigner初始化
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        # 添加loss模块，包括分类损失，回归损失和方向损失并初始化
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE  # reg_loss_name:WeightedSmoothL1Loss
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:
            all_targets_dict = {
                'box_cls_labels': cls_labels, # (4，321408）
                'box_reg_targets': bbox_targets, # (4，321408，7）
                'reg_weights': reg_weights # (4，321408）
            }
        """
        # anchors-->list:3 [(1，248，216，1，2，7），(1，248，216，1，2，7），(1，248，216，1，2，7)]
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def get_cls_layer_loss(self):
        # (batch_size, 248, 216, 18) 网络类别预测
        cls_preds = self.forward_ret_dict['cls_preds']
        # (batch_size, 321408) 前景anchor类别
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        # [batch_szie, num_anchors]--> (batch_size, 321408)
        # 关心的anchor  选取出前景背景anchor, 在0.45到0.6之间的设置为仍然是-1，不参与loss计算
        cared = box_cls_labels >= 0
        # (batch_size, 321408) 前景anchor
        positives = box_cls_labels > 0
        # (batch_size, 321408) 背景anchor
        negatives = box_cls_labels == 0
        # 背景anchor赋予权重
        negative_cls_weights = negatives * 1.0
        # 将每个anchor分类的损失权重都设置为1
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        # 每个正样本anchor的回归损失权重，设置为1
        reg_weights = positives.float()
        # 如果只有一类
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        # 正则化并计算权重     求出每个数据中有多少个正例，即shape=（batch， 1）
        pos_normalizer = positives.sum(1, keepdim=True).float()  # (4,1) 所有正例的和 eg:[[162.],[166.],[155.],[108.]]
        # 正则化回归损失-->(batch_size, 321408)，最小值为1,根据论文中所述，用正样本数量来正则化回归损失
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        # 正则化分类损失-->(batch_size, 321408)，根据论文中所述，用正样本数量来正则化分类损失
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        # care包含了背景和前景的anchor，但是这里只需要得到前景部分的类别即可不关注-1和0
        # cared.type_as(box_cls_labels) 将cared中为False的那部分不需要计算loss的anchor变成了0
        # 对应位置相乘后，所有背景和iou介于match_threshold和unmatch_threshold之间的anchor都设置为0
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        # 在最后一个维度扩展一次
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )  # (batch_size, 321408, 4)，这里的类别数+1是考虑背景

        # target.scatter(dim, index, src)
        # scatter_函数的一个典型应用就是在分类问题中，
        # 将目标标签转换为one-hot编码形式 https://blog.csdn.net/guofei_fly/article/details/104308528
        # 这里表示在最后一个维度，将cls_targets.unsqueeze(dim=-1)所索引的位置设置为1

        """
            dim=1: 表示按照列进行填充
            index=batch_data.label:表示把batch_data.label里面的元素值作为下标，
            去下标对应位置(这里的"对应位置"解释为列，如果dim=0，那就解释为行)进行填充
            src=1:表示填充的元素值为1
        """
        # (batch_size, 321408, 4)
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        # (batch_size, 248, 216, 18) --> (batch_size, 321408, 3)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        # (batch_size, 321408, 3) 不计算背景分类损失
        one_hot_targets = one_hot_targets[..., 1:]

        # 计算分类损失 # [N, M] # (batch_size, 321408, 3)
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)
        # 求和并除以batch数目
        cls_loss = cls_loss_src.sum() / batch_size
        # loss乘以分类权重 --> cls_weight=1.0
        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        # 针对角度添加sin损失，有效防止-pi和pi方向相反时损失过大
        assert dim != -1  # 角度=180°×弧度÷π   弧度=角度×π÷180°
        # (batch_size, 321408, 1)  torch.sin() - torch.cos() 的 input (Tensor) 都是弧度制数据，不是角度制数据。
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        # (batch_size, 321408, 1)
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        # (batch_size, 321408, 7) 旋转角在前
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        # (batch_size, 321408, 7) 旋转角在前
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        # (batch_size, 321408, 7)
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        # (batch_size, 321408)角度是直接相加的关系，在-pi到pi之间
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        # (batch_size, 321408)将角度限制在0到2*pi之间
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        # (batch_size, 321408) 取值为0和1，num_bins=2
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        # (batch_size, 321408)
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            # (batch_size, 321408, 2)
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            # one-hot编码，只存在两个方向:正向和反向 (batch_size, 321408, 2)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        # (batch_size, 248, 216, 42） anchor_box的7个回归参数
        box_preds = self.forward_ret_dict['box_preds']
        # (batch_size, 248, 216, 12） anchor_box的方向预测
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        # (batch_size, 321408, 7) 每个anchor和GT编码的结果
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        # (batch_size, 321408)
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])
        # 获取所有anchor中属于前景anchor的mask  shape : (batch_size, 321408)
        positives = box_cls_labels > 0
        # 设置回归参数为1.    [True, False] * 1. = [1., 0.]
        reg_weights = positives.float()  # (4, 211200) 只保留标签>0的值
        # 同cls处理
        pos_normalizer = positives.sum(1,
                                       keepdim=True).float()  # (batch_size, 1) 所有正例的和 eg:[[162.],[166.],[155.],[108.]]

        reg_weights /= torch.clamp(pos_normalizer, min=1.0)  # (batch_size, 321408)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)  # (1, 248, 216, 3, 2, 7)
        else:
            anchors = self.anchors
        # (1, 248*216, 7） --> (batch_size, 248*216, 7）
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        # (batch_size, 248*216, 7）
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        # (batch_size, 321408, 7)
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']  # loc_weight = 2.0 损失乘以回归权重
        box_loss = loc_loss
        tb_dict = {
            # pytorch中的item()方法，返回张量中的元素值，与python中针对dict的item方法不同
            'rpn_loss_loc': loc_loss.item()
        }

        # 如果存在方向预测，则添加方向损失
        if box_dir_cls_preds is not None:
            # (batch_size, 321408, 2)
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,  # 方向偏移量 0.78539 = π/4
                num_bins=self.model_cfg.NUM_DIR_BINS  # BINS的方向数 = 2
            )
            # 方向预测值 (batch_size, 321408, 2)
            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            # 只要正样本的方向预测值 (batch_size, 321408)
            weights = positives.type_as(dir_logits)
            # (4, 211200) 除正例数量，使得每个样本的损失与样本中目标的数量无关
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            # 方向损失计算
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            # 损失权重，dir_weight: 0.2
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            # 将方向损失加入box损失
            box_loss += dir_loss

            tb_dict['rpn_loss_dir'] = dir_loss.item()
        return box_loss, tb_dict

    def get_loss(self):
        # 计算classfiction layer的loss，tb_dict内容和cls_loss相同，形式不同，一个是torch.tensor一个是字典值
        cls_loss, tb_dict = self.get_cls_layer_loss()
        # 计算regression layer的loss
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        # 在tb_dict中添加tb_dict_box，在python的字典中添加值，如果添加的也是字典，用update方法，如果是键值对则采用赋值的方式
        tb_dict.update(tb_dict_box)
        # rpn_loss是分类和回归的总损失
        rpn_loss = cls_loss + box_loss

        # 在tb_dict中添加rpn_loss，此时tb_dict中包含cls_loss,reg_loss和rpn_loss
        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors  # (1, 248, 216, 3, 2, 7])
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]  # (211200,)
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)  # (1, 211200, 7)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds,
                              list) else cls_preds  # cls_preds:（1, 200, 176, 18）--> batch_cls_preds:(1, 211200, 3)
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)  # (1, 200, 176, 42) --> (1, 211200, 7)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET  # 0.78539
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET  # 0
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)  # (1, 321408, 2)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]  # (1, 321408) --> 这里是一个分类:正向和反向

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)  # pi
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )  # 限制在0到pi之间
            # 转化0.25pi到2.5pi
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
