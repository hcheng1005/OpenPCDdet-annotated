import torch.nn as nn


class HeightCompression(nn.Module):  # 在高度方向上进行压缩
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()  # 转化为密集tensor,[bacth_size, 128, 2, 200, 176] # 结合batch，spatial_shape、indice和feature将特征还原的对应位置
        N, C, D, H, W = spatial_features.shape  # batch_size，128，2，200，176
        spatial_features = spatial_features.view(N, C * D, H,
                                                 W)  # reshape为2D鸟瞰特征    将两个深度方向内的voxel特征拼接成一个 shape : (batch_size, 256, 200, 176)
        batch_dict['spatial_features'] = spatial_features  # 将特征和采样尺度加入batch_dict
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']  # 8
        return batch_dict
