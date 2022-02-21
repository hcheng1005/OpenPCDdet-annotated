import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    """
       对应到论文中就是stacked pillars，将生成的pillar按照坐标索引还原到原空间中
    """

    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES  # 64
        self.nx, self.ny, self.nz = grid_size  # [432,496,1]
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        """
       Args:
           pillar_features:(M,64)
           coords:(M, 4) 第一维是batch_index 其余维度为xyz
       Returns:
           batch_spatial_features:(batch_size, 64, 496, 432)
       """
        # 拿到经过前面pointnet处理过后的pillar数据和每个pillar所在点云中的坐标位置
        # pillar_features 维度 （M， 64）
        # coords 维度 （M， 4）
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

        # 将转换成为伪图像的数据存在到该列表中
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1

        # batch中的每个数据独立处理
        for batch_idx in range(batch_size):
            # 创建一个空间坐标所有用来接受pillar中的数据
            # self.num_bev_features是64
            # self.nz * self.nx * self.ny是生成的空间坐标索引 [496, 432, 1]的乘积
            # spatial_feature 维度 (64,214272)
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)  # (64,214272)-->1x432x496=214272

            # 从coords[:, 0]取出该batch_idx的数据mask
            batch_mask = coords[:, 0] == batch_idx
            # 根据mask提取坐标
            this_coords = coords[batch_mask, :]
            # this_coords中存储的坐标是z,y和x的形式,且只有一层，因此计算索引的方式如下
            # 平铺后需要计算前面有多少个pillar 一直到当前pillar的索引
            """
            因为前面是将所有数据flatten成一维的了，相当于一个图片宽高为[496, 432]的图片
            被flatten成一维的图片数据了，变成了496*432=214272;
            而this_coords中存储的是平面（不需要考虑Z轴）中一个点的信息，所以要
            将这个点的位置放回被flatten的一位数据时，需要计算在该点之前所有行的点总和加上
            该点所在的列即可
            """
            # 这里得到所有非空pillar在伪图像的对应索引位置
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            # 转换数据类型
            indices = indices.type(torch.long)
            # 根据mask提取pillar_features
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            # 在索引位置填充pillars
            spatial_feature[:, indices] = pillars
            # 将空间特征加入list,每个元素为(64, 214272)
            batch_spatial_features.append(spatial_feature)

        # 在第0个维度将所有的数据堆叠在一起
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        # reshape回原空间(伪图像)    （4, 64, 214272）--> (4, 64, 496, 432)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny,
                                                             self.nx)
        # 将结果加入batch_dict
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
