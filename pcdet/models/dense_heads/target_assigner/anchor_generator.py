import torch


class AnchorGenerator(object):
    def __init__(self, anchor_range, anchor_generator_config):
        super().__init__()
        self.anchor_generator_cfg = anchor_generator_config  # list:3
        # 得到anchor在点云中的分布范围[0, -39.68, -3, 69.12, 39.68, 1]
        self.anchor_range = anchor_range
        # 得到配置参数中所有尺度anchor的长宽高
        # list:3 --> 车、人、自行车[[[3.9, 1.6, 1.56]],[[0.8, 0.6, 1.73]],[[1.76, 0.6, 1.73]]]
        self.anchor_sizes = [config['anchor_sizes'] for config in anchor_generator_config]
        # 得到anchor的旋转角度，这是是弧度，也就是0度和90度
        # list:3 --> [[0, 1.57],[0, 1.57],[0, 1.57]]
        self.anchor_rotations = [config['anchor_rotations'] for config in anchor_generator_config]
        # 得到每个anchor初始化在点云中z轴的位置，其中在kitti中点云的z轴范围是-3米到1米
        # list:3 -->  [[-1.78],[-0.6],[-0.6]]
        self.anchor_heights = [config['anchor_bottom_heights'] for config in anchor_generator_config]
        # 每个先验框产生的时候是否需要在每个格子的中间，
        # 例如坐标点为[1,1]，如果需要对齐中心点的话，需要加上0.5变成[1.5, 1.5]
        # 默认为False
        # list:3 --> [False, False, False]
        self.align_center = [config.get('align_center', False) for config in anchor_generator_config]

        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(self.anchor_heights)
        self.num_of_anchor_sets = len(self.anchor_sizes)  # 3

    def generate_anchors(self, grid_sizes):
        assert len(grid_sizes) == self.num_of_anchor_sets
        # 1.初始化
        all_anchors = []
        num_anchors_per_location = []
        # 2.三个类别的先验框逐类别生成
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights, self.align_center):
            # 2 = 2x1x1 --> 每个位置产生2个anchor，这里的2代表两个方向
            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            # 　不需要对齐中心点来生成先验框
            if align_center:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                # 中心对齐，平移半个网格
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                # 2.1计算每个网格的在点云空间中的实际大小
                # 用于将每个anchor映射回实际点云中的大小
                # (69.12 - 0) / (216 - 1) = 0.3214883848678234  单位：米
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)
                # (39.68 - (-39.68.)) / (248 - 1) = 0.3212955490297634  单位：米
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)
                # 由于没有进行中心对齐，所有每个点相对于左上角坐标的偏移量都是0
                x_offset, y_offset = 0, 0

            # 2.2 生成单个维度x_shifts，y_shifts和z_shifts
            # 以x_stride为step，在self.anchor_range[0] + x_offset和self.anchor_range[3] + 1e-5，
            # 产生x坐标 --> 216个点 [0, 69.12]
            x_shifts = torch.arange(
                self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            # 产生y坐标 --> 248个点 [0, 79.36]
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            """
            new_tensor函数可以返回一个新的张量数据，该张量数据与指定的有相同的属性
            如拥有相同的数据类型和张量所在的设备情况等属性；
            并使用anchor_height数值个来填充这个张量
            """
            # [-1.78]
            z_shifts = x_shifts.new_tensor(anchor_height)
            # num_anchor_size = 1
            # num_anchor_rotation = 2
            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()  # 1, 2
            #  [0, 1.57] 弧度制
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            # [[3.9, 1.6, 1.56]]
            anchor_size = x_shifts.new_tensor(anchor_size)

            # 2.3 调用meshgrid生成网格坐标
            x_shifts, y_shifts, z_shifts = torch.meshgrid([
                x_shifts, y_shifts, z_shifts
            ])
            # meshgrid可以理解为在原来的维度上进行扩展,例如:
            # x原来为（216，）-->（216，1, 1）--> (216,248,1)
            # y原来为（248，）--> (1，248，1）--> (216,248,1)
            # z原来为 (1, )  --> (1,1,1)    --> (216,248,1)

            # 2.4.anchor各个维度堆叠组合，生成最终anchor(1,432,496,1,2,7）
            # 2.4.1.堆叠anchor的位置 
            # [x, y, z, 3]-->[216, 248, 1, 3] 代表了每个anchor的位置信息
            # 其中3为该点所在映射tensor中的（z, y, x）数值
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  
            # 2.4.2.将anchor的位置和大小进行组合，编程为将anchor扩展并复制为相同维度（除了最后一维），然后进行组合
            # (216, 248, 1, 3) --> (216, 248, 1 , 1, 3)
            # 维度分别代表了： z，y，x， 该类别anchor的尺度数量，该个anchor的位置信息
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)
            # (1, 1, 1, 1, 3) --> (216, 248, 1, 1, 3)
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            # anchors生成的最终结果需要有位置信息和大小信息 --> (216, 248, 1, 1, 6)
            # 最后一个纬度中表示（z, y, x, l, w, h）
            anchors = torch.cat((anchors, anchor_size), dim=-1)
            # 2.4.3.将anchor的位置和大小和旋转角进行组合
            # 在倒数第二个维度上增加一个维度，然后复制该维度一次
            # (216, 248, 1, 1, 2, 6)        长， 宽， 深， anchor尺度数量， 该尺度旋转角个数，anchor的6个参数
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)
            # (216, 248, 1, 1, 2, 1)        两个不同方向先验框的旋转角度
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat(
                [*anchors.shape[0:3], num_anchor_size, 1, 1])
            # [z, y, x, num_size, num_rot, 7] --> (216, 248, 1, 1, 2, 7)
            # 最后一个纬度表示为anchors的位置+大小+旋转角度（z, y, x, l, w, h, theta）
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)  # [z, y, x, num_size, num_rot, 7]

            # 2.5 置换anchor的维度
            # [z, y, x, num_anchor_size, num_rot, 7]-->[x, y, z, num_anchor_zie, num_rot, 7]
            # 最后一个纬度代表了 : [x, y, z, dx, dy, dz, rot]
            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
            # 使得各类anchor的z轴方向从anchor的底部移动到该anchor的中心点位置
            # 车 ： -1.78 + 1.56/2 = -1.0
            # 人、自行车 ： -0.6 + 1.73/2 = 0.23
            anchors[..., 2] += anchors[..., 5] / 2
            all_anchors.append(anchors)
        # all_anchors： [(1，248，216，1，2，7），(1，248，216，1，2，7），(1，248，216，1，2，7）]
        # num_anchors_per_location：[2,2,2]
        return all_anchors, num_anchors_per_location


if __name__ == '__main__':
    from easydict import EasyDict

    config = [
        EasyDict({
            'anchor_sizes': [[2.1, 4.7, 1.7], [0.86, 0.91, 1.73], [0.84, 1.78, 1.78]],
            'anchor_rotations': [0, 1.57],
            'anchor_heights': [0, 0.5]
        })
    ]

    A = AnchorGenerator(
        anchor_range=[-75.2, -75.2, -2, 75.2, 75.2, 4],
        anchor_generator_config=config
    )
    import pdb

    pdb.set_trace()
    A.generate_anchors([[188, 188]])
