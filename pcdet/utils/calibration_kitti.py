import numpy as np

"""
============================================================
P0, P1, P2, P3, RO_Rect, Tr_velo_to_cam, Tr_imu_to_velo	
============================================================

0, 1, 2, 3 代表相机编号:   3×4的相机投影矩阵
0~3分别对应左侧灰度相机、右侧灰度相机、左侧彩色相机、右侧彩色相机

P0, P1, P2, P3 分别代表对应的相机内参矩阵, 大小为 3x4：
         |f_u   0   c_u  -f_u*b_x|
P_rect = | 0   fv   c_v     0    |
         |0     0   1       0    |

fu 和 fv 是指相机的焦距;cu 和 cv 是指主点偏移： 相机的主轴是与图像平面垂直且穿过真空的线，
它与图像平面的焦点称为主点。 主点偏移就是主点位置相对于图像平面(投影面)的位置。
上图中，增加x0的值相当于把针孔向右移动，等价将投影面向左移动同时保持针孔位置不变。 
bi 是指第i个摄像头到0号摄像头的距离偏移（x方向）。



R0_rect 为0号相机的修正矩阵.
Tr_velo_to_cam 为velodyne到camera的矩阵 大小为3x4，包含了旋转矩阵 R 和 平移向量 t.

         |r11   r12   r13  | t1 |
[R|t] =  |r21   r22   r23  | t2 |
         |r31   r32   r33  | t3 |


Tr_imu_to_vel 为imu到camera的矩阵 大小为3x4，包含了旋转矩阵 R 和 平移向量 t.
"""

"""
[fu, 0, cu, tx   其中tx和ty是相机CAM0像素坐标系在其他相机像素坐标系下的像素坐标值即P_02
 0 , fv,cv, ty
 0 , 0 , 0 , 1]   
如果有点云为P=[1,3],Ph表示其齐次坐标.
则p_cam= (ph·(R0·P2).T)=[1,4]·[4,3]=[1,3] 这样计算可避免4x4矩阵中0做的额外运算

R0_rect是3x3矩阵,表示CAM0的旋转矫正矩阵
Tr_velo_to_cam是3x4矩阵,表示lidar到CAM0的坐标变换也就是激光雷达和相机的外参.
"""


def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


class Calibration(object):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """  camera coordinate(in meters) to lidar coordinate
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1
        # 激光雷达坐标系中的一点Pv映射到相机坐标系下的一点Pc的转换公式为 P_c = R0_rect_ext * Tr_velo_to_cam * P_v(1)
        # 现在我们已知相机坐标系下的点Pv，反过来求Pc，变成了一个矩阵求逆的过程。 np.linalg.inv 矩阵求逆
        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner
