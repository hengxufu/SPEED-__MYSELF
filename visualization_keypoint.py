import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from src.utils.utils    import set_all_seeds, setup_logger, load_tango_3d_keypoints, load_camera_intrinsics, project_keypoints
import os.path as osp
import os
import numpy as np
from scipy.io import loadmat
import torch
from PIL import Image
from src.utils.utils import load_tango_3d_keypoints, load_camera_intrinsics
import torchvision.transforms.functional as T
import matplotlib.pyplot as plt
import scipy
import cv2
import csv
print("czl")
def _to_numpy_image(image):
    return image.mul(255).clamp(0,255).permute(1,2,0).byte().cpu().numpy()
def scatter_keypoints(image, x_pr, y_pr, normalized=False):
    ''' Show image with keypoints
    Arguments:
        image: (3,H,W) torch.tensor image
        x_pr:  (11,) numpy.ndarray
        y_pr:  (11,) numpy.ndarray
        normalized: True if keypoints are normalized w.r.t. image size
                    False if keypoints are in pixels
    '''
    import matplotlib
    matplotlib.use('TkAgg')   # 或者 'Qt5Agg'
    import matplotlib.pyplot as plt
    _, h, w = image.shape
    data  = _to_numpy_image(image)

    if normalized:
        x_pr = x_pr * w
        y_pr = y_pr * h

    # figure
    fig = plt.figure()
    plt.imshow(data)
    #plt.scatter(x_pr , y_pr, c='lime', marker='+')
    plt.scatter(
    x_pr, y_pr,
    c='#EE82EE',         # 或者 color='red'，支持 HTML/CSS 颜色名、十六进制码、RGB 元组等
    marker='+',      # 点的形状：'o' 圆点，'+' 加号，'x' 叉号，'s' 方块，'*' 星号，'^' 三角，等等
)
    plt.show()
def quat2dcm(q):
    """ Computing direction cosine matrix from quaternion, adapted from PyNav. 
    Arguments:
        q: (4,) numpy.ndarray - unit quaternion (scalar-first)
    Returns:
        dcm: (3,3) numpy.ndarray - corresponding DCM
    """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm
t_pr = np.array([-0.129896,
            0.069519,
            6.457073])	
q_pr = np.array([-0.336458,
-0.093409,
-0.8286,
0.437599])



KEYPOINTS_3D_FILE = 'tangoPoints.mat'
# keypts3d = np.matrix(load_tango_3d_keypoints(KEYPOINTS_3D_FILE))
# print(keypts3d)
keypts3d = np.array(load_tango_3d_keypoints(KEYPOINTS_3D_FILE))
cameraMatrix, distCoeffs = load_camera_intrinsics(
                osp.join('camera.json'))


points2D = project_keypoints(q_pr, t_pr,  cameraMatrix, distCoeffs, keypts3d)
x_pr = points2D[0]
       
y_pr = points2D[1]

image_path = r'D:\进阶项目\CNN\speedplusv2\synthetic\images\img000001.jpg'  # 替换为实际的图像路径
        
image = Image.open(image_path).convert('RGB')

image = T.to_tensor(image).type(torch.float32)  # 转换为torch.Tensor

scatter_keypoints(image, x_pr, y_pr, normalized=False)
