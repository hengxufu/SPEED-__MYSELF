import numpy as np
import pandas as pd
import csv
import random
import math
#这个文件是UE4相机光心轴为Y情景 用来生成训练集，选用的是taes模型  四元数为左手四元数
def euler_to_quaternion(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr

    return np.array([qw, qx, qy, qz])

def euler2quatt(pitch, yaw, roll):
    """Convert euler angles in degrees to a quaternion"""

    cos_pitch = np.cos(pitch * 0.5)
    sin_pitch = np.sin(pitch * 0.5)
    cos_yaw = np.cos(yaw * 0.5)
    sin_yaw = np.sin(yaw * 0.5)
    cos_roll = np.cos(roll * 0.5)
    sin_roll = np.sin(roll * 0.5)

    qx = -sin_yaw * cos_roll * cos_pitch - cos_yaw * sin_roll * sin_pitch
    qy = -cos_yaw * sin_roll * cos_pitch + sin_yaw * cos_roll * sin_pitch
    qz = sin_yaw * sin_roll * cos_pitch - cos_yaw * cos_roll * sin_pitch
    qw = cos_yaw * cos_roll * cos_pitch + sin_yaw * sin_roll * sin_pitch

    return np.array([qx, qy, qz, qw])
def quatt2euler(quatt):
    """ Convert left-handed quaternion to euler angles (X,Y,Z) (valid)"""
    q = np.zeros((4), dtype=np.float32)
    q[0]=quatt[2]
    q[1]=quatt[0]
    q[2]=quatt[1]
    q[3]=quatt[3]
    sqx = q[0] * q[0]
    sqy = q[1] * q[1]
    sqz = q[2] * q[2]
    test = q[0]*q[2] + q[1]*q[3]
    if test > 0.499: # singularity at north pole
        pitch = 2 * np.arctan2(q[0], q[3])
        yaw = - np.pi / 2
        roll = 0
    elif test < -0.499: # singularity at south pole
        pitch = -2 * np.arctan2(q[0], q[3])
        yaw = np.pi / 2
        roll = 0
    else:
        pitch = np.arctan2(2*(q[1]*q[2] - q[0]*q[3]), 1-2*sqx-2*sqy)
        yaw = np.arcsin(-2*(q[0]*q[2]+q[1]*q[3]))
        roll = np.arctan2(2*(q[0]*q[1] - q[2]*q[3]), 1-2*sqy-2*sqz)

    # Keeps pitch between [-180, 180] under singularities
    if pitch > np.pi:
        pitch = 2*np.pi - pitch
    if pitch < -np.pi:
        pitch = 2*np.pi + pitch

    return pitch*180/np.pi, yaw*180/np.pi, roll*180/np.pi
# 定义欧拉角通道的离散级数
num_steps = 32
#num_steps = 28
# 欧拉角范围
min_angle = 0
max_angle = 2 * np.pi

# 生成所有可能的欧拉角组合
roll_values = np.linspace(min_angle, max_angle, num_steps)
pitch_values = np.linspace(min_angle, max_angle, num_steps)
yaw_values = np.linspace(min_angle, max_angle, num_steps)
print(yaw_values)
# 创建四元数集合和欧拉角列表
quaternion_set = []
pose_data = []
gt_data = []
count = 0
# 遍历姿态空间
for roll in roll_values:
    for pitch in pitch_values:
        for yaw in yaw_values:
            # 将欧拉角转换为四元数
            #quaternion = euler_to_quaternion(roll, pitch, yaw)
            quaternion = euler2quatt(-roll, -pitch, yaw)
            # 判断四元数是否与集合中的某个四元数相同
            if not any(np.allclose(np.abs(quaternion), np.abs(existing_quaternion)) for existing_quaternion in quaternion_set):
                # 将四元数添加到集合中
                quaternion_set.append(quaternion)

                # 生成姿态数据
                pose = [roll / np.pi * 180, pitch / np.pi * 180, yaw / np.pi * 180]

                # 将姿态数据和x, y, z值合并
                pose_data.append([*pose])
            count = count + 1
            print(count)
posee_data = []
indices = list(range(len(pose_data)))
print(len(indices))
random.shuffle(pose_data)
focal_length = 35  # 焦距（单位：毫米）
horizontal_fov = 90  # 水平视场角（单位：度）
vertical_fov = 73.7  # 垂直视场角（单位：度）

# 卫星参数
satellite_size = 3  # 卫星尺寸（单位：米）
depth_min = 12  # 卫星相对于相机的深度范围下限（单位：米）
depth_max = 18.5  # 卫星相对于相机的深度范围上限（单位：米）

# 生成1000个不同组合的(x, y, z)数据
data = []
for _ in range(1):
    for _ in range(5000):
    # 生成随机的深度值
        depth = random.uniform(depth_min, depth_max)

    # 计算相机视场范围
        horizontal_range = 2 * depth * math.tan(math.radians(horizontal_fov / 2)) - satellite_size
        vertical_range = 2 * depth * math.tan(math.radians(vertical_fov / 2)) - satellite_size

    # 生成随机的y和z值，使卫星位于相机视场中
        y = random.uniform(-horizontal_range / 2, horizontal_range / 2)
        z = random.uniform(-vertical_range / 2, vertical_range / 2)

    # 将(x, y, z)组合添加到数据列表中,光心为Y轴的UE4
        data.append([-y, depth, z])
#print("after shuffle", indices)
j = 0
dataa = []
for k in range(1):
    for i in range(5000):
        j = j + 1
        posee = pose_data[i]
        dataa = data[i + k * len(pose_data)]
        posee_data.append([j, 100 * dataa[0], 100 * dataa[1], 100 * dataa[2], posee[0], posee[1], posee[2]+90])
        quater = euler2quatt(-posee[0] / 180 * np.pi, -posee[1] / 180 * np.pi, posee[2] / 180 * np.pi)
        gt_data.append([-dataa[0], dataa[2], dataa[1], quater[0], quater[1], quater[2], quater[3]])
# 写入CSV文件
filename = "pose_data.csv"
header = ["Index", "x", "y", "z", "ROLL", "PITCH", "YAW"]

with open(filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(posee_data)

print("CSV文件保存成功。")
filename = "gt.csv"
header = ["x", "y", "z", "q1", "q2", "q3", "q4"]

with open(filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(gt_data)

print("CSV文件保存成功/gt。")


