import math

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def line(point1, point2, const_dict):
    length = math.sqrt((point1[0] - point2[0]) ** 2 +
                       (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

    lin = np.linspace(0, length, num=int(length * const_dict['ProteinDensity']))
    x = point1[0] + lin * (point2[0] - point1[0]) / length
    y = point1[1] + lin * (point2[1] - point1[1]) / length
    z = point1[2] + lin * (point2[2] - point1[2]) / length

    return x, y, z


def background_noise(p, const_dict, xx, yy, zz):
    points = np.c_[xx, yy, zz]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # bbox: AxisAlignedBoundingBox = pcd.get_axis_aligned_bounding_box()
    # box: np.array = np.array(bbox.get_box_points())
    # ext_box: np.array = box * 1.2
    # box = ext_box + (ext_box.mean() - box.mean())
    center = pcd.get_center()

    # length in nm
    image_length = 120

    # volume in micrometer^3
    image_volume = image_length ** 3

    # fluorophore for nm^3
    NoiseDensity = 0.00002

    x = []
    y = []
    z = []

    for i in range(round(NoiseDensity * image_volume)):
        x.append(random.uniform(center[0] - image_length / 2, center[0] + image_length / 2))
        y.append(random.uniform(center[1] - image_length / 2, center[1] + image_length / 2))
        z.append(random.uniform(center[2] - image_length / 2, center[2] + image_length / 2))

    xx = np.concatenate((xx, x))
    yy = np.concatenate((yy, y))
    zz = np.concatenate((zz, z))

    return xx, yy, zz


def rect(p1, p2, p3, p4, const_dict):
    x0, y0, z0 = line(p1, p2, const_dict)
    x1, y1, z1 = line(p2, p3, const_dict)
    x2, y2, z2 = line(p3, p4, const_dict)
    x3, y3, z3 = line(p4, p1, const_dict)
    x0 = np.concatenate((x0, x1, x2, x3))
    y0 = np.concatenate((y0, y1, y2, y3))
    z0 = np.concatenate((z0, z1, z2, z3))
    p = [p1, p2, p3, p4]
    x0, y0, z0 = background_noise(p, const_dict, x0, y0, z0)
    return x0, y0, z0


def triangle(p1, p2, p3, const_dict):
    x, y, z = line(p1, p2, const_dict)
    x1, y1, z1 = line(p2, p3, const_dict)
    x2, y2, z2 = line(p3, p1, const_dict)
    x = np.concatenate((x, x1, x2))
    y = np.concatenate((y, y1, y2))
    z = np.concatenate((z, z1, z2))
    p = [p1, p2, p3]
    return x, y, z


def add_noise(xdata, ydata, zdata, const_dict):
    x = []
    y = []
    z = []
    for i in range(len(xdata)):
        tempx = random.gauss(0, math.sqrt(
            const_dict['xy_resolution'] ** 2 + const_dict['protein_Location_var'] ** 2 + const_dict['gfp_var'] ** 2))
        tempy = random.gauss(0, math.sqrt(
            const_dict['xy_resolution'] ** 2 + const_dict['protein_Location_var'] ** 2 + const_dict['gfp_var'] ** 2))
        tempz = random.gauss(0, math.sqrt(
            const_dict['z_resolution'] ** 2 + const_dict['protein_Location_var'] ** 2 + const_dict['gfp_var'] ** 2))

        x.append(tempx)
        y.append(tempy)
        z.append(tempz)

    xdata = xdata + x
    ydata = ydata + y
    zdata = zdata + z
    return xdata, ydata, zdata


def rotate(x, y, z):
    rotation_degrees = random.uniform(0, 360)
    rotation_radians = np.radians(rotation_degrees)
    rotation_axis = np.array(
        [random.uniform(-1, 1),
         random.uniform(-1, 1),
         random.uniform(-1, 1)])

    for i in range(len(x)):
        vec = [x[i], y[i], z[i]]
        rotation_vector = rotation_radians * rotation_axis
        rotation = R.from_rotvec(rotation_vector)
        rotated_vec = rotation.apply(vec)
        x[i], y[i], z[i] = rotated_vec

    return x, y, z


def ellipse(a, b, const_dict):
    ProteinDensity = const_dict['ProteinDensity']

    # Calculating perimeter using infinite series
    h = ((a - b) / (a + b)) ** 2
    perimeter = math.pi * (a + b) * (64 - 3 * h ** 2) / (64 + 16 * h)

    # Calculate ellipse points
    lin = np.linspace(0, math.pi * 2, num=int(ProteinDensity * perimeter))
    x = a * np.cos(lin)
    y = b * np.sin(lin)
    z = np.zeros_like(lin)
    p = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    x, y, z = background_noise(p, const_dict, x, y, z)

    # print(f'A = {a}, B = {b}, P = {perimeter}, Points = {lin.__len__()}')

    return x, y, z


def create_triangles(const_dict, n=100):
    min_len = 60  # nm
    max_len = 70
    shapes = []
    for _ in range(n):
        # rib >20 nm  and <50
        p1 = [0, 0, 0]
        p2 = [0, 0, random.uniform(min_len, max_len)]
        p3 = [0, random.gauss((min_len + max_len) / 2, (max_len - min_len)),
              random.gauss(1.7 * p2[2], p2[2] / 2)]
        x, y, z = triangle(p1, p2, p3, const_dict)

        x, y, z = add_noise(x, y, z, const_dict)
        x, y, z = background_noise([], const_dict, x, y, z)
        x, y, z = rotate(x, y, z)

        shapes.append(np.array([x, y, z]))

    return shapes


def create_rects(const_dict, n=100):
    min_len = 60  # nm
    max_len = 70
    shapes = []
    for _ in range(n):
        # rib >20 nm  and <50
        p1 = [0, 0, 0]
        p2 = [0, 0, random.uniform(min_len, max_len)]
        p3 = [0, random.uniform(min_len, max_len),
              random.uniform(0.7 * p2[2], 1.3 * p2[2])]
        p4 = [random.uniform(-0.5 * min_len, 0.5 * min_len),
              random.uniform(0.7 * p3[1], 1.3 * p3[1]), random.uniform(-0.5 * min_len, 0.5 * min_len)]

        x, y, z = rect(p1, p2, p3, p4, const_dict)
        x, y, z = rotate(x, y, z)
        x, y, z = add_noise(x, y, z, const_dict)
        shapes.append(np.array([x, y, z]))

    return shapes


import random


def create_ellipses(const_dict, n=100):
    min_len = 40
    max_len = 50
    shapes = []

    for _ in range(n):
        a = random.uniform(min_len, max_len)
        b = random.uniform(min_len, max_len)
        x, y, z = ellipse(a, b, const_dict)
        x, y, z = add_noise(x, y, z, const_dict)
        x, y, z = rotate(x, y, z)
        shapes.append(np.array([x, y, z]))

    return shapes


def uniform_trunc(xyz):
    sampled = []
    m = min(p.shape[1] for p in xyz)
    for point in xyz:
        s = np.random.choice(point.shape[1], size=m, replace=False)
        sampled.append(point[:, s])
    return sampled
