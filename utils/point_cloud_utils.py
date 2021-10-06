import math

import numpy as np
from scipy.spatial.transform import Rotation as R


def line(point1, point2, const_dict):
    length = math.sqrt((point1[0] - point2[0]) ** 2 +
                       (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

    l = np.linspace(0, length, num=int(
        round(length * const_dict['ProteinDensity'])))
    x = point1[0] + l * (point2[0] - point1[0]) / length
    y = point1[1] + l * (point2[1] - point1[1]) / length
    z = point1[2] + l * (point2[2] - point1[2]) / length

    return x, y, z


def backround_noise(p, const_dict, xx, yy, zz):
    l = len(p)
    x_mean = 0
    y_mean = 0
    z_mean = 0
    for i in range(len(p)):
        pp = p[i]
        x_mean = x_mean + pp[0]
        y_mean = x_mean + pp[1]
        z_mean = x_mean + pp[2]

    x_mean = x_mean / l
    y_mean = y_mean / l
    z_mean = z_mean / l

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
        x.append(random.uniform(x_mean - image_length / 2, x_mean + image_length / 2))
        y.append(random.uniform(y_mean - image_length / 2, y_mean + image_length / 2))
        z.append(random.uniform(z_mean - image_length / 2, z_mean + image_length / 2))

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
    x0, y0, z0 = backround_noise(p, const_dict, x0, y0, z0)
    return x0, y0, z0


def triangle(p1, p2, p3, const_dict):
    x, y, z = line(p1, p2, const_dict)
    x1, y1, z1 = line(p2, p3, const_dict)
    x2, y2, z2 = line(p3, p1, const_dict)
    x = np.concatenate((x, x1, x2))
    y = np.concatenate((y, y1, y2))
    z = np.concatenate((z, z1, z2))
    p = [p1, p2, p3]
    x, y, z = backround_noise(p, const_dict, x, y, z)
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
    h = ((a - b) / (a + b)) ** 2
    perimeter = math.pi * (a + b) * (64 - 3 * h ** 2) / (64 + 16 * h)
    l = np.linspace(0, math.pi * 2, num=int(round(ProteinDensity * perimeter)))
    x = a * np.cos(l)
    y = b * np.sin(l)
    z = np.zeros_like(l)
    p = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    x, y, z = backround_noise(p, const_dict, x, y, z)
    return x, y, z


def create_triangles(const_dict, n=100):
    min_len = 20  # nm
    max_len = 80
    shapes = []
    for _ in range(n):
        # rib >20 nm  and <50
        p1 = [0, 0, 0]
        p2 = [0, 0, random.uniform(min_len, max_len)]
        p3 = [0, random.uniform(p2[2] * 0.5, p2[2] * 1.5),
              random.uniform(0, 1.2 * p2[2])]
        xdata, ydata, zdata = triangle(p1, p2, p3, const_dict)
        xdata, ydata, zdata = rotate(xdata, ydata, zdata)
        xdata, ydata, zdata = add_noise(xdata, ydata, zdata, const_dict)
        shapes.append(np.array([xdata, ydata, zdata]))

    return shapes


def create_rects(const_dict, n=100):
    min_len = 20  # nm
    max_len = 80
    shapes = []
    for _ in range(n):
        # rib >20 nm  and <50
        p1 = [0, 0, 0]
        p2 = [0, 0, random.uniform(min_len, max_len)]
        p3 = [0, random.uniform(min_len, max_len),
              random.uniform(0.7 * p2[2], 1.3 * p2[2])]
        p4 = [random.uniform(-0.5 * min_len, 0.5 * min_len),
              random.uniform(0.7 * p3[1], 1.3 * p3[1]), random.uniform(-0.5 * min_len, 0.5 * min_len)]

        xdata, ydata, zdata = rect(p1, p2, p3, p4, const_dict)
        xdata, ydata, zdata = rotate(xdata, ydata, zdata)
        xdata, ydata, zdata = add_noise(xdata, ydata, zdata, const_dict)
        shapes.append(np.array([xdata, ydata, zdata]))

    return shapes


import random


def create_ellipses(const_dict, n=100):
    min_len = 30
    max_len = 70
    shapes = []

    for _ in range(n):
        a = random.uniform(min_len, max_len)
        b = random.uniform(min_len, max_len)
        xdata, ydata, zdata = ellipse(a, b, const_dict)

        xdata, ydata, zdata = rotate(xdata, ydata, zdata)
        xdata, ydata, zdata = add_noise(xdata, ydata, zdata, const_dict)
        shapes.append(np.array([xdata, ydata, zdata]))

    return shapes


def uniform_trunc(xyz):
    sampled = []
    m = min(p.shape[1] for p in xyz)
    for point in xyz:
        s = np.random.choice(point.shape[1], size=m, replace=False)
        sampled.append(point[:, s])
    return sampled
