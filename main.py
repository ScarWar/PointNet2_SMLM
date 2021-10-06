import importlib
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from utils import net_utils, point_cloud_utils


def load_model(cls_num=40, use_cpu=True, is_notebook=True):
    # Find module
    if is_notebook:
        f = __file__
    else:
        f = './Pointnet_Pointnet2_pytorch/test_classification.py'

    BASE_DIR = os.path.dirname(os.path.abspath(f))
    ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(ROOT_DIR, 'models'))

    # Load module and model
    model = importlib.import_module('pointnet2_cls_ssg')
    classifier = model.get_model(cls_num, normal_channel=False)
    cls_param_path = './log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'
    checkpoint = torch.load(cls_param_path, map_location='cpu')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # Remove last layer
    classifier.fc3 = net_utils.Identity()

    # Support cpu
    if not use_cpu:
        classifier = classifier.cude()

    return classifier


def load_config():
    xy_resolution = 5  # nm
    z_resolution = 5

    gfp_var = 1
    protein_Location_var = 4

    ProteinDensity = 1  # Protein to nm

    const_dict = {
        "xy_resolution": xy_resolution,
        "z_resolution": z_resolution,
        "protein_Location_var": protein_Location_var,
        "ProteinDensity": ProteinDensity,
        "gfp_var": gfp_var,
        "num_shapes": 50
    }
    return const_dict


def main():
    config = load_config()

    # Simulate Point Clouds
    xyz1 = point_cloud_utils.create_ellipses(const_dict=config, n=config['num_shapes'])
    xyz2 = point_cloud_utils.create_triangles(const_dict=config, n=config['num_shapes'])
    xyz3 = point_cloud_utils.create_rects(const_dict=config, n=config['num_shapes'])
    xyz = xyz1 + xyz2 + xyz3
    labels = np.array([*[0 for _ in range(config['num_shapes'])],
                       *[1 for _ in range(config['num_shapes'])],
                       *[2 for _ in range(config['num_shapes'])]])
    xyz = point_cloud_utils.uniform_trunc(xyz)
    xyz = np.array(xyz, dtype=np.float32)
    xyz = torch.from_numpy(xyz)

    # Build model and find representation
    classifier = load_model()
    x, y = classifier(xyz)

    # Visualize in lower dimension
    proj = TSNE(n_components=3, perplexity=30)
    y = y.detach().numpy()
    y_ = proj.fit_transform(y.reshape(y.shape[:2]))

    # 3D Plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(y_[:, 0], y_[:, 1], y_[:, 2], c=labels)
    plt.show()


if __name__ == '__main__':
    main()
