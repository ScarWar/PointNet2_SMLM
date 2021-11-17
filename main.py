import importlib
import os
import sys
from pprint import pprint

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
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
    xy_resolution = 2  # nm
    z_resolution = 2

    gfp_var = 1
    protein_Location_var = 1

    ProteinDensity = 1  # Protein to nm

    const_dict = {
        "xy_resolution": xy_resolution,
        "z_resolution": z_resolution,
        "protein_Location_var": protein_Location_var,
        "ProteinDensity": ProteinDensity,
        "gfp_var": gfp_var,
        "num_shapes": 200
    }
    return const_dict


def visualize_samples(geometries):
    fig, axes = plt.subplots(3, 3, subplot_kw=dict(projection='3d'),
                             figsize=(14, 13))

    for r, ax_c in enumerate(axes):
        # print(f'type = {r}')
        for c, ax in enumerate(ax_c):
            rand_point_cloud = geometries[r][c].T

            ax.scatter(rand_point_cloud[:, 0],
                       rand_point_cloud[:, 1],
                       rand_point_cloud[:, 2], marker=',')
            # print(f'Points = {rand_point_cloud.__len__()}')

    fig.show()
    #
    # for seq in geometries:
    #     rand_point_cloud = random.choice(seq).T
    #
    #     # Open3D plot
    #     pcl = o3d.geometry.PointCloud()
    #     pcl.points = o3d.utility.Vector3dVector(rand_point_cloud)
    #     print(f"Number of points {np.asarray(pcl.points).shape[0]}")
    #     alpha = 0.3
    #     # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcl, alpha)
    #     # mesh.compute_vertex_normals()
    #     o3d.visualization.draw_geometries([pcl], mesh_show_back_face=True)


def visualize_lower_dim(vectors, labels, shapes, colors):
    vis = [('perplexity = 10', TSNE(n_components=2, perplexity=10)),
           ('perplexity = 20', TSNE(n_components=2, perplexity=20)),
           ('perplexity = 50', TSNE(n_components=2, perplexity=50)),
           ('perplexity = 100', TSNE(n_components=2, perplexity=100))]

    vis = [('PCA', PCA(n_components=2))]

    y = vectors.reshape(-1, 1024)

    for v in vis:
        name, proj = v
        y_ = proj.fit_transform(y)

        # 2D Plot
        fig = plt.figure()
        ax = fig.add_subplot()

        for i, (p, l) in enumerate(zip(y_, labels)):
            ax.scatter(p[0], p[1], c=colors[l], marker=shapes[l])
        plt.title(name)
        plt.show()


def main():
    print('Load configuration...')
    config = load_config()
    print('Done loading...')
    pprint(config)

    # Simulate Point Clouds
    print('Create point cloud simulations...')
    xyz1 = point_cloud_utils.create_ellipses(const_dict=config, n=config['num_shapes'])
    xyz2 = point_cloud_utils.create_triangles(const_dict=config, n=config['num_shapes'])
    xyz3 = point_cloud_utils.create_rects(const_dict=config, n=config['num_shapes'])
    print('Done creating...')

    xyz = [xyz1, xyz2, xyz3]

    labels = np.array([*[0 for _ in range(config['num_shapes'])],
                       *[1 for _ in range(config['num_shapes'])],
                       *[2 for _ in range(config['num_shapes'])]])
    shapes = ['o', '^', 's']
    colors = ['tab:orange', 'tab:blue', 'tab:green']

    # Visualize samples
    print('Visualizing simulations...')
    visualize_samples(xyz)
    print('Done visualizing...')

    # Build model and find representation
    print('Loading network...')
    classifier = load_model()
    classifier.train(mode=False)
    print('Done loading...')

    print('Creating representations...')
    y = []
    for geometry in xyz:
        for pc in geometry:
            pc = pc.astype(np.float32)
            pc = pc[np.newaxis, :]
            pc = torch.from_numpy(pc)
            x0, y0 = classifier(pc)
            y0 = y0.detach().numpy()
            y.append(y0)
    y = np.array(y)
    print('Done creating...')

    # Visualize in lower dimension
    print('Creating dimension reduced visualization of embeddings...')
    visualize_lower_dim(y, labels, shapes, colors)
    print('Done.')


if __name__ == '__main__':
    main()
