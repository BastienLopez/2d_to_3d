# Génération du Nuage de Points en un Maillage 3D sans IA 


import numpy as np
import open3d as o3d
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2

# Fonction pour estimer la profondeur
def estimate_depth(image_path, model_type="DPT_Large"):
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if model_type == "DPT_Large" else midas_transforms.small_transform
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        prediction = midas(img)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[2:],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    depth_map = prediction.numpy()
    return depth_map

# Fonction pour convertir la carte de profondeur en nuage de points
def depth_to_pointcloud(depth_map, image_path):
    image = cv2.imread(image_path)
    height, width = depth_map.shape
    points = []
    colors = []
    for y in range(height):
        for x in range(width):
            z = depth_map[y, x]
            points.append([x, y, z])
            colors.append(image[y, x] / 255.0)
    points = np.array(points)
    colors = np.array(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# Fonction pour convertir le nuage de points en maillage
def pointcloud_to_mesh(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh

if __name__ == "__main__":
    image_path = "path/to/your/image.png"
    depth_map = estimate_depth(image_path)
    np.save("depth_map.npy", depth_map)
    pcd = depth_to_pointcloud(depth_map, image_path)
    o3d.io.write_point_cloud("output_without_generation.ply", pcd)
    mesh = pointcloud_to_mesh(pcd)
    o3d.io.write_triangle_mesh("output_mesh_without_generation.ply", mesh)
    o3d.visualization.draw_geometries([mesh])
