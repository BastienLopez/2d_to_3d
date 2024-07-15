# Génération Complète avec IA 


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

# Charger le modèle génératif pré-entrainé (exemple VQ-VAE-2 ou GAN)
model = torch.load("path/to/generative_model.pth")
model.eval()

def generate_hidden_parts(image_path):
    img = Image.open(image_path)
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
    with torch.no_grad():
        generated_img_tensor = model(img_tensor)
    generated_img = transforms.ToPILImage()(generated_img_tensor.squeeze())
    return generated_img

if __name__ == "__main__":
    image_path = "path/to/your/image.png"
    
    # Générer les parties cachées
    generated_img = generate_hidden_parts(image_path)
    generated_img.save("generated_image.png")
    
    # Estimer la profondeur avec l'image générée
    depth_map = estimate_depth("generated_image.png")
    np.save("depth_map_with_generation.npy", depth_map)
    
    # Convertir la carte de profondeur en nuage de points
    pcd = depth_to_pointcloud(depth_map, "generated_image.png")
    o3d.io.write_point_cloud("output_with_generation.ply", pcd)
    
    # Convertir le nuage de points en maillage
    mesh = pointcloud_to_mesh(pcd)
    o3d.io.write_triangle_mesh("output_mesh_with_generation.ply", mesh)
    o3d.visualization.draw_geometries([mesh])
