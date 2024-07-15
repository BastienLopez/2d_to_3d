# Conversion de la Carte de Profondeur en Nuage de Points 3D
# Créer un nuage de points 3D à partir de la carte de profondeur.

import numpy as np
import open3d as o3d
import cv2

def depth_to_pointcloud(depth_map_path, image_path):
    depth_map = np.load(depth_map_path)
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
    
    # Créer un nuage de points Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Enregistrer le nuage de points
    o3d.io.write_point_cloud("output.ply", pcd)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    depth_map_path = "depth_map.npy"
    image_path = "path/to/your/image.png"
    depth_to_pointcloud(depth_map_path, image_path)
