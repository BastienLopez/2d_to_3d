# Génération de Maillage 3D
# Convertir le nuage de points en un maillage 3D.

import open3d as o3d

def pointcloud_to_mesh(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # Estimation des normales
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Reconstruction de surface
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    # Nettoyer le maillage
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    o3d.io.write_triangle_mesh("output_mesh.ply", mesh)
    o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    ply_path = "output.ply"
    pointcloud_to_mesh(ply_path)
