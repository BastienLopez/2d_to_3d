import os
import json
from flask import Flask, request, jsonify, render_template
import numpy as np
import open3d as o3d
import torch
from PIL import Image
import requests
import cv2

# Charger les informations de project_structure.json
with open('project_structure.json', 'r') as f:
    project_structure = json.load(f)

# Charger les clés API depuis api_key.json
with open('api_key.json', 'r') as f:
    config = json.load(f)

deepai_api_key = config['deepai_api_key']
roboflow_api_key = config['roboflow_api_key']
sketchfab_api_key = config['sketchfab_api_key']
removebg_api_key = config['removebg_api_key']

# Récupérer les chemins du projet
root_dir = project_structure['root']
main_file = project_structure['main_file']
index_file = project_structure['index_file']
directories = project_structure['directories']
files = project_structure['files']

# Initialiser Flask
app = Flask(__name__)

# Assurez-vous que les répertoires existent
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Prétraitement de l'image avec super-résolution
def super_resolution(image_path):
    print("Enhancing image quality using super-resolution...")
    api_url = "https://api.deepai.org/api/super-resolution"
    with open(image_path, 'rb') as image_file:
        response = requests.post(
            api_url,
            files={'image': image_file},
            headers={'api-key': deepai_api_key}
        )
    if response.status_code == 200:
        result = response.json()
        enhanced_image_url = result['output_url']
        enhanced_image_response = requests.get(enhanced_image_url)
        enhanced_image_path = image_path.replace('.jpg', '_enhanced.jpg')
        with open(enhanced_image_path, 'wb') as enhanced_image_file:
            enhanced_image_file.write(enhanced_image_response.content)
        print("Super-resolution completed.")
        return enhanced_image_path
    else:
        print("Failed to enhance image:", response.status_code, response.text)
        return image_path

# Segmentation d'objets
def segment_image(image_path):
    print("Segmenting image to isolate objects...")
    api_url = "https://detect.roboflow.com/your-model/1"
    params = {"api_key": roboflow_api_key}
    with open(image_path, 'rb') as image_file:
        response = requests.post(api_url, params=params, files={"file": image_file})
    if response.status_code == 200:
        result = response.json()
        # Logique pour traiter les résultats de segmentation
        print("Segmentation completed.")
    else:
        print("Failed to segment image:", response.status_code, response.text)

# Estimation de la profondeur
def estimate_depth(image_path, model_type="DPT_Large"):
    print("Starting depth estimation...")
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if model_type == "DPT_Large" else midas_transforms.small_transform
    img = Image.open(image_path)

    # Convertir l'image en tableau NumPy et normaliser
    img = np.array(img) / 255.0  

    # Vérifier et ajuster les canaux de l'image
    if img.shape[2] == 4:  # Si l'image a 4 canaux (RGBA), convertir en 3 canaux (RVB)
        img = img[:, :, :3]
    elif img.shape[2] == 1:  # Si l'image est en niveaux de gris, convertir en 3 canaux (RVB)
        img = np.stack((img,) * 3, axis=-1)

    print(f"Image shape after processing: {img.shape}")

    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Convertir en tenseur PyTorch float32
    print(f"Torch tensor shape: {img.shape}")

    with torch.no_grad():
        prediction = midas(img)
        print("Prediction done.")
    
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[2:],  # Garder la taille originale de l'image
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    print(f"Prediction shape after interpolation: {prediction.shape}")

    if prediction.shape[0] == 0 or prediction.shape[1] == 0:
        print(f"Invalid prediction shape: {prediction.shape}")
        raise ValueError(f"Invalid prediction shape: {prediction.shape}")

    depth_map = prediction.numpy()
    print("Depth map created.")
    return depth_map

# Conversion de la carte de profondeur en nuage de points
def depth_to_pointcloud(depth_map, image_path):
    print("Converting depth map to point cloud...")
    image = cv2.imread(image_path)
    height, width = depth_map.shape
    points = []
    colors = []
    for y in range(height):
        for x in range(width):
            z = depth_map[y, x] * 10  # Augmenter le facteur de mise à l'échelle de la profondeur pour une profondeur plus prononcée
            points.append([x, y, z])
            colors.append(image[y, x, :3] / 255.0)  # Garder les couleurs en float
    points = np.array(points)
    colors = np.array(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print("Point cloud created.")
    return pcd

# Génération du maillage à partir du nuage de points
def pointcloud_to_mesh(pcd):
    print("Converting point cloud to mesh...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()  # Calculer les normales pour une meilleure qualité de maillage
    print("Mesh created.")
    return mesh

# Prétraitement de l'image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image)
    l = cv2.equalizeHist(l)
    image = cv2.merge((l, a, b))
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    # Utiliser une meilleure méthode de débruitage
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return image

def save_preprocessed_images(image_path, output_path):
    image = preprocess_image(image_path)
    # Sauvegarder l'image en qualité maximale
    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])

# Téléchargement sur Sketchfab
def upload_to_sketchfab(file_path, title, description):
    api_url = "https://api.sketchfab.com/v3/models"
    with open(file_path, 'rb') as f:
        files = {
            'modelFile': f
        }
        data = {
            'token': sketchfab_api_key,
            'name': title,
            'description': description,
            'isPublished': True
        }
        response = requests.post(api_url, files=files, data=data)
        if response.status_code == 201:
            result = response.json()
            print(f"Model uploaded to Sketchfab: {result['uid']}")
            return result['uid']
        else:
            print(f"Failed to upload model to Sketchfab: {response.status_code}")
            print(response.json())
            return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    print("Processing image...")
    file = request.files['image']
    file_path = f'uploads/{file.filename}'
    file.save(file_path)

    # Pré-traitement de l'image
    preprocessed_path = f'uploads/preprocessed_{file.filename}'
    save_preprocessed_images(file_path, preprocessed_path)

    # Amélioration de l'image avec super-résolution
    enhanced_image_path = super_resolution(preprocessed_path)

    # Estimer la profondeur avec l'image originale
    print("Estimating depth for original image...")
    depth_map = estimate_depth(enhanced_image_path)
    pcd = depth_to_pointcloud(depth_map, enhanced_image_path)
    original_ply_path = 'static/scan_base/original.ply'
    o3d.io.write_point_cloud(original_ply_path, pcd)
    print(f"Original point cloud saved to {original_ply_path}")

    # Générer le maillage final
    print("Generating final mesh...")
    mesh = pointcloud_to_mesh(pcd)
    final_ply_path = 'static/final_scan/final.ply'
    o3d.io.write_triangle_mesh(final_ply_path, mesh)
    print(f"Final mesh saved to {final_ply_path}")

    # Télécharger les fichiers PLY sur Sketchfab
    original_model_id = upload_to_sketchfab(original_ply_path, "Original 3D Image", "This is the original 3D image.")
    final_model_id = upload_to_sketchfab(final_ply_path, "Final 3D Image", "This is the final 3D image.")

    return jsonify({
        'original_model_id': original_model_id,
        'final_model_id': final_model_id,
        'logs': [
            f"Original point cloud saved to {original_ply_path}",
            f"Final mesh saved to {final_ply_path}"
        ]
    })

if __name__ == "__main__":
    app.run(debug=True)
