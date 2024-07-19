import os
import json
from flask import Flask, request, jsonify, render_template
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import requests
import cv2

# Charger les informations de project_structure.json
with open('project_structure.json', 'r') as f:
    project_structure = json.load(f)

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

# Définition du modèle VAE
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, z_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2_mu(h), self.fc2_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae(model, train_loader, optimizer, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for epoch in range(1, epochs + 1):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item() / len(data):.6f}')
        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

# Définition des hyperparamètres
input_dim = 784
hidden_dim = 400
z_dim = 20
lr = 1e-3
batch_size = 128
epochs = 10

# Chemin du modèle
model_path = "all_fichiers_need/generative_model.pth"

# Vérifier si le modèle existe déjà
if not os.path.exists(model_path):
    print("Model not found. Training a new VAE model.")
    
    # Chargement des données
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('all_fichiers_need/data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialisation du modèle et de l'optimiseur
    model = VAE(input_dim, hidden_dim, z_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Entraîner le modèle
    train_vae(model, train_loader, optimizer, epochs)

    # Sauvegarder le modèle entraîné
    torch.save(model.state_dict(), model_path)
    print('Model saved to all_fichiers_need/generative_model.pth')
else:
    # Charger le modèle pré-entraîné
    model = VAE(input_dim, hidden_dim, z_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(torch.load(model_path))
    model.eval()

def estimate_depth(image_path, model_type="DPT_Large"):
    print("Starting depth estimation...")
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if model_type == "DPT_Large" else midas_transforms.small_transform
    img = Image.open(image_path)

    # Ne pas redimensionner l'image pour conserver la qualité maximale
    img = np.array(img) / 255.0  # Convertir l'image en tableau NumPy et normaliser

    # Gérer les images en niveaux de gris
    if len(img.shape) == 2:
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
            colors.append(image[y, x] / 255.0)  # Garder les couleurs en float
    points = np.array(points)
    colors = np.array(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print("Point cloud created.")
    return pcd

def pointcloud_to_mesh(pcd):
    print("Converting point cloud to mesh...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()  # Calculer les normales pour une meilleure qualité de maillage
    print("Mesh created.")
    return mesh

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

def upload_to_sketchfab(file_path, title, description):
    api_url = "https://api.sketchfab.com/v3/models"
    api_token = "9b7d7fb9860f4aceb368cdca9f458ab8"  # Remplacez par votre propre token d'API Sketchfab

    with open(file_path, 'rb') as f:
        files = {
            'modelFile': f
        }
        data = {
            'token': api_token,
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

    # Estimer la profondeur avec l'image originale
    print("Estimating depth for original image...")
    depth_map = estimate_depth(preprocessed_path)
    pcd = depth_to_pointcloud(depth_map, preprocessed_path)
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
