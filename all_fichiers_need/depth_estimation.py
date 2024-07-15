# Télécharger le modèle MiDaS pour l'estimation de profondeur


import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

def estimate_depth(image_path, model_type="DPT_Large"):
    # Charger le modèle MiDaS
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # Charger le transformateur MiDaS
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if model_type == "DPT_Large" else midas_transforms.small_transform

    # Charger l'image
    img = Image.open(image_path)

    # Préparer l'image
    img = transform(img).unsqueeze(0)

    # Faire l'estimation de profondeur
    with torch.no_grad():
        prediction = midas(img)

    # Redimensionner la profondeur estimée à la taille de l'image d'origine
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[2:],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_map = prediction.numpy()
    return depth_map

if __name__ == "__main__":
    image_path = "path/to/your/image.png"
    depth_map = estimate_depth(image_path)
    np.save("depth_map.npy", depth_map)
    cv2.imshow("Depth Map", depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
