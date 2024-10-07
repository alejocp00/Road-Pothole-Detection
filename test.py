import subprocess
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import models


images_to_predict = os.listdir("./testing/images")
mask_images_path = "./dataset/masks/"
output_path = "./predicted/"

# Definir el modelos
models = [model for model in os.listdir("./models/") if model.endswith(".pth")]

for model_name in models:
    model_output_path = output_path + model_name

    model = torch.load("./models/"+model_name)
    # Poner el modelo en modo de evaluación
    model.eval()

    # Crear carpeta para los resultados del modelo
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    for image in images_to_predict:
        # Cargar la imagen
        input_image = Image.open("./testing/images/"+image).convert("RGB")
        input_mask = np.array(Image.open(mask_images_path+image).convert("L"))

        # Definir las transformaciones (deben ser las mismas que en el entrenamiento)
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # Aplicar las transformaciones
        input_tensor = preprocess(input_image)

        # Añadir una dimensión extra para el batch (batch size = 1)
        input_batch = input_tensor.unsqueeze(0)

        # Mover el tensor a la GPU si está disponible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_batch = input_batch.to(device)

        # Deshabilitar el cálculo de gradientes para acelerar
        with torch.no_grad():
            output = model(input_batch)["out"][0]  # Obtener la salida y eliminar dimensión de batch

        # Obtener las predicciones de clase por píxel
        # Si ya se utilizó CrossEntropyLoss, no es necesario aplicar Softmax para obtener las clases
        predicted_classes = output.argmax(0).cpu().numpy()

        # Definir una paleta de colores para las clases
        # 0 = negro (fondo), 1 = verde (calle buena), 2 = rojo (calle mala)
        color_map = {
            0: [0, 0, 0],  # Fondo - negro
            1: [128, 0, 0],  # Calle buena - rojo
            2: [0, 128, 0],  # Calle mala - verde
        }

        # Crear una imagen RGB vacía
        height, width = predicted_classes.shape
        mask_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Mapear cada clase a su color
        for class_value, color in color_map.items():
            mask_image[predicted_classes == class_value] = color

        # Convertir la imagen original a numpy para visualizar
        original_image = np.array(input_image)
        # Convertir el array a imagen
        mask_image = Image.fromarray(mask_image)

        # Superponer la máscara sobre la imagen original con cierta transparencia
        alpha = 0.6  # Transparencia de la máscara
        overlay_image = original_image.copy()
        overlay_image = Image.fromarray(overlay_image)
        overlay_mask = mask_image.convert("RGBA")
        overlay_mask.putalpha(int(255 * alpha))

        # Superponer
        overlay_image.paste(overlay_mask, (0, 0), overlay_mask)
        overlay_image = np.array(overlay_image)

        # Save Original image, original mask, prediction mask and overlay image in the


        # Guardar la máscara como imagen
        mask_image.save(model_output_path + "/" + image.replace(".png", " prediction.png"))
        
        # Guardar la imagen original
        input_image.save(model_output_path + "/" + image)
        
        # Guardar la imagen con la superposición
        overlay_image = Image.fromarray(overlay_image)
        overlay_image.save(model_output_path + "/" + image.replace(".png", " overlay.png"))
        
        # Guardar la máscara original
        subprocess.run(f"cp ./dataset/masks/{image} {model_output_path}/{image.replace('.png', '_original_mask.png')}", shell=True)
        
        