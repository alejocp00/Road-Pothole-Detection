import subprocess
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import models
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

images_to_predict = os.listdir("./testing/images")
mask_images_path = "./dataset/masks/"
output_path = "./predicted/"


def compute_confusion_matrix(output, target, num_classes):
    # Verificar que los tensores tengan las mismas dimensiones
    # if output.shape != target.shape:
    #     raise ValueError("Output and target shapes do not match!")

    # Si output y target son tensores de PyTorch, convertir a numpy
    if isinstance(target, torch.Tensor):
        labels = target.cpu().numpy().astype(np.int64).flatten()
    else:
        labels = target.astype(np.int64).flatten()

    if isinstance(output, torch.Tensor):
        predictions = output.cpu().numpy().astype(np.int64).flatten()
    else:
        predictions = output.astype(np.int64).flatten()

    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    return conf_matrix


def save_confusion_matrix(conf_matrix, class_names, file_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Guardar la figura como archivo de imagen
    plt.savefig(file_path, format="png")
    plt.close()  # Cerrar la figura para liberar memoria


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
        
        conf_matrix = compute_confusion_matrix(model(input_batch)["out"].argmax(dim=1), input_mask, 3)

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

        save_confusion_matrix(conf_matrix, ["Background", "good_street", "bad_street"], model_output_path + "/" + image.replace(".png", "_confusion_matrix.png"))
