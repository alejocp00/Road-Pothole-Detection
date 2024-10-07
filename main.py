import gc
import torch
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import os
from PIL import Image

# Liberar memoria no gestionada por PyTorch
gc.collect()
torch.cuda.empty_cache()

# Si estás usando CUDA, resetear caché
if torch.cuda.is_available():
    try:
        torch.cuda.ipc_collect()
    except Exception as e:
        print(f"Error al intentar resetear la caché de CUDA: {e}")


class RoadConditionDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # unique_mask_values = set(mask.getdata())
        # print("Valores únicos en la máscara original:", unique_mask_values)
        # input()
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Definir transformaciones
transform_resize = (315, 315)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(transform_resize),
    ]
)



# Crear dataset y dataloader
dataset = RoadConditionDataset(
    images_dir="./dataset/images", masks_dir="./dataset/masks", transform=transform
)
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Cargar modelo DeepLabv3 preentrenado
model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=3)
model.classifier[-1] = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=(1, 1))

# Optimizar solo la última capa
optimizer = optim.Adam(model.classifier[-1].parameters(), lr=0.000001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Definir pesos para cada clase (por ejemplo: fondo con peso 0.1, calles buenas y malas con peso 1.0)
class_weights = torch.tensor([0.00001, 2, 1.5]).to(device)

# Definir la función de pérdida con los pesos asignados
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Entrenamiento
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, masks in dataloader:
        torch.cuda.empty_cache()

        inputs, masks = inputs.to(device), masks.to(device)
        outputs = model(inputs)["out"]
        masks = masks.squeeze(1).long()
        # print(masks)
        # input()
        # Cálculo del loss con mask de pesos
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

# Guardar el modelo entrenado
model_name = f"model_deeplabv3_resnet101_ts{transform_resize[0]},{transform_resize[1]}_bs{batch_size}_ep{num_epochs}_w"
torch.save(model, f"{model_name}.pth")
