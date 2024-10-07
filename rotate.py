import glob
import subprocess

# Cargar todos los archivos .png en el directorio actual
png_files = glob.glob("*.png")

# Ejecutar un comando de consola para cada archivo usando f-string
for file in png_files:
    # get file number
    file_number = file.split(".")[0]
    subprocess.run(
        f"convert -rotate 90 {file_number}.png {file_number}_r90.png ", shell=True
    )
    subprocess.run(
        f"convert -rotate 180 {file_number}.png {file_number}_r180.png ", shell=True
    )
    subprocess.run(
        f"convert -rotate 270 {file_number}.png {file_number}_r270.png ", shell=True
    )
