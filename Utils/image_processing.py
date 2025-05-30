import os
from PIL import Image
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import string
import onnxruntime as ort
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration for form processing
ANCHO_IMAGEN = 1500
ALTO_IMAGEN = 893
output_folder_campos = "campos"
os.makedirs(output_folder_campos, exist_ok=True)
output_folder_lineas = os.path.join(output_folder_campos, "observaciones_lineas")
os.makedirs(output_folder_lineas, exist_ok=True)

campos = {
    "fecha_ingreso": (1053, 42, 1488, 152),
    "fecha_exhumacion": (177, 176, 642, 218),
    "protocolo_necropsia": (373, 244, 643, 285),
    "causa_muerte": (798, 214, 1443, 288),
    "primer_apellido": (33, 381, 361, 420),
    "segundo_apellido": (386, 381, 727, 426),
    "nombres": (757, 381, 1090, 426),
    "fecha_nacimiento": (1111, 381, 1435, 426),
    "fecha_defuncion": (11, 484, 355, 546),
    "documento_identidad": (398, 488, 918, 552),
    "fecha_inhumacion": (964, 494, 1478, 529),
    "estado_inhumado": (66, 689, 94, 715),
    "estado_exhumado": (376, 689, 406, 715),
    "observaciones": (668, 598, 1489, 746),
    "autoridad_remitente": (29, 776, 540, 811),
    "cargo_remitente": (29, 842, 540, 875),
    "funcionario_receptor": (588, 776, 1004, 811),
    "cargo_funcionario": (588, 842, 1004, 875),
    "autoridad_exhumacion": (1051, 776, 1476, 811),
    "cargo_exhumacion": (1051, 842, 1476, 875),
}

# Configuration for text prediction
IMG_HEIGHT = 128
IMG_WIDTH = 512
CHARSET = string.ascii_lowercase + string.ascii_uppercase + string.digits + " .,!?'-"
char_to_idx = {char: idx + 1 for idx, char in enumerate(CHARSET)}
char_to_idx[''] = 0

def decode_output(output, char_to_idx):
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    output = output.permute(1, 0, 2)
    decoded = []
    for seq in output:
        _, max_idx = torch.max(seq, dim=1)
        chars = []
        prev = None
        for idx in max_idx:
            idx = idx.item()
            if idx != 0 and idx != prev:
                chars.append(idx_to_char.get(idx, ''))
            prev = idx
        decoded.append(''.join(chars))
    return decoded

def predict(image_path, model_path, char_to_idx):
    try:
        session = ort.InferenceSession(model_path)
        transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        img = Image.open(image_path).convert('L')
        img = transform(img).numpy()
        img = np.expand_dims(img, axis=0)
        
        outputs = session.run(None, {'input': img})[0]
        decoded = decode_output(torch.tensor(outputs), char_to_idx)
        return decoded[0]
    except Exception as e:
        logging.error(f"Error in prediction for {image_path}: {e}")
        return ""

def separar_lineas_imagen(ruta_imagen, carpeta_salida, nombre_campo):
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        logging.error(f"Could not load image {ruta_imagen}.")
        return []
    
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binarizada = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proyeccion_horizontal = np.sum(binarizada, axis=1)
    umbral_linea = np.max(proyeccion_horizontal) * 0.05
    altura_minima_linea = 5
    
    lineas = []
    inicio_linea = None
    
    for i in range(len(proyeccion_horizontal)):
        if proyeccion_horizontal[i] > umbral_linea and inicio_linea is None:
            inicio_linea = i
        elif proyeccion_horizontal[i] <= umbral_linea and inicio_linea is not None:
            if i - inicio_linea >= altura_minima_linea:
                lineas.append((inicio_linea, i))
            inicio_linea = None
    
    if inicio_linea is not None and len(proyeccion_horizontal) - inicio_linea >= altura_minima_linea:
        lineas.append((inicio_linea, len(proyeccion_horizontal)))
    
    lineas_texto = []
    for i, (inicio, fin) in enumerate(lineas):
        margen = 5
        y_inicio = max(0, inicio - margen)
        y_fin = min(imagen.shape[0], fin + margen)
        linea = imagen[y_inicio:y_fin, :]
        ruta_linea = os.path.join(carpeta_salida, f"{nombre_campo}_linea_{i+1}.png")
        cv2.imwrite(ruta_linea, linea)
        lineas_texto.append(ruta_linea)
        logging.info(f"Saved line {i+1} of field '{nombre_campo}' at {ruta_linea}")
    
    return lineas_texto

def process_form_image(imagen_path: str, model_path: str) -> dict:
    """
    Process a form image and return digitized fields.
    """
    resultados = {}
    
    if not os.path.exists(imagen_path):
        logging.error(f"No se encontró la imagen en {imagen_path}")
        return {"error": f"No se encontró la imagen en {imagen_path}"}
    
    try:
        imagen = Image.open(imagen_path)
        # Resize the image to 1500x893 pixels
        imagen = imagen.resize((ANCHO_IMAGEN, ALTO_IMAGEN), Image.Resampling.LANCZOS)
        logging.info(f"Imagen redimensionada a {ANCHO_IMAGEN}x{ALTO_IMAGEN} píxeles")
    except Exception as e:
        logging.error(f"Error al abrir o redimensionar la imagen {imagen_path}: {e}")
        return {"error": f"Error al abrir o redimensionar la imagen: {str(e)}"}
    
    for nombre, (x1, y1, x2, y2) in campos.items():
        if x1 < 0 or x2 > ANCHO_IMAGEN or y1 < 0 or y2 > ALTO_IMAGEN or x1 >= x2 or y1 >= y2:
            logging.error(f"Coordenadas inválidas para el campo '{nombre}': ({x1}, {y1}, {x2}, {y2})")
            resultados[nombre] = ""
            continue
        
        try:
            campo = imagen.crop((x1, y1, x2, y2))
            ruta = os.path.join(output_folder_campos, f"{nombre}.png")
            campo.save(ruta)
            logging.info(f"Campo '{nombre}' guardado en {ruta}")
            
            if nombre == "observaciones":
                lineas = separar_lineas_imagen(ruta, output_folder_lineas, nombre)
                textos_lineas = [predict(linea, model_path, char_to_idx) for linea in lineas]
                resultados[nombre] = textos_lineas
                logging.info(f"Textos predichos para 'observaciones': {textos_lineas}")
            else:
                texto_predicho = predict(ruta, model_path, char_to_idx)
                resultados[nombre] = texto_predicho
                logging.info(f"Texto predicho para '{nombre}': {texto_predicho}")
                
        except Exception as e:
            logging.error(f"Error al procesar el campo '{nombre}': {e}")
            resultados[nombre] = ""
    
    return resultados