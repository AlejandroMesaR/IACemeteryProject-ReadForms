import re
import os
from PIL import Image
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from spellchecker import SpellChecker
import string
import onnxruntime as ort
import logging
from Utils.HandwrittenTextExtractor import HandwrittenTextExtractor

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuración para el procesamiento de formularios
ANCHO_IMAGEN = 1900
ALTO_IMAGEN = 1131
output_folder_campos = "campos"
os.makedirs(output_folder_campos, exist_ok=True)
output_folder_lineas = os.path.join(output_folder_campos, "observaciones_lineas")
os.makedirs(output_folder_lineas, exist_ok=True)

campos = {
    "fecha_ingreso": (1370, 94, 1792, 153),
    "fecha_exhumacion": (329, 237, 682, 279),
    "protocolo_necropsia": (470, 320, 875, 373),
    "causa_muerte": (1067, 308, 1726, 363),
    "primer_apellido": (90, 491, 395, 535),
    "segundo_apellido": (563, 491, 888, 535),
    "nombres": (1000, 491, 1339, 535),
    "fecha_nacimiento": (1440, 493, 1780, 540),
    "fecha_defuncion": (41, 650, 391, 696),
    "documento_identidad": (603, 650, 1044, 700),
    "fecha_inhumacion": (1324, 629, 1762, 671),
    "estado_inhumado": (84, 873, 118, 906),
    "estado_exhumado": (475, 873, 514, 905),
    "observaciones": (864, 757, 1879, 950),
    "autoridad_remitente": (30, 984, 548, 1029),
    "cargo_remitente": (23, 1069, 490, 1115),
    "funcionario_receptor": (740, 984, 1210, 1032),
    "cargo_funcionario": (740, 1069, 1260, 1111),
    "autoridad_exhumacion": (1333, 986, 1820, 1030),
    "cargo_exhumacion": (1333, 1069, 1820, 1111),
}

# Configuración para la predicción de texto
IMG_HEIGHT = 90
IMG_WIDTH = 1650
CHARSET = string.ascii_lowercase + string.ascii_uppercase + string.digits + " .,!?'-/:"
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
        result = decoded[0]
        logging.info(f"Predicción para {image_path}: {result}")
        return result
    except Exception as e:
        logging.error(f"Error en la predicción para {image_path}: {e}")
        return ""

def separar_lineas_imagen(ruta_imagen, carpeta_salida, nombre_campo):
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        logging.error(f"No se pudo cargar la imagen {ruta_imagen}.")
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
        logging.info(f"Guardada línea {i+1} del campo '{nombre_campo}' en {ruta_linea}")
    
    return lineas_texto

def postprocess_text(text, is_date_field=False, skip_spell_check=False):
    """
    Postprocesa el texto para añadir espacios, corregir formatos de fecha, limpiar caracteres repetidos
    y corregir ortografía en español.
    
    Args:
        text (str): Texto a procesar.
        is_date_field (bool): Indica si el campo es una fecha para aplicar formato específico.
    
    Returns:
        str: Texto procesado con espacios corregidos, formato limpio y ortografía corregida.
    """
    if not text:
        return ""
    
    # Limpiar caracteres repetidos (e.g., "Fuerrte" → "Fuerte", "perforaciion" → "perforacion")
    # Eliminar caracteres repetidos excepto 'r' y 'c'
    def remove_repeats_except_rc(match):
        char = match.group(1)
        if char in 'rc1234567890':
            return match.group(0)
        return char
    
    text = re.sub(r'(.)\1+', remove_repeats_except_rc, text)
    
   
    text = re.sub(r'([a-záéíóúñ])([A-ZÁÉÍÓÚÑ])', r'\1 \2', text)
    
    # Normalizar separadores en fechas si es un campo de fecha
    if is_date_field:
        # Reemplazar múltiples guiones o espacios por un solo guión
        
        text = re.sub(r'[-]+', '-', text)
        text = re.sub(r'\s+', '-', text)
        # Corregir años largos (e.g., "22025" → "2025")
        text = re.sub(r'(\d{1,2})-(\d{1,2})-(\d{4})', r'\1-\2-\3', text)
               
    elif not skip_spell_check:
        # Aplicar corrección ortográfica para campos que no son fechas ni nombres
        spell = SpellChecker(language='es')
        words = text.split()
        corrected_words = []
        for word in words:
            # No corregir números o palabras con guiones
            if word.isdigit() or '-' in word:
                corrected_words.append(word)
            else:
                corrected = spell.correction(word)
                corrected_words.append(corrected if corrected else word)
        text = ' '.join(corrected_words)
    
    # Normalizar espacios múltiples a un solo espacio
    text = re.sub(r'\s+', ' ', text)
    
    # Eliminar espacios al inicio y al final
    return text.strip()

def process_form_image(imagen_path: str, model_path: str) -> dict:
    """
    Procesa una imagen de formulario y devuelve los campos digitalizados.
    """
    resultados = {}
    
    if not os.path.exists(imagen_path):
        logging.error(f"No se encontró la imagen en {imagen_path}")
        return {"error": f"No se encontró la imagen en {imagen_path}"}
    
    try:
        imagen = Image.open(imagen_path)
        imagen = imagen.resize((ANCHO_IMAGEN, ALTO_IMAGEN), Image.Resampling.LANCZOS)
        logging.info(f"Imagen redimensionada a {ANCHO_IMAGEN}x{ALTO_IMAGEN} píxeles")
    except Exception as e:
        logging.error(f"Error al abrir o redimensionar la imagen {imagen_path}: {e}")
        return {"error": f"Error al abrir o redimensionar la imagen: {str(e)}"}
    
    extractor = HandwrittenTextExtractor(target_width=1650, target_height=90)
    
    # Lista de campos que son fechas
    date_fields = [
        "fecha_ingreso", "fecha_exhumacion", "fecha_nacimiento",
        "fecha_defuncion", "fecha_inhumacion"
    ]
    
    # Lista de campos que contienen nombres o títulos (omitir corrección ortográfica)
    name_fields = [
        "nombres", "primer_apellido", "segundo_apellido","protocolo_necropsia",
        "funcionario_receptor", "autoridad_remitente", "autoridad_exhumacion", 
    ]
    
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
                textos_lineas = []
                lineas = separar_lineas_imagen(ruta, output_folder_lineas, nombre)
                for linea in lineas:
                    processed_image_path = extractor.extract_individual_regions(linea)
                    if not processed_image_path:
                        logging.error(f"No se pudo procesar la imagen para la línea de 'observaciones': {linea}")
                        textos_lineas.append("")
                        continue
                    texto_predicho = predict(processed_image_path, model_path, char_to_idx)
                    # Aplicar postprocesamiento sin corrección ortográfica para observaciones
                    textos_lineas.append(postprocess_text(texto_predicho, is_date_field=False, skip_spell_check=False))
                resultados[nombre] = textos_lineas
                logging.info(f"Textos predichos para 'observaciones': {textos_lineas}")
            else:
                processed_image_path = extractor.extract_individual_regions(ruta)
                if not processed_image_path:
                    logging.error(f"No se pudo procesar la imagen para el campo '{nombre}'. Intenta de nuevo.")
                    resultados[nombre] = ""
                    continue
                texto_predicho = predict(processed_image_path, model_path, char_to_idx)
                # Determinar si es campo de fecha o de nombres
                is_date = nombre in date_fields
                skip_spell = nombre in name_fields
                resultados[nombre] = postprocess_text(texto_predicho, is_date_field=is_date, skip_spell_check=skip_spell)
                logging.info(f"Texto predicho para '{nombre}': {resultados[nombre]}")
                
        except Exception as e:
            logging.error(f"Error al procesar el campo '{nombre}': {e}")
            resultados[nombre] = ""
    
    return resultados