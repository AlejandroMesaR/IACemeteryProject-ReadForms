import cv2
import numpy as np
import easyocr
import os

class HandwrittenTextExtractor:
    def __init__(self, target_width=1100, target_height=40):
        self.target_width = target_width
        self.target_height = target_height
        # Inicializar EasyOCR con español e inglés
        self.reader = easyocr.Reader(['es', 'en'])
    
    def resize_and_pad_image(self, image_path, output_path):
        """
        Redimensiona y rellena la imagen para ajustarla al tamaño objetivo
        """
        try:
            # Cargar la imagen original
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("No se pudo cargar la imagen")
            
            original_height, original_width = img.shape[:2]
            
            # Calcular el factor de escala manteniendo la proporción
            scale_width = self.target_width / original_width
            scale_height = self.target_height / original_height
            scale = min(scale_width, scale_height)
            
            # Calcular nuevas dimensiones
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Redimensionar la imagen manteniendo la proporción
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Crear una imagen blanca del tamaño objetivo
            final_img = np.ones((self.target_height, self.target_width, 3), dtype=np.uint8) * 255
            
            # Calcular la posición para centrar la imagen redimensionada
            x_offset = (self.target_width - new_width) // 2
            y_offset = (self.target_height - new_height) // 2
            
            # Colocar la imagen redimensionada en el centro
            final_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img
            
            # Guardar la imagen procesada
            cv2.imwrite(output_path, final_img)
            
            return final_img, output_path
        
        except Exception as e:
            print(f"Error al procesar la imagen: {e}")
            return None, None
    
    def extract_individual_regions(self, image_path):
        """
        Extrae cada renglón de texto completo y lo procesa a 1100x40.
        Retorna una lista con los paths de las imágenes procesadas finales.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("No se pudo cargar la imagen")
            
            results = self.reader.readtext(image_path)
            
            if not results:
                print("No se detectaron regiones de texto")
                return []
            
            # Filtrar regiones con confianza suficiente
            text_regions = []
            text_contents = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:
                    text_regions.append(bbox)
                    text_contents.append(text)
            
            if not text_regions:
                print("No se encontraron regiones con suficiente confianza")
                return []
            
            # Agrupar regiones por renglones basándose en la coordenada Y
            line_groups = []
            current_line = []
            current_y_center = None
            y_threshold = 20
            
            for i, bbox in enumerate(text_regions):
                y_coords = [point[1] for point in bbox]
                y_center = (min(y_coords) + max(y_coords)) / 2
                
                if current_y_center is None:
                    current_y_center = y_center
                    current_line.append((bbox, text_contents[i]))
                elif abs(y_center - current_y_center) <= y_threshold:
                    current_line.append((bbox, text_contents[i]))
                else:
                    line_groups.append(current_line)
                    current_line = [(bbox, text_contents[i])]
                    current_y_center = y_center
            
            if current_line:
                line_groups.append(current_line)
            
            # Crear directorios
            name, ext = os.path.splitext(image_path)
            output_dir = f"{name}_individual_lines"
            processed_dir = f"{name}_processed_lines"
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(processed_dir, exist_ok=True)
            
            processed_paths = []
            
            # Procesar cada renglón
            for i, line in enumerate(line_groups):
                # Combinar bounding boxes del renglón
                min_x = float('inf')
                min_y = float('inf')
                max_x = 0
                max_y = 0
                
                for bbox, _ in line:
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    min_x = min(min_x, min(x_coords))
                    max_x = max(max_x, max(x_coords))
                    min_y = min(min_y, min(y_coords))
                    max_y = max(max_y, max(y_coords))
                
                # Añadir margen
                margin = 5
                min_x = max(0, int(min_x) - margin)
                min_y = max(0, int(min_y) - margin)
                max_x = min(img.shape[1], int(max_x) + margin)
                max_y = min(img.shape[0], int(max_y) + margin)
                
                # Recortar el renglón
                cropped_region = img[min_y:max_y, min_x:max_x]
                
                # Guardar la imagen del renglón
                region_path = os.path.join(output_dir, f"line_{i+1}.png")
                cv2.imwrite(region_path, cropped_region)
                
                # Procesar la imagen recortada
                processed_name = f"processed_line_{i+1}.png"
                processed_path = os.path.join(processed_dir, processed_name)
                
                processed_img, final_path = self.resize_and_pad_image(region_path, processed_path)
                
                if final_path:
                    processed_paths.append(final_path)
            
            return final_path
        
        except Exception as e:
            print(f"Error al extraer renglones individuales: {e}")
            return []

