import os
from Utils.image_processing import process_form_image as process_image
from Utils.DateFormatValidator import correct_date
import logging
from datetime import datetime, date
from uuid import uuid4

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_numeric_text(text: str, camp: str) -> str:
    """
    Limpia el texto reemplazando caracteres erróneos por sus equivalentes numéricos.
    """
    if not text:
        return text
    # Reemplazar letras comunes confundidas con números
    replacements = {
        'O': '0', 'o': '0','D': '0', 'Q': '0',
        'I': '1', 'i': '1', 'l': '1',
        'Z': '2', 'z': '2',
        'S': '5', 's': '2',
        'B': '8', 'b': '8',
        't': '7', 'T': '7',
        'G': '6',
        
    }
    cleaned = text
    for wrong, correct in replacements.items():
        cleaned = cleaned.replace(wrong, correct)
    # Asegurar que solo queden números, guiones, barras y espacios para fechas
    
    if not camp == "documento_identidad":
        cleaned = correct_date(cleaned)
        
    cleaned = ''.join(c for c in cleaned if c.isdigit() or c in '-/: ')
    logging.info(f"Texto limpio: {text} -> {cleaned}")
    return cleaned

def process_form_image(image_path: str) -> dict:
    """
    Servicio para procesar una imagen de formulario y devolver los campos digitalizados.
    """
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'crnn_model.onnx')
    if not os.path.exists(model_path):
        logging.error(f"Modelo {model_path} no encontrado")
        return {"error": "Modelo ONNX no encontrado"}
    
    resultados = process_image(image_path, model_path)
    if "error" in resultados:
        return resultados
    
    # Mapear los campos predichos a los del modelo CuerpoInhumadoResponse
    try:
        observaciones_str = ""
        # Concatenar las líneas de observaciones en un solo string
        if resultados.get("observaciones") != None:
            observaciones = resultados.get("observaciones", [])
            observaciones_str = " ".join([line for line in observaciones if line]) if isinstance(observaciones, list) else str(observaciones)
        
        # Limpiar campos numéricos y de fechas
        fecha_nacimiento = clean_numeric_text(resultados.get("fecha_nacimiento", ""), "fecha_nacimiento")
        fecha_defuncion = clean_numeric_text(resultados.get("fecha_defuncion", ""), "fecha_defuncion")
        fecha_ingreso = clean_numeric_text(resultados.get("fecha_ingreso", ""), "fecha_ingreso")
        fecha_inhumacion = clean_numeric_text(resultados.get("fecha_inhumacion", ""), "fecha_inhumacion")
        fecha_exhumacion = clean_numeric_text(resultados.get("fecha_exhumacion", ""), "fecha_exhumacion")
        documento_identidad = clean_numeric_text(resultados.get("documento_identidad", ""), "documento_identidad")
        
        # Validar fechas
        print(f'fecha_nacimiento: {fecha_nacimiento}, fecha_defuncion: {fecha_defuncion}, fecha_ingreso: {fecha_ingreso}, fecha_inhumacion: {fecha_inhumacion}, fecha_exhumacion: {fecha_exhumacion}')
        try:
            if fecha_nacimiento:
                fecha_nacimiento = datetime.strptime(fecha_nacimiento, '%d-%m-%Y').date()
            if fecha_defuncion:
                fecha_defuncion = datetime.strptime(fecha_defuncion, '%d-%m-%Y').date()
            if fecha_ingreso:
                fecha_ingreso = datetime.strptime(fecha_ingreso, '%d-%m-%Y').date()
            if fecha_inhumacion:
                fecha_inhumacion = datetime.strptime(fecha_inhumacion, '%d-%m-%Y').date()
            if fecha_exhumacion and fecha_exhumacion.strip():  # Verificar que no esté vacío
                fecha_exhumacion = datetime.strptime(fecha_exhumacion, '%d-%m-%Y').date()
            else:
                fecha_exhumacion = None  # Asignar una fecha mínima válida
        except ValueError as e:
            logging.error(f"Error al parsear fechas: {e}")
            return {"error": f"Error al parsear fechas: {str(e)}"}
        
        mapped_resultados = {
            "nombre": resultados.get("nombres", ""),
            "apellido": resultados.get("primer_apellido", "") + " " + resultados.get("segundo_apellido", ""),
            "documentoIdentidad": documento_identidad,
            "numeroProtocoloNecropsia": resultados.get("protocolo_necropsia", ""),
            "causaMuerte": resultados.get("causa_muerte", ""),
            "fechaNacimiento": fecha_nacimiento,
            "fechaDefuncion": fecha_defuncion, 
            "fechaIngreso": fecha_ingreso, 
            "fechaInhumacion": fecha_inhumacion, 
            "fechaExhumacion": fecha_exhumacion,
            "funcionarioReceptor": resultados.get("funcionario_receptor", ""),
            "cargoFuncionario": resultados.get("cargo_funcionario", ""),
            "autoridadRemitente": resultados.get("autoridad_remitente", ""),
            "cargoAutoridadRemitente": resultados.get("cargo_remitente", ""),
            "autoridadExhumacion": resultados.get("autoridad_exhumacion", ""),
            "cargoAutoridadExhumacion": resultados.get("cargo_exhumacion", ""),
            "estado": "EXHUMADO" if resultados.get("estado_inhumado") == "." or  resultados.get("estado_inhumado") == None or resultados.get("estado_exhumado") == "X" or resultados.get("estado_exhumado") == "x"
                    else "INHUMADO" if resultados.get("estado_exhumado") == "." or resultados.get("estado_exhumado") == None or resultados.get("estado_inhumado") == "X" or resultados.get("estado_inhumado") == "x"
                    else  "EXHUMADO",
            "observaciones": observaciones_str
        }
        return mapped_resultados
    except Exception as e:
        logging.error(f"Error al mapear los resultados: {e}")
        return {"error": f"Error al mapear los resultados: {str(e)}"}
    
    