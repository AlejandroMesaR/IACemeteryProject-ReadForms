import os
from Utils.image_processing import process_form_image as process_image
import logging
from datetime import datetime, date
from uuid import uuid4

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        
        mapped_resultados = {
            "nombre": resultados.get("nombres", ""),
            "apellido": resultados.get("primer_apellido", ""),
            "documentoIdentidad": resultados.get("documento_identidad", ""),
            "numeroProtocoloNecropsia": resultados.get("protocolo_necropsia", ""),
            "causaMuerte": resultados.get("causa_muerte", ""),
            "fechaNacimiento": resultados.get("fecha_nacimiento", ""),
            "fechaDefuncion": resultados.get("fecha_defuncion", ""), 
            "fechaIngreso": resultados.get("fecha_ingreso", ""), 
            "fechaInhumacion": resultados.get("fecha_inhumacion", ""), 
            "fechaExhumacion": resultados.get("fecha_exhumacion", ""),
            "funcionarioReceptor": resultados.get("funcionario_receptor", ""),
            "cargoFuncionario": resultados.get("cargo_funcionario", ""),
            "autoridadRemitente": resultados.get("autoridad_remitente", ""),
            "cargoAutoridadRemitente": resultados.get("cargo_remitente", ""),
            "autoridadExhumacion": resultados.get("autoridad_exhumacion", ""),
            "cargoAutoridadExhumacion": resultados.get("cargo_exhumacion", ""),
            "estado": "INHUMADO" if resultados.get("estado_inhumado") == "." else "INHUMADO" if resultados.get("estado_exhumado") == "." else "EXHUMADO",
            "observaciones": observaciones_str
        }
        return mapped_resultados
    except Exception as e:
        logging.error(f"Error al mapear los resultados: {e}")
        return {"error": f"Error al mapear los resultados: {str(e)}"}
    
    