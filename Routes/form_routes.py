from fastapi import APIRouter, File, UploadFile, HTTPException
from Services.form_service import process_form_image
from Models.form_models import CuerpoInhumadoResponse
import os
import uuid
import shutil

router = APIRouter()

@router.post("/process-form", response_model=CuerpoInhumadoResponse)
async def process_form(file: UploadFile = File(...)):
    """
    Endpoint para subir y procesar una imagen de formulario, devolviendo los campos digitalizados.
    """
    # Validar la extensión del archivo
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PNG, JPG o JPEG")
    
    # Validar el tamaño del archivo (máximo 10 MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB en bytes
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="El archivo excede el tamaño máximo de 10 MB")
    
    # Guardar el archivo subido temporalmente
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        # Guardar el archivo
        with open(temp_file_path, "wb") as buffer:
            buffer.write(content)
        
        # Procesar la imagen
        resultados = process_form_image(temp_file_path)
        
        # Verificar si hubo un error
        if "error" in resultados:
            raise HTTPException(status_code=400, detail=resultados["error"])
        
        return CuerpoInhumadoResponse(**resultados)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")
    
    finally:
        # Eliminar el archivo temporal
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)