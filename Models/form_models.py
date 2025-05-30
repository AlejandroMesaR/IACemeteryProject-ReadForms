from pydantic import BaseModel, Field
from datetime import date, datetime
from typing import Literal
from uuid import UUID

class CuerpoInhumadoResponse(BaseModel):
    nombre: str
    apellido: str
    documentoIdentidad: str
    numeroProtocoloNecropsia: str
    causaMuerte: str
    fechaNacimiento: str
    fechaDefuncion: str
    fechaIngreso: str
    fechaInhumacion: str
    fechaExhumacion: str
    funcionarioReceptor: str
    cargoFuncionario: str
    autoridadRemitente: str
    cargoAutoridadRemitente: str
    autoridadExhumacion: str
    cargoAutoridadExhumacion: str
    estado: Literal["INHUMADO", "EXHUMADO"]
    observaciones: str