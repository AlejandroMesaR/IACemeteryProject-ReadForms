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
    fechaNacimiento: date
    fechaDefuncion: date
    fechaIngreso: date
    fechaInhumacion: date
    fechaExhumacion: date | None = Field(default=None, description="Fecha de exhumación, puede ser None si no aplica")
    funcionarioReceptor: str
    cargoFuncionario: str
    autoridadRemitente: str
    cargoAutoridadRemitente: str
    autoridadExhumacion: str
    cargoAutoridadExhumacion: str
    estado: Literal["INHUMADO", "EXHUMADO"]
    observaciones: str