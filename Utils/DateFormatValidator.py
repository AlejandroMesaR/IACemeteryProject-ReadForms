import re
from datetime import datetime

def correct_component(valor_str, tipo):

    AÑO_ACTUAL = datetime.now().year
    # Tratar valores mal segmentados como "225" o "95"
    valor_str = valor_str.strip()
    if tipo == "anio":
        valor_str = valor_str[:4].ljust(4, '0')
    else:
        valor_str = valor_str[-2:].rjust(2, '0')  # Mantiene los 2 últimos dígitos para día/mes

    try:
        valor = int(valor_str)
        if tipo == "dia" and 1 <= valor <= 31:
            return f"{valor:02d}"
        elif tipo == "mes" and 1 <= valor <= 12:
            return f"{valor:02d}"
        elif tipo == "anio" and 1900 <= valor <= AÑO_ACTUAL:
            return f"{valor:04d}"
    except:
        pass

    # Si es inválido, hacer reemplazos de dígitos
    valor_corregido = valor_str.translate(str.maketrans({'7': '1', '9': '0'}))

    try:
        valor = int(valor_corregido)
    except:
        valor = 1 if tipo != "anio" else 2000

    if tipo == "dia":
        valor = max(1, min(valor, 31))
        return f"{valor:02d}"
    elif tipo == "mes":
        valor = max(1, min(valor, 12))
        return f"{valor:02d}"
    elif tipo == "anio":
        valor = max(1900, min(valor, AÑO_ACTUAL))
        return f"{valor:04d}"

def max_day_in_month(mes: int, anio: int) -> int:
    try:
        if mes == 12:
            siguiente = datetime(anio + 1, 1, 1)
        else:
            siguiente = datetime(anio, mes + 1, 1)
        actual = datetime(anio, mes, 1)
        return (siguiente - actual).days
    except:
        return 28

def correct_date(fecha_str):
    numeros = re.findall(r'\d+', fecha_str)
    if len(numeros) < 3:
        return "FECHA_INVALIDA"

    dia = correct_component(numeros[0], "dia")
    mes = correct_component(numeros[1], "mes")
    anio = correct_component(numeros[2], "anio")

    try:
        dia_int, mes_int, anio_int = int(dia), int(mes), int(anio)
        max_dia = max_day_in_month(mes_int, anio_int)
        if dia_int > max_dia:
            dia_int = max_dia
        return f"{dia_int:02d}-{mes_int:02d}-{anio_int:04d}"
    except:
        return "FECHA_INVALIDA"
