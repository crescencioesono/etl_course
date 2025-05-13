import pandas as pd
import numpy as np
import os
from datetime import datetime
import sqlite3

# Configuración de rutas
DATA_RAW = os.path.join('data', 'raw')
DATA_PROCESSED = os.path.join('data', 'processed')
DATA_FINAL = os.path.join('data', 'final')

# Asegurar que los directorios existen
for dir_path in [DATA_RAW, DATA_PROCESSED, DATA_FINAL]:
    os.makedirs(dir_path, exist_ok=True)

def log_step(message):
    # Registra los pasos del proceso ETL
    print(f"{datetime.now()}: {message}")

def extract():
    # Extrae datos de muestra para nuestro ETL
    log_step("Iniciando la extracción de datos.")

    # Creamos datos de muestra para un dataset de ventas
    data = {
        'fecha': pd.date_range(start='2023-01-01', periods=100),
        'producto_id': [f'PROD-{i:03d}' for i in range(1, 101)],
        'categoria': ['Electrónica', 'Ropa', 'Hogar', 'Alimentos', 'Juguetes'] * 20,
        'precio': [round(100 * (0.5 + np.random.rand()), 2) for _ in range(100)],
        'cantidad': [int(1 + 10 * np.random.random()) for _ in range(100)],
        'calificacion': [round(1 + 4 * np.random.random(), 1) for _ in range(100)],
    }

    df = pd.DataFrame(data)

    # Guardar datos crudos
    raw_path = os.path.join(DATA_RAW, 'ventas_raw.csv')
    df.to_csv(raw_path, index=False)
    log_step(f"Datos extraídos y guardados en {raw_path}.")

    return df

def transform(df):
    # Transforma los datos
    log_step("Iniciando la transformación de datos.")

    # Crear nuevas características
    df['ingresos'] = df['precio'] * df['cantidad']
    df['anio_mes'] = df['fecha'].dt.strftime('%Y-%m')
    df['es_fin_de_semana'] = df['fecha'].dt.dayofweek >= 5

    # Limpiar datos - ejemplo: categoría con formato consistente
    df['categoria'] = df['categoria'].str.upper()

    # Filtrar datos sin precio
    df = df[df['precio'] > 0]

    # Guardar datos procesados
    processed_path = os.path.join(DATA_PROCESSED, 'ventas_processed.csv')
    df.to_csv(processed_path, index=False)
    log_step(f"Datos transformados y guardados en {processed_path}.")

    return df

def load(df):
    # Cargar datos a destino final
    log_step("Iniciando la carga de datos.")

    # Crear resumen por categoría
    resumen = df.groupby('categoria').agg(
        total_ventas=('ingresos', 'sum'),
        promedio_precio=('precio', 'mean'),
        total_productos=('producto_id', 'count'),
        calificacion_promedio=('calificacion', 'mean')
    ).reset_index()

    # Guardar en CSV
    final_path = os.path.join(DATA_FINAL, 'resumen_ventas.csv')
    resumen.to_csv(final_path, index=False)
    log_step(f"Resumen cargado en {final_path}.")

    # Simulamos carga a una base de datos
    log_step("Simulando carga a base de datos SQLite.")
    conn = sqlite3.connect(os.path.join(DATA_FINAL, 'ventas.db'))
    df.to_sql('ventas', conn, if_exists='replace', index=False)
    resumen.to_sql('resumen', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

    return True

def run_etl():
    # Ejecutar el proceso ETL completo
    try:
        log_step("Iniciando el proceso ETL.")
        df = extract()
        df = transform(df)
        load(df)
        log_step("Proceso ETL completado con éxito.")
        return True
    except Exception as e:
        log_step(f"Error en el proceso ETL: {str(e)}")
        return False

if __name__ == "__main__":
    run_etl()