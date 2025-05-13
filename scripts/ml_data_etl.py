import pandas as pd
import numpy as np
import requests
import sqlite3
import os
import json
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'ml_etl.log')),
        logging.StreamHandler()
    ]
  )

logger = logging.getLogger("ml_etl")

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
DATA_FINAL = os.path.join(BASE_DIR, 'data', 'final')

# Asegurar que los directorios existen
for dir_path in [DATA_RAW, DATA_PROCESSED, DATA_FINAL, os.path.join(BASE_DIR, 'logs')]:
    os.makedirs(dir_path, exist_ok=True)

class ETLPipeline:
    # Pipeline ETL para preparar datos para un modelo de Machine Learning
    def __init__(self):
        self.db_path = os.path.join(DATA_RAW, 'ml_data.db')
        self.features_df = None
        self.target_df = None

    def extract_from_csv(self):
        # Extrae datos del CSV generado en el ejercicio anterior
        try:
            csv_path = os.path.join(DATA_RAW, 'ventas_raw.csv')
            if not os.path.exists(csv_path):
                # Si no existe, ejecutamos el script anterior para generarlo
                from basic_etl import run_etl
                run_etl()
            df = pd.read_csv(csv_path)
            df['fecha'] = pd.to_datetime(df['fecha'])
            logger.info(f"Datos extraidos de CSV: {df.shape[0]} registros.")

            return df
        except Exception as e:
            logger.error(f"Error al extraer datos de CSV: {str(e)}")
            raise

    def extract_from_api(self):
        # Simula extracción de datos de una API
        try:
            # En una situación real, haríamos:
            # response = requests.get('https://api.example.com/data')
            # data = response.json()

            # Simulamos datos de tendencias de mercado
            np.random.seed(42) # Para reproducibilidad
            data = {
                'producto_id': [f'PROD-{i:03d}' for i in range(1, 101)],
                'tendencia_mercado': np.random.choice(['SUBIENDO', 'BAJANDO', 'ESTABLE'], size=100),
                'competidores': np.random.randint(1, 10, size=100),
                'sentimiento_social': np.random.uniform(0, 1, size=100).round(2),
            }

            df = pd.DataFrame(data)
            api_path = os.path.join(DATA_RAW, 'api_data.csv')
            df.to_csv(api_path, index=False)
            logger.info(f"Datos extraidos de API: {df.shape[0]} registros.")

            return df
        except Exception as e:
            logger.error(f"Error al extraer datos de API: {str(e)}")
            raise

    def extract_from_database(self):
        # Simula extracción de datos de una base de datos SQLite
        try:
            # Creamos una base de datos temporal con informaci'on de clientes
            conn = sqlite3.connect(':memory:')
            cursor = conn.cursor()

            # Creamos una tabla de clientes
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS clientes (
                producto_id TEXT PRIMARY KEY,
                segmento_cliente TEXT,
                frecuencia_compra INTEGER,
                valor_cliente REAL
            )
            ''')

            # Insertamos datos aleatorios
            np.random.seed(42)
            for i in range(1, 101):
                cursor.execute(
                    'INSERT INTO clientes VALUES (?, ?, ?, ?)',
                    (f'PROD-{i:03d}', 
                     np.random.choice(['PREMIUM', 'ESTANDAR', 'BASICO']),
                     np.random.randint(1, 20),
                     round(np.random.uniform(100, 1000), 2)
                    )
                )
            conn.commit()

            # Consultamos los datos
            query = 'SELECT * FROM clientes'
            df = pd.read_sql_query(query, conn)
            conn.close()

            db_path = os.path.join(DATA_RAW, 'db_data.csv')
            df.to_csv(db_path, index=False)
            logger.info(f"Datos extraidos de base de datos: {df.shape[0]} registros.")

            return df
        except Exception as e:
            logger.error(f"Error al extraer datos de base de datos: {str(e)}")
            raise

    def extract(self):
        # Coordina la extracción de datos de todas las fuentes
        try:
            logger.info("Iniciando la extracción de datos.")

            #Extraer datos de diferentes fuentes
            df_ventas = self.extract_from_csv()
            df_api = self.extract_from_api()
            df_db = self.extract_from_database()

            # Guardar los datos extraídos
            return {
                'ventas': df_ventas,
                'api': df_api,
                'db': df_db
            }
        except Exception as e:
            logger.error(f"Error en la extracción de datos: {str(e)}")
            raise

    def transform(self, data_dict):
        # Transforma y combina los datos extraídos para el modelo ML
        try:
            logger.info("Iniciando la transformación de datos.")

            # Extraer los DataFrames
            df_ventas = data_dict['ventas']
            df_api = data_dict['api']
            df_db = data_dict['db']

            # 1. Preparar datos de ventas (df_ventas)
            # Convertir la fecha a datetime si no lo es
            if not pd.api.types.is_datetime64_any_dtype(df_ventas['fecha']):
                df_ventas['fecha'] = pd.to_datetime(df_ventas['fecha'])

            # Crear características temporales
            df_ventas['dia_semana'] = df_ventas['fecha'].dt.dayofweek
            df_ventas['mes'] = df_ventas['fecha'].dt.month
            df_ventas['trimestre'] = df_ventas['fecha'].dt.quarter

            # One-hot encoding de categorías
            df_ventas = pd.get_dummies(df_ventas, columns=['categoria'], prefix='categoria')

            # Calcular ingresos y otros indicadores
            df_ventas['ingresos'] = df_ventas['precio'] * df_ventas['cantidad']

            # 2. Combinar datos
            # Unir df_ventas con df_api
            df_combinado = pd.merge(df_ventas, df_api, on='producto_id', how='left')

            # Unir df_combinado con df_db
            df_combinado = pd.merge(df_combinado, df_db, on='producto_id', how='left')

            # 3. Limpiar datos, manejo de valores flotantes
            # Imputar valores faltantes numéricos con la media
            for col in df_combinado.select_dtypes(include=['number']).columns:
                df_combinado[col] = df_combinado[col].fillna(df_combinado[col].mean())

            # Importar categorías faltantes con el valor más frecuente
            for col in df_combinado.select_dtypes(include=['object']).columns:
                df_combinado[col] = df_combinado[col].fillna(df_combinado[col].mode()[0])

            # 4. Feature engineering para el modelo ML
            # Onehot encoding de variables categóricas restantes
            for col in ['segmento_cliente', 'tendencia_mercado']:
                df_combinado = pd.get_dummies(df_combinado, columns=[col], prefix=col)
            
            # Normalizar variables numéricas (importante para muchos modelos ML)
            numerical_cols = ['precio', 'cantidad', 'calificacion', 'competidores', 'sentimiento_social', 'frecuencia_compra', 'valor_cliente']

            for col in numerical_cols:
                if col in df_combinado.columns:
                    mean = df_combinado[col].mean()
                    std = df_combinado[col].std()
                    if std > 0:
                        df_combinado[f'{col}_norm'] = (df_combinado[col] - mean) / std

            # 5. Preparar features y target
            # Para este ejemplo, predecimos ingresos con el resto de características
            target = df_combinado['ingresos']

            # Excluir columnas que no queremos como features
            features = df_combinado.drop(['fecha', 'ingresos', 'producto_id'], axis=1)

            # Guardar transformaciones
            self.features_df = features
            self.target_df = target

            processed_path = os.path.join(DATA_PROCESSED, 'ml_features.csv')
            features.to_csv(processed_path, index=False)
            target.to_csv(os.path.join(DATA_PROCESSED, 'ml_target.csv'), index=False)
            logger.info(f"Transformación de datos completada: {features.shape[1]} características generadas.")

            return features, target
        except Exception as e:
            logger.error(f"Error en la transformación de datos: {str(e)}")
            raise

    def load(self, features, target):
        # Carga los datos en formaos adecuados para el modelo ML
        try:
            logger.info("Iniciando la carga de datos.")

            # 1. Guardar los datos en CSV para fácil acceso
            final_features_path = os.path.join(DATA_FINAL, 'ml_features_final.csv')
            final_target_path = os.path.join(DATA_FINAL, 'ml_target_final.csv')

            features.to_csv(final_features_path, index=False)
            target.to_csv(final_target_path, index=False)

            # 2. Guardar en SQLite para consultas
            conn = sqlite3.connect(self.db_path)
            features.to_sql('features', conn, if_exists='replace', index=False)

            # Convertir target a DataFrame si es Series
            if isinstance(target, pd.Series):
                target_df = target.to_frame()
            else:
                target_df = target

            target_df.to_sql('target', conn, if_exists='replace', index=False)

            # 3. Guardar metadata (útil para reproducibilidad)
            metadata = {
                'features_count': features.shape[1],
                'samples_count': features.shape[0],
                'features_names': list(features.columns),
                'target_name': 'ingresos',
                'data_created': datetime.now().isoformat(),
                'numerical_features': list(features.select_dtypes(include=['number']).columns),
                'categorical_features': list(features.select_dtypes(include=['object']).columns),
            }

            with open(os.path.join(DATA_FINAL, 'ml_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)

            logger.info(f"Datos cargados y guardados en {self.db_path}.")
            
            return True
        except Exception as e:
            logger.error(f"Error en la carga de datos: {str(e)}")
            raise

    def run(self):
        # Ejecuta el pipeline ETL completo
        try:
            logger.info("Iniciando pipeline ETL para preparar datos de ML.")
            data_dict = self.extract()
            features, target = self.transform(data_dict)
            self.load(features, target)
            logger.info("Pipeline ETL completado con éxito.")
            return True
        except Exception as e:
            logger.error(f"Error en el proceso ETL: {str(e)}")
            return False
        
if __name__ == "__main__":
    pipeline = ETLPipeline()
    pipeline.run()