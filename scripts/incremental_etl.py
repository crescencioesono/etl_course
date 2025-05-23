import pandas as pd
import numpy as np
import os
import json
import logging
import hashlib
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'incremental_etl.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('incremental_etl')

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
DATA_FINAL = os.path.join(BASE_DIR, 'data', 'final')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'data', 'checkpoints')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Asegurar que los directorios existen
for dir_path in [DATA_RAW, DATA_PROCESSED, DATA_FINAL, CHECKPOINT_DIR, MODEL_DIR, os.path.join(BASE_DIR, 'logs')]:
    os.makedirs(dir_path, exist_ok=True)

class IncrementalETLPipeline:
    # Pipeline ETL incremental para sistemas de IA en producción
    def __init__(self):
        self.db_path = os.path.join(DATA_FINAL, 'production_data.db')
        self.checkpoint_file = os.path.join(CHECKPOINT_DIR, 'etl_checkpoint.json')
        self.last_processed_date = self._load_checkpoint()
        self.scaler = None
        self._load_or_create_scaler()

    def _load_checkpoint(self):
        # Carga el checkpoint del último procesamiento
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                return datetime.fromisocalendar(checkpoint.get('last_processed_date', '2000-01-01T00:00:00'))
            return datetime(2000, 1, 1) # Fecha por defecto si no hay checkpoint
        except Exception as e:
            logger.error(f"Error al cargar el checkpoint: {str(e)}")
            return datetime(2000, 1, 1)
        
    def _save_checkpoint(self, date=None):
        # Guarda el checkpoint del procesamiento actual
        try:
            if date is None:
                date = datetime.now()

            checkpoint = {
                'last_processed_date': date.isoformat(),
                'records_processed': self.records_processed,
                'last_run': datetime.now().isoformat()
            }

            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=4)

            logger.info(f"Checkpoint guardado: {date.isoformat()}")
        except Exception as e:
            logger.error(f"Error al guardar checkpoint: {str(e)}")

    def _load_or_create_scaler(self):
        # Carga o crea un nuevo scaler para normalización
        scaler_path = os.path.join(MODEL_DIR, 'standard_scaler.pkl')

        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Scaler cargado correctamente")
            except Exception as e:
                logger.error(f"Error al cargar el scaler: {str(e)}")
                self.scaler = StandardScaler()
        else:
            self.scaler = StandardScaler()
            logger.info("Nuevo scaler creado")

    def _save_scaler(self):
        # Guarda el scaler entrenado
        try:
            scaler_path = os.path.join(MODEL_DIR, 'standard_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info("Scaler guardado correctamente")
        except Exception as e:
            logger.error(f"Error al guardar el scaler: {str(e)}")

    def _validate_data(self, df, validation_rules=None):
        # Valida los datos según reglas definidas
        if validation_rules is None:
            validation_rules = {
                'precio': {'min': 0, 'max': 10000},
                'cantidad': {'min': 0, 'max': 1000},
                'calificacion': {'min': 0, 'max': 5}
            }

        validation_results = {
            'total_records': len(df),
            'valid_records': 0,
            'invalid_records': 0,
            'validation_errors': {}
        }

        # Crear máscara de registro válido
        valid_mask = pd.Series(True, index=df.index)

        # Aplicar reglas de validación
        for column, rules in validation_rules.items():
            if column in df.columns:
                # Verificar valores mínimos
                if 'min' in rules:
                    min_mask = df[column] >= rules['min']
                    valid_mask = valid_mask & min_mask
                    invalid_count = (~min_mask).sum()
                    if invalid_count > 0:
                        validation_results['validation_errors'][f'{column}_below_min'] = invalid_count

                # Verificar valores máximos
                if 'max' in rules:
                    max_mask = df[column] <= rules['max']
                    valid_mask = valid_mask & max_mask
                    invalid_count = (~max_mask).sum()
                    if invalid_count > 0:
                        validation_results['validation_errors'][f'{column}_null'] = invalid_count

        # Verificar valores nulos en columnas críticas
        critical_columns = ['producto_id', 'precio', 'cantidad']
        for column in critical_columns:
            if column in df.columns:
                null_mask = df[column].notnull()
                valid_mask = valid_mask & null_mask
                invalid_count = (~null_mask).sum()
                if invalid_count > 0:
                    validation_results['validation_errors'][f'{column}_null'] = invalid_count

        # Filtrar registros válidos
        valid_df = df[valid_mask]

        validation_results['valid_records'] = len(valid_df)
        validation_results['invalid_records'] = len(df) - len(valid_df)

        return valid_df, validation_results
    
    def _generate_data_hash(self, df):
        #Genera un hash MD5 para los datos
        return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
    
    def extract_new_data(self):
        # Extrae solo datos nuevos desde la última ejecución
        try:
            logger.info(f"Extreyendo datos nuevos desde: {self.last_processed_date}")

            # Simulamos la generación de datos nuevos (en un caso real serían APIs, bases de datos, etc.)
            # Generamos datos para los últimos días
            current_date = datetime.now()

            # Número de días desde el último checkpoint hasta hoy
            days_diff = (current_date - self.last_processed_date).days
            days_to_generate = max(1, min(days_diff, 30)) # Generamos entre 1 y 30 días

            # Generar datos para cada día
            all_data = []

            for day in range(days_to_generate):
                date = current_date - timedelta(days=day)

                # Número de registros por día (aleatorio entre 5 y 20)
                n_records = np.random.randint(5, 21)

                for i in range(n_records):
                    record = {
                        'fecha': date,
                        'producto_id': f'PROD-{np.random.randint(1, 101):03d}',
                        'categoria': np.random.choice(['ELECTRÓNICA', 'HOGAR', 'ALIMENTOS', 'JUGUETES']),
                        'precio': round(100 * (0.5 + np.random.random()), 2),
                        'cantidad': int(1 + 10 * np.random.random)
                    }

        except Exception as e:
            pass