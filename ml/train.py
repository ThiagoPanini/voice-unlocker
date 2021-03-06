"""
Script desenvolvido para o treinamento de um modelo de
aprendizado supervisionado para o reconhecimento de 
locutores a partir da fala. 

O código python aqui desenvolvido considera a presença
de arquivos de áudio já disponibilizados em diretório
específico do projeto.

------------------------------------------------------
                        SUMÁRIO
------------------------------------------------------
1. Configuração inicial
    1.1 Importação de bibliotecas
    1.2 Instanciando objetos de log
2. Definição de variáveis do projeto
3. Leitura da base de dados
4. Preparação e extração de features
    4.1 Pré processamento na base
    4.2 Aplicação de pipeline
    4.3 Separação em treino, validação e teste
5. Treinamento de modelo preditivo
    5.1 Instanciando objetos
    5.2 Realizando treinamento via pycomp
"""

# Autor: Thiago Panini
# Data de Criação: 29/03/2021


"""
------------------------------------------------------
-------------- 1. CONFIGURAÇÃO INICIAL ---------------
            1.1 Importação de bibliotecas
------------------------------------------------------ 
"""

# Bibliotecas python
import pandas as pd
import numpy as np
import os
import shutil
import librosa
import joblib
import logging
from warnings import filterwarnings
filterwarnings('ignore')

# Machine Learning
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

# Third-party e self-made
from prep import *
from pycomp.log.log_config import log_config
from pycomp.ml.trainer import ClassificadorMulticlasse


"""
------------------------------------------------------
-------------- 1. CONFIGURAÇÃO INICIAL ---------------
           1.2 Instanciando objetos de log
------------------------------------------------------ 
"""

# Definindo objeto de log
logger = logging.getLogger(__file__)
logger = log_config(logger)


"""
------------------------------------------------------
-------- 2. DEFINIÇÃO DE VARIÁVEIS DO PROJETO --------
------------------------------------------------------ 
"""

# Definindo variáveis de diretórios
PROJECT_PATH = '/home/paninit/workspaces/voice-unlocker'
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
AUDIOS_PATH = os.path.join(PROJECT_PATH, 'data/audios')
MCV_PATH = os.path.join(PROJECT_PATH, 'data/mozilla-common-voice/cv-corpus-6.1-2020-12-11/pt')
MCV_TRAIN_PATH = os.path.join(MCV_PATH, 'train.tsv')
MCV_CLIPS_PATH = os.path.join(MCV_PATH, 'clips')
PIPELINES_PATH = os.path.join(PROJECT_PATH, 'pipelines')
PIPELINE_NAME = 'audio_fe_pipeline.pkl'

# Definindo variáveis de processamento de áudios
SAMPLE_RATE = 22050

# Definindo variáveis de referências auxiliares
SIGNAL_COL = 'signal'
TARGET_COL = 'y_class'
ENCODED_TARGET = False
N_CLASSES = 4
TARGET_NAMES = [f'Locutor 0{i}' for i in range(1, N_CLASSES + 1)]

# Definindo variáveis relacionadas a enriquecimento com Mozilla Common Voice
COPY_MCV = False
LABEL_PREFIX = 'interlocutor_'
NEW_FOLDER = False
N_MCV_FILES = 200

# Definindo variáveis relacionadas a separação de datasets (train, val, test)
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10

# Definindo variáveis relacionadas a modelagem de dados
MODEL_KEY = 'LGBMClassifier'
ML_PATH = os.path.join(os.getcwd(), 'ml')
MODEL_PATH = os.path.join(os.getcwd(), 'model')
MODEL_FILENAME = 'lgbm_clf.pkl'
METRICS_FILENAME = 'metrics.csv'
SAVE_METRICS = True


"""
------------------------------------------------------
------------ 3. LEITURA DA BASE DE DADOS -------------
------------------------------------------------------ 
"""

# Copiando arquivos Mozilla Common Voice
if COPY_MCV:
    logger.debug(f'Flag de cópia de áudios do Common Voice ativada. Copiando {N_MCV_FILES} arquivos.')
    try:
        copy_common_voice(mcv_train_path=MCV_TRAIN_PATH, mcv_clips_path=MCV_CLIPS_PATH, data_path=AUDIOS_PATH, 
                        n_mcv_files=N_MCV_FILES, label_prefix=LABEL_PREFIX, new_folder=NEW_FOLDER)
        logger.info(f'Cópia de arquivos do Common Voice realizada com sucesso.')
    except Exception as e:
        logger.error(f'Erro ao copiar arquivos do Common Voice. Exception: {e}')

# Lendo áudios oficiais e gerando base inicial
logger.debug(f'Realizando leitura de arquivos de áudios armazenados localmente.')
try:
    df = read_data(data_path=AUDIOS_PATH, sr=SAMPLE_RATE, signal_col=SIGNAL_COL, target_col=TARGET_COL)
    logger.info(f'Base de dados lida com sucesso. Dimensões: {df.shape}')
except Exception as e:
    logger.error(f'Erro ao ler base de dados. Exception: {e}')
    exit()


"""
------------------------------------------------------
-------- 4. PREPARAÇÃO E EXTRAÇÃO DE FEATURES --------
            4.1 Pré-processamento na base
------------------------------------------------------ 
"""

# Separando apenas colunas do sinal e do target
logger.debug('Aplicando pré-processamento na base.')
try:
    X, y = data_pre_processing(df=df, signal_col=SIGNAL_COL, target_col=TARGET_COL, 
                               encoded_target=ENCODED_TARGET)
    logger.info('Base X (sinal) e y (target) gerada com sucesso.')
except Exception as e:
    logger.error(f'Erro ao aplicar pré-processamento. Exception: {e}')
    exit()


"""
------------------------------------------------------
-------- 4. PREPARAÇÃO E EXTRAÇÃO DE FEATURES --------
               4.2 Aplicação de pipeline
------------------------------------------------------ 
"""

# Leitura do pipeline
logger.debug(f'Realizando leitura do pipeline {PIPELINE_NAME} em {PIPELINES_PATH}')
try:
    audio_fe_pipeline = joblib.load(os.path.join(PIPELINES_PATH, PIPELINE_NAME))
    logger.info(f'Pipeline {PIPELINE_NAME} lido com sucesso.')
except Exception as e:
    logger.error(f'Erro ao ler o pipeline de preparação. Exception: {e}')
    exit()

# Aplicação do pipeline
logger.debug('Aplicando o pipeline na base de sinais de áudio.')
try:
    X_prep = audio_fe_pipeline.fit_transform(X)
    X_prep.drop(SIGNAL_COL, axis=1, inplace=True)
    logger.info(f'Base final preparada com sucesso. Dimensões: {X_prep.shape}')
except Exception as e:
    logger.error(f'Erro ao aplicar o pipeline. Exception: {e}')
    exit()


"""
------------------------------------------------------
-------- 4. PREPARAÇÃO E EXTRAÇÃO DE FEATURES --------
      4.3 Separação em treino, validação e teste
------------------------------------------------------ 
"""

# Gerando bases de treino, validação e teste 
logger.debug(f'Separando base em treino, validação e teste.')
try:
    X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=1-TRAIN_RATIO, 
                                                        random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=TEST_RATIO/(TEST_RATIO+VAL_RATIO), 
                                                    random_state=42)

    pct_train = round(100 * (X_train.shape[0] / len(df)), 2)
    pct_val = round(100 * (X_val.shape[0] / len(df)), 2)
    pct_test = round(100 * (X_test.shape[0] / len(df)), 2)
    logger.info(f'Separação feita com sucesso. {pct_train}% treino, {pct_val}% validação e {pct_test}% teste.')
except Exception as e:
    logger.error(f'Erro ao aplicar separação da base. Exception: {e}')
    exit()


"""
------------------------------------------------------
--------- 5. TREINAMENTO DE MODELO PREDITIVO ---------
               5.1 Instanciando objetos
------------------------------------------------------ 
"""

logger.debug(f'Iniciando preparação do(s) modelo(s).')
try:
    # Instanciando modelo a ser treinado
    lgbm = LGBMClassifier(objective='multiclass', num_class=N_CLASSES)

    # Construindo dicionário para treinamento
    model_obj = [lgbm]
    model_names = [type(model).__name__ for model in model_obj]
    set_classifiers = {name: {'model': obj, 'params': {}} for (name, obj) in zip(model_names, model_obj)}
    logger.info(f'Preparação para treinamento realizada com sucesso. Modelo(s): {model_names}')
except Exception as e:
    logger.error(f'Erro ao preparar modelo(s). Exception: {e}')


"""
------------------------------------------------------
--------- 5. TREINAMENTO DE MODELO PREDITIVO ---------
        5.2 Realizando treinamento via pycomp
------------------------------------------------------ 
"""

logger.debug(f'Inicializando objeto e realizando treinamento via pycomp.')
try:
    # Instanciando objeto e treinando modelo
    trainer = ClassificadorMulticlasse(encoded_target=ENCODED_TARGET)
    trainer.fit(set_classifiers, X_train, y_train, random_search=False)

    # Salvando modelo treinado
    model = trainer.get_estimator(MODEL_KEY)
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    joblib.dump(model, os.path.join(MODEL_PATH, MODEL_FILENAME))

    # Gerando report de métricas de performance
    if SAVE_METRICS:
        logger.debug(f'Avaliando performance do modelo treinado.')
        metrics = trainer.evaluate_performance(X_train, y_train, X_val, y_val, target_names=TARGET_NAMES)

        # Salvando resultado em arquivo csv
        if not os.path.isdir(ML_PATH):
            os.makedirs(ML_PATH)
        metrics.to_csv(os.path.join(ML_PATH, METRICS_FILENAME), index=False)

    logger.info(f'Treinamento realizado com sucesso.')
except Exception as e:
    logger.error(f'Erro ao treinar modelo. Exception: {e}')
    exit()


