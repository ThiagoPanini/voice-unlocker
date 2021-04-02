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
"""

# Autor: Thiago Panini
# Data de Criação: 29/03/2021

"""
------------------------------------------------------
-------------- 1. CONFIGURAÇÃO INICIAL ---------------
            1.1 Importação de bibliotecas
------------------------------------------------------ 
"""

import pandas as pd
import numpy as np
import os
import shutil
import librosa
import joblib
import logging
from warnings import filterwarnings
filterwarnings('ignore')

from prep import *

from pycomp.log.log_config import log_config

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

# Definindo variáveis para visualizações gráficas
DONUT_COLORS = ['cadetblue', 'salmon', 'seagreen', 'navy']


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
    logger.info(f'Base final preparada com sucesso. Dimensões: {X_prep.shape}')
except Exception as e:
    logger.error(f'Erro ao aplicar o pipeline. Exception: {e}')
    exit()