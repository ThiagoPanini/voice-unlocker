"""
Script principal do projeto Voice Unlocker responsável
por coordenar o recebimento de novos áudios a serem
validados e a execução do modelo preditivo treinado
previamente de modo a apresentar um resultado ao
executor do código.

------------------------------------------------------
                        SUMÁRIO
------------------------------------------------------
1. Configuração inicial
    1.1 Importação de bibliotecas
    1.2 Instanciando objetos de log
2. Definição de variáveis do projeto
3. Definição de função com as regras de validação
"""

# Autor: Thiago Panini
# Data de Criação: 04/04/2021


"""
------------------------------------------------------
-------------- 1. CONFIGURAÇÃO INICIAL ---------------
            1.1 Importação de bibliotecas
------------------------------------------------------ 
"""

# Bibliotecas python
import os
import pandas as pd
import joblib
import logging
from datetime import datetime
import time
import librosa
from warnings import filterwarnings
filterwarnings('ignore')

# Third-party e self-made
from ml.prep import *
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
PREDICTIONS_PATH = os.path.join(PROJECT_PATH, 'predictions')
TARGET_PATH = os.path.join(PREDICTIONS_PATH, 'audios')
RESULTS_PATH = os.path.join(PREDICTIONS_PATH, 'results')
RESULTS_FILE = 'audio_verification_log.csv'

# Definindo variáveis estruturantes de regras
AUDIO_EXT = '.mp3'
MOST_RECENT = True
DROP_AUDIOS = True

# Extraindo a quantidade de arquivos válidos na pasta alvo
VALID_FILES = [file for file in os.listdir(TARGET_PATH) if os.path.splitext(file)[-1] == AUDIO_EXT]
QTD_FILES = len(VALID_FILES)

# Definindo variáveis para extração de artefatos
PIPELINE_PATH = os.path.join(PROJECT_PATH, 'pipelines')
PIPELINE_NAME = 'audio_fe_pipeline.pkl'
MODEL_PATH = os.path.join(PROJECT_PATH, 'model')
MODEL_NAME = 'lgbm_clf.pkl'

# Definindo variáveis de leitura e predição de áudios
SAMPLE_RATE = 22050
SIGNAL_COL = 'signal'
SAVE_RESULTS = True


"""
------------------------------------------------------
---------- 3. REGRAS DE VALIDAÇÃO DE ÁUDIOS ----------
------------------------------------------------------ 
"""
 
if QTD_FILES == 0:
    # Nenhum áudio válido presente no diretório alvo
    logger.warning(f'Nenhum arquivo {AUDIO_EXT} encontrado no diretório. Verificar extensão do áudio disponibilizado.')
    exit()
elif QTD_FILES > 1:
    # Mais de um áudio válido presente no diretório alvo
    logger.warning(f'Foram encontrados {QTD_FILES} arquivos {AUDIO_EXT} no diretório. Necessário validar um áudio por vez.')
    if MOST_RECENT:
        logger.debug(f'Considerando o áudio mais recente presente no diretório.')
        try:
            # Extraindo data de criação dos múltiplos áudios
            ctimes = [os.path.getctime(os.path.join(TARGET_PATH, file)) for file in VALID_FILES]
            audio_ctimes = [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ct)) for ct in ctimes]

            # Indexando áudio mais recente
            idx_max_ctime = audio_ctimes.index(max(audio_ctimes))
            audio_file = VALID_FILES[idx_max_ctime]
        except Exception as e:
            logger.error(f'Erro ao extrair o áudio mais recente. Exception: {e}')
            exit()
    else:
        exit()
else:
    # Nenhuma regra de invalidação selecionada: referenciando áudio válido
    audio_file = VALID_FILES[0]
    logger.info(f'Regras de validação aplicadas. Áudio considerado: {audio_file}')


"""
------------------------------------------------------
------ 4. RESPOSTA A IDENTIFICAÇÃO DE LOCUTORES ------
              4.1 Importando artefatos
------------------------------------------------------ 
"""

# Lendo pipeline de preparação e modelo treinado
pipeline = joblib.load(os.path.join(PIPELINE_PATH, PIPELINE_NAME))
model = joblib.load(os.path.join(MODEL_PATH, MODEL_NAME))


"""
------------------------------------------------------
------ 4. RESPOSTA A IDENTIFICAÇÃO DE LOCUTORES ------
     4.2 Lendo, preparando e realizando predições
------------------------------------------------------ 
"""

# Lendo sinal de áudio selecionado
t0 = time.time()
logger.debug(f'Lendo sinal de áudio via librosa e transformando em DataFrame')
try:
    audio = [librosa.load(os.path.join(TARGET_PATH, audio_file), sr=SAMPLE_RATE)[0]]
    audio_df = pd.DataFrame([audio])
    audio_df.columns = [SIGNAL_COL]
except Exception as e:
    logger.error(f'Erro ao ler ou transformar áudio. Exception: {e}')
    exit()

# Aplicando pipeline no áudio selecionado
logger.debug(f'Aplicando pipeline de preparação no áudio lido')
try:
    audio_prep = pipeline.fit_transform(audio_df)
    audio_prep.drop(SIGNAL_COL, axis=1, inplace=True)
except Exception as e:
    logger.error(f'Erro ao aplicar pipeline de preparação. Exception: {e}')
    exit()

# Realizando predições
logger.debug(f'Realizando predições no áudio selecionado')
try:
    y_pred = model.predict(audio_prep)
except Exception as e:
    logger.error(f'Erro ao realizar predição. Exception: {e}')
    exit()
t1 = time.time()


"""
------------------------------------------------------
------------- 5. COORDENANDO RESULTADOS --------------
------------------------------------------------------ 
"""

if SAVE_RESULTS:
    logger.debug('Armazenando resultados')
    try:
        df_results = pd.DataFrame({})
        df_results['audio_file'] = [audio_file]
        df_results['prediction'] = ['Locutor ' + str(y_pred[0])]
        df_results['exec_time'] = round((t1 - t0), 3)
        df_results['datetime'] = datetime.now()
        
        # Criando diretório de resultados, caso inexistente
        if not os.path.isdir(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)

        # Lendo arquivo de resultados, caso já existente
        if RESULTS_FILE not in  os.listdir(RESULTS_PATH):
            df_results.to_csv(os.path.join(RESULTS_PATH, RESULTS_FILE), index=False)
        else:
            old_results = pd.read_csv(os.path.join(RESULTS_PATH, RESULTS_FILE))
            df_results = old_results.append(df_results)
            df_results.to_csv(os.path.join(RESULTS_PATH, RESULTS_FILE), index=False)
    except Exception as e:
        logger.error(f'Erro ao salvar os resultados. Exception: {e}')

# Eliminando áudios
if DROP_AUDIOS:
    logger.debug(f'Eliminando áudios no diretório alvo')
    try:
        for file in os.listdir(TARGET_PATH):
            os.remove(os.path.join(TARGET_PATH, file))
    except Exception as e:
        logger.error(f'Erro ao eliminar áudios. Exception: {e}')

logger.info('Fim do processamento')