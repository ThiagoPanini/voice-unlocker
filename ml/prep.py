"""
Script responsável por alocar funções e classes 
auxiliares relacionadas a leitura e preparação dos 
sinais de áudio.

O código aqui disponibilizado pode ser importado em
outros módulos para ser utilizado de forma encapsulada
nas etapas de preparação e construção de pipelines.

------------------------------------------------------
                        SUMÁRIO
------------------------------------------------------
1. Importação de bibliotecas
1. Funções de leitura e enriquecimento de base
"""

# Autor: Thiago Panini
# Data de Criação: 29/03/2021

"""
------------------------------------------------------
             1. IMPORTAÇÃO DE BIBLIOTECAS
------------------------------------------------------ 
"""

import os 
import pandas as pd
import shutil
import librosa
from warnings import filterwarnings
filterwarnings('ignore')


"""
------------------------------------------------------
    2. FUNÇÕES DE LEITURA E ENRIQUECIMENTO DE BASE
------------------------------------------------------ 
"""

# Definindo função para cópia de arquivos de áudio do portal Mozilla Common Voice
def copy_common_voice(mcv_train_path, mcv_clips_path, data_path, n_mcv_files=10, 
                      label_prefix='interlocutor_', new_folder=False):
    """
    Função responsável por copiar arquivos de áudio do diretório local Mozilla Common Voice
    
    Parâmetros
    ----------
    :param mcv_train_path: referência de arquivo train.tsv do Mozilla Common Voice [type: string]
    :param mcv_clips_path: diretório contendo áudios do Mozilla Common Voice [type: string]
    :param data_path: diretório de dados do projeto Voice Unlocker [type: string]
    :param n_mcv_files: quantidade de arquivos mp3 a serem copiados do MCV [type: int, default=10]
    :param label_prefix: prefixo de label no diretório do Voice Unlcoker [type: string, default='interlocutor_']
    :param new_folder: flag para criação de nova pasta de interlocutor [type: bool, default=False]
    
    Retorno
    -------
    Essa função não retorna nenhum parâmetro além da devida cópia dos arquivos do diretório
    Mozilla Common Voice para o diretório do projeto Voice Unlocker na pasta adequada de interlocutor
    """
    
    labels = os.listdir(data_path)
    if new_folder:
        # Definindo nomenclatura para nova pasta a ser criada
        qtd_labels = len(labels)
        if qtd_labels < 9:
            other_label = label_prefix + '0' + str(qtd_labels + 1)
        else:
            other_label = label_prefix + str(qtd_labels + 1)

        # Criando nova pasta
        print(f'Pastas presentes antes da criação: \n{os.listdir(data_path)}')
        os.mkdir(os.path.join(DATA_PATH, other_label))
        print(f'\nPastas presentes após a criação: \n{os.listdir(data_path)}')
    else:
        other_label = sorted(labels)[-1]
        print(f'Pastas presentes no diretório destino: \n{os.listdir(data_path)}')
    
    # Definindo diretório de destino baseado no label definido acima
    dst_path = os.path.join(data_path, other_label)
    print(f'\nDiretório destino das amostras de áudio da classe N-1:\n{dst_path}')
    
    # Lendo base de referência de dados a serem copiados
    mcv_train = pd.read_csv(mcv_train_path, sep='\t')
    mcv_train = mcv_train.head(n_mcv_files)
    mcv_train['src_path'] = mcv_train['path'].apply(lambda x: os.path.join(mcv_clips_path, x))
    mcv_train['dst_path'] = mcv_train['path'].apply(lambda x: os.path.join(dst_path, x))
    
    # Copiando arquivos
    for src, dst in mcv_train.loc[:, ['src_path', 'dst_path']].values:
        shutil.copy(src=src, dst=dst)

    # Validando cópia
    new_files = os.listdir(dst_path)
    print(f'\nQuantidade de novos arquivos copiados pra pasta do projeto: {len(new_files)}')
    
# Definindo função para leitura de arquivos de áudio em diretório do projeto
def read_data(data_path, sr, signal_col='signal', target_col='y_class'):
    """
    Leitura e armazenagem de arquivos de áudio e seus respectivos metadados
    
    Parâmetros
    ----------
    :param data_path: diretório alvo contendo as pastas para os interlocutores [type: string]
    :param sr: taxa de amostragem utilizada na leitura dos áudios [type: int]
    :param signal_col: referência da coluna de armazenamento do sinal [type: string, default='signal']
    :param target_col: referência da coluna de armazenamento do target [type: string, default='y_class']
    
    Retorno
    -------
    :return df: pandas DataFrame com áudios armazenados [type: pd.DataFrame]
    """
    
    # Extraindo informações dos sinais de áudio armazenados localmente
    roots = [root for root, dirs, files in os.walk(data_path)][1:]
    files = [files for root, dirs, files in os.walk(data_path)][1:]
    paths = [os.path.join(root, f) for root, file in zip(roots, files) for f in file]
    filenames = [p.split('/')[-1] for p in paths]
    file_formats = [f.split('.')[-1] for f in filenames]
    labels = [p.split('/')[-2] for p in paths]

    # Realizando a leitura dos sinais
    signals = [librosa.load(path, sr=sr)[0] for path in paths]
    durations = [librosa.get_duration(s) for s in signals]

    # Criando DataFrame para armazenagem de sinais
    df = pd.DataFrame()
    df['audio_path'] = paths
    df['filename'] = filenames
    df['file_format'] = file_formats
    df[signal_col] = signals
    df['duration'] = durations
    df['label_class'] = labels

    # Definindo variável resposta
    unique_class = df['label_class'].sort_values().unique()
    class_dict = {c: i for c, i in zip(unique_class, range(1, len(unique_class) + 1))}
    df[target_col] = df['label_class'].map(class_dict)

    return df

