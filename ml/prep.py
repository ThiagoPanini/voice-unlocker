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
2. Funções de leitura e enriquecimento de base
3. Funções de pré-processamento de dados
4. Funções auxiliares de extração de features
5. Classes transformadoras do pipeline
"""

# Autor: Thiago Panini
# Data de Criação: 29/03/2021

"""
------------------------------------------------------
------------ 1. IMPORTAÇÃO DE BIBLIOTECAS ------------
------------------------------------------------------ 
"""

import os 
import pandas as pd
import numpy as np
import shutil
import librosa
from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin


"""
------------------------------------------------------
--- 2. FUNÇÕES DE LEITURA E ENRIQUECIMENTO DE BASE ---
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


"""
------------------------------------------------------
------ 3. FUNÇÕES DE PRÉ-PROCESSAMENTO DE DADOS ------
------------------------------------------------------ 
"""

# Definindo função para pré processamento da base
def data_pre_processing(df, signal_col='signal', target_col='y_class', encoded_target=True):
    """
    Função responsável por filtrar as colunas utilizadas na preparação e aplicar o processo
    de encoding na variável resposta
    
    Parâmetros
    ----------
    :param df: base de dados original contendo informações de áudios [type: pd.DataFrame]
    :param signal_col: coluna de armazenamento do sinal temporal [type: string, default='signal']
    :param target_col: coluna de armazenamento da variável target [type: string, default='y_class']
    :param encoded_target: define a aplicação do encoding no array target [type: bool, default=True]
    
    Retorno
    -------
    :return X: base de dados contendo apenas o sinal de entrada dos áudios [type: pd.DataFrame]
    :return y: array multidimensional contendo informações sobre as classes [type: np.array]
    """
    
    # Filtrando dataset inicial
    X_df = df.loc[:, [signal_col]]
    y_df = df.loc[:, [target_col]]

    # Codificando variável target
    if encoded_target:
        y = pd.get_dummies(y_df[target_col]).values
    else:
        y = y_df.values.reshape(-1)

    return X_df, y


"""
------------------------------------------------------
--- 4. FUNÇÕES AUXILIARES DE EXTRAÇÃO DE FEATURES ----
------------------------------------------------------ 
"""

# Definindo função para separação de faixas de frequências (BER)
def calc_split_freq_bin(spec, split_freq, sr):
    """
    Função responsável por calcular o índice da frequência de separação F
    no espectro discreto de frequências

    Parâmetros
    ----------
    :param spec: espectrograma calculado via STFT [type: ndarray]
    :param split_freq: frequência de separação F [type: int]
    :param sr: taxa de amostragem do sinal [type: int]

    Retorno
    -------
    :return idx_split_freq: retorna o índice relacionado ao parâmetro F no espectro discreto [type: int]
    :return split_freq_bin: retorna a frequência discreta relacionada ao parâmetro F [type: float]
    """

    # Intervalo de frequências (Nyquist)
    f_range = sr / 2

    # Intervalo de frequências para cada faixa discreta individual
    qtd_freq_bins = spec.shape[0]
    f_delta_bin = f_range / qtd_freq_bins

    # Calculando índice do parâmetro F nas faixas discretas
    idx_split_freq = int(np.floor(split_freq / f_delta_bin))

    # Calculando faixa de frequência presente na matriz espectral
    freq_bins = np.linspace(0, f_range, qtd_freq_bins)
    split_freq_bin = freq_bins[idx_split_freq]

    return idx_split_freq, split_freq_bin

# Definindo função para o cálculo da Taxa de Energia de Banda (BER)
def calc_ber(spec, split_freq, sr):
    """
    Função responsável por calcular a taxa de energia de banda (BER)

    Parâmetros
    ----------
    :param spec: espectrograma calculado via STFT [type: ndarray]
    :param split_freq: frequência de separação F [type: int]
    :param sr: taxa de amostragem do sinal [type: int]

    Retorno
    -------
    :return ber: taxa de energia de banda para cada frame t [type: np.array]
    """

    # Calculando faixa de frequência discreta do parâmetro F
    idx_split_freq, split_freq_bin = calc_split_freq_bin(spec, split_freq, sr)
    bers = []

    # Transformando amplitudes do espectro em potências
    power_spec = np.abs(spec) ** 2

    # Aplicando transpose para iteração em cada frame
    power_spec = power_spec.T

    # Calculando somatório para cada frame
    for frame in power_spec:
        sum_power_low_freq = frame[:idx_split_freq].sum()
        sum_power_high_freq = frame[idx_split_freq:].sum()
        ber_frame = sum_power_low_freq / sum_power_high_freq
        bers.append(ber_frame)

    return np.array(bers)


"""
------------------------------------------------------
------- 5. CLASSES TRANSFORMADORAS DO PIPELINE -------
------------------------------------------------------ 
"""

# Definindo transformador para envelope de amplitude
class AmplitudeEnvelop(BaseEstimator, TransformerMixin):
    """
    Classe responsável por extrair o envelope de amplitude de sinais de áudio
    considerando agregados estatísticos pré definidos.

    Parâmetros
    ----------
    :param frame_size: quantidade de amostrar por enquadramento do sinal [type: int]
    :param hop_length: parâmetro de overlapping de quadros do sinal [type: int]
    :param signal_col: referência da coluna de armazenamento do sinal na base [type: string, default='signal']
    :param feature_aggreg: lista de agregadores estatísticos aplicados após a extração da features
        *default=['mean', 'median', 'std', 'var', 'max', 'min']

    Retorno
    -------
    :return X: base de dados contendo os agregados estatísticos para o envelope de amplitude [type: pd.DataFrame]

    Aplicação
    ---------
    ae_extractor = AmplitudeEnvelop(frame_size=FRAME_SIZE, hop_length=HOP_LENGTH, 
                                signal_col='signal', feature_aggreg=FEATURE_AGGREG)
    X_ae = ae_extractor.fit_transform(X)                          
    """
    
    def __init__(self, frame_size, hop_length, signal_col='signal',
                 feature_aggreg=['mean', 'median', 'std', 'var', 'max', 'min']):
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.signal_col = signal_col
        self.feature_aggreg = feature_aggreg
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        # Retornando o envelope de amplitude para cada frame do sinal
        X['ae'] = X[self.signal_col].apply(lambda x: np.array([max(x[i:i+self.frame_size]) for i in range(0, len(x), self.hop_length)]))
        
        # Criando dicionário com agregações do envelope de amplitude de cada sinal
        X['aggreg_dict'] = X['ae'].apply(lambda x: pd.DataFrame(x).agg(self.feature_aggreg))
        
        # Extraindo agregações e enriquecendo dataset
        for agg in self.feature_aggreg:
            X['ae_' + agg] = X['aggreg_dict'].apply(lambda x: x[0][agg])
            
        # Eliminando colunas adicionais
        X = X.drop(['ae', 'aggreg_dict'], axis=1)
            
        return X
    
# Definindo transformador para RMS Energy
class RMSEnergy(BaseEstimator, TransformerMixin):
    """
    Classe responsável por extrair a raíz da energia média quadrática de sinais de áudio
    considerando agregados estatísticos pré definidos.

    Parâmetros
    ----------
    :param frame_size: quantidade de amostrar por enquadramento do sinal [type: int]
    :param hop_length: parâmetro de overlapping de quadros do sinal [type: int]
    :param signal_col: referência da coluna de armazenamento do sinal na base [type: string, default='signal']
    :param feature_aggreg: lista de agregadores estatísticos aplicados após a extração da features
        *default=['mean', 'median', 'std', 'var', 'max', 'min']

    Retorno
    -------
    :return X: base de dados contendo os agregados estatísticos para a raíz da energia média quadrática [type: pd.DataFrame]

    Aplicação
    ---------
    rms_extractor = RMSEnergy(frame_size=FRAME_SIZE, hop_length=HOP_LENGTH, 
                              signal_col='signal', feature_aggreg=FEATURE_AGGREG)
    X_rms = rms_extractor.fit_transform(X)
    """
    
    def __init__(self, frame_size, hop_length, signal_col='signal',
                 feature_aggreg=['mean', 'median', 'std', 'var', 'max', 'min']):
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.signal_col = signal_col
        self.feature_aggreg = feature_aggreg
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        # Extraindo feature para cada sinal
        X['rms_engy'] = X[self.signal_col].apply(lambda x: librosa.feature.rms(x, frame_length=self.frame_size, 
                                                                               hop_length=self.hop_length)[0])
        
        # Criando dicionário com agregações
        X['aggreg_dict'] = X['rms_engy'].apply(lambda x: pd.DataFrame(x).agg(self.feature_aggreg))
        
        # Extraindo agregações e enriquecendo dataset
        for agg in self.feature_aggreg:
            X['rms_engy_' + agg] = X['aggreg_dict'].apply(lambda x: x[0][agg])
            
        # Eliminando colunas adicionais
        X = X.drop(['rms_engy', 'aggreg_dict'], axis=1)
            
        return X
    
# Definindo transformador para Zero Crossing Rate
class ZeroCrossingRate(BaseEstimator, TransformerMixin):
    """
    Classe responsável por extrair a taxa de cruzamento de zero de sinais de áudio
    considerando agregados estatísticos pré definidos.

    Parâmetros
    ----------
    :param frame_size: quantidade de amostrar por enquadramento do sinal [type: int]
    :param hop_length: parâmetro de overlapping de quadros do sinal [type: int]
    :param signal_col: referência da coluna de armazenamento do sinal na base [type: string, default='signal']
    :param feature_aggreg: lista de agregadores estatísticos aplicados após a extração da features
        *default=['mean', 'median', 'std', 'var', 'max', 'min']

    Retorno
    -------
    :return X: base de dados contendo os agregados estatísticos para a taxa de cruzamento de zero [type: pd.DataFrame]

    Aplicação
    ---------
    zcr_extractor = ZeroCrossingRate(frame_size=FRAME_SIZE, hop_length=HOP_LENGTH, 
                                     signal_col='signal', feature_aggreg=FEATURE_AGGREG)
    X_zcr = zcr_extractor.fit_transform(X)
    """
    
    def __init__(self, frame_size, hop_length, signal_col='signal',
                 feature_aggreg=['mean', 'median', 'std', 'var', 'max', 'min']):
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.signal_col = signal_col
        self.feature_aggreg = feature_aggreg
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        # Extraindo feature para cada sinal
        X['zcr'] = X[self.signal_col].apply(lambda x: librosa.feature.zero_crossing_rate(x, frame_length=self.frame_size, 
                                                                                         hop_length=self.hop_length)[0])
        
        # Criando dicionário com agregações
        X['aggreg_dict'] = X['zcr'].apply(lambda x: pd.DataFrame(x).agg(self.feature_aggreg))
        
        # Extraindo agregações e enriquecendo dataset
        for agg in self.feature_aggreg:
            X['zcr_' + agg] = X['aggreg_dict'].apply(lambda x: x[0][agg])
            
        # Eliminando colunas adicionais
        X = X.drop(['zcr', 'aggreg_dict'], axis=1)
            
        return X

# Definindo transformador para BER
class BandEnergyRatio(BaseEstimator, TransformerMixin):
    """
    Classe responsável por extrair a taxa de energia de banda de sinais de áudio
    considerando agregados estatísticos pré definidos.

    Parâmetros
    ----------
    :param frame_size: quantidade de amostrar por enquadramento do sinal [type: int]
    :param hop_length: parâmetro de overlapping de quadros do sinal [type: int]
    :param split_freq: frequência de separação entre altas e baixas frequências [type: int]
    :param sr: taxa de amostragem do sinal de áudio [type: int]
    :param signal_col: referência da coluna de armazenamento do sinal na base [type: string, default='signal']
    :param feature_aggreg: lista de agregadores estatísticos aplicados após a extração da features
        *default=['mean', 'median', 'std', 'var', 'max', 'min']

    Retorno
    -------
    :return X: base de dados contendo os agregados estatísticos para a taxa de energia de banda [type: pd.DataFrame]

    Aplicação
    ---------
    ber_extractor = BandEnergyRatio(frame_size=FRAME_SIZE, hop_length=HOP_LENGTH, 
                                    signal_col='signal', feature_aggreg=FEATURE_AGGREG)
    X_ber = ber_extractor.fit_transform(X)
    """
    
    def __init__(self, frame_size, hop_length, split_freq, sr, signal_col='signal',
                 feature_aggreg=['mean', 'median', 'std', 'var', 'max', 'min']):
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.split_freq = split_freq
        self.sr = sr
        self.signal_col = signal_col
        self.feature_aggreg = feature_aggreg
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        # Calculando espectrograma dos sinais
        X['spec'] = X[self.signal_col].apply(lambda x: librosa.stft(y=x, n_fft=self.frame_size, 
                                                                    hop_length=self.hop_length))
        
        # Calculando BER
        X['ber'] = X['spec'].apply(lambda x: calc_ber(spec=x, split_freq=self.split_freq, sr=self.sr))
        
        # Criando dicionário com agregações
        X['aggreg_dict'] = X['ber'].apply(lambda x: pd.DataFrame(x).agg(self.feature_aggreg))
        
        # Extraindo agregações e enriquecendo dataset
        for agg in self.feature_aggreg:
            X['ber_' + agg] = X['aggreg_dict'].apply(lambda x: x[0][agg])
            
        # Eliminando colunas adicionais
        X = X.drop(['spec', 'ber', 'aggreg_dict'], axis=1)
            
        return X
    
# Definindo transformador para Spectral Centroid
class SpectralCentroid(BaseEstimator, TransformerMixin):
    """
    Classe responsável por extrair o centroide espectral de sinais de áudio
    considerando agregados estatísticos pré definidos.

    Parâmetros
    ----------
    :param frame_size: quantidade de amostrar por enquadramento do sinal [type: int]
    :param hop_length: parâmetro de overlapping de quadros do sinal [type: int]
    :param sr: taxa de amostragem do sinal de áudio [type: int]
    :param signal_col: referência da coluna de armazenamento do sinal na base [type: string, default='signal']
    :param feature_aggreg: lista de agregadores estatísticos aplicados após a extração da features
        *default=['mean', 'median', 'std', 'var', 'max', 'min']

    Retorno
    -------
    :return X: base de dados contendo os agregados estatísticos para o centroide espectral [type: pd.DataFrame]

    Aplicação
    ---------
    sc_extractor = SpectralCentroid(frame_size=FRAME_SIZE, hop_length=HOP_LENGTH, 
                                     signal_col='signal', feature_aggreg=FEATURE_AGGREG)
    X_sc = sc_extractor.fit_transform(X)
    """
    
    def __init__(self, frame_size, hop_length, sr, signal_col='signal',
                 feature_aggreg=['mean', 'median', 'std', 'var', 'max', 'min']):
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.sr = sr
        self.signal_col = signal_col
        self.feature_aggreg = feature_aggreg
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        # Calculando feature
        X['sc'] = X[self.signal_col].apply(lambda x: librosa.feature.spectral_centroid(y=x, sr=self.sr,
                                                                                       n_fft=self.frame_size,
                                                                                       hop_length=self.hop_length)[0])
        
        # Criando dicionário com agregações
        X['aggreg_dict'] = X['sc'].apply(lambda x: pd.DataFrame(x).agg(self.feature_aggreg))
        
        # Extraindo agregações e enriquecendo dataset
        for agg in self.feature_aggreg:
            X['sc_' + agg] = X['aggreg_dict'].apply(lambda x: x[0][agg])
            
        # Eliminando colunas adicionais
        X = X.drop(['sc', 'aggreg_dict'], axis=1)
            
        return X
    
# Definindo transformador para BandWidth
class BandWidth(BaseEstimator, TransformerMixin):
    """
    Classe responsável por extrair a largura de banda de sinais de áudio
    considerando agregados estatísticos pré definidos.

    Parâmetros
    ----------
    :param frame_size: quantidade de amostrar por enquadramento do sinal [type: int]
    :param hop_length: parâmetro de overlapping de quadros do sinal [type: int]
    :param sr: taxa de amostragem do sinal de áudio [type: int]
    :param signal_col: referência da coluna de armazenamento do sinal na base [type: string, default='signal']
    :param feature_aggreg: lista de agregadores estatísticos aplicados após a extração da features
        *default=['mean', 'median', 'std', 'var', 'max', 'min']

    Retorno
    -------
    :return X: base de dados contendo os agregados estatísticos para a largura de banda [type: pd.DataFrame]

    Aplicação
    ---------
    bw_extractor = BandWidth(frame_size=FRAME_SIZE, hop_length=HOP_LENGTH, 
                             signal_col='signal', feature_aggreg=FEATURE_AGGREG)
    X_bw = bw_extractor.fit_transform(X)
    """
    
    def __init__(self, frame_size, hop_length, sr, signal_col='signal',
                 feature_aggreg=['mean', 'median', 'std', 'var', 'max', 'min']):
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.sr = sr
        self.signal_col = signal_col
        self.feature_aggreg = feature_aggreg
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        # Calculando feature
        X['bw'] = X[self.signal_col].apply(lambda x: librosa.feature.spectral_bandwidth(y=x, sr=self.sr,
                                                                                        n_fft=self.frame_size,
                                                                                        hop_length=self.hop_length)[0])
        
        # Criando dicionário com agregações
        X['aggreg_dict'] = X['bw'].apply(lambda x: pd.DataFrame(x).agg(self.feature_aggreg))
        
        # Extraindo agregações e enriquecendo dataset
        for agg in self.feature_aggreg:
            X['bw_' + agg] = X['aggreg_dict'].apply(lambda x: x[0][agg])
            
        # Eliminando colunas adicionais
        X = X.drop(['bw', 'aggreg_dict'], axis=1)
            
        return X
    
# Definindo transformador para agregação de espectrograma em grupos
class GroupSpecAggreg(BaseEstimator, TransformerMixin):
    """
    Classe responsável por extrair a potência espectral de altas e baixas frequências
    de sinais de áudio considerando agregados estatísticos pré definidos.

    Parâmetros
    ----------
    :param frame_size: quantidade de amostrar por enquadramento do sinal [type: int]
    :param hop_length: parâmetro de overlapping de quadros do sinal [type: int]
    :param sr: taxa de amostragem do sinal de áudio [type: int]
    :param split_freq: frequência de separação entre altas e baixas frequências [type: int]
    :param freq_cat_aggreg: agregador aplicado no agrupamento das potências [type: int, default='sum']
    :param signal_col: referência da coluna de armazenamento do sinal na base [type: string, default='signal']
    :param feature_aggreg: lista de agregadores estatísticos aplicados após a extração da features
        *default=['mean', 'median', 'std', 'var', 'max', 'min']

    Retorno
    -------
    :return X: base de dados contendo os agregados estatísticos para a potência espectral agrupada [type: pd.DataFrame]

    Aplicação
    ---------
    spec_extractor = GroupSpecAggreg(frame_size=FRAME_SIZE, hop_length=HOP_LENGTH, 
                                     signal_col='signal', feature_aggreg=FEATURE_AGGREG)
    X_spec = spec_extractor.fit_transform(X)
    """
    
    def __init__(self, frame_size, hop_length, sr, split_freq, freq_cat_aggreg='sum',
                 signal_col='signal', feature_aggreg=['mean', 'median', 'std', 'var', 'max', 'min']):
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.sr = sr
        self.split_freq = split_freq
        self.freq_cat_aggreg = freq_cat_aggreg
        self.signal_col = signal_col
        self.feature_aggreg = feature_aggreg
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        # Criando DataFrame vazio e aplicando STFT no sinal
        all_spec_agg = pd.DataFrame()
        X['spec'] = X[self.signal_col].apply(lambda x: np.abs(librosa.stft(y=x, n_fft=self.frame_size, 
                                                                           hop_length=self.hop_length))**2)
        
        idx_split, split_freq = calc_split_freq_bin(X['spec'][0], split_freq=self.split_freq, sr=self.sr)
        frequency_bins = np.linspace(0, self.sr/2, 1025)
        
        # Iterando sobre cada espectrograma de cada sinal
        for spec in X['spec']:
            # DataFrame intermediário para agregações de cada sinal
            signal_spec_agg = pd.DataFrame()
            i = 0
            
            # Separando frequências de acordo com threshold estabelecido
            spec_data = pd.DataFrame(spec)
            spec_data.reset_index(inplace=True)
            spec_data['freq_cat'] = spec_data['index'].apply(lambda x: 'low_freq_pwr' if x <= idx_split else 'high_freq_pwr')
            
            # Somando potências de baixas e altas frequências
            spec_data_sum = spec_data.groupby(by='freq_cat').agg(self.freq_cat_aggreg)
            spec_data_sum.drop('index', axis=1, inplace=True)
            
            # Agregando resultado agregado separado por grupo de frequências
            S_aggreg = pd.DataFrame(spec_data_sum).agg(self.feature_aggreg, axis=1)
            #print(spec_data_sum)
            #print(S_aggreg)

            # Iterando sobre cada agregador para gerar um novo DataFrame
            for agg in self.feature_aggreg:
                S_agg = pd.DataFrame(S_aggreg[agg]).T
                S_agg.reset_index(inplace=True, drop=True)
                S_agg.columns = [col + '_' + agg for col in S_agg.columns]
                
                # Unindo agregadores em DataFrame intermediário do sinal
                if i == 0:
                    signal_spec_agg = S_agg.copy()
                else:
                    signal_spec_agg = signal_spec_agg.merge(S_agg, left_index=True, right_index=True)
                i += 1
            
            # Empilhando compilado agregado de cada sinal
            all_spec_agg = all_spec_agg.append(signal_spec_agg)
        
        # Enriquecendo dataset com agregações geradas
        all_spec_agg.reset_index(inplace=True, drop=True)
        X = X.merge(all_spec_agg, left_index=True, right_index=True)
        
        # Dropando colunas auxiliares
        X.drop('spec', axis=1, inplace=True)
        
        return X
    
# Definindo transformador para agregação individual de espectrograma
class MFCCsAggreg(BaseEstimator, TransformerMixin):
    """
    Classe responsável por extrair as componentes MFCCs (primeira e segunda derivada)
    de sinais de áudio considerando agregados estatísticos pré definidos.

    Parâmetros
    ----------
    :param n_mfcc: quantidade de componentes MFCCs extraídas [type: int]
    :param order: ordem das derivadas extraídas [type: int, default=0]
    :param signal_col: referência da coluna de armazenamento do sinal na base [type: string, default='signal']
    :param feature_aggreg: lista de agregadores estatísticos aplicados após a extração da features
        *default=['mean', 'median', 'std', 'var', 'max', 'min']

    Retorno
    -------
    :return X: base de dados contendo os agregados estatísticos para as componentes MFCCs [type: pd.DataFrame]

    Aplicação
    ---------
    mfcc_extractor = MFCCsAggreg(frame_size=FRAME_SIZE, hop_length=HOP_LENGTH, 
                                 signal_col='signal', feature_aggreg=FEATURE_AGGREG)
    X_mfcc = mfcc_extractor.fit_transform(X)
    """
    
    def __init__(self, n_mfcc, order=0, signal_col='signal',
                 feature_aggreg=['mean', 'median', 'std', 'var', 'max', 'min']):
        self.n_mfcc = n_mfcc
        self.order = order
        self.signal_col = signal_col
        self.feature_aggreg = feature_aggreg
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        # Criando DataFrame vazio e retornando mfccs
        all_mfcc_agg = pd.DataFrame()
        X['mfcc'] = X[self.signal_col].apply(lambda x: librosa.feature.mfcc(y=x, n_mfcc=self.n_mfcc))
        
        # Calculando derivadas (se aplicáveis)
        if self.order == 1:
            X['d1_mfcc'] = X['mfcc'].apply(lambda x: librosa.feature.delta(x, order=1))
        
        if self.order == 2:
            X['d1_mfcc'] = X['mfcc'].apply(lambda x: librosa.feature.delta(x, order=1))
            X['d2_mfcc'] = X['mfcc'].apply(lambda x: librosa.feature.delta(x, order=2))
        
        # Iterando sobre cada conjunto de coeficientes mfccs de cada sinal
        if self.order == 0:
            for mfcc in X['mfcc']:
                # DataFrame intermediário para agregações de cada sinal
                signal_mfcc_agg = pd.DataFrame()
                i = 0

                # Agregando dimensão temporal do espectrograma (eixo 1)
                M_aggreg = pd.DataFrame(mfcc).agg(self.feature_aggreg, axis=1)
                M_aggreg.index = ['mfcc_c' + str(i) for i in range(1, self.n_mfcc + 1)]

                # Iterando sobre cada agregador para gerar um novo DataFrame
                for agg in self.feature_aggreg:
                    M_agg = pd.DataFrame(M_aggreg[agg]).T
                    M_agg.reset_index(inplace=True, drop=True)
                    M_agg.columns = [col + '_' + agg for col in M_agg.columns]

                    # Unindo agregadores em DataFrame intermediário do sinal
                    if i == 0:
                        signal_mfcc_agg = M_agg.copy()
                    else:
                        signal_mfcc_agg = signal_mfcc_agg.merge(M_agg, left_index=True, right_index=True)
                    i += 1

                # Empilhando compilado agregado de cada sinal
                all_mfcc_agg = all_mfcc_agg.append(signal_mfcc_agg)
                to_drop = ['mfcc']
                
        elif self.order == 1:
            for mfcc, d1_mfcc in X.loc[:, ['mfcc', 'd1_mfcc']].values:
                # DataFrame intermediário para agregações de cada sinal
                signal_mfcc_agg = pd.DataFrame()
                i = 0

                # Agregando dimensão temporal do espectrograma (eixo 1)
                M_aggreg_d0 = pd.DataFrame(mfcc).agg(self.feature_aggreg, axis=1)
                M_aggreg_d0.index = ['mfcc_c' + str(i) for i in range(1, self.n_mfcc + 1)]
                
                M_aggreg_d1 = pd.DataFrame(d1_mfcc).agg(self.feature_aggreg, axis=1)
                M_aggreg_d1.index = ['d1_mfcc_c' + str(i) for i in range(1, self.n_mfcc + 1)]

                # Iterando sobre cada agregador para gerar um novo DataFrame
                for agg in self.feature_aggreg:
                    M_agg_d0 = pd.DataFrame(M_aggreg_d0[agg]).T
                    M_agg_d0.reset_index(inplace=True, drop=True)
                    M_agg_d0.columns = [col + '_' + agg for col in M_agg_d0.columns]
                                       
                    M_agg_d1 = pd.DataFrame(M_aggreg_d1[agg]).T
                    M_agg_d1.reset_index(inplace=True, drop=True)
                    M_agg_d1.columns = [col + '_' + agg for col in M_agg_d1.columns]
                    
                    M_agg = M_agg_d0.merge(M_agg_d1, left_index=True, right_index=True)

                    # Unindo agregadores em DataFrame intermediário do sinal
                    if i == 0:
                        signal_mfcc_agg = M_agg.copy()
                    else:
                        signal_mfcc_agg = signal_mfcc_agg.merge(M_agg, left_index=True, right_index=True)
                    i += 1

                # Empilhando compilado agregado de cada sinal
                all_mfcc_agg = all_mfcc_agg.append(signal_mfcc_agg)
                to_drop = ['mfcc', 'd1_mfcc']
                
        elif self.order == 2:
            for mfcc, d1_mfcc, d2_mfcc in X.loc[:, ['mfcc', 'd1_mfcc', 'd2_mfcc']].values:
                # DataFrame intermediário para agregações de cada sinal
                signal_mfcc_agg = pd.DataFrame()
                i = 0

                # Agregando dimensão temporal do espectrograma (eixo 1)
                M_aggreg_d0 = pd.DataFrame(mfcc).agg(self.feature_aggreg, axis=1)
                M_aggreg_d0.index = ['mfcc_c' + str(i) for i in range(1, self.n_mfcc + 1)]
                
                M_aggreg_d1 = pd.DataFrame(d1_mfcc).agg(self.feature_aggreg, axis=1)
                M_aggreg_d1.index = ['d1_mfcc_c' + str(i) for i in range(1, self.n_mfcc + 1)]
                
                M_aggreg_d2 = pd.DataFrame(d2_mfcc).agg(self.feature_aggreg, axis=1)
                M_aggreg_d2.index = ['d2_mfcc_c' + str(i) for i in range(1, self.n_mfcc + 1)]

                # Iterando sobre cada agregador para gerar um novo DataFrame
                for agg in self.feature_aggreg:
                    M_agg_d0 = pd.DataFrame(M_aggreg_d0[agg]).T
                    M_agg_d0.reset_index(inplace=True, drop=True)
                    M_agg_d0.columns = [col + '_' + agg for col in M_agg_d0.columns]
                                       
                    M_agg_d1 = pd.DataFrame(M_aggreg_d1[agg]).T
                    M_agg_d1.reset_index(inplace=True, drop=True)
                    M_agg_d1.columns = [col + '_' + agg for col in M_agg_d1.columns]
                    
                    M_agg_d2 = pd.DataFrame(M_aggreg_d2[agg]).T
                    M_agg_d2.reset_index(inplace=True, drop=True)
                    M_agg_d2.columns = [col + '_' + agg for col in M_agg_d2.columns]
                    
                    M_agg = M_agg_d0.merge(M_agg_d1, left_index=True, right_index=True)
                    M_agg = M_agg.merge(M_agg_d2, left_index=True, right_index=True)

                    # Unindo agregadores em DataFrame intermediário do sinal
                    if i == 0:
                        signal_mfcc_agg = M_agg.copy()
                    else:
                        signal_mfcc_agg = signal_mfcc_agg.merge(M_agg, left_index=True, right_index=True)
                    i += 1

                # Empilhando compilado agregado de cada sinal
                all_mfcc_agg = all_mfcc_agg.append(signal_mfcc_agg)
                to_drop = ['mfcc', 'd1_mfcc', 'd2_mfcc']
        
        # Enriquecendo dataset com agregações geradas
        all_mfcc_agg.reset_index(inplace=True, drop=True)
        X = X.merge(all_mfcc_agg, left_index=True, right_index=True)
        
        # Dropando colunas auxiliares
        X.drop(to_drop, axis=1, inplace=True)
        
        return X

