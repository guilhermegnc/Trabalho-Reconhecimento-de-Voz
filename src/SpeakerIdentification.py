from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras import backend as K
from matplotlib import pyplot
from keras.callbacks import EarlyStopping
import keras
from keras import layers
from keras import models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import os
import csv
import librosa
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
#from ann_visualizer.visualize import ann_viz;

CREATE_CSV_FILES = False    # Se True, irá criar os arquivos CSV

# Define o nome dos arquivos CSV
TRAIN_CSV_FILE = "train.csv"
TEST_CSV_FILE = "test.csv"
MORE_TRAIN_CSV_FILE = "more_train.csv"
MORE_TEST_CSV_FILE = "more_test.csv"


def extractWavFeatures(soundFilesFolder, csvFileName):
    print("As features do arquivo na pasta " +
          soundFilesFolder+" serao salvos em "+csvFileName)
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    print('CSV Header: ', header)
    file = open(csvFileName, 'w', newline='')
    # with file:
    writer = csv.writer(file)
    writer.writerow(header)
    genres = '1 2 3 4 5 6 7 8 9 0'.split()
    for filename in os.listdir(soundFilesFolder):
        number = f'{soundFilesFolder}/{filename}'
        y, sr = librosa.load(number, mono=True, duration=30)
        # remove leading and trailing silence
        y, index = librosa.effects.trim(y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        writer.writerow(to_append.split())
    file.close()
    print("Fim de extractWavFeatures")


if (CREATE_CSV_FILES == True):
    extractWavFeatures("../data/recordings/train", TRAIN_CSV_FILE)
    extractWavFeatures("../data/recordings/test", TEST_CSV_FILE)
    extractWavFeatures(
        "../data/recordings/moreSpeakersTrain", MORE_TRAIN_CSV_FILE)
    extractWavFeatures(
        "../data/recordings/moreSpeakersTest", MORE_TEST_CSV_FILE)
    print("Arquivos CSV sao criados")
else:
    print("Criacao dos arquivos CSV foi pulada")

# =====================================================================================
print("\nNomes dos audios no dataset original: jackson, nicolas, theo\n")
# AUDIOS ADICIONAIS ESTÃO NAS PASTAS COMEÇADAS COM "MORE" EM Recordings

flag = (input("Houve a adicao de algum audio fora do dataset?    [1] - Sim   [# != 1] - Nao\n"))
flag2 = (input("Houve modificacao de algum arquivo original do dataset?     [1] - Sim   [# != 1] - Nao\n"))
if flag == "1" or flag2 == "1":
    names = []
    if flag2 != "1":
        names = ["jackson", "nicolas", "theo"]
    while True:
            if flag2 != "1":
                name = (input(
                    "Entre com o nome da pessoa no arquivo de audio adicionado.\n [-1] Caso ja tenha adicionado todos\n"))
                if name != "-1":
                    names.append(name.lower())
                else:
                    break
            else:
                name = (input(
                    "Entre com o nome de cada pessoa nos arquivos de audio.\n [-1] Caso ja tenha adicionado todos\n"))
                if name != "-1":
                    names.append(name.lower())
                else:
                    break
else:
    names = ["jackson", "nicolas", "theo"]
print(len(names))

# Lê o dataset e converte para o respectivo número


def preProcessData(csvFileName):
    print(csvFileName + " sera pre-processado")
    data = pd.read_csv(csvFileName)
    # Há três pessoas no datase original:
    # 0: Jackson
    # 1: Nicolas
    # 2: Theo
    filenameArray = data['filename']
    speakerArray = []
    # print(filenameArray)
    for i in range(len(filenameArray)):
        speaker = ""
        # print(filenameArray[i])
        for j in range(len(names)):
            if names[j] in filenameArray[i]:
                speaker = str(j)
                break
        # print(speaker)
        speakerArray.append(speaker)
    data['number'] = speakerArray
    # Tirando colunas desnecessarias
    data = data.drop(['filename'], axis=1)
    data = data.drop(['label'], axis=1)
    data = data.drop(['chroma_stft'], axis=1)
    data.shape

    print("Pre-processamento terminou")
    print(data.head())
    return data


trainData = preProcessData(TRAIN_CSV_FILE)
testData = preProcessData(TEST_CSV_FILE)
moreTrainData = preProcessData(MORE_TRAIN_CSV_FILE)
moreTestData = preProcessData(MORE_TEST_CSV_FILE)

# ========================================================================================

# Dividindo o dataset para treinamento, validacao e teste
X = np.array(trainData.iloc[:, :-1], dtype=float)
y = trainData.iloc[:, -1]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42)


X_test = np.array(testData.iloc[:, :-1], dtype=float)
y_test = testData.iloc[:, -1]

print("Y do dado de treinamento:", y_train.shape)
print("Y do dado de validacao:", y_val.shape)
print("Y do dado de teste:", y_test.shape)

# ==========================================================================================

# Normalizando o dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("X do dado de treinamento", X_train.shape)
print("X do dado de validacao", X_val.shape)
print("X do dado de teste", X_test.shape)

# ==============================================================================

# Criando o modelo

# modelo 1
model = models.Sequential()
model.add(layers.Dense(256, activation='relu',
                       input_shape=(X_train.shape[1],)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Processo de aprendizado do modelo 1
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

# Treinamento com EarlyStopping para evitar overfitting
history = model.fit(X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=128,
                    callbacks=[es])

# =====================================================================================

# plota a variavel history do treinamento
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# ======================================================================================

# Funcoes auxiliares para mostrar os resultados


def getSpeaker(speaker):
    speaker = int(speaker)
    if speaker >= len(names):
        speaker = "Desconhecido"
        return "Desconhecido"
    else:
        return names[speaker]


def printPrediction(X_data, y_data, printDigit):
    print('\n# Gerando predicoes')
    for i in range(len(y_data)):
        prediction = getSpeaker(model.predict_classes(X_data[i:i+1])[0])
        speaker = getSpeaker(y_data[i])
        if printDigit == True:
            print("Numero={0:d}, y={1:10s}- predicao={2:10s}- match={3}".format(i,
                                                                                speaker, prediction, speaker == prediction))
        else:
            print("y={0:10s}- predicao={1:10s}- match={2}".format(speaker,
                                                                  prediction, speaker == prediction))


# =============================================================================================================================


def report(X_data, y_data):
    # Matriz de confusão
    Y_pred = model.predict_classes(X_data)
    y_test_num = y_data.astype(np.int64)
    conf_mt = confusion_matrix(y_test_num, Y_pred)
    print(conf_mt)
    plt.matshow(conf_mt)
    plt.show()
    print('\nClassificacao')
    target_names = names.append("Desconhecido")
    print(classification_report(y_test_num, Y_pred))


# ===============================================================================================

# Desempenho do modelo

print('\n# Dado de teste #\n')
score = model.evaluate(X_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# Predicao
printPrediction(X_test[0:10], y_test[0:10], False)

print("Classificacao para dado de teste\n")
report(X_test, y_test)

# Dividindo o dataset para treinamento, validacao e teste

fullTrainData = trainData.append(moreTrainData)

X = np.array(fullTrainData.iloc[:, :-1], dtype=float)
y = fullTrainData.iloc[:, -1]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42)

X_test = np.array(testData.iloc[:, :-1], dtype=float)
y_test = testData.iloc[:, -1]

X_more_test = np.array(moreTestData.iloc[:, :-1], dtype=float)
y_more_test = moreTestData.iloc[:, -1]

print("Y do dado de treinamento:", y_train.shape)
print("Y do dado de validacao:", y_val.shape)
print("Y do dado de teste:", y_test.shape)
print("Y do dado de teste das outras pessoas:", y_more_test.shape)


# Normalizando o dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_more_test = scaler.transform(X_more_test)

print("X do dado de treinamento:", X_train.shape)
print("X do dado de validacao:", X_val.shape)
print("X do dado de teste:", X_test.shape)
print("X do dado de teste das outras pessoas:", X_more_test.shape)


# Criando o modelo


# modelo 1
model = models.Sequential()
model.add(layers.Dense(256, activation='relu',
                       input_shape=(X_train.shape[1],)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Processo de aprendizado do modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

# Treinamento com EarlyStopping para evitar overfitting
history = model.fit(X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=128,
                    callbacks=[es])

# plota o history do treinamento
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# =============================================================================

# Mostra o desempenho do modelo

print('\n# DADOS DE TESTE #\n')
score = model.evaluate(X_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# Predicao
printPrediction(X_test[0:10], y_test[0:10], False)


print('\n# DADOS DAS OUTRAS PESSOAS #\n')
score = model.evaluate(X_more_test, y_more_test)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# Predicao
printPrediction(X_more_test[0:10], y_more_test[0:10], False)

print("Classificacao\n")
report(X_test, y_test)

print("Classificacao para as outras pessoas\n")
report(X_more_test, y_more_test)

#ann_viz(model, title="Neural network")

allowed = []
while True:
    name2 = (input(
                    "Entre com o nome de todas as pessoas autorizadas.\n [-1] Caso ja tenha adicionado todos\n"))
    if name2 != "-1":
        allowed.append(name2.lower())
    else:
        break
        
def printAllowedOrNot(X_data, y_data):
    for i in range(len(y_data)):
        prediction = getSpeaker(model.predict_classes(X_data[i:i+1])[0])
        allow = False
        correct = False
        speaker = getSpeaker(y_data[i])
        for j in range(len(allowed)):
            if prediction in allowed[j]:
                allow = True
                break
        print("speaker={0:10s}- predicao={1:10s}- autorizacao={2}\t- autorizacao correta={3}".format(speaker, prediction, allow, speaker in allowed))

print("\nPermitidos: ")        
printAllowedOrNot(X_test, y_test)
print("\n\nPermitidos das demais pessoas: ")
printAllowedOrNot(X_more_test, y_more_test)
