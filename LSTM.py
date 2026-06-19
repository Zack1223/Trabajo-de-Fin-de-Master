import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.layers import TextVectorization
import keras
import tensorflow
import pandas
# Descargar recursos
nltk.download('punkt')
nltk.download('stopwords')
df=pandas.read_csv('csic_database.csv')
X=df['URL'].values
Y=df['Result'].map({"Normal":1,"Anomalous":0})

# Datos
textos = X
etiquetas = Y

# Dividir
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=42)
vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=250)
vectorizer.adapt(X_train) 



# Modelo
model = Sequential([
    tensorflow.keras.Input(shape=(1,), dtype=tensorflow.string),
    vectorizer,
    Embedding(10000, 128, input_length=500,mask_zero=True),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(
    monitor='loss',     
    patience=3,              
    restore_best_weights=True 
)
# Entrenar y evaluar
model.fit(np.array(X_train), np.array(y_train), epochs=3,callbacks=[early_stop],validation_data=(X_val,y_val))
# Usar la función



print(model.evaluate(X_test, y_test))
def ListaPeticiones(fichero : str):
    log=open(fichero)
    content=log.read()
    lineas=content.split('\n')

    peticiones=[]
    for linea in lineas:
        import re
        aux=re.search(""" "([^"]*)" """,linea)
        if aux:
            peticiones.append(aux.group(1))
    return peticiones


Lista_Accesos=ListaPeticiones("access.log")

nuevos_textos = np.array(Lista_Accesos,dtype=object
)
predicciones = model.predict(nuevos_textos)

print(predicciones)
model.save("LSTMModel.keras")

