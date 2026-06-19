import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from transformers import TFAutoModelForSequenceClassification,AutoTokenizer,pipeline
import tensorflow
import numpy as np 
import pandas
from sklearn.model_selection import train_test_split


def Modelo_bert(lista_peticiones : list):
    tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
    modelo=TFAutoModelForSequenceClassification.from_pretrained(
    "./BertModel",
    num_labels = 2,
    )
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    p=pipeline("text-classification",model=modelo,tokenizer=tokenizer)
    print("Leyendo Peticiones")
    for pe in lista_peticiones:
        pe=pe.strip()
        pe="http://localhost:8080"+pe
        resultado=p(pe)
        if resultado[0]['label'] =='LABEL_0' and resultado[0]['score']>0.7:
            
            print(f"Petición sospechosa de ser anomala:{pe}  {resultado[0]['score']}")
        else:
            print(f"Peticion normal: {pe}")

        





