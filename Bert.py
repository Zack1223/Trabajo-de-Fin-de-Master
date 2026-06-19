
import numpy as np
from sklearn.model_selection import train_test_split


from transformers import  AutoTokenizer,TFAutoModelForSequenceClassification,pipeline
import pandas
import tensorflow
import keras

df=pandas.read_csv('csic_database.csv')
x=df['URL']
y=df['Result'].map({"Normal":1,"Anomalous":0})
# Datos

tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
x=x.astype(str).tolist()



# Dividir en 3 conjuntos train test y validacion

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

x_train = tokenizer(
    x_train,
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="tf"
)
x_test = tokenizer(
    x_test,
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="tf"
)
x_val = tokenizer(
    x_val,
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="tf"
)
y_train=np.array(y_train)
y_test=np.array(y_test)
y_val=np.array(y_val)
train_dataset=tensorflow.data.Dataset.from_tensor_slices((dict(x_train),y_train)).batch(32)
test_dataset=tensorflow.data.Dataset.from_tensor_slices((dict(x_test),y_test)).batch(32)
val_dataset=tensorflow.data.Dataset.from_tensor_slices((dict(x_val),y_val)).batch(32)

model = TFAutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2,

)

optimizer = tensorflow.keras.optimizers.Adam(learning_rate=2e-5)
loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3,
    batch_size=32,
    verbose=1
)
results=model.evaluate(test_dataset,verbose=0)
print(f"Porcentaje de acierto:{results[1]} | Perdida {results[0]}")


model.save_pretrained('./BertModel')










