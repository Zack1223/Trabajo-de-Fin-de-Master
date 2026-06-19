import tensorflow as tf
import numpy as np
import lime, lime.lime_text
import shap
import numpy as np
from lime.lime_text import LimeTextExplainer
model = tf.keras.models.load_model('LSTMModel.keras')

def lime_predict_fn(text):
    # 1. Convertir la lista de textos a un tensor/array compatible con Keras
    # Si tu capa 'vectorizer' ya está integrada al inicio del modelo, 
    # solo pasas el texto plano directamente al modelo.
    
    
    # 2. Obtener las predicciones del modelo LSTM
    predictions = model.predict(np.array(text,dtype=object), verbose=0)
    
    # 3. Formatear la salida según el tipo de clasificación:
    
    # CASO A: Si es Clasificación Binaria con activación 'sigmoid' (salida de 1 columna)
    if predictions.shape[1] == 1:
        # Reconstruimos la probabilidad de la clase 0 (1 - p) y de la clase 1 (p)
        probs = np.hstack([1 - predictions, predictions])
        return probs
        
    # CASO B: Si es Clasificación Multiclase con activación 'softmax'
    else:
        return predictions

def explicar(texto):
    # Definir los nombres de tus clases/etiquetas
    class_names = ['Anomalo', 'Normal']  # Cambia según tu problema

    # Inicializar el explicador de texto de LIME
    explainer = LimeTextExplainer(class_names=class_names)

    # Seleccionar el texto que deseas analizar
    # Generar la explicación local de la instancia
    exp = explainer.explain_instance(
        text_instance=texto, 
        classifier_fn=lime_predict_fn,  # Nuestra función personalizada
        num_features=6                  # Número de palabras clave a destacar
    )
    import webbrowser

# 1. Guardar la explicación como un archivo HTML local
    exp.save_to_file('explicacion_lime.html')

# 2. Abrir automáticamente el archivo en tu navegador web predeterminado
    webbrowser.open('explicacion_lime.html')

##explicar('http://localhost:8080/tienda1/index.jsp HTTP/1.1')
def Modelo_LSTM(lista_peticiones : list):
   

    print("Leyendo Peticiones")
    
    for pe in lista_peticiones:
       
        pe=pe.strip()
        pe="http://localhost:8080"+pe
        texto=np.array([pe],dtype=object)
        
        prediccion=model.predict(texto)
        if prediccion[0][0]<0.5:
            # 2. Mira cómo transforma tu texto a números
            
            print(f"Peticion anomala{prediccion[0][0]}")
        else:
            print(f"Peticion Normal{prediccion[0][0]}")
        explicar(pe)
    

        prediccion=0


