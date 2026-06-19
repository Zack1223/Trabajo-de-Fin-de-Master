import nltk 
import re
from UsarModeloBERT import Modelo_bert
from UsarModeloLSTM import Modelo_LSTM
def ListaPeticiones(fichero : str):
    log=open(fichero)
    content=log.read()
    lineas=content.split('\n')

    peticiones=[]
    for linea in lineas:
        aux=re.search(""" "([^"]*)" """,linea)
        if aux:
            peticiones.append(aux.group(1))
    return peticiones
print("Bienvenido al detector de intrusiones ")
try :
    logfile=input("Introduzca el fichero de log : ")
    Lista_Accesos=ListaPeticiones(logfile)

    i=0
    while i<len(Lista_Accesos):
            Lista_Accesos[i]=Lista_Accesos[i].replace("GET","").replace("POST","")
            i+=1

        #print(Lista_Accesos)
    ModeloAUsar=0
    while ModeloAUsar!=1 and ModeloAUsar!= 2:
            ModeloAUsar=int(input("Introduzca el Modelo a Usar 1:BERT 2:LSTM:"))
    if ModeloAUsar == 1:
            Modelo_bert(Lista_Accesos)
    if ModeloAUsar == 2:
            Modelo_LSTM(Lista_Accesos)
except Exception as e:
    
    print(f"Error el Archivo no ha sido encontrado o no es un fichero log valido{e}")

    