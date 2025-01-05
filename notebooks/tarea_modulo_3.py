import json
import nltk
import re
import string
import matplotlib.pyplot as plt

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline


# Lista para almacenar los valores de "text"
texts = []

def leer_jsonl():
  # Leer el archivo JSONL
  with open('Gift_Cards_reviews.jsonl', 'r') as file:
      for line in file:
          review = json.loads(line.strip())  # Eliminar el salto de línea
          texts.append(review["text"])  # Almacenar el valor de "text" en la lista

# Convertir texto a tokens
def tokenizacion(texts):
  return [word_tokenize(i) for i in texts]

# Eliminar palabras comunes que no aportan un significado significativo
def stop_words(lema):

  stop_words_texts = [] 

  # Cargar las stop words en ingles
  stop_words = set(stopwords.words('english'))

  # Imprimir la lista de textos
  tokens = [word_tokenize(i) for i in lema]
  for i in tokens:
    # Filtrar las stop words
    tokens_filtrados = [word for word in i if word.lower() not in stop_words]
    stop_words_texts.append(" ".join(tokens_filtrados))
  return stop_words_texts

# Normalizar texto
def normalizacion_texto(texto_stop_words):

  texto_normalizado = []

  # Imprimir la lista de textos
  texto = tokenizacion(texto_stop_words)

  for i in texto:
    nuevo_texto = " ".join(i).lower()

    # Eliminación de puntuación
    nuevo_texto = nuevo_texto.translate(str.maketrans('', '', string.punctuation + '¡¿'))

    # Eliminación de números
    nuevo_texto = re.sub(r'\d+', '', nuevo_texto)

    # Eliminación de espacios en blanco adicionales
    nuevo_texto = re.sub(r'\s+', ' ', nuevo_texto).strip()

    texto_normalizado.append(nuevo_texto)

  return texto_normalizado

# Lematizar texto
def lematizacion(texto_normalizado):
  # Inicializar wordnet lemmatizer
  wnl = WordNetLemmatizer()

  lemmatized_texts = []
  for line in texto_normalizado:  # Iterar a través de la lista de textos tokenizados
      lemmatized_line = []
      for word in word_tokenize(line):
          lemmatized_line.append(wnl.lemmatize(word, pos="v"))
      lemmatized_texts.append(" ".join(lemmatized_line))

  return lemmatized_texts

# Análisis de Sentimiento
def analisis_sentimiento():

  # Procesamiento de Texto
  texto_normalizado = normalizacion_texto(texts)
  lema = lematizacion(texto_normalizado)
  texto_stop_words = stop_words(lema)

  # Cargar el modelo y el tokenizador BERT preentrenado
  tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
  model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

  # Crear un pipeline de clasificación de texto
  clasificador = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

  #Preparo las clasificaciones para hacer la construir la gráfica
  tags = ['Insatisfecho','Neutral','Satisfecho']
  bert_values = [0,0,0]  

  for line in texto_normalizado:
    resultado = clasificador(line) # Clasificar Texto

    if 0 < resultado[0]["score"] <= 0.3: # Insatisfecho: Mayor a 0 y menor o igual a 0.3
        bert_values[0] += 1
    elif 0.3 < resultado[0]["score"] <= 0.5: # Neutral: Mayor 0.3 y menor o igual a 0.5
        bert_values[1] +=1
    elif 0.5 < resultado[0]["score"]: # Satisfecho: Mayor a 0.5
        bert_values[2] +=1

  #Mostrar Gráfica
  plt.bar(tags, bert_values)
  plt.title('Análisis de sentimiento de Reseñas de Gift Cards de Amazon')
  plt.xlabel('RANGO')
  plt.ylabel('CANTIDAD')
  plt.show()
 

#--------------------------INICIO DE PROGRAMA-----------------------------------
#-------------------------------------------------------------------------------
#Leer archivo
leer_jsonl()
#Análisis de sentimiento reseñas de Gift Cards
analisis_sentimiento()

