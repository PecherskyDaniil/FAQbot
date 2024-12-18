import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
import json
tf.get_logger().setLevel('ERROR')


with open('QA.json', 'r') as file:
    data = json.load(file)
questions=data['questions']
answers=data['answers']

bert="https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3"
bpre="https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"

print("Loading bert")
bert_preprocess_model=hub.KerasLayer(bpre)
bert_model = hub.KerasLayer(bert)
loaded_model = tf.keras.saving.load_model("model.keras")
dataset=np.load("dataset.npy")


def show_message():
    question=[entry.get().lower()]
    emb1=bert_model(bert_preprocess_model(question[0:1]))["pooled_output"].numpy()[0]
    p=[]
    for i in range(dataset.shape[0]):
        emb2=dataset[i,1]
        emb3=np.concatenate([emb1,emb2])
        p.append(emb3)
    p=np.array(p)
    label["text"] = answers[loaded_model.predict(p).argmax()]
    
root = Tk()
root.title("Чат бот")
root.geometry("250x200") 
 
entry = ttk.Entry()
entry.pack(anchor=NW, padx=6, pady=6)
  
btn = ttk.Button(text="Задать вопрос", command=show_message)
btn.pack(anchor=NW, padx=6, pady=6)
 
label = ttk.Label()
label.pack(anchor=NW, padx=6, pady=6)
  
root.mainloop()
