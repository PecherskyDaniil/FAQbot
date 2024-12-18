import torch
import json
import tensorflow as tf
import numpy as np
from tkinter import *
from tkinter import ttk
from transformers import AutoTokenizer, AutoModel



loaded_model = tf.keras.models.load_model("torch_model.keras")
dataset=np.load("torch_dataset.npy")

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
bert_model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")

with open('../QA.json', 'r') as file:
    data = json.load(file)
questions=data['questions']
answers=data['answers']

def show_message():
    question=[entry.get().lower()]
    qtokenized=tokenizer(question,return_tensors='pt')
    emb1=bert_model(**{k: v.to(bert_model.device) for k, v in qtokenized.items()}).pooler_output.detach().numpy()[0]
    p=[]
    for i in range(dataset.shape[0]):
        emb2=dataset[i,1]
        emb3=np.concatenate([emb1,emb2])
        p.append(emb3)
    p=np.array(p)
    label["text"] = answers[loaded_model.predict(p).argmax()]
    
root = Tk()
root.title("Чат бот")
root.geometry("500x500") 
 
entry = ttk.Entry(width=400)
entry.pack(anchor=NW, padx=50, pady=50)
  
btn = ttk.Button(text="Задать вопрос", command=show_message)
btn.pack(anchor=NW, padx=50, pady=50)
 
label = ttk.Label()
label.pack(anchor=NW, padx=50, pady=50)
  
root.mainloop()
