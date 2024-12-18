import torch
import json
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
bert_model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")

with open('../QA.json', 'r') as file:
    data = json.load(file)
questions=data['questions']
answers=data['answers']

dataset=[]
questions=list(map(lambda x: x.lower(),questions))
answers=list(map(lambda x: x.lower(),answers))
for i in range(len(questions)):
  qt=tokenizer(questions[i],return_tensors='pt')
  q_emb=bert_model(**{k: v.to(bert_model.device) for k, v in qt.items()}).pooler_output.detach().numpy()#.sum(axis=1)
  #print(q_emb.shape)
  at=tokenizer(answers[i],return_tensors='pt')
  a_emb=bert_model(**{k: v.to(bert_model.device) for k, v in at.items()}).pooler_output.detach().numpy()#.sum(axis=1)
  dataset.append([np.array(q_emb[0]),np.array(a_emb[0])])
dataset=np.array(dataset)
np.save('torch_dataset', dataset)
X,Y=[],[]
for i in range(dataset.shape[0]):
  for j in range(dataset.shape[0]):
    X.append(np.concatenate([dataset[i,0,:],dataset[j,1,:]],axis=0))
    if i==j:
      Y.append(1)
    else:
      Y.append(0)
X=np.array(X)
Y=np.array(Y)
print("model compile")
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(1536,)))
model.add(tf.keras.layers.Dense(200,activation="relu"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss="binary_crossentropy",metrics=[tf.keras.metrics.AUC(curve="pr",name="auc")])
print("model fit")
model.fit(X,Y,epochs=1000)
model.summary()
model.save("torch_model.keras")
