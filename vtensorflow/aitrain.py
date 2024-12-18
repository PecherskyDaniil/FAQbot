
import json
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
import numpy as np
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')

bert="https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3"
bpre="https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"

print("Loading bert")
bert_preprocess_model=hub.KerasLayer(bpre)
bert_model = hub.KerasLayer(bert)

with open('QA.json', 'r') as file:
    data = json.load(file)
questions=data['questions']
answers=data['answers']
print(questions[0])
print("Creating dataset")
dataset=[]
questions=list(map(lambda x: x.lower(),questions))
answers=list(map(lambda x: x.lower(),answers))
for i in range(len(questions)):
  q_emb=bert_model(bert_preprocess_model(questions[i:i+1]))["pooled_output"].numpy()#.sum(axis=1)
  #print(q_emb.shape)
  a_emb=bert_model(bert_preprocess_model(answers[i:i+1]))["pooled_output"].numpy()#.sum(axis=1)
  dataset.append([np.array(q_emb[0]),np.array(a_emb[0])])
dataset=np.array(dataset)
np.save('dataset', dataset)
print("X and Y creating")
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

model.save("model.keras")