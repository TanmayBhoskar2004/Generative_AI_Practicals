#RNN

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense,Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 1. Load and Tokenize Data
with open('input.txt','r')
 as f:
  text=f.read().lower().split('\n')

  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(text)
  total_words = len(tokenizer.word_index)+1
# 2. Prepare N0gram Sequences
input_sequences = []
for line in text:
  token_list = tokenizer.texts_to_sequences([line])[0]
  for i in range(1,len(token_list)):
    input_sequences.append(token_list[:i+1])

max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences,maxlen=max_sequence_len,padding='pre'))
X,y = input_sequences[:,:-1],input_sequences[:,-1]

#3. Build and train RNN Model

model = Sequential([
    Embedding(total_words,10,input_length=max_sequence_len-1),
  SimpleRNN(50),
  Dense(total_words,activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
model.fit(X,y,epochs=100,verbose=1)

# 4. Predict Next word from user Input

while True:
  user_text = input("Enter text(or 'quit'to exit): ")
  if user_text.lower() == 'quit': break

  seq = pad_sequences(tokenizer.texts_to_sequences([user_text]),maxlen=max_sequence_len-1,padding='pre')
  pred_idx = np.argmax(model.predict(seq, verbose = 0),axis=-1)[0]

  print(f"Predicted word: {tokenizer.index_word.get(pred_idx, ' ')}\n")

