import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Embedding, LSTM, concatenate, Dense
from keras.models import Model
import matplotlib.pyplot as plt

data = pd.read_excel('data/sample_finish.xlsx')
data['TEXT1_padded'] = [eval(data['TEXT1_padded'].iloc[i]) for i in range(len(data['TEXT1_padded']))]
data['TEXT2_padded'] = [eval(data['TEXT2_padded'].iloc[i]) for i in range(len(data['TEXT2_padded']))]

vocab_size=20023
# 定义Siamese网络结构
def build_siamese_model(vocab_size, embedding_dim, lstm_units):
	input1 = Input(shape=(None,))
	input2 = Input(shape=(None,))

	embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)

	lstm_layer = LSTM(units=lstm_units)

	vector1 = lstm_layer(embedding_layer(input1))
	vector2 = lstm_layer(embedding_layer(input2))

	concatenated_vector = concatenate([vector1, vector2], axis=-1)

	output = Dense(1, activation='sigmoid')(concatenated_vector)

	model = Model(inputs=[input1, input2], outputs=output)
	return model


# 构建Siamese网络
siamese_model = build_siamese_model(vocab_size=vocab_size, embedding_dim=100, lstm_units=128)

# 编译模型
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 准备训练数据
train_cut=20
X1 = keras.preprocessing.sequence.pad_sequences(data['TEXT1_padded'], padding='post')
X2 = keras.preprocessing.sequence.pad_sequences(data['TEXT2_padded'], padding='post')
y = data['label']
X_train1 = X1[:-train_cut,:]
X_train2 = X2[:-train_cut,:]
y_train =y[:-train_cut]

x_test1 = X1[-train_cut:,:]
x_test2 = X2[-train_cut:,:]
y_test = y[-train_cut:]


# 训练模型
history=siamese_model.fit([X_train1, X_train2], y_train, epochs=20, batch_size=64, validation_split=0.2)
siamese_model.save("siamese_model.h5")
# 评估模型


# 训练模型并记录历史数据

# 可视化训练过程中的损失值
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 可视化训练过程中的准确率
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# 评估模型
siamese_model.evaluate([X_train1, X_train2], y_train)


tmp=siamese_model.predict([X_train1,X_train2]).tolist()
pre=[]
for i in range(len(tmp)):
	if tmp[i][0]>0.5:
		pre.append(1)
	else:
		pre.append(0)
test=y_train.values.tolist()
cnt=0
for i in range(len(tmp)):
	if pre[i] == test[i]:
		cnt+=1
print(cnt/len(tmp))