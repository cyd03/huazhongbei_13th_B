import pandas as pd
import keras


data = pd.read_excel('data/sample_finish.xlsx')
data['TEXT1_padded'] = [eval(data['TEXT1_padded'].iloc[i]) for i in range(len(data['TEXT1_padded']))]
data['TEXT2_padded'] = [eval(data['TEXT2_padded'].iloc[i]) for i in range(len(data['TEXT2_padded']))]

loaded_model = keras.models.load_model("siamese_model.h5")

# 准备训练数据

X1 = keras.preprocessing.sequence.pad_sequences(data['TEXT1_padded'], padding='post')
X2 = keras.preprocessing.sequence.pad_sequences(data['TEXT2_padded'], padding='post')
y = data['label']

tmp=loaded_model.predict([X1,X2]).tolist()
pre=[]
for i in range(len(tmp)):
	if tmp[i][0]>0.5:
		pre.append(1)
	else:
		pre.append(0)
test=y.values.tolist()
cnt=0
for i in range(len(tmp)):
	if pre[i] == test[i]:
		cnt+=1
print(cnt/len(tmp))