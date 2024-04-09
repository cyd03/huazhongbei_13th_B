import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.preprocessing.text import Tokenizer

# 1. 加载数据
data = pd.read_excel('data/sample_initial.xlsx')

# 2. 文本预处理
def preprocess_text(text):
    text = text.lower()  # 转换为小写
    text = text.strip()  # 去除首尾空格
    # text = text.replace('\n', '')
    # 其他文本预处理步骤，如去除标点符号、特殊字符等
    return text

data['TEXT1'] = data['TEXT1'].apply(preprocess_text)
data['TEXT2'] = data['TEXT2'].apply(preprocess_text)

# 3. 建立词汇表
all_texts = np.concatenate([data['TEXT1'].values, data['TEXT2'].values])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

# 4. 文本编码
data['TEXT1_encoded'] = tokenizer.texts_to_sequences(data['TEXT1'].values)
data['TEXT2_encoded'] = tokenizer.texts_to_sequences(data['TEXT2'].values)


tmp1=[len(data['TEXT1_encoded'].iloc[i]) for i in range(len(data['TEXT1_encoded']))]
tmp2=[len(data['TEXT2_encoded'].iloc[i]) for i in range(len(data['TEXT2_encoded']))]
# 5. 填充或截断序列
# max_sequence_length = int(np.mean(tmp1+tmp2))+50
max_sequence_length = 50
print(max_sequence_length)
data['TEXT1_padded'] = pad_sequences(data['TEXT1_encoded'], maxlen=max_sequence_length, padding='post', truncating='post').tolist()
data['TEXT2_padded'] = pad_sequences(data['TEXT2_encoded'], maxlen=max_sequence_length, padding='post', truncating='post').tolist()

data.to_excel('data/sample_finish.xlsx',index=False)
