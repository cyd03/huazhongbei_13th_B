import pandas as pd
import random
import numpy as np
import pandas as pd
import keras
df = pd.read_excel('data/sample_finish.xlsx')
df_id = df['ID1'].tolist()
# random.seed(521)
random_id=random.sample(df_id,1)[0]
print(random_id)
random_row=df[df['ID1'] ==random_id]
random_row=eval(random_row['TEXT1_padded'].iloc[0])
df['TEXT1_padded'] = [eval(df['TEXT1_padded'].iloc[i]) for i in range(len(df['TEXT1_padded']))]
df['TEXT2_padded'] = [eval(df['TEXT2_padded'].iloc[i]) for i in range(len(df['TEXT2_padded']))]
test2_row = np.vstack((df['TEXT1_padded'].tolist(),df['TEXT2_padded'].tolist()))
answer_id = df['ID1'].tolist()+df['ID2'].tolist()
random_row = np.array(random_row)
random_row = np.reshape(random_row,(-1,50))
random_row=np.tile(random_row,(test2_row.shape[0],1))
loaded_model = keras.models.load_model("siamese_model.h5")
X1 = keras.preprocessing.sequence.pad_sequences(random_row, padding='post')
X2 = keras.preprocessing.sequence.pad_sequences(test2_row, padding='post')
data=loaded_model.predict([X1,X2])
max_indices = np.argsort(data.flatten())[-30:][::-1]
max_values = np.sort(data.flatten())[-30:][::-1]
max_indices = set([answer_id[i] for i in max_indices])


print(max_indices)


