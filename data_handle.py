import pandas as pd
import numpy as np

df1 = pd.read_excel('data/sample.xlsx')
df2 = pd.read_csv('data/附件1.csv')
text1=[]
text2=[]
df2_id = df2['id'].tolist()
df2_text = df2['text'].tolist()
id1 = df1['ID1'].tolist()
id2 = df1['ID2'].tolist()
my_dict = dict(zip(df2_id,df2_text))
for i in id1:
    text1.append(my_dict[i])
for i in id2:
    text2.append(my_dict[i])
np_text1 = np.array(text1)
np_text2 = np.array(text2)
np_text1 = np.reshape(np_text1,(2000,-1))
np_text2 = np.reshape(np_text2,(2000,-1))
np_text = np.hstack((np_text1,np_text2))
df = pd.DataFrame(np_text,columns=['TEXT1','TEXT2'])
df = pd.concat([df1,df],axis=1)
df.to_excel('data/sample_initial.xlsx',index=False)