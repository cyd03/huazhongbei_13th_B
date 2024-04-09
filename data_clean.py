import pandas as pd
import numpy as np
import re
import random
def get_num(s):
    text=s
    number = None
    matches = re.findall(r'\d+',text)
    if len(matches) > 0:
        number=int(matches[0])
    return number


df1 = pd.read_csv('data/附件1.csv')
df2 = pd.read_csv('data/附件2.csv')

id_df1= df1['id'].tolist()
id_df2 = df2['questionID'].tolist()
cnt = 0
for i in id_df2:
    if i not in id_df1:
        cnt+=1
print(cnt)
df2_repeat=df2[df2['duplicates'].notnull()]
sep_quesid=df2_repeat['questionID'].tolist()
seq_reid_str=df2_repeat['duplicates'].tolist()
seq_reid=[]
for s in seq_reid_str:
    num=get_num(s)
    seq_reid.append(num)
com_id = np.column_stack((sep_quesid,seq_reid))
for i in sep_quesid:
    id_df1.remove(i)
for i in seq_reid:
    if i in id_df1:
        id_df1.remove(i)
com_id2=[]
for i in range(2000-len(com_id)):
    random.seed(i**3+50+i**2)
    random_numbers = random.sample(id_df1, 2)
    com_id2.append(random_numbers)
com_id2=np.array(com_id2)
com_id_new = np.vstack((com_id,com_id2))
label=[1]*len(com_id)+[0]*len(com_id2)
label= np.array(label)
label=label.reshape(-1,1)
sample = np.hstack((com_id_new,label))
random.seed(521)
indices = np.arange(sample.shape[0])
random.shuffle(indices)
sample=sample[indices]

# 创建一个DataFrame，可以根据需要指定列名
df = pd.DataFrame(sample, columns=['ID1', 'ID2', 'label'])
df.to_excel('data/sample.xlsx',index=False)

