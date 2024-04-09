import pandas as pd
import numpy as np


tmp_list=[]
label =[]
for i in range(4):
	for j in range(6):
		path='yourpath/'+str(i)+'/pic'+str(j)+'.png'
		print(path)
		tmp_list.append(path)
		label.append(i)


tmp_list=np.array(tmp_list).transpose()
label=np.array(label).transpose()

sample = np.vstack((tmp_list,label)).transpose()


sample=pd.DataFrame(sample,columns = ['path','label'])
sample.to_excel('tmp.xlsx',index=False)