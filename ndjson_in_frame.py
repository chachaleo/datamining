import ujson as json
import pandas as pd
import ndjson
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Dataload
path = "full_simplified_apple.ndjson"
records = map(json.loads, open((path),  encoding="utf-8", newline="\n" ))
data = pd.DataFrame.from_records(records, nrows=100, columns=['word', 'drawing'])

path = "full_simplified_donut.ndjson"
records = map(json.loads, open((path),  encoding="utf-8", newline="\n" ))
data2 = pd.DataFrame.from_records(records, nrows=100, columns=['word', 'drawing'])

result = data.append(data2, ignore_index=True)

path = "full_simplified_eye.ndjson"
records = map(json.loads, open((path),  encoding="utf-8", newline="\n" ))
data3 = pd.DataFrame.from_records(records, nrows=100, columns=['word', 'drawing'])

df = result.append(data3, ignore_index=True)

#shuffled dataset
df_shuffled=df.sample(frac=1).reset_index(drop=True)

#splitting in train dataset (70%) and test dataset (30%)
train, test = train_test_split(df_shuffled, test_size=0.3)
train_data = df_shuffled[:train.shape[0]]
test_data = df_shuffled[train.shape[0]:]

print(train_data)
print(test_data)




#image example

n=1

for i in range(len(df_shuffled['drawing'][n])):
    plt.plot(df_shuffled['drawing'][n][i][0],df_shuffled['drawing'][n][i][1],'-')
plt.show()
0)
