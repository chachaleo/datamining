import ndjson
import matplotlib.pyplot as plt

# load from file-like objects
with open('apple.ndjson') as f:
    data = ndjson.load(f)

print(data[1]['drawing'])

print (len(data[1]['drawing']))
n=33

for i in range(len(data[n]['drawing'])):
	print(i)
	plt.plot(data[n]['drawing'][i][0],data[n]['drawing'][i][1],'o')
plt.show()	
