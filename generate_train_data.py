import random
import ljqpy
from faq_preprocess import LoadCandidateData

data = LoadCandidateData()
max = 2082
d = []

for i in range(3000):
    a = random.randint(0, max)
    b = random.randint(0, max)
    if a != b:
        d.append(data[str(a)][0] + '\t' + data[str(b)][0] + '\t0' )

for i in range(2000):
    a = random.randint(0, max)
    st = data[str(a)][0]
    stn = data[str(a)][0]
    if random.random()>0.5:
        k = random.randint(0, len(st) - 1)
        st = st[:k]+st[k+1:]
    d.append(stn+'\t'+st+'\t1')

random.shuffle(d)

ljqpy.SaveList(d,'data/train_data.txt')