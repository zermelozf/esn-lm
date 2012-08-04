
import cPickle as pickle 

with open('./datasets/t5') as f:
    a, b = pickle.load(f)
print a[0][:10]
print a[1][:10]

c = [w for s in a for w in s]
d = [w for s in b for w in s]
print c[:60]
print d[:60]

with open('./datasets/t5_train', 'w') as f:
    pickle.dump(c, f, protocol=0)
    
with open('./datasets/t5_test', 'w') as f:
    pickle.dump(d, f, protocol=0)