seq = ['one', 'two', 'three']
for i, element in enumerate(seq):
    print(i, element)

alist = [10,11,12]
atuple = (2,3,4)
print([x for x in zip(alist,atuple)])

adict = {'a':1,'b':2,'c':3}
aset = {19,20,21,18}
print(zip(adict,aset))
print(tuple(zip(adict,aset)))
print(set(zip(aset,alist)))