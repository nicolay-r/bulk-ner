from src.batch_iter import BatchIterator

data = iter([1,2,3])
x = BatchIterator(data, batch_size=2)
for i in x:
    print(i)