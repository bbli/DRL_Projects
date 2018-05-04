import itertools

l = [1,2,3,4]

b= itertools.groupby(l,lambda i: i%2 ==0)

b
