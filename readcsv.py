import numpy as np
x = np.array([[[1, 2, 3], [3,4,5]], [[6,7,8], [9,10,11]]])
print(x[1,1,1])
print(x[0,1,2])
print(x[1,0,1])
print(x[0,1,1])
z = np.zeros(5)
print(z)
np.shape(z)
z2 = np.zeros((4,5))
print(z2)
np.shape(z2)
y = np.ones((2,3))
print(y)

F = np.full((7,8),11)
print(F)

x = np.linspace(0,5,100)
print(x)

x2 = np.arange(0,500,0.2)
print(x2)

a = 1
b = 6
amount = 50
nopat = np.random.randint(a,b,amount)

print(nopat)
x = np.random.randn(100,100)
print(x)