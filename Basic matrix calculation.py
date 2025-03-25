import numpy as np
x = np.array([[1,2], [3,4]])
print(x)
print(x.ndim)
print(x.shape)
y = np.array([[5,5], [6,6]])

add= np.add(x,y)
add1 =x+y
print(add)
print(add1)

comp1 =x>1
print(comp1)

comp2 = y<=1
print(comp2)

sub = x-y
print(sub)

mul = x*y
print(mul)

div = x/y
print(div)

sin = np.sin(x)
print(sin)

cos = np.cos(x)
print(cos)

tan = np.tan(x)
print(tan)

sort = np.sort(x)
print(sort)

y =np.array([1,2,3,4,5,6,7,8,9,10,11,12])
x = y.reshape(3,4)
n,m =np.shape(x)
print(n)
print(m)

New1 =np.array([1,2,13,4,15,16,7,8,9,10,11,22])
New2 = New1.reshape(4,3)
print(New2)
print(New2[:,])
print(New2[:,0])
print(New2[:,2])






