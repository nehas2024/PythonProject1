"""import numpy as np

# Define two arrays
x = np.array([1, 2, 3, 4, 5])
y = np.array([6, 7, 8, 9, 10])

# Stack arrays horizontally
hstack_result = np.hstack((x, y))
print("Horizontal Stack:\n", hstack_result)

# Stack arrays vertically
vstack_result = np.vstack((x, y))
print("Vertical Stack:\n", vstack_result)

#traversing array
a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
a = a.reshape(4,3)
n,m =np.shape(a)
print(a)
for i in range(n):
    for j in range(m):
        print(a[i,j])

# Deleting elements from an array
a = np.delete(a,[0], 1)
print(a)

# nditer
a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
for i in np.nditer(a):
    print(i)

# nditer in 2d array
a = np.array([[1,2,3,4],[5,6,7,8]])
for i in np.nditer(a):
    print(i)
"""
import numpy as np
a = np.array([[1,2,3,4],[5,6,7,8]])
print(a)
b = np.delete(a,[0],1)
print(b)