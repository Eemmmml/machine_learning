import numpy as np

arr1 = np.zeros((4, 4), dtype=int)
print(arr1)
print(arr1.dtype)
print(arr1.shape)

arr2 = np.ones((3, 4))
print(arr2)
print(arr2.dtype)
print(arr2.shape)

arr3 = np.eye(5)
print(arr3)
print(arr3.dtype)
print(arr3.shape)

arr4 = np.identity(3)
print(arr4)
print(arr4.dtype)
print(arr4.shape)

arr5 = np.empty((2, 3))
print(arr5)
print(arr5.dtype)
print(arr5.shape)
