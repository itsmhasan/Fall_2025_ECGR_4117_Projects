import numpy as np

# Load the data
train = np.load("train.npy", allow_pickle=True)
test = np.load("test.npy", allow_pickle=True)

print("First 3 training samples:")
print(train[0])
print(train[1])
print(train[2])
print()

print("First 3 testing samples:")
print(test[0])
print(test[1])
print(test[2])
