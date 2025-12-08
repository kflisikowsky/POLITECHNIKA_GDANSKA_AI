import numpy as np
from simplenn import SimpleNeuralNetwork

network = SimpleNeuralNetwork()

print(network.weights)

train_inputs = np.array([[1,1,0],[1,1,1],[1,1,0],[1,0,0],[0,1,1],[0,1,0],])
train_outputs = np.array([[0,1,0,0,1,0]]).T
train_iterators = 50000

network.train(train_inputs,train_outputs,train_iterators)
print(network.weights)

print("testowamnie sieci neuronowej")
test_data = np.array([[1,1,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]])

for data in test_data:
    print(f'wynik dla {data} wynosi: {network.propagation(data)}')
