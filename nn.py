import numpy as np
import matplotlib.pyplot as plt
class Module:
    def __init__(self):
        self.with_weights = False
        pass
    def forward(self):
        pass
    def backward(self):
        pass
    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)
# linear layer
class Linear(Module):
    def __init__(self,input_dim,output_dim):
        super(Linear, self).__init__()
        # initilize weights
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.b = np.zeros((1, output_dim))
        self.with_weights = True
    def forward(self, input_array):
        res = np.zeros((input_array.shape[0], self.W.shape[1]))
        
        ## start of your code
        res = input_array @ self.W + self.b
        ## end of your code
        
        return res
    def backward(self, input_array, output_gradient, lr = 0.05):
        res = np.zeros_like(input_array)
        
        ## start of your code
        # 1. compute new output_gradient, which will be backward-passed to previous layer
        res = np.dot(output_gradient, self.W.T)
        # 2. compute the gradient and update W, b
        self.W = self.W - lr * np.dot(input_array.T, output_gradient)
        self.b = self.b - lr * np.dot( np.ones((1, input_array.shape[0])), output_gradient)
        ## end of your code
        return res
# ReLU layer
class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        pass
    def forward(self, input_array):
        res = np.zeros_like(input_array)
        
        ## start of your code
        res = (np.abs(input_array) + input_array) / 2
        ## end of your code
        
        return res
    def backward(self, input_array, output_gradient):
        res = np.zeros_like(input_array)
        ## start of your code
        res = (input_array > 0) * output_gradient
        ## end of your code
        return res

class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()   
        pass
    def forward(self, predicted_y, y):
        res = 1e5
        
        ## start of your code
        res = np.sum(np.power(predicted_y - y, 2)) / y.shape[0]
        ## end of your code
        return res
    def backward(self, predicted_y, y):
        
        ## start of your code
        res = 2 * (predicted_y - y) / y.shape[0] 
        ## end of your code
        return res

class SimpleNN(Module):
    def __init__(self, layers, loss, lr = 0.005):
        super(SimpleNN, self).__init__()    
        self.layers = layers
        self.loss = loss
        self.inputs = [None for _ in range(len(self.layers))]
        self.output = None
        self.loss_value = 1e5
        self.lr = lr
        pass
    def forward(self, input_array):
        current_input = input_array
        for i in range(len(self.layers)):
            self.inputs[i] = current_input
            current_input = self.layers[i](current_input)
        self.output = current_input
        return self.output
    def backward(self, y):
        if self.inputs[-1] is None:
            print("call forward first.")
            return
        self.loss_value = self.loss(self.output, y)
        output_gradient = self.loss.backward(self.output, y)
        for i in range(len(self.layers)-1, -1, -1):
            if self.layers[i].with_weights:
                output_gradient = self.layers[i].backward(self.inputs[i], output_gradient, self.lr)
            else:
                output_gradient = self.layers[i].backward(self.inputs[i], output_gradient)
        self.output = None

if __name__ == "__main__":
    layers = [
        # input layer is input data
        # L1
        Linear(1, 80),
        # L2
        ReLU(),
        # L3
        Linear(80, 80),
        # L4
        ReLU(),
        # L5
        Linear(80, 1)
    ]
    loss = MSELoss()
    model = SimpleNN(layers, loss, lr=0.1)

    x = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1) 
    y = np.sin(x)

    epoch = 0
    max_iteration = 20000
    plt.figure(figsize=(15, 8))
    pred_y = model(x)
    plt.plot(x, pred_y, label="init")
    plt.plot(x, y, label="true")
    while epoch < max_iteration and model.loss_value > 1e-4:
        pred_y = model(x)
        model.backward(y)
        if epoch % 500 == 0:
            print(r'epoch {}/{}, loss: {}'.format(epoch, max_iteration, model.loss_value))
        epoch += 1
    plt.plot(x, pred_y, label="pred")
    plt.legend()
    plt.show()