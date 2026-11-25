import numpy as np
class Dense:
    def __init__(self,input_size : int, output_size : int):
        """
        Initialize a Dense layer with random weights and biases.

        Parameters:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        
        if input_size <= 0 or output_size <= 0:
            raise ValueError("Input and output sizes must be positive integers.")
        
        
    def forward(self, X : np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the Dense layer.

        Parameters:
        X (numpy.ndarray): Input data of shape (batch_size, input_size).

        Returns:
        numpy.ndarray: Output data of shape (batch_size, output_size).
        """
        
        if X.ndim != 2 or X.shape[1] != self.input_size:
            raise ValueError(f"Input data must be of shape (batch_size, {self.input_size}).")
        self.input = X
        return X @ self.weights + self.biases
    
    
    def backward(self, dZ : np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the Dense layer.

        Parameters:
        dZ (numpy.ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
        numpy.ndarray: Gradient of the loss with respect to the input of this layer.
        """
        
        if dZ.ndim != 2 or dZ.shape[1] != self.output_size:
            raise ValueError(f"dZ must be of shape (batch_size, {self.output_size}).")
        
        self.dW = self.input.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)
        return dZ @ self.weights.T