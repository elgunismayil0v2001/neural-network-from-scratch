import numpy as np
import pytest
from neural_network.layers import Dense


def test_dense_forward():
    np.random.seed(0)
    input_size = 3
    output_size = 2
    batch_size = 4

    # Create a Dense layer
    dense_layer = Dense(input_size, output_size)

    # Create random input data
    X = np.random.randn(batch_size, input_size)

    # Perform forward pass
    output = dense_layer.forward(X)

    # Check the shape of the output
    assert output.shape == (batch_size, output_size), f"Expected output shape {(batch_size, output_size)}, but got {output.shape}"
    
    
    
def test_dense_backward():
    np.random.seed(0)
    input_size = 3
    output_size = 2
    batch_size = 4

    # Create a Dense layer
    dense_layer = Dense(input_size, output_size)

    # Create random input data and perform forward pass
    X = np.random.randn(batch_size, input_size)
    dense_layer.forward(X)

    # Create random gradient for the output
    dZ = np.random.randn(batch_size, output_size)

    # Perform backward pass
    dX = dense_layer.backward(dZ)
    
    expected_dW = X.T @ dZ
    expected_db = np.sum(dZ, axis=0, keepdims=True)
    expected_dX = dZ @ dense_layer.weights.T

    np.testing.assert_allclose(dense_layer.dW, expected_dW)
    np.testing.assert_allclose(dense_layer.db, expected_db)
    np.testing.assert_allclose(dX, expected_dX)

    # Check the shape of the gradients
    assert dX.shape == (batch_size, input_size), f"Expected dX shape {(batch_size, input_size)}, but got {dX.shape}"
    assert dense_layer.dW.shape == (input_size, output_size), f"Expected dW shape {(input_size, output_size)}, but got {dense_layer.dW.shape}"
    assert dense_layer.db.shape == (1, output_size), f"Expected db shape {(1, output_size)}, but got {dense_layer.db.shape}"