"""Simple convolutional neural network implemented with NumPy.

This module implements a lightweight CNN for estimating action values in the
Sequence game.  It is deliberately kept small so that training can run
quickly on a CPU without external dependencies such as PyTorch or TensorFlow.

The network operates on a state tensor of shape (C, H, W) where C is the
number of channels (currently 5: my chips, opponent chips, empty squares,
corners, protected), and H = W = 10.  The output is a vector of length
100 representing scores for each board cell.  Invalid moves will be
masked by the agent before computing probabilities.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


class SequenceCNN:
    """A tiny CNN with two convolutional layers and two dense layers.

    Architecture:
        Conv2d(C, 32, kernel=3, padding=1) -> ReLU
        Conv2d(32, 32, kernel=3, padding=1) -> ReLU
        Flatten -> Dense(32 * 10 * 10, 256) -> ReLU
        Dense(256, 100)

    Parameters are initialised with small random values.  Gradients are
    computed manually for policy gradient updates.
    """

    def __init__(self, input_channels: int = 5, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        # Convolutional layer weights and biases
        self.W1 = rng.normal(scale=0.1, size=(32, input_channels, 3, 3))
        self.b1 = np.zeros((32,), dtype=np.float32)
        self.W2 = rng.normal(scale=0.1, size=(32, 32, 3, 3))
        self.b2 = np.zeros((32,), dtype=np.float32)
        # Dense layers
        self.W3 = rng.normal(scale=0.1, size=(32 * 10 * 10, 256))
        self.b3 = np.zeros((256,), dtype=np.float32)
        self.W4 = rng.normal(scale=0.1, size=(256, 100))
        self.b4 = np.zeros((100,), dtype=np.float32)

        # Adam optimiser parameters
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self._param_keys()}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self._param_keys()}
        self.t = 0

    def _param_keys(self) -> Tuple[str, ...]:
        return ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4')

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Forward pass through the network.

        :param x: input state tensor of shape (C, 10, 10)
        :returns: logits of shape (100,) and a cache for backpropagation
        """
        # Add batch dimension
        x_in = x[np.newaxis, ...]  # (1, C, 10, 10)
        # Conv1
        z1 = self._conv2d(x_in, self.W1, self.b1)
        a1 = np.maximum(z1, 0)  # ReLU
        # Conv2
        z2 = self._conv2d(a1, self.W2, self.b2)
        a2 = np.maximum(z2, 0)
        # Flatten
        flat = a2.reshape(1, -1)  # (1, 32*10*10)
        # Dense1
        z3 = flat @ self.W3 + self.b3
        a3 = np.maximum(z3, 0)
        # Dense2
        logits = a3 @ self.W4 + self.b4  # (1, 100)
        cache = {
            'x_in': x_in,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'a2': a2,
            'flat': flat,
            'z3': z3,
            'a3': a3
        }
        return logits[0], cache

    def _conv2d(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Manual convolution with padding=1 and stride=1."""
        batch, c_in, h, w_in = x.shape
        c_out, _, kh, kw = w.shape
        # pad input
        pad = 1
        x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        h_out, w_out = h, w_in
        out = np.zeros((batch, c_out, h_out, w_out), dtype=np.float32)
        # Compute convolution
        for i in range(h_out):
            for j in range(w_out):
                # extract receptive field
                region = x_padded[:, :, i:i+kh, j:j+kw]  # (batch, c_in, kh, kw)
                # Multiply and sum over input channels and kernel dims
                out[:, :, i, j] = np.tensordot(region, w, axes=((1, 2, 3), (1, 2, 3))) + b
        return out

    def backward(self, cache: dict, d_logits: np.ndarray) -> None:
        """Backpropagate the gradient of the loss w.r.t. logits through the network.

        :param cache: dictionary returned by forward()
        :param d_logits: gradient of loss w.r.t. logits, shape (1, 100)
        """
        # Gradients for dense2
        dW4 = cache['a3'].T @ d_logits  # (256,100)
        db4 = d_logits.sum(axis=0)  # (100,)
        da3 = d_logits @ self.W4.T
        # ReLU on layer 3
        dz3 = da3 * (cache['z3'] > 0)
        dW3 = cache['flat'].T @ dz3
        db3 = dz3.sum(axis=0)
        dflat = dz3 @ self.W3.T  # (1, 32*10*10)
        da2 = dflat.reshape(1, 32, 10, 10)
        # ReLU on layer 2
        dz2 = da2 * (cache['z2'] > 0)
        # Conv2 gradients
        dW2, db2, da1 = self._conv2d_backward(cache['a1'], self.W2, dz2)
        # ReLU on layer 1
        dz1 = da1 * (cache['z1'] > 0)
        # Conv1 gradients
        dW1, db1, _ = self._conv2d_backward(cache['x_in'], self.W1, dz1)
        # Update parameters using Adam
        grads = {
            'W4': dW4, 'b4': db4,
            'W3': dW3, 'b3': db3,
            'W2': dW2, 'b2': db2,
            'W1': dW1, 'b1': db1
        }
        self._update_params(grads)

    def _conv2d_backward(self, x_in: np.ndarray, w: np.ndarray, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass for a convolutional layer.

        Returns gradients with respect to weights, biases and input.
        """
        batch, c_in, h_in, w_in = x_in.shape
        c_out, _, kh, kw = w.shape
        pad = 1
        x_padded = np.pad(x_in, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        h_out, w_out = h_in, w_in
        dW = np.zeros_like(w)
        db = dout.sum(axis=(0, 2, 3))
        d_x_padded = np.zeros_like(x_padded)
        # Compute gradients via explicit loops
        for n in range(batch):
            for i in range(h_out):
                for j in range(w_out):
                    for o in range(c_out):
                        # Weight gradient: accumulate region * dout
                        dW[o] += dout[n, o, i, j] * x_padded[n, :, i:i+kh, j:j+kw]
                        # Input gradient: distribute dout through weight
                        d_x_padded[n, :, i:i+kh, j:j+kw] += w[o] * dout[n, o, i, j]
        d_input = d_x_padded[:, :, pad:-pad, pad:-pad]
        return dW, db, d_input

    def _update_params(self, grads: dict, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        """Adam optimiser update for all parameters."""
        self.t += 1
        for k, grad in grads.items():
            m = self.m[k] = beta1 * self.m[k] + (1 - beta1) * grad
            v = self.v[k] = beta2 * self.v[k] + (1 - beta2) * (grad * grad)
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)
            param = getattr(self, k)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def save(self, path: str) -> None:
        """Save model weights to an .npz file."""
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3, W4=self.W4, b4=self.b4)

    def load(self, path: str) -> None:
        """Load model weights from an .npz file."""
        data = np.load(path, allow_pickle=False)
        for key in self._param_keys():
            if key in data:
                setattr(self, key, data[key])