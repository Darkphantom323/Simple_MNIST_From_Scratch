# Simple Two-Layer Neural Network for MNIST Digit Recognition

This repository contains the implementation of a simple two-layer neural network trained on the MNIST digit recognizer dataset. The dataset can be downloaded from [Kaggle Digit Recognizer Competition](https://www.kaggle.com/competitions/digit-recognizer).

## Neural Network Architecture

### Forward Propagation

The neural network has the following architecture:

- **Input Layer**: $a^{[0]}$ with 784 units corresponding to the 784 pixels in each 28x28 input image.
- **Hidden Layer**: $a^{[1]}$ with 10 units and ReLU activation.
- **Output Layer**: $a^{[2]}$ with 10 units corresponding to the ten-digit classes and softmax activation.

Forward propagation involves the following steps:

1. $Z^{[1]} = W^{[1]} X + b^{[1]}$
2. $A^{[1]} = g_{\text{ReLU}}(Z^{[1]})$
3. $Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$
4. $A^{[2]} = g_{\text{softmax}}(Z^{[2]})$

### Backward Propagation

The backward propagation process for updating the parameters involves:

1. Compute gradients:
   - $dZ^{[2]} = A^{[2]} - Y$
   - $dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}$
   - $dB^{[2]} = \frac{1}{m} \sum {dZ^{[2]}}$
   - $dZ^{[1]} = W^{[2]T} dZ^{[2]} .* g^{[1]\prime} (z^{[1]})$
   - $dW^{[1]} = \frac{1}{m} dZ^{[1]} A^{[0]T}$
   - $dB^{[1]} = \frac{1}{m} \sum {dZ^{[1]}}$

2. Update parameters:
   - $W^{[2]} := W^{[2]} - \alpha dW^{[2]}$
   - $b^{[2]} := b^{[2]} - \alpha db^{[2]}$
   - $W^{[1]} := W^{[1]} - \alpha dW^{[1]}$
   - $b^{[1]} := b^{[1]} - \alpha db^{[1]}$

### Variable Shapes

#### Forward Propagation

- $A^{[0]} = X$: 784 x m
- $Z^{[1]} \sim A^{[1]}$: 10 x m
- $W^{[1]}$: 10 x 784
- $B^{[1]}$: 10 x 1
- $Z^{[2]} \sim A^{[2]}$: 10 x m
- $W^{[1]}$: 10 x 10
- $B^{[2]}$: 10 x 1

#### Backward Propagation

- $dZ^{[2]}$: 10 x m
- $dW^{[2]}$: 10 x 10
- $dB^{[2]}$: 10 x 1
- $dZ^{[1]}$: 10 x m
- $dW^{[1]}$: 10 x 10
- $dB^{[1]}$: 10 x 1
