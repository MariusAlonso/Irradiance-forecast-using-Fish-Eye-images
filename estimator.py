import torch
import torch.nn as nn

class MLP(nn.Module):
  """
  The MLP class plays the role of the model estimator. It is a convolutional 
  neural network, with the following layers (for a 64x64 input image):
  - Conv 3x3, dimensions of the layer : 64x64x8
  - Max Pooling 2x2
  - Conv 3x3, dimensions of the layer : 32x32x16
  - Max Pooling 4x4
  - Conv 3x3, dimensions of the layer : 8x8x32
  - Max Pooling 4x4
  - Flatten, dimensions of the layer : 128
  - Dense, dimensions of the layer : 32
  - Dense, dimensions of the layer : 8
  - Dense, dimensions of the layer : 1

  Construction :: MLP(input_size, n_channels, output_dim)

  Parameters
  ----------

  input_size : int
    The size of the input image
  n_channels : int
    The number of channels of the input image
  output_dim : int, optional
    The size of the output array (1 for an estimator : the estimation of solar irradiance)
  """

  def __init__(self, input_size, n_channels, output_dim=1):
    super().__init__()
    self.input_size = input_size
    self.output_dim = output_dim

    self.layers = nn.Sequential(
        nn.Conv2d(n_channels, 8, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(4,8),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(4,16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=4, stride=4),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.GroupNorm(4,32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=4, stride=4),
        nn.Flatten(),
        nn.Linear(128, 32),
        #nn.Dropout(p=0.2),
        nn.ReLU(inplace=True),
        nn.Linear(32, 8),
        #nn.Dropout(p=0.2),
        nn.ReLU(inplace=True),
        nn.Linear(8, self.output_dim)
    )

  def forward(self, x):
      return self.layers(x)

