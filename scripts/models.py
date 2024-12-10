import torch.nn as nn
import torch

from loss import mse_loss
from datasets import SpectrogramDataset
    
class SpectrVelCNNRegr(nn.Module):
    """Baseline model for regression to the velocity

    Use this to benchmark your model performance.
    """

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=37120,out_features=1024)
        self.linear2=nn.Linear(in_features=1024,out_features=256)
        self.linear3=nn.Linear(in_features=256,out_features=1)
    
    def _input_layer(self, input_data):
        return self.conv1(input_data)

    def _hidden_layer(self, x):
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.flatten(x)
        x=self.linear1(x)
        return self.linear2(x)

    def _output_layer(self, x):
        return self.linear3(x)

    def forward(self, input_data):
        x = self._input_layer(input_data)
        x = self._hidden_layer(x)
        return self._output_layer(x)

class SpectrCNN_5_layers(nn.Module):
    """Define your model here.

    I suggest make your changes initial changes
    to the hidden layers defined in _hidden_layer below.
    This will preserve the input and output dimensionality.
    
    Eventually, you will need to update the output dimensionality
    of the input layer and the input dimensionality of your output
    layer.

    """
    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()

        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=23040,out_features=1024)
        self.linear2=nn.Linear(in_features=1024,out_features=256)
        self.linear3=nn.Linear(in_features=256,out_features=1)
    
    def _input_layer(self, input_data):
        return self.conv1(input_data)

    def _hidden_layer(self, x):
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.flatten(x)
        x=self.linear1(x)
        return self.linear2(x)

    def _output_layer(self, x):
        return self.linear3(x)

    def forward(self, input_data):
        x = self._input_layer(input_data)
        x = self._hidden_layer(x)
        return self._output_layer(x)

class SpectrCNN_5_layers_dropout(nn.Module):
    """Define your model here.

    I suggest make your changes initial changes
    to the hidden layers defined in _hidden_layer below.
    This will preserve the input and output dimensionality.
    
    Eventually, you will need to update the output dimensionality
    of the input layer and the input dimensionality of your output
    layer.

    """
    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self, dropput_p = 0.2):
        super().__init__()

        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.Dropout(dropput_p),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.Dropout(dropput_p),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.Dropout(dropput_p),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.Dropout(dropput_p),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(dropput_p),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=14848,out_features=1024)
        self.linear2=nn.Linear(in_features=1024,out_features=256)
        self.linear3=nn.Linear(in_features=256,out_features=1)
    
    def _input_layer(self, input_data):
        return self.conv1(input_data)

    def _hidden_layer(self, x):
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.flatten(x)
        x=self.linear1(x)
        return self.linear2(x)

    def _output_layer(self, x):
        return self.linear3(x)

    def forward(self, input_data):
        x = self._input_layer(input_data)
        x = self._hidden_layer(x)
        return self._output_layer(x)

class SpectrVelCNNRegr_no_linear(nn.Module):
    """Baseline model for regression to the velocity

    Use this to benchmark your model performance.
    """

    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self):
        super().__init__()
        
        
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten=nn.Flatten()
        self.linear3=nn.Linear(in_features=256,out_features=1)
    
    def _input_layer(self, input_data):
        return self.conv1(input_data)

    def _hidden_layer(self, x):
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.flatten(x)
        return x

    def _output_layer(self, x):
        return self.linear3(x)

    def forward(self, input_data):
        x = self._input_layer(input_data)
        x = self._hidden_layer(x)
        return self._output_layer(x)

class SpectrRNN(nn.Module):
    """Define your model here.

    I suggest make your changes initial changes
    to the hidden layers defined in _hidden_layer below.
    This will preserve the input and output dimensionality.
    
    Eventually, you will need to update the output dimensionality
    of the input layer and the input dimensionality of your output
    layer.

    """
    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self, dropout_rate=0.2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=7,
                      stride=2,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=2,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten(start_dim=1, end_dim=3)

        self.rnn1 = nn.Sequential(
            nn.RNN(input_size=512,
                   hidden_size=256,
                   dropout=dropout_rate,
                   nonlinearity="tanh",
                   num_layers=5)
        )

        self.rnn2 = nn.Sequential(
            nn.RNN(input_size=256,
                   hidden_size=128,
                   dropout=dropout_rate,
                   nonlinearity="tanh",
                   num_layers=5)
        )
        
        self.linear1 = nn.Linear(in_features=128, out_features=76)
        self.linear2 = nn.Linear(in_features=76, out_features=10)
        self.linear3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

class ResidualBlock(nn.Module):
    """
    A residual block for a ResNet model.
    The residual block is used for the ResNet18 and ResNet34 models.
    For deeper models, a bottleneck block is used instead for computational efficiency.
    """
    def __init__(self, in_channels : int, out_channels : int, stride : int, downsample : nn.Module = None, expansion : int = 1):
        super().__init__()
        self.downsample = downsample
        self.expansion = expansion # Expansion is 1 for ResNet18 and ResNet34

        # Define the layers of the residual block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion) # Dimensions are expanded by the expansion factor deeper layer architectures
        self.relu2 = nn.ReLU(inplace=True)
    
    # Define the forward function
    def forward(self, x):
        # Save the input for the identity mapping
        identity = x

        # Forward pass through the first convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # Forward pass through the second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)

        # If a downsampling model is passed in, perform downsampling
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add the identity to the output and apply the ReLU activation function
        out += identity
        out = self.relu2(out)
        return out

class BottleNeck(nn.Module):
    """
     A Bottleneck block for a ResNet model.
     The bottleneck block is used for the ResNet50, ResNet101, and ResNet152 models.
     However, it may be used for ResNet18 and ResNet34 as well, as in the ETP paper.
    """
    def __init__(self, in_channels : int, out_channels : int, stride : int, downsample : nn.Module = None, expansion : int = 4):
        super().__init__()
        self.downsample = downsample
        self.expansion = expansion # Expansion is 4 for ResNet50, ResNet101, and ResNet152

        # Define the layers of the bottleneck block
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv1d(in_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Save the input for the identity mapping
        identity = x

        # Forward pass through the first convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # Forward pass through the second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # Forward pass through the third convolutional layer
        out = self.conv3(out)
        out = self.bn3(out)

        # If a downsampling model is passed in, perform downsampling
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add the identity to the output and apply the ReLU activation function
        out += identity
        out = self.relu3(out)
        return out

class ResNet(nn.Module):
    """
    ResNet modified for signal processing. The main modifications are the use of 1D layers instead of 2D layers.
    Follows the implementation as described in "Deep Residual Learning for Image Recognition" (https://arxiv.org/abs/1512.03385).
    The architecture is built as in Table 1.
    Uses either a Residual Block or Bottleneck as "block", depending on depth.
    """
    loss_fn = mse_loss
    dataset = SpectrogramDataset

    def __init__(self, data_in_channels : int = 6, num_classes : int = 1, num_layers : int = 34, block : object = ResidualBlock):
        super().__init__()
        """
        The number of layers in the network is specified by the num_layers parameter. Must match those from the paper, currently either 18 or 34.
        """
        if num_layers == 18:
            self.layers = [2, 2, 2, 2]
            self.expansion = 1 # 1 for ResNet18
            self.name = "ResNet-18"
        elif num_layers == 34:
            self.layers = [3, 4, 6, 3]
            self.expansion = 1 # 1 for ResNet34
            self.name = "ResNet-34"
        elif num_layers == 50:
            self.layers = [3, 4, 6, 3]
            self.expansion = 4
            self.name = "ResNet-50"
        elif num_layers == 101:
            self.layers = [3, 4, 23, 3]
            self.expansion = 4
            self.name = "ResNet-101"
        elif num_layers == 152:
            self.layers = [3, 8, 36, 3]
            self.expansion = 4
            self.name = "ResNet-152"
        else:
            raise ValueError("The number of layers must be either 18 or 34.")

        # Define the "stem" convolutional layer before the residual layers
        self.in_channels = 64 # in_channels defined by the paper, which can be modified for each layer by multiplication
        self.conv1 = nn.Conv2d(in_channels = data_in_channels, out_channels = self.in_channels, kernel_size=7, stride=2 , padding=3, bias=False) # The first layer has a kernel size of 7
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # Define the maxpool layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define the residual layers
        self.conv2_x = self._make_layer(block, 64, self.layers[0])
        self.conv3_x = self._make_layer(block, 128, self.layers[1], stride = 2)
        self.conv4_x = self._make_layer(block, 256, self.layers[2], stride = 2)
        self.conv5_x = self._make_layer(block, 512, self.layers[3], stride = 2)

        # Define the average pool layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Define the fully connected layer
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def _make_layer(self, block, out_channels : int, blocks : int, stride : int = 1):
            downsample = None # Initialize the downsampling layer as None
            if stride != 1 or self.in_channels != out_channels * self.expansion:
                # If stride is not 1, then the dimensions of the input must be changed to match the output
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels * self.expansion, kernel_size = 1, stride = stride, bias = False),
                    nn.BatchNorm2d(out_channels * self.expansion)
                )
            layers = []
            # Create the first block
            layers.append(block(
                self.in_channels, out_channels, stride, downsample, self.expansion
            ))
            self.in_channels = out_channels * self.expansion

            # Create the remaining blocks
            for i in range(1, blocks):
                layers.append(block(self.in_channels, out_channels, stride = 1, expansion = self.expansion))
            return nn.Sequential(*layers)
        
    def forward(self, x):
            # Forward pass through the first convolutional layer
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            # Forward pass through the residual layers
            x = self.conv2_x(x)
            x = self.conv3_x(x)
            x = self.conv4_x(x)
            x = self.conv5_x(x)

            # Forward pass through the final layers
            x = self.avgpool(x)
            #x = x.view(x.size(0), -1)
            x = torch.flatten(x, 1)
            
            x = self.fc(x) # Apply the linear projection head

            return x

class SpectrHybridNet1(nn.Module):
    loss_fn = mse_loss
    dataset = SpectrogramDataset
    
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        
        # Enhanced CNN Feature Extractor
        self.cnn_features = nn.Sequential(
            # First Conv Block
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Second Conv Block with Residual Connection
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Third Conv Block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((8, 25))  # Reduced width for smaller feature size
        )
        
        # Calculate LSTM input size
        self.lstm_input_size = 64 * 25  # channels * width after adaptive pooling
        
        # Bidirectional LSTM with attention
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 64),  # 256 from bidirectional LSTM
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Enhanced Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate/2),
            nn.Linear(32, 1)
        )
        
    def attention_net(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN Feature Extraction
        cnn_out = self.cnn_features(x)
        
        # Reshape for LSTM
        cnn_out = cnn_out.permute(0, 2, 1, 3)  # [batch, height, channels, width]
        cnn_out = cnn_out.reshape(batch_size, 8, self.lstm_input_size)  # [batch, seq_len, features]
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_out)
        
        # Apply attention
        context = self.attention_net(lstm_out)
        
        # Final regression
        return self.regressor(context)

class SpectrHybridNet2(nn.Module):
    loss_fn = mse_loss
    dataset = SpectrogramDataset
    
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        
        # Increased CNN Feature Extractor
        self.cnn_features = nn.Sequential(
            # First Conv Block
            nn.Conv2d(in_channels=6, out_channels=48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Second Conv Block with Residual Connection
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Third Conv Block
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((8, 25))
        )
        
        self.lstm_input_size = 96 * 25  # Increased channels * width
        
        # Larger LSTM
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=192,  # Increased hidden size
            num_layers=3,     # Added one more layer
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        # Fixed attention mechanism dimensions
        self.attention = nn.Sequential(
            nn.Linear(384, 96),  # 384 from bidirectional LSTM (192*2)
            nn.Tanh(),
            nn.Linear(96, 1)
        )
        
        # Enhanced Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(384, 128),  # Changed from 256 to 384
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate/2),
            nn.Linear(32, 1)
        )
        
    def attention_net(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN Feature Extraction
        cnn_out = self.cnn_features(x)
        
        # Reshape for LSTM
        cnn_out = cnn_out.permute(0, 2, 1, 3)  # [batch, height, channels, width]
        cnn_out = cnn_out.reshape(batch_size, 8, self.lstm_input_size)  # [batch, seq_len, features]
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_out)
        
        # Apply attention
        context = self.attention_net(lstm_out)
        
        # Final regression
        return self.regressor(context)
    
# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/n**.5
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)
