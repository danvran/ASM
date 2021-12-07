import torch
import torch.nn as nn

class PyTorchGRU(nn.Module):
    """
    A recurrent layer based network to make single step predictions on 1D time series data.
    """
    def __init__(self, h1, prefix):
        super(PyTorchGRU, self).__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=h1, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=h1, out_features=1)
        self.modeldict = copy.deepcopy(self.state_dict())
        self.prefix = prefix
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, x):
        output, _status = self.gru(x)
        output = output[:, -1, :]
        output = torch.relu(output)
        output = self.fc1(output)
        return output


class PyTorchLSTM(nn.Module):
    """
    A recurrent layer based network to make single step predictions on 1D time series data.
    """
    def __init__(self, h1, prefix):
        super(PyTorchLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=h1, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=h1, out_features=1)
        self.modeldict = copy.deepcopy(self.state_dict())
        self.prefix = prefix
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, x):
        output, _status = self.lstm(x)
        output = output[:, -1, :]
        output = torch.relu(output)
        output = self.fc1(output)
        return output


class PyTorchCNN(nn.Module):
    """
    A convolutional layer based network to make single step predictions on 1D time series data.
    """
    def __init__(self, inputsize, channels, prefix):
        super(PyTorchCNN, self).__init__()
        self.kernel = 3
        self.channels = channels
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.channels, kernel_size=self.kernel, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=self.channels, out_channels=10, kernel_size=self.kernel, stride=1, padding=0)
        self.drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(in_features=(inputsize-(self.kernel-1))*self.channels, out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=20)
        self.fc3 = nn.Linear(in_features=20, out_features=1)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.prefix = prefix
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, x):
        x = x.transpose(2, 1)
        output = self.conv1(x)
        output = self.act(output)
        output = self.flat(output)
        output = self.fc1(output)
        output = self.act(output)
        output = self.fc2(output)
        output = self.act(output)
        output = self.fc3(output)
        return output


class PyTorchMLP(nn.Module):
    """
    A feed forward network to make single step predictions on 1D time series data.
    """
    def __init__(self, inputsize, prefix):
        super(PyTorchMLP, self).__init__()
        self.fc1 = nn.Linear(in_features=inputsize, out_features=round(inputsize/2))
        self.fc2 = nn.Linear(in_features=round(inputsize/2), out_features=1)
        self.act = nn.ReLU()
        self.prefix = prefix

    def forward(self, x):
        y = torch.squeeze(x)
        output = self.fc1(y)
        output = self.act(output)
        output = self.fc2(output)
        return output

class PyTorchAE(nn.Module):
    """
    An undercomplete autoendcoder feed forward network for 1D time series data.
    """
    def __init__(self, in_samples):
        super(PyTorchAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_samples, out_features=in_samples - 20),
            nn.ReLU(),
            nn.Linear(in_features=in_samples - 20, out_features=in_samples - 40),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=in_samples - 40, out_features=in_samples - 20),
            nn.ReLU(),
            nn.Linear(in_features=in_samples - 20, out_features=in_samples)
        )

    def forward(self, x):
        code = self.encoder(x)
        output = self.decoder(code)
        return output
