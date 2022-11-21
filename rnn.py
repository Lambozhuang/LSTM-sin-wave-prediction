import torch
import torch.nn as nn


class RNN_Predictor(nn.Module):
    def __init__(self, n_hidden=128, device=torch.device("cpu")):
        super(RNN_Predictor, self).__init__()
        self.n_hidden = n_hidden

        self.rnn = nn.RNNCell(1, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x, future=0, device=torch.device("cpu")):
        outputs = []
        n_samples = x.size(0) # == batch_size

        # initial hidden state
        hidden = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(device)

        for input_t in x.split(1, dim=1):
            hidden = self.rnn(input_t, hidden)
            output = self.linear(hidden)
            outputs.append(output)

        # if need predict future
        for i in range(future):
            hidden = self.rnn(output, hidden)
            output = self.linear(hidden)
            outputs.append(output)
        
        outputs = torch.cat(outputs, dim=1)
        return outputs