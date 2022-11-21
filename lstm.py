import torch
import torch.nn as nn


class LSTM_Predictor(nn.Module):
    def __init__(self, n_hidden=51, device=torch.device("cpu")):
        super(LSTM_Predictor, self).__init__()
        self.n_hidden = n_hidden

        # lstm1, lstm2, linear
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x, future=0, device=torch.device("cpu")):
        outputs = []
        n_samples = x.size(0) # x.shape = 97, 999 (n_samples, L(length for each sample))

        # initial hidden state
        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(device)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(device)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(device)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32).to(device)

        for input_t in x.split(1, dim=1):
            # input_t.shape = 97, 1 (n_samples, 1(1 time step of the length in each sample))
            h_t, c_t = self.lstm1(input_t, (h_t, c_t)) # h_t.shape = c_t.shape = 97, 50 (n_samples, n_hidden)
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2) # output.shape = 97, 1 (n_samples, 1)
            outputs.append(output)

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs