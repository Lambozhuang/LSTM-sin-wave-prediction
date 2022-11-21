import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

N = 100 # number of samples
L = 1000 # length for each sample
T = 20 # width of the wave

x = np.empty((N, L), np.float32)
x[:] = np.array(range(L)) + np.random.randint(-4*T, 4*T, N).reshape(N, 1)
y = np.sin(x/1.0/T).astype(np.float32)

plt.figure(figsize=(10, 8))
plt.title("Sine wave")
plt.xlabel("x")
plt.ylabel("y")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(np.arange(x.shape[1]), y[0, :], 'r', linewidth=2.0)
plt.show()


class LSTM_Predictor(nn.Module):
    def __init__(self, n_hidden=50):
        super(LSTM_Predictor, self).__init__()
        self.n_hidden = n_hidden

        # lstm1, lstm2, linear
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x, future=0):
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


if __name__ == "__main__":

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # y.shape = 100, 1000
    # 4-100
    train_input = torch.from_numpy(y[3:, :-1]).to(device) # 97, 999
    train_target = torch.from_numpy(y[3:, 1:]).to(device) # 97, 999
    # 1,2,3
    test_input = torch.from_numpy(y[:3, :-1]).to(device) # 3, 999
    test_target = torch.from_numpy(y[:3, 1:]).to(device) # 3, 999

    model = LSTM_Predictor().to(device)
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.8)

    n_steps = 10
    for i in range(n_steps):
        print("Step", i)
        
        model.train()

        def closure():
            optimizer.zero_grad()
            out = model(train_input).to(device)
            loss = criterion(out, train_target)
            print("loss", loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)


        model.eval()
        future = 1000
        pred = model(test_input, future=future) # pred.shape = 3, 1999
        loss = criterion(pred[:, :-future], test_target)
        print("Test loss", loss.item())
        y = pred.cpu().detach().numpy()


        plt.figure(figsize=(12, 6))
        plt.title(f"Step {i+1}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        n = train_input.shape[1]
        def draw(y_i, color):
            plt.plot(np.arange(n), y_i[:n], color, linewidth=2.0)
            plt.plot(np.arange(n, n+future), y_i[n:], color + ":", linewidth=2.0)
        draw(y[0], 'r')
        draw(y[1], 'b')
        draw(y[2], 'g')
        plt.savefig("predict%d.pdf"%i)
        plt.close()

