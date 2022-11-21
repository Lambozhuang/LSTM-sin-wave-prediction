import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from lstm import LSTM_Predictor
from rnn import RNN_Predictor

N = 100 # number of samples
L = 1000 # length for each sample
T = 50 # width of the wave


if __name__ == "__main__":

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
    plt.plot(np.arange(x.shape[1]), y[1, :], 'b', linewidth=2.0)
    plt.plot(np.arange(x.shape[1]), y[2, :], 'g', linewidth=2.0)
    plt.savefig("target.pdf")
    plt.close()

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

    model = LSTM_Predictor(device=device).to(device)
    # model = RNN_Predictor(device=device).to(device)

    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.8)

    n_steps = 15
    for i in range(n_steps):
        print("Step", i)
        
        def closure():
            optimizer.zero_grad()
            out = model(train_input, device=device).to(device)
            loss = criterion(out, train_target)
            print("loss", loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)


        with torch.no_grad():
            future = 1000
            pred = model(test_input, future=future, device=device) # pred.shape = 3, 1999
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

