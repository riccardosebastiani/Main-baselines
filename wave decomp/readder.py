import torch
import numpy as np

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_lines = [line for line in lines if not line.startswith('%')]
    data = np.array([list(map(float, line.split())) for line in data_lines])

    x = data[:, 0]
    y = data[:, 1]
    values = data[:, 2:]
    return x, y, values


def create_tensor_from_data(file_path):
    x, y, values = read_file(file_path)

    num_times = values.shape[1]
    times = np.linspace(0, 0.5, num_times)

    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    times_tensor = torch.tensor(times, dtype=torch.float32)
    values_tensor = torch.tensor(values, dtype=torch.float32)

    return x_tensor, y_tensor, times_tensor, values_tensor


def extract_values(x_tensor, y_tensor, t_tensor, values_tensor, x, y, t):

    x_np = x_tensor.numpy()
    y_np = y_tensor.numpy()
    t_np = t_tensor.numpy()
    values_np = values_tensor.numpy()

    x = x.clone().cpu().detach().numpy()
    y = y.clone().cpu().detach().numpy()
    t = t.clone().cpu().detach().numpy()

    # Nearest indexes m
    result = []
    for xi, yi, ti in zip(x, y, t):
        idx = np.argmin(np.abs(x_np - xi) + np.abs(y_np - yi))
        t_index = np.argmin(np.abs(t_np - ti))
        if 0 <= t_index < values_np.shape[1]:
            value = values_np[idx, t_index]
            if t_index == 0:
                value = value * 10 ** 2
            else:
                value = value * 10 ** 2
            result.append(value)
        else:
            result.append(np.nan)

    return torch.tensor(result, dtype=torch.float32).view(-1, 1)


def solu(X, Y, T, file_path):
    res = torch.zeros(X.shape)
    X_new = X[:, 0, 0].view(1,-1).squeeze()
    Y_new = Y[0,:,0].view(1,-1).squeeze()
    x,y = torch.meshgrid(X_new, Y_new, indexing="ij")
    x = x.reshape(1,-1).squeeze()
    y = y.reshape(1,-1).squeeze()
    t_vals = torch.linspace(0, 0.5, 21)
    for i in range(21):
        t = t_vals[i]
        t_tensor = torch.ones(x.shape)
        t_tensor = t_tensor*t
        x_tensor, y_tensor, times_tensor, values_tensor = create_tensor_from_data(file_path)
        res[:,:,i] = extract_values(x_tensor, y_tensor, times_tensor, values_tensor, x, y, t_tensor).reshape(50,50)
    return res