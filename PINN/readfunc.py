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

    # Transform in PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    times_tensor = torch.tensor(times, dtype=torch.float32)
    values_tensor = torch.tensor(values, dtype=torch.float32)

    return x_tensor, y_tensor, times_tensor, values_tensor


def extract_values(x_tensor, y_tensor, t_tensor, values_tensor, x, y, t):
    # Find index for each thruple (x, y, t)
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
                value = value * 10**12
            else:
                value = value * 10**4
            result.append(value)
        else:
            result.append(np.nan)  #If index is out of range

    return torch.tensor(result, dtype=torch.float32).view(-1, 1)  #Shape (N, 1)

def sol(x,y,t, file_path):
    x_tensor, y_tensor, t_tensor, values_tensor = create_tensor_from_data(file_path)
    result_tensor = extract_values(x_tensor, y_tensor, t_tensor, values_tensor, x, y, t)
    return result_tensor
