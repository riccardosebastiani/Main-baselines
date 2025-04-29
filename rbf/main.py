from scipy.interpolate import Rbf
import math
import torch
import seaborn as sns
import numpy as np
from readder import solu
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def show_image(img, axis=None, title='', x_label='x [m]', y_label='y [m]', z_label='', max=None, min=None):
    im = axis.imshow(np.rot90(img, k=1), cmap='plasma', origin='upper', aspect='auto', vmin=min, vmax=max)
    cb = plt.colorbar(im, label=z_label, ax=axis)
    axis.set_aspect(0.5)
    axis.set_xticks([0, img.shape[0] - 1])
    axis.set_xticklabels([0, 0.4])
    axis.set_yticks([0, img.shape[1] - 1])
    axis.set_yticklabels([0, 0.2])
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    return im

def calc_nmse(ur, up):
    mse = np.mean((ur - up) ** 2)
    var_u_true = np.var(ur)
    nmse = mse / var_u_true
    return nmse

def rbf_interpolate_torch(X, Y, F, imgn, mask, method='thin_plate'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smooth = 5*1e-5

    X, Y, F, imgn, mask = map(lambda x: torch.tensor(x, dtype=torch.float32, device=device), [X, Y, F, imgn, mask])


    known_idx = torch.where(mask == 1)
    unknown_idx = torch.where(mask == 0)


    X_k = X[known_idx]
    Y_k = Y[known_idx]
    F_k = F[known_idx]
    V_k = imgn[known_idx]


    X_u = X[unknown_idx]
    Y_u = Y[unknown_idx]
    F_u = F[unknown_idx]


    pk = torch.stack([X_k, Y_k, F_k], dim=1).cpu().numpy()
    vk = V_k.cpu().numpy()
    pu = torch.stack([X_u, Y_u, F_u], dim=1).cpu().numpy()


    rbfi = Rbf(pk[:, 0], pk[:, 1], pk[:, 2], vk, function=method, smooth=smooth)


    pred = rbfi(pu[:, 0], pu[:, 1], pu[:, 2])


    ti = imgn.clone()


    ti[unknown_idx] = torch.tensor(pred, dtype=torch.float32, device=device)


    nmse = 20 * torch.log10(torch.mean((V_k - ti[known_idx]) ** 2) / torch.mean(V_k ** 2))
    ncc = 100 * torch.abs(
        torch.dot(V_k.flatten(), ti[known_idx].flatten()) / (torch.norm(V_k) * torch.norm(ti[known_idx]))
    )

    return ti.cpu().numpy(), V_k.cpu().numpy(), nmse.item(), ncc.item()



dim_x, dim_y, dim_t = 50, 50, 21


x_vals = torch.linspace(0, 0.4, dim_x)
y_vals = torch.linspace(0, 0.2, dim_y)
t_vals = torch.linspace(0, 0.5, dim_t)


X, Y, T = torch.meshgrid(x_vals, y_vals, t_vals, indexing="ij")

imgn = solu(X,Y,T,'datiesportati').numpy()


X, Y, T = X.numpy(), Y.numpy(), T.numpy()

percentages = [0.2, 0.15, 0.1, 0.05, 0.01, 0.005]

nmses_prec = torch.zeros(len(percentages), 12)
data_pde = torch.zeros(len(percentages), 12)
pp = 0
for p in percentages:
    mask = np.zeros_like(imgn)

    n_dir = 3
    num_x, num_y, num_t = X.shape
    sf = n_dir / num_t
    n = num_x * num_y * sf
    pps = p*n+2*n_dir
    num_known_points = int(math.sqrt(pps))

    x_indices = np.linspace(0, num_x - 1, num_known_points, dtype=int)
    y_indices = np.linspace(0, num_y - 1, num_known_points, dtype=int)
    t_indices = np.linspace(0, num_t - 1, 21, dtype=int)


    for t in t_indices:
        for x in x_indices:
            for y in y_indices:
                mask[x, y, t] = 1


    ti, V_u, nmse, ncc = rbf_interpolate_torch(X, Y, T, imgn, mask, method="thin_plate")

    nmses = torch.zeros(21)

    for tidx in range(4):
        fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
        show_image(imgn[:, :, tidx], axs[0], "Dati Originali con Mancanze")
        show_image(ti[:, :, tidx], axs[1], "Dati Ricostruiti (Interpolazione RBF)")
        for ax in axs.flat:
            ax.label_outer()

        plt.tight_layout()
        plt.show()

    for t_idx in range(21):
        ur = imgn[:, :, t_idx].reshape(1, -1)
        up = ti[:, :, t_idx].reshape(1, -1)
        NMSE = calc_nmse(ur, up)
        nmses[t_idx] = float(NMSE)

    print('nmses:', nmses)

    times = [0.0, 0.025, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    tt = [0, 1, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    nmses_prec[pp, :] = nmses[tt]
    NMSES = nmses_prec[pp, :]
    for t in range(21):
        if t in tt:
            data_pde[pp, :] = NMSES[:].reshape(1, -1).squeeze().clone().detach().to(dtype=torch.float32)
    pp = pp + 1
print('data_pde:', data_pde, data_pde.shape, '\n')

def to_db(tensor):
    return 10 * torch.log10(tensor + 1e-10)


data_pde_db = to_db(data_pde)
print('data_pde_db:', data_pde_db, data_pde_db.shape, '\n')

# Creazione delle heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(data_pde_db.numpy(), ax=axes[0], cmap="viridis", xticklabels=times, yticklabels=percentages, annot=True, fmt=".2f", vmin=-21, vmax=7)
axes[0].set_title("rbf (dB)")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Known Points Percentage")

plt.tight_layout()
plt.show()
