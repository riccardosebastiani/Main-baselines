import torch
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import dataio
from readfunc import sol
import readder

EPS = 1e-6


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def normalize(x,y,t):
    return x / 0.4, y / 0.2, t

def denormalize(x,y,t):
    return x * 0.4, y * 0.2, t


def compute_derivatives(x, y, t, u):

    dudx = gradient(u, x)
    dudy = gradient(u, y)
    dudt = gradient(u, t)

    dudxx = gradient(dudx, x)
    dudyy = gradient(dudy, y)
    dudtt = gradient(dudt, t)

    dudxxx = gradient(dudxx, x)
    dudxxy = gradient(dudxx, y)
    dudyyy = gradient(dudy, y)

    dudxxxx = gradient(dudxxx, x)
    dudxxyy = gradient(dudxxy, y)
    dudyyyy = gradient(dudyyy, y)

    return dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy, dudtt, dudt


def compute_moments(Dx, Dy, Dxy, dudxx, dudyy):
    mx = - Dx * dudxx - Dxy * dudyy
    my = - Dxy * dudxx - Dy * dudyy

    return mx, my

def unpack(x, y, t, preds, norm, sx, sy):
    preds = np.squeeze(preds, axis=0)
    x = np.squeeze(x, axis=0)
    y = np.squeeze(y, axis=0)
    t = np.squeeze(t, axis=0)

    u = np.squeeze(preds[:, 0:1])

    if norm=='y':
        dudxx = np.squeeze((sx**2)*preds[:, 1:2])
        dudyy = np.squeeze((sy**2)*preds[:, 2:3])
        dudxxxx = np.squeeze((sx**4)*preds[:, 3:4])
        dudyyyy = np.squeeze((sy**4)*preds[:, 4:5])
        dudxxyy = np.squeeze((sy**2)*(sx**2)*preds[:, 5:6])
        dudtt = np.squeeze(preds[:, 6:7])
        dudt = np.squeeze(preds[:, 7:8])

    elif norm=='n':

        sx = 1
        sy = 1

        dudxx = np.squeeze((sx ** 2) * preds[:, 1:2])
        dudyy = np.squeeze((sy ** 2) * preds[:, 2:3])
        dudxxxx = np.squeeze((sx ** 4) * preds[:, 3:4])
        dudyyyy = np.squeeze((sy ** 4) * preds[:, 4:5])
        dudxxyy = np.squeeze((sy ** 2) * (sx ** 2) * preds[:, 5:6])
        dudtt = np.squeeze(preds[:, 6:7])
        dudt = np.squeeze(preds[:, 7:8])


    return u, x, y, t, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy, dudtt, dudt


class KirchhoffDataset(Dataset):

    def __init__(self, T, H, W, D, load, T0, total_length, den, device):

        self.T = T
        self.H = H
        self.W = W
        self.sx = int(1/0.4)
        self.sy = int(1/0.2)
        self.w = self.W * self.sx
        self.h = self.H * self.sy
        self.num_terms = 5
        self.total_length = total_length
        self.den = den
        self.D = D
        self.p = load
        self.T0 = T0
        self.device = device
        self.count = 0
        self.full_count = 20


    def __getitem__(self, item):
        x, y, t = self.training_batch()
        x,y,t = normalize(x,y,t)
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        xyt = torch.cat([x, y, t], dim=-1)
        return {'coords': xyt}

    def __len__(self):
        return self.total_length

    def training_batch(self) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:

        cc = dataio.get_mgrid(50, dim=2)
        x_in, y_in = cc[:, 0], cc[:, 1]
        x = (x_in + 1) * (self.W / 2)
        y = (y_in + 1) * (self.H / 2)

        t = torch.zeros(x.shape[0]).uniform_(0, self.T0 * (self.count / self.full_count)).squeeze()

        x = x[..., None].clone().detach().to(self.device)
        y = y[..., None].clone().detach().to(self.device)
        t = t[..., None].clone().detach().to(self.device)

        if self.count == self.full_count:
            self.count = 1
        else:
            self.count += 1
        return x, y, t

    def compute_loss(self, x, y, t, preds):

        u, x, y, t, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy, dudtt, dudt = unpack(x, y, t, preds, 'y',self.sx, self.sy)

        #masks for boundaries
        maskbcx = ((x <= 0.0 + EPS) | (x >= self.w - EPS))
        maskbcy = ((y <= 0.0 + EPS) | (y >= self.h - EPS))
        maskbc = (maskbcx | maskbcy)

        #First boundary condition
        L_b0 = u[maskbc]**2

        #Second boundary condition
        mx, my = compute_moments(self.D[0, 0], self.D[1, 1], self.D[0, 1], dudxx, dudyy)

        L_b2_x = mx[maskbcx] ** 2
        L_b2_y = my[maskbcy] ** 2
        L_b2 = torch.mean(L_b2_x) + torch.mean(L_b2_y)

        #Governing equation loss
        x,y,t = denormalize(x,y,t)
        f = self.D[0, 0] * dudxxxx + 2 * (self.D[0, 1] + 2 * self.D[2, 2]) * dudxxyy + self.D[1, 1] * dudyyyy - self.den * self.T * dudtt + self.p(x, y, t)
        L_f = f ** 2

        #mask for point selection percentage
        mask = np.arange(u.shape[0]) % 5 == 1 #20% known points

        u_real = sol(x, y, t, 'datiesportati').squeeze().to(self.device)
        err_t = u_real - u
        err_t = err_t[mask]

        L_err = err_t ** 2

        return 1e-16*L_f, 0*L_b0, 0*L_b2, L_err

    def validation_batch(self, snap, grid_width=32, grid_height=32):
        x, y = np.mgrid[0:self.W:complex(0, grid_width), 0:self.H:complex(0, grid_height)]

        x = torch.tensor(x.reshape(1, grid_width * grid_height, 1), dtype=torch.float32, device=self.device)
        y = torch.tensor(y.reshape(1, grid_width * grid_height, 1), dtype=torch.float32, device=self.device)
        t = torch.full((1, grid_width * grid_height, 1), snap, dtype=torch.float32, device=self.device)

        return x, y, t

    def __validation_results(self, pinn, snap, image_width=32, image_height=32):
        x, y, t = self.validation_batch(snap, image_width, image_height)

        x,y,t = normalize(x,y,t)

        c = {'coords': torch.cat([x, y, t], dim=-1).float()}
        pred = pinn(c)['model_out']

        u_pred, x, y, t, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy, dudtt, dudt = unpack(x, y, t, pred, 'n', self.sx, self.sy)
        x,y,t = denormalize(x,y,t)
        x = x.reshape(image_width * image_height, 1)
        y = y.reshape(image_width * image_height, 1)
        t = t.reshape(image_width * image_height, 1)
        mx, my = compute_moments(self.D[0,0], self.D[1,1], self.D[0,1], dudxx, dudyy)
        f = self.D[0, 0] * dudxxxx + 2 * (self.D[0, 1] + 2 * self.D[2, 2]) * dudxxyy + self.D[1, 1] * dudyyyy - self.den * self.T * dudtt
        p = self.p(x, y, t)
        u_real = sol(x,y,t, 'datiesportati').squeeze().to(self.device)
        uxx, uyy, utt = readder.sol(x, y, t, 'newtimes')
        return u_real, u_pred, mx, my, f, p, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy, dudtt, dudt, uxx.squeeze(), uyy.squeeze(), utt.squeeze()

    def visualise(self, pinn=None, snap=0, image_width=64, image_height=64):
        u_real, u_pred, mx, my, f, p, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy, dudtt, dudt, uxx, uyy, utt = self.__validation_results(pinn, snap, image_width, image_height)

        if snap==0.0:
            u_real = u_real/(10**12)
            u_pred = u_pred/(10**12)
        else:
            u_real = u_real / (10**4)
            u_pred = u_pred / (10**4)

        maxim = torch.max(u_real)
        minim = torch.min(u_real)

        u_real = u_real.cpu().detach().numpy().reshape(image_width, image_height)
        u_pred = u_pred.cpu().detach().numpy().reshape(image_width, image_height)

        NMSE = self.calc_nmse(u_real, u_pred)

        fig, axs = plt.subplots(1, 3, figsize=(16, 6.2))
        self.__show_image(u_pred, axs[0], 'Predicted Displacement (m)', max=maxim, min=minim)
        self.__show_image(u_real, axs[1], 'Real Displacement (m)', max=maxim, min=minim)
        self.__show_image((u_real - u_pred)**2, axs[2],'Squared Error Displacement (m)')

        for ax in axs.flat:
            ax.label_outer()

        plt.tight_layout()
        plt.show()

        return NMSE

    def __show_image(self, img, axis=None, title='', x_label='x [m]', y_label='y [m]', z_label='', max=None, min=None):

        im = axis.imshow(np.rot90(img, k=1), cmap='plasma', origin='upper', aspect='auto', vmin=min, vmax=max)
        cb = plt.colorbar(im, label=z_label, ax=axis)
        axis.set_aspect(0.5)
        axis.set_xticks([0, img.shape[0] - 1])
        axis.set_xticklabels([0, self.W])
        axis.set_yticks([0, img.shape[1] - 1])
        axis.set_yticklabels([0, self.H])
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(title)
        return im

    def calc_nmse(self, ur,up):
        mse = np.mean((ur - up) ** 2)
        var_u_true = np.var(ur)
        nmse = mse / var_u_true
        return nmse

    def ncc(self, ur,up):

        mu_r = np.mean(ur)
        mu_p = np.mean(up)
        num = np.sum((ur - mu_r) * (up - mu_p))
        den_r = np.sqrt(np.sum((ur - mu_r)**2))
        den_p = np.sqrt(np.sum((up - mu_p)**2))
        den = den_r * den_p

        ncc = num/den
        return ncc