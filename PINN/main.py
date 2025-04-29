from torch.utils.data import DataLoader
import Data_Set as dataSet
import torch
import WAlg as loss
import modules
import training
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_epochs = 2000
lr = 0.001
steps_til_summary = 10
opt_model = 'sine'
mode = 'pinn'
clip_grad = 1.0
total_length = 1
max_epochs_without_improvement = 100

W = 0.4
H = 0.2
T = 0.002
p0 = 0.15 * 1e4
den = 405
T0 = 0.5

paf = 1e9
C = torch.tensor([
    [10.8, 0, 0],
    [0, 0.8424, 0],
    [0, 0, 0.697]
])

ni = torch.tensor([0.372, 0.04, 0])
fac = T**3/(12*(1-ni[0]*ni[1]))
gac = (1/12) * T**3
D = [[0 for _ in range(3)] for _ in range(3)]
D[0][0] = fac * C[0][0] * paf
D[1][1] = fac * C[1][1] * paf
D[1][0] = fac * C[1][1] * ni[0] * paf
D[0][1] = fac * C[1][1] * ni[0] * paf
D[2][2] = gac * C[2][2] * paf
D = torch.tensor(D)
print('D:', D)

x0 = torch.tensor(0.1)
y0 = torch.tensor(0.1)
t0 = torch.tensor(0.0)
epsilon = torch.tensor(0.01)
epsilon_t = torch.tensor(0.01)


def load(x, y, t):
    dx = (x - x0) ** 2
    dy = (y - y0) ** 2
    dt = (t - t0) ** 2
    g_x = torch.exp(-dx / (2.0 * epsilon**2))
    g_y = torch.exp(-dy / (2.0 * epsilon**2))
    g_t = torch.exp(-dt / (2.0 * epsilon_t**2))
    g = g_x*g_y*g_t

    g = p0*g
    return g

plate = dataSet.KirchhoffDataset(T=T, W=W, H=H, D=D, load=load, T0=T0,
                                 total_length=total_length, den=den, device=device)
data_loader = DataLoader(plate, shuffle=True, pin_memory=False, num_workers=0)
model = modules.PINNet(out_features=1, type=opt_model, mode=mode)
model.to(device)

history_loss = {'L_f': [], 'L_b0': [], 'L_b2': [], 'L_err': []}
history_lambda = {'L_f_lambda': [], 'L_b0_lambda': [], 'L_b2_lambda': [], 'L_err_lambda': []}

loss_fn = loss.KirchhoffLoss(model, plate)
kirchhoff_metric = loss.KirchhoffMetric(plate)
metric_lam = loss.LambdaMetric(loss_fn)

training.train(model=model, train_dataloader=data_loader, epochs=num_epochs, lr=lr,
               steps_til_summary=steps_til_summary, loss_fn=loss_fn, history_loss=history_loss,
               history_lambda=history_lambda,
               metric=kirchhoff_metric, metric_lam=metric_lam, clip_grad=clip_grad,
               use_lbfgs=False, max_epochs_without_improvement=max_epochs_without_improvement)
model.eval()

times = [0.025, 0.1, 0.3, 0.425]
nmses = []

for time in times:
    nmse = plate.visualise(model, time)
    nmses.append(nmse)
print('nmses:', nmses)