import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from readder import solu

########################################################################################################################
############################               FIRST STEP             #####################################################
########################################################################################################################

#PLOT function
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

# 50x50 spatial points, 20 temporal points
dim_x, dim_y, dim_t = 50, 50, 21

#X, Y, T sampling
x_val = torch.linspace(0, 0.4, dim_x)  # X between 0 and 0.4
y_val = torch.linspace(0, 0.2, dim_y)  # Y between tra 0 and 0.2
t_vals = torch.linspace(0, 0.5, dim_t)  # Time between 0 and 0.5
x, y = torch.meshgrid(x_val, y_val, indexing="ij")
x_vals = x.reshape(-1, 1).squeeze()
y_vals = y.reshape(-1, 1).squeeze()

t_max, num_t = 2.0, 21  #Temporal interval parameters

#Data Structure
t_val = torch.linspace(0, 0.5, 21)  # Time between 0 and 0.5
X,Y,T = torch.meshgrid(x_val, y_val, t_val, indexing="ij")
U_nomask = solu(X,Y,T,'datiesportati').numpy()

#Sparse masking
msk = np.zeros_like(U_nomask)
perc = 0.5 #percentage of known points
mak = np.zeros_like(U_nomask)
nm_x, nm_y, nm_t = X.shape
n = nm_x * nm_y
num_known_points = int(math.sqrt(perc * n))  # 20% dei punti x,y

x_indices = np.linspace(0, nm_x - 1, num_known_points, dtype=int)
y_indices = np.linspace(0, nm_y - 1, num_known_points, dtype=int)
t_indices = np.linspace(0, nm_t - 1, 20, dtype=int)

#Imposing mask=1 solely for selected known points
for tt in t_indices:
    for xx in x_indices:
        for yy in y_indices:
            msk[xx, yy, tt] = 1
U_nomask = torch.tensor(U_nomask).reshape(2500,21)
msk = msk.reshape(2500,21)
U = (U_nomask*msk).T
U_nomask = U_nomask.T


#Frequency, Damping and Phase deduction using SOMP
def somp(Y, D, limit_residue=1e-3, max_iters=100):
    assert D.shape[0] == Y.shape[0], "Dimentions Mismatch"

    residual = Y.clone()
    support = []
    X = torch.zeros((D.shape[1], Y.shape[1]))

    norma_Y = torch.norm(Y)

    for it in range(max_iters):
        #Correlation calculation
        corr = D.T @ residual  # [n_atoms, n_signals]
        summed_corr = torch.sum(torch.abs(corr), dim=1)

        #Best index selection
        best_idx = torch.argmax(summed_corr).item()

        if best_idx in support:
            break

        support.append(best_idx)

        #Update solution
        D_sub = D[:, support]
        X_sub = torch.linalg.lstsq(D_sub, Y).solution
        X[support, :] = X_sub

        #Update Residual
        residual = Y - D_sub @ X_sub
        residuo_norm_relativo = torch.norm(residual) / norma_Y

        #Check convergence
        if residuo_norm_relativo.item() < limit_residue:
            print(f"Convergence at step {it + 1}!")
            break

    return support, X


n_freq = 28 #number of frequency samples
n_damps = 40 #numer of damping coefficient samples
n_phi = 2 #number of phases samples
M = 18 #number of spatial directions
Ksteps = 41 #number of k values samples

Kmax = 100 #Max value for k sampling range
omega = 2*torch.pi #frequency range factor [-omega, omega]


#Coefficient sampling
freqs = torch.linspace(0, omega - omega/(n_freq + 1), n_freq + 1)     # Frequencies
damps = torch.linspace(0, 1, n_damps)                            #Damping
phi = torch.linspace(-torch.pi, torch.pi - torch.pi/(n_phi + 1), n_phi + 1) #Phases

#List of coefficients combinations
freq_damp_phi_truples = [(f.item(), a.item(), p.item()) for f in freqs for a in damps for p in phi]

D = torch.zeros(dim_t, (n_freq+1) * n_damps * (n_phi+1))

for i, (f, a, p) in enumerate(freq_damp_phi_truples):
    D[:, i] = torch.exp(-a * t_vals) * torch.cos(f * t_vals + p)


def orthogonalize_dictionary_torch(D):
    """
    Ortogonalizza il dizionario D (torch) senza normalizzare.
    """
    Q, R = torch.linalg.qr(D)
    return Q


D = orthogonalize_dictionary_torch(D)



#SOMP call
support, coeffs = somp(U, D)




###########Evaluate coefficients

#Creating dictionary that associates each index to a truple (f, a, phi)
index_to_params = {i: truple for i, truple in enumerate(freq_damp_phi_truples)}
selected_params = [index_to_params[i] for i in support]

#Print the results
for i, (freq, damp, phi) in enumerate(selected_params):
    print(f"Modo {i+1}: Frequenza = {freq:.3f}, Damping = {damp:.3f}, Phase = {phi:.3f}")
print('support and coeffs:', support, coeffs.shape)


########################################################################################################################
############################               SECOND STEP             #####################################################
########################################################################################################################



def calculate_Wk(k0, grid_points, M):

    #angle sampling
    theta = torch.linspace(-math.pi, math.pi- 2*math.pi/M, M)

    #Calculate u direction on x and y
    u_theta = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

    #x*cos(theta) + y*sin(theta)
    scalar_products = grid_points @ u_theta.T

    num_points = grid_points.shape[0]  # Number of points on grid (N * N)
    Wk = torch.zeros((num_points, 2 * M), dtype=torch.complex64)  #2M for both complex and real waves

    for i in range(M):
        Wk[:, i] = torch.exp(1j * k0 * scalar_products[:, i])
        Wk[:, i+M] = torch.exp(k0 * scalar_products[:, i])/torch.max(torch.exp(k0 * scalar_products[:, i]))

    return Wk

def projection_operator(Wk):
    Wk_t = Wk.t()  # Transpose  Wk
    Wk_t_Wk = Wk_t @ Wk  # Calculate Wk^T Wk
    Wk_t_Wk_inv = torch.pinverse(Wk_t_Wk)  # Calculate pseudo-inverse Wk^T Wk
    Pk = Wk @ Wk_t_Wk_inv @ Wk_t  # Calculate Pk = Wk (Wk^T Wk)^-1 Wk^T
    return Pk


def optimize_k(m, k_values, grid_points, M, i):

    best_k0 = None
    max_norm = -float('inf')  #Initialization of norm's highest value
    projection_norms = []

    #cycle over k values
    for k0 in k_values:
        Wk = calculate_Wk(k0, grid_points, M)

        #Calculate Pk operator
        Pk = projection_operator(Wk)

        #Calculate projection Pk * m
        proj_m = torch.matmul(Pk, m)

        #Calculate projection norm
        norm = torch.norm(proj_m)
        projection_norms.append(norm.item())
        #Update optimal k0 value if necessary
        if norm > max_norm:
            max_norm = norm
            best_k0 = k0
    return best_k0

#turn it into a list of couples
grid_points = torch.stack([x.ravel(), y.ravel()], dim=1)

modes = coeffs[support, :] #recovering the spatial coefficients
k_values = torch.linspace(0, Kmax, Ksteps) #k sampling

#find k0 values in k_values using optimize_k
k0 = torch.zeros(len(support))
for i in range(len(support)):
    m = modes[i, :]
    m = torch.tensor(m, dtype=torch.complex64)
    k0[i] = optimize_k(m, k_values, grid_points, M, i)

print('k0:', k0)


########################################################################################################################
############################               LAST STEP             #####################################################
########################################################################################################################


w_nm = coeffs[support, :]

grid_points = torch.stack([x.ravel(), y.ravel()], dim=1) #turn it into a list of couples

w_xy = []

for modo in range(w_nm.shape[0]):

    Wk = calculate_Wk(k0[modo], grid_points, M)  #Waves matrix

    # Calculate Pk operator
    Pk = projection_operator(Wk)

    Wk_t = Wk.t()  # Transpose  Wk
    Wk_t_Wk = Wk_t @ Wk  # Calculate Wk^T Wk
    Wk_t_Wk_inv = torch.pinverse(Wk_t_Wk)  # Calculate pseudo-inverse Wk^T Wk
    Wk_penrose = Wk_t_Wk_inv @ Wk_t
    # Calculate projection Pk * m
    proj_m = Wk_penrose @ torch.tensor(modes[modo, :], dtype=torch.complex64)


    w_n_xy = (Wk * proj_m).T
    w_n = torch.zeros(w_n_xy.shape[1], dtype=torch.complex64)

    for mm in range(M):
        w_n += w_n_xy[mm, :] + w_n_xy[mm + M, :]

    w_n = w_n.real
    w_xy.append(w_n)

wxy = torch.stack(w_xy)



#Displacement recomposition
disp = D[:, support] @ wxy

#Coefficient reconstruction
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, t_idx in enumerate([0, 5, 7]):  # Tre istanti nel tempo
    show_image(w_xy[i].reshape(50, 50), axis=axes[i], title='coeffs')

plt.show()

#Reconstructed Displacement
t_vals = torch.linspace(0, 0.5, dim_t)  # Tempo tra 0 e 2s

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, t_idx in enumerate([0, 5, 7]):  # Tre istanti nel tempo
    show_image(disp[t_idx, :].reshape(50, 50), axis=axes[i], title=f't = {t_vals[t_idx]:.2f} s')

plt.show()

#Real Function
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, t_idx in enumerate([0, 5, 7]):  # Tre istanti nel tempo
    show_image(U_nomask[t_idx, :].reshape(50, 50), axis=axes[i], title=f'Real t = {t_vals[t_idx]:.2f} s')

plt.show()

#Masked Real Function
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, t_idx in enumerate([0, 5, 7]):  # Tre istanti nel tempo
    show_image(U[t_idx, :].reshape(50, 50), axis=axes[i], title=f'Masked t = {t_vals[t_idx]:.2f} s')

plt.show()