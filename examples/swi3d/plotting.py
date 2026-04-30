import numpy as np
import matplotlib.pyplot as plt



samples=np.load("/home/sixtine/Documents/SANT/GeoPVI-SpecInv/examples/swi3d/output/samples_intermediate_ite8000.npy")

nsamples, ndim = samples.shape
nx, ny, nz = 29, 28, 20

Vs = samples.reshape(nsamples, nx, ny, nz)

Vs_mean = Vs.mean(axis=0)   # (nx, ny, nz)

z_mid = nz // 2

plt.figure(figsize=(6,5))
plt.imshow(Vs_mean[:, :, z_mid].T, origin='lower', cmap='viridis')
plt.colorbar(label="Vs (km/s)")
plt.title(f"Mean Vs model (z = {z_mid})")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("mean_vs_slice.png")



# Reshape
samples_3d = samples.reshape(2000, nx, ny, nz)

# Mean and std of posterior
mean_model = samples_3d.mean(axis=0)   # (nx, ny, nz)
std_model  = samples_3d.std(axis=0)    # uncertainty

# Compare to true
Vs_true = np.load('/home/sixtine/Documents/SANT/GeoPVI-SpecInv/examples/swi3d/input/vs3d_true_xyz.npy')
residual = mean_model - Vs_true

print(f"RMSE: {np.sqrt((residual**2).mean()):.4f} km/s")
print(f"Max error: {np.abs(residual).max():.4f} km/s")
print(f"Mean uncertainty: {std_model.mean():.4f} km/s")

fig, axes = plt.subplots(3, nz//4, figsize=(16, 10))
depths = [5, 10, 15]  # layer indices

for i, iz in enumerate(depths):
    axes[0,i].imshow(Vs_true[:,:,iz].T, origin='lower', vmin=1.5, vmax=3.5)
    axes[0,i].set_title(f'True, layer {iz}')
    
    axes[1,i].imshow(mean_model[:,:,iz].T, origin='lower', vmin=1.5, vmax=3.5)
    axes[1,i].set_title(f'Recovered, layer {iz}')
    
    im = axes[2,i].imshow(std_model[:,:,iz].T, origin='lower', cmap='hot_r')
    axes[2,i].set_title(f'Uncertainty, layer {iz}')

plt.savefig("compare.png")