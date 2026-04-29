import numpy as np
import matplotlib.pyplot as plt

samples=np.load("/home/sixtine/Documents/SANT/GeoPVI-SpecInv/examples/specinv/output/Linear_structured_samples.npy")

nsamples, ndim = samples.shape
nx, ny, nz = 10, 10, 20

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
plt.show()