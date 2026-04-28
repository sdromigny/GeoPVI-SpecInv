import numpy as np
from create_array import *
from geopvi.forward.swi.posterior import forward_sw
import matplotlib.pyplot as plt


def build_spectrum(c_obs, c_axis, sigma_c):
    """
    c_obs: (Nperiods,)
    c_axis: (Nc,)
    """
    Nperiods = len(c_obs)
    Nc = len(c_axis)

    E = np.zeros((Nperiods, Nc))

    for i in range(Nperiods):
        E[i] = np.exp(-0.5 * ((c_axis - c_obs[i]) / sigma_c[i])**2)

    return E

# Load true Vs model
Vs_true=np.load("/home/sixtine/Documents/SANT/GeoPVI-SpecInv/examples/specinv/input/vs3d_true_xyz.npy")

print(Vs_true.shape)


# Define periods in seconds, and grid dimensions
prior_bounds = np.loadtxt('/home/sixtine/Documents/SANT/GeoPVI-SpecInv/examples/specinv/input/prior_20_top.txt')
thickness = prior_bounds[:,0]

periods = np.array([4,6,8,9,10,11,12,15])
Nperiods = len(periods)
nx, ny, nz = Vs_true.shape
Ngrid = nx * ny   # assuming 1D model per (x,y)
dx = 1.0  # km or whatever
dy = 1.0

x = np.arange(nx) * dx
y = np.arange(ny) * dy

grid_x, grid_y = np.meshgrid(x, y, indexing='ij')

# Define true phase velocity maps
c_true = np.zeros((Ngrid, len(periods)))

idx = 0
for ix in range(nx):
    for iy in range(ny):
        vs_profile = Vs_true[ix, iy, :]   # shape (nz,)
        c_true[idx], _ = forward_sw(
            vs_profile,
            periods,
            thickness,
            requires_grad=False
        )
        idx += 1


# Apply averaging kernel

stations=define_station_geometry(
    x,
    y,
    nstations=200,
    geometry="random",
    seed=42,
    margin=0,
    line_axis="x",)

A, centers, pair_counts = build_A_subarrays(
    x,   # 1D
    y,   # 1D
    stations,
    radius=3,
    n_samples=100
)

np.save("/home/sixtine/Documents/SANT/GeoPVI-SpecInv/examples/specinv/input/A_matrix.npy",A)

s_true = 1.0 / c_true          # (Ngrid, Nperiods)
s_obs  = A @ s_true            # (Nobs, Nperiods)
valid = A.sum(axis=1) > 0

c_obs = np.full_like(s_obs, np.nan)
c_obs[valid] = 1.0 / s_obs[valid]

c_axis = np.linspace(0.5, 4, 200)

Nobs = c_obs.shape[0]

# Spectra
spectra = []

for p in range(Nobs):
    if not valid[p]:
        continue

    sigma_c = 0.05 * c_obs[p]
    sigma_c = np.clip(sigma_c, 0.05, 0.1)

    E = build_spectrum(c_obs[p], c_axis, sigma_c)
    E /= (E.max(axis=1, keepdims=True) + 1e-12)

    E += 0.01 * np.random.randn(*E.shape)
    E = np.clip(E, 0, None)

    spectra.append({
        "E": E,
        "c_axis": c_axis,
        "idx": p   # 👈 CRITICAL
    })

spec = spectra[0]   # pick any valid one
p = spec["idx"]

E = spec["E"]

plt.figure()
plt.imshow(E, aspect='auto',
           extent=[c_axis.min(), c_axis.max(),
                   periods.max(), periods.min()])

plt.scatter(c_obs[p], periods, color='r', label='c_obs')
plt.plot(c_true[p], periods, color='white', linewidth=2, label='c_true')

plt.gca().invert_yaxis()
plt.legend()

plt.title(f"Spectrum at node {p}")
plt.savefig("synth_spec.png")


coverage = A.sum(axis=1)   # (Nnodes,)

coverage_map = coverage.reshape(nx, ny)

plt.figure()

plt.imshow(coverage_map.T, origin='lower', aspect='auto')
plt.scatter(stations["x"],stations["y"])
plt.colorbar(label="Ray coverage (sum of weights)")
plt.title("A matrix coverage")
plt.xlabel("x index")
plt.ylabel("y index")
plt.savefig("coverage_map.png")

pair_map = pair_counts.reshape(nx, ny)

plt.figure()
plt.imshow(pair_map.T, origin='lower', aspect='auto')
plt.colorbar(label="Number of station pairs")
plt.title("Subarray pair counts")
plt.savefig("pair_counts.png")

plt.figure(figsize=(10, 4))

for i, p in enumerate([0, Ngrid//4, Ngrid//2, 3*Ngrid//4]):
    plt.subplot(1, 4, i+1)
    kernel = A[p].reshape(nx, ny)
    plt.imshow(kernel.T, origin='lower')
    plt.title(f"Node {p}")
    plt.axis('off')

plt.suptitle("Example averaging kernels")
plt.savefig("A_kernels.png")

np.save('/home/sixtine/Documents/SANT/GeoPVI-SpecInv/examples/specinv/input/specs.npy',E)
