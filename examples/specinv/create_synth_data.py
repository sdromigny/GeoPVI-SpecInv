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
# Vs_true=np.load("/home/sixtine/Documents/SANT/GeoPVI-SpecInv/examples/specinv/input/vs3d_true_xyz.npy")

import numpy as np

def build_synthetic_vs(nx=29, ny=28, nz=20):
    # --- coordinates ---
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(0, 1, nz)   # depth normalized

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # --- background: increasing with depth ---
    Vs_bg = 2.0 + 1.0 * Z   # from ~2.0 to ~3.0 km/s

    # --- low velocity anomaly (centered) ---
    low_anomaly = -0.5 * np.exp(
        -((X/0.4)**2 + (Y/0.4)**2 + ((Z-0.5)/0.3)**2)
    )

    # --- high velocity anomaly (slightly offset) ---
    high_anomaly = +0.4 * np.exp(
        -(((X-0.4)/0.3)**2 + ((Y+0.3)/0.3)**2 + ((Z-0.6)/0.2)**2)
    )

    Vs = Vs_bg + low_anomaly + high_anomaly

    return Vs


Vs_true = build_synthetic_vs(10, 10, 20)

print(Vs_true.shape)

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(Vs_true[:, :, 20//2].T, origin='lower')
plt.colorbar(label="Vs (km/s)")
plt.title("Mid-depth slice")
plt.savefig("results/true_vel.png")


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

radius=2

A, centers, pair_counts = build_A_subarrays(
    x,   # 1D
    y,   # 1D
    stations,
    radius=radius,
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


import os

out_dir = "/home/sixtine/Documents/SANT/GeoPVI-SpecInv/examples/specinv/input/spectra_nodes"
os.makedirs(out_dir, exist_ok=True)


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

    # --- get spatial coordinates ---
    ix = p // ny
    iy = p % ny

    xp = x[ix]
    yp = y[iy]

    # --- package data ---
    spec_dict = {
        "E": E.astype(np.float32),
        "c_axis": c_axis.astype(np.float32),
        "periods": periods.astype(np.float32),
        "c_obs": c_obs[p].astype(np.float32),
        "c_true": c_true[p].astype(np.float32),
        "idx": int(p),
        "x": float(xp),
        "y": float(yp),
        "pair_count": int(pair_counts[p]),
        "radius": float(radius),
        "A_row": A[p].astype(np.float32),
    }

    # --- save ---
    fname = os.path.join(out_dir, f"spectrum_node_{p:04d}.npy")
    np.save(fname, spec_dict)

    spectra.append(spec_dict)

np.save(os.path.join(out_dir, "valid_mask.npy"), valid)

index = []

for spec in spectra:
    index.append({
        "idx": spec["idx"],
        "x": spec["x"],
        "y": spec["y"],
        "file": f"spectrum_node_{spec['idx']:04d}.npy"
    })

np.save(os.path.join(out_dir, "index.npy"), index)

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

