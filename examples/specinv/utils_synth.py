
import numpy as np
import numpy
import numpy as np

def define_station_geometry(
    x,
    y,
    nstations=20,
    geometry="random",
    side="left",
    seed=None,
    margin=0,
    line_axis="x",
):
    """
    Define station positions at the surface of a 3D velocity model.

    Parameters
    ----------
    x, y : array-like
        1D model coordinates in km (or whatever unit your model uses).
    nstations : int
        Number of stations to place.
    geometry : str
        One of:
        - "random"   : random surface stations across the whole domain
        - "side"     : stations placed along one side
        - "line"     : stations placed along a straight line at the surface
        - "grid"     : stations on a regular surface grid
        - "cluster"  : stations clustered near one edge/corner
    side : str
        Used when geometry="side" or "cluster".
        Options: "left", "right", "top", "bottom", "corner_ul",
                 "corner_ur", "corner_ll", "corner_lr"
    seed : int or None
        Random seed for reproducibility.
    margin : int
        Number of grid cells to avoid near boundaries.
    line_axis : str
        Used when geometry="line".
        "x" -> stations spread in x, fixed y
        "y" -> stations spread in y, fixed x

    Returns
    -------
    stations : dict
        {
            "x": station_x positions,
            "y": station_y positions,
            "z": station_z positions (all zeros),
            "ix": nearest x indices,
            "iy": nearest y indices
        }
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")

    nx = x.size
    ny = y.size

    if margin < 0:
        raise ValueError("margin must be >= 0.")
    if margin * 2 >= nx or margin * 2 >= ny:
        raise ValueError("margin is too large for the grid size.")

    x_use = x[margin:nx - margin] if margin > 0 else x
    y_use = y[margin:ny - margin] if margin > 0 else y

    def nearest_index(arr, vals):
        return np.array([np.argmin(np.abs(arr - v)) for v in vals], dtype=int)

    if geometry == "random":
        sx = rng.choice(x_use, size=nstations, replace=False if nstations <= x_use.size else True)
        sy = rng.choice(y_use, size=nstations, replace=False if nstations <= y_use.size else True)

    elif geometry == "side":
        if side == "left":
            sx = np.full(nstations, x_use[0])
            sy = rng.choice(y_use, size=nstations, replace=True)
        elif side == "right":
            sx = np.full(nstations, x_use[-1])
            sy = rng.choice(y_use, size=nstations, replace=True)
        elif side == "bottom":
            sy = np.full(nstations, y_use[0])
            sx = rng.choice(x_use, size=nstations, replace=True)
        elif side == "top":
            sy = np.full(nstations, y_use[-1])
            sx = rng.choice(x_use, size=nstations, replace=True)
        else:
            raise ValueError("side must be 'left', 'right', 'top', or 'bottom' for geometry='side'.")

    elif geometry == "line":
        if line_axis == "x":
            sx = np.linspace(x_use[0], x_use[-1], nstations)
            sy = np.full(nstations, y_use[len(y_use) // 2])
        elif line_axis == "y":
            sy = np.linspace(y_use[0], y_use[-1], nstations)
            sx = np.full(nstations, x_use[len(x_use) // 2])
        else:
            raise ValueError("line_axis must be 'x' or 'y'.")

    elif geometry == "grid":
        n_side = int(np.ceil(np.sqrt(nstations)))
        gx = np.linspace(x_use[0], x_use[-1], n_side)
        gy = np.linspace(y_use[0], y_use[-1], n_side)
        Xg, Yg = np.meshgrid(gx, gy, indexing="ij")
        sx = Xg.ravel()[:nstations]
        sy = Yg.ravel()[:nstations]

    elif geometry == "cluster":
        spread_x = (x_use[-1] - x_use[0]) * 0.08
        spread_y = (y_use[-1] - y_use[0]) * 0.08

        if side == "left":
            cx, cy = x_use[0], y_use[len(y_use)//2]
        elif side == "right":
            cx, cy = x_use[-1], y_use[len(y_use)//2]
        elif side == "bottom":
            cx, cy = x_use[len(x_use)//2], y_use[0]
        elif side == "top":
            cx, cy = x_use[len(x_use)//2], y_use[-1]
        elif side == "corner_ul":
            cx, cy = x_use[0], y_use[-1]
        elif side == "corner_ur":
            cx, cy = x_use[-1], y_use[-1]
        elif side == "corner_ll":
            cx, cy = x_use[0], y_use[0]
        elif side == "corner_lr":
            cx, cy = x_use[-1], y_use[0]
        else:
            raise ValueError(
                "side must be one of: left, right, top, bottom, corner_ul, corner_ur, corner_ll, corner_lr"
            )

        sx = np.clip(rng.normal(cx, spread_x, size=nstations), x_use[0], x_use[-1])
        sy = np.clip(rng.normal(cy, spread_y, size=nstations), y_use[0], y_use[-1])

    else:
        raise ValueError("geometry must be one of: random, side, line, grid, cluster")

    ix = nearest_index(x, sx)
    iy = nearest_index(y, sy)
    sz = np.zeros(nstations, dtype=float)

    return {
        "x": sx,
        "y": sy,
        "z": sz,
        "ix": ix,
        "iy": iy,
    }

import numpy as np
import pandas as pd

def read_station_csv(
    filepath,
    x,
    y,
    ref_lat=None,
    ref_lon=None,
):
    """
    Read station locations from CSV and convert to model coordinates.

    Parameters
    ----------
    filepath : str
        Path to CSV file with columns: Latitude, Longitude, Elevation, Name
    x, y : 1D arrays
        Model coordinates (in km)
    ref_lat, ref_lon : float or None
        Reference point for projection (defaults to mean of stations)

    Returns
    -------
    stations : dict
        Same structure as define_station_geometry()
    """

    df = pd.read_csv(filepath)

    lat = df["Latitude"].values
    lon = df["Longitude"].values
    elev = df["Elevation"].values
    names = df["Name"].values

    # --- Reference point ---
    if ref_lat is None:
        ref_lat = lat.mean()
    if ref_lon is None:
        ref_lon = lon.mean()

    # --- Convert to km ---
    # Convert to km (same as before)
    lat_km = (lat - lat.min()) * 111.0
    lon_km = (lon - lon.min()) * 111.0 * np.cos(np.deg2rad(lat.mean()))

    # --- Normalize to [0, 1]
    lat_norm = (lat_km - lat_km.min()) / (lat_km.max() - lat_km.min())
    lon_norm = (lon_km - lon_km.min()) / (lon_km.max() - lon_km.min())

    # --- Scale to model domain
    sx = x.min() + lon_norm * (x.max() - x.min())
    sy = y.min() + lat_norm * (y.max() - y.min())
    sz = elev     # or np.zeros_like(elev) if you want surface only

    # --- Snap to grid ---
    def nearest_index(arr, vals):
        return np.array([np.argmin(np.abs(arr - v)) for v in vals], dtype=int)

    ix = nearest_index(x, sx)
    iy = nearest_index(y, sy)

    return {
        "x": sx,
        "y": sy,
        "z": sz,
        "ix": ix,
        "iy": iy,
        "name": names,
    }

import numpy as np
from itertools import combinations

def build_A_subarrays(
    grid_x,
    grid_y,
    stations,
    radius,
    n_samples=100
):
    """
    Build averaging kernel A for FJ-style subarrays

    Parameters
    ----------
    grid_x, grid_y : 1D arrays
        Model grid coordinates
    stations : dict
        Output of define_station_geometry()
    radius : float
        Subarray radius (same units as x/y)
    n_samples : int
        Number of samples along each ray

    Returns
    -------
    A : (Nnodes, Ngrid) array
    centers : (Nnodes, 2) array of grid node coordinates
    pair_counts : (Nnodes,) number of pairs used
    """

    nx = len(grid_x)
    ny = len(grid_y)
    Ngrid = nx * ny

    # grid nodes (centers where you compute FJ)
    centers = np.array([
        [x, y] for x in grid_x for y in grid_y
    ])

    Nnodes = len(centers)

    # station coords
    sx = stations["x"]
    sy = stations["y"]
    station_coords = np.column_stack([sx, sy])

    def idx(ix, iy):
        return iy * nx + ix

    A = np.zeros((Nnodes, Ngrid))
    pair_counts = np.zeros(Nnodes, dtype=int)

    # loop over subarray centers
    for p_idx, (px, py) in enumerate(centers):

        # --- 1. select stations in radius ---
        d = np.sqrt((sx - px)**2 + (sy - py)**2)
        mask = d <= radius

        sub_stations = station_coords[mask]

        if len(sub_stations) < 2:
            continue

        # --- 2. build station pairs ---
        pairs = list(combinations(range(len(sub_stations)), 2))
        pair_counts[p_idx] = len(pairs)

        weights = np.zeros(Ngrid)

        # --- 3. loop over pairs ---
        for (i1, i2) in pairs:

            x1, y1 = sub_stations[i1]
            x2, y2 = sub_stations[i2]

            dx = x2 - x1
            dy = y2 - y1
            L = np.sqrt(dx**2 + dy**2)

            if L == 0:
                continue

            t = np.linspace(0, 1, n_samples)
            xs = x1 + t * dx
            ys = y1 + t * dy

            ds = L / (n_samples - 1)

            # --- 4. accumulate path density ---
            for x, y in zip(xs, ys):

                ix = np.searchsorted(grid_x, x) - 1
                iy = np.searchsorted(grid_y, y) - 1

                if ix < 0 or ix >= nx-1 or iy < 0 or iy >= ny-1:
                    continue

                # bilinear weights
                x0, x1g = grid_x[ix], grid_x[ix+1]
                y0, y1g = grid_y[iy], grid_y[iy+1]

                wx = (x - x0) / (x1g - x0 + 1e-12)
                wy = (y - y0) / (y1g - y0 + 1e-12)

                w00 = (1-wx)*(1-wy)
                w10 = wx*(1-wy)
                w01 = (1-wx)*wy
                w11 = wx*wy

                weights[idx(ix,   iy  )] += w00 * ds / L
                weights[idx(ix+1, iy  )] += w10 * ds / L
                weights[idx(ix,   iy+1)] += w01 * ds / L
                weights[idx(ix+1, iy+1)] += w11 * ds / L

        # --- 5. normalize row ---
        if weights.sum() > 0:
            A[p_idx] = weights / weights.sum()

    return A, centers, pair_counts

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


def build_synthetic_vs(nx=29, ny=28, nz=20):
    # --- coordinates ---
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(0, 1, nz)   # depth normalized

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # --- background: increasing with depth ---
    Vs_bg = 2.0 + 1.0 * Z   # from ~2.0 to ~3.0 km/s

    # --- low velocity anomaly (centered) ---
    low_anomaly = -1 * np.exp(
        -((X/0.7)**2 + (Y/0.7)**2 + ((Z-0.5)/0.3)**2)
    )

    # --- high velocity anomaly (slightly offset) ---
    high_anomaly = +0.8 * np.exp(
        -(((X-0.4)/0.7)**2 + ((Y+0.3)/0.7)**2 + ((Z-0.6)/0.4)**2)
    )

    Vs = Vs_bg + low_anomaly + high_anomaly

    return Vs

import numpy as np

def build_checkerboard_vs(nx=29, ny=28, nz=20,
                         v_low=1.8, v_high=3.2,
                         n_blocks_x=5, n_blocks_y=5, n_blocks_z=4):
    """
    3D checkerboard Vs model
    """

    # base grid
    Vs = np.zeros((nx, ny, nz))

    # block sizes
    bx = nx // n_blocks_x
    by = ny // n_blocks_y
    bz = nz // n_blocks_z

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                ix = i // bx
                iy = j // by
                iz = k // bz

                # alternating pattern
                if (ix + iy + iz) % 2 == 0:
                    Vs[i, j, k] = v_low
                else:
                    Vs[i, j, k] = v_high

    return Vs