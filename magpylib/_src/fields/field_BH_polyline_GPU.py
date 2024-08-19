import numpy as np
from numpy.linalg import norm
from scipy.constants import mu_0 as MU0
import torch

from magpylib._src.input_checks import check_field_input


def current_vertices_field(
    field: str,
    observers: np.ndarray,
    current: np.ndarray,
    vertices: np.ndarray = None,
    segment_start=None,  # list of mix3 ndarrays
    segment_end=None,
) -> np.ndarray:
    if vertices is None:
        return BHJM_current_polyline(
            field=field,
            observers=observers,
            current=current,
            segment_start=segment_start,
            segment_end=segment_end,
        )

    nvs = np.array([f.shape[0] for f in vertices])  # lengths of vertices sets
    if all(v == nvs[0] for v in nvs):  # if all vertices sets have the same lengths
        n0, n1, *_ = vertices.shape
        BH = BHJM_current_polyline(
            field=field,
            observers=np.repeat(observers, n1 - 1, axis=0),
            current=np.repeat(current, n1 - 1, axis=0),
            segment_start=vertices[:, :-1].reshape(-1, 3),
            segment_end=vertices[:, 1:].reshape(-1, 3),
        )
        BH = BH.reshape((n0, n1 - 1, 3))
        BH = np.sum(BH, axis=1)
    else:
        split_indices = np.cumsum(nvs - 1)[:-1]  # remove last to avoid empty split
        BH = BHJM_current_polyline(
            field=field,
            observers=np.repeat(observers, nvs - 1, axis=0),
            current=np.repeat(current, nvs - 1, axis=0),
            segment_start=np.concatenate([vert[:-1] for vert in vertices]),
            segment_end=np.concatenate([vert[1:] for vert in vertices]),
        )
        bh_split = np.split(BH, split_indices)
        BH = np.array([np.sum(bh, axis=0) for bh in bh_split])
    return BH


# CORE
def current_polyline_Hfield(
    observers: np.ndarray,
    segments_start: np.ndarray,
    segments_end: np.ndarray,
    currents: np.ndarray,
) -> np.ndarray:
    """B-field of line current segments."""
    # Convert inputs to tensors and move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    p1 = torch.tensor(segments_start, dtype=torch.float64, device=device)
    p2 = torch.tensor(segments_end, dtype=torch.float64, device=device)
    po = torch.tensor(observers, dtype=torch.float64, device=device)
    currents = torch.tensor(currents, dtype=torch.float64, device=device)

    # make dimensionless (avoid all large/small input problems) by introducing
    # the segment length as characteristic length scale.
    norm_12 = torch.norm(p1 - p2, dim=1)
    p1 = p1 / norm_12[:, None]
    p2 = p2 / norm_12[:, None]
    po = po / norm_12[:, None]

    # p4 = projection of pos_obs onto line p1-p2
    t = torch.sum((po - p1) * (p1 - p2), dim=1)
    p4 = p1 + (t[:, None] * (p1 - p2))

    # distance of observers from line
    norm_o4 = torch.norm(po - p4, dim=1)

    # separate on-line cases (-> B=0)
    mask1 = norm_o4 < 1e-15  # account for numerical issues

    # continue only with general off-line cases
    if torch.any(mask1):
        not_mask1 = ~mask1
        po = po[not_mask1]
        p1 = p1[not_mask1]
        p2 = p2[not_mask1]
        p4 = p4[not_mask1]
        norm_12 = norm_12[not_mask1]
        norm_o4 = norm_o4[not_mask1]
        currents = currents[not_mask1]

    # determine field direction
    cros = torch.cross(p2 - p1, po - p4, dim=1)
    norm_cros = torch.norm(cros, dim=1)
    eB = cros / norm_cros[:, None]

    # compute angles
    norm_o1 = torch.norm(po - p1, dim=1)  # improve performance by computing all norms at once
    norm_o2 = torch.norm(po - p2, dim=1)
    norm_41 = torch.norm(p4 - p1, dim=1)
    norm_42 = torch.norm(p4 - p2, dim=1)
    sinTh1 = norm_41 / norm_o1
    sinTh2 = norm_42 / norm_o2
    deltaSin = torch.empty((len(po),), dtype=torch.float64, device=device)

    # determine how p1, p2, p4 are sorted on the line (to get sinTH signs)
    # both points below
    mask2 = (norm_41 > 1) & (norm_41 > norm_42)
    deltaSin[mask2] = torch.abs(sinTh1[mask2] - sinTh2[mask2])
    # both points above
    mask3 = (norm_42 > 1) & (norm_42 > norm_41)
    deltaSin[mask3] = torch.abs(sinTh2[mask3] - sinTh1[mask3])
    # one above one below or one equals p4
    mask4 = ~mask2 & ~mask3
    deltaSin[mask4] = torch.abs(sinTh1[mask4] + sinTh2[mask4])

    # Perform element-wise multiplication and ensure dimensions align
    H = (deltaSin / norm_o4)[:, None] * eB / norm_12[:, None] * currents[:, None] / (4 * np.pi)

    # avoid array creation if possible
    if torch.any(mask1):
        H_full = torch.zeros_like(po, dtype=torch.float64, device=device)
        H_full[~mask1] = H
        return H_full.cpu().numpy()  # convert back to numpy array
    return H.cpu().numpy()


def BHJM_current_polyline(
    field: str,
    observers: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    check_field_input(field)

    BHJM = np.zeros_like(observers, dtype=float)

    if field in "MJ":
        return BHJM

    # Check for zero-length segments (or discontinuous)
    mask_nan_start = np.isnan(segment_start).all(axis=1)
    mask_nan_end = np.isnan(segment_end).all(axis=1)
    mask_equal = np.all(segment_start == segment_end, axis=1)
    mask0 = mask_equal | mask_nan_start | mask_nan_end
    not_mask0 = ~mask0  # avoid multiple computation of ~mask

    if np.all(mask0):
        return BHJM

    # continue only with non-zero segments
    if np.any(mask0):
        current = current[not_mask0]
        segment_start = segment_start[not_mask0]
        segment_end = segment_end[not_mask0]
        observers = observers[not_mask0]

    BHJM[not_mask0] = current_polyline_Hfield(
        observers=observers,
        segments_start=segment_start,
        segments_end=segment_end,
        currents=current,
    )

    if field == "H":
        return BHJM

    if field == "B":
        return BHJM * MU0

    raise ValueError(
        "`output_field_type` must be one of ('B', 'H', 'M', 'J'), " f"got {field!r}"
    )
