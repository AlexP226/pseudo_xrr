import numpy as np


def compare_nested_dicts(d1, d2, path="root", rtol=1e-12, atol=1e-15, verbose=True):
    """
    Recursively compare two nested dictionaries / structures.

    Supports:
    - dict
    - list / tuple
    - numpy arrays
    - numpy scalars
    - Python scalars / strings / None

    Parameters
    ----------
    d1, d2 : any
        Objects to compare.

    path : str
        Internal path tracker.

    rtol, atol : float
        Tolerances for float comparison with np.allclose().

    verbose : bool
        If True, print the first mismatch.

    Returns
    -------
    identical : bool
        True if structures and values are identical (within tolerance).
    """

    # -----------------------------
    # type normalization
    # -----------------------------
    if isinstance(d1, np.generic):
        d1 = d1.item()
    if isinstance(d2, np.generic):
        d2 = d2.item()

    # -----------------------------
    # dict
    # -----------------------------
    if isinstance(d1, dict) and isinstance(d2, dict):
        keys1 = set(d1.keys())
        keys2 = set(d2.keys())

        if keys1 != keys2:
            if verbose:
                print(f"[KEY MISMATCH] at {path}")
                print("Only in first :", sorted(keys1 - keys2))
                print("Only in second:", sorted(keys2 - keys1))
            return False

        for k in sorted(keys1):
            if not compare_nested_dicts(d1[k], d2[k], path=f"{path}['{k}']", rtol=rtol, atol=atol, verbose=verbose):
                return False
        return True

    # -----------------------------
    # list / tuple
    # -----------------------------
    if isinstance(d1, (list, tuple)) and isinstance(d2, (list, tuple)):
        if len(d1) != len(d2):
            if verbose:
                print(f"[LENGTH MISMATCH] at {path}: {len(d1)} != {len(d2)}")
            return False

        for i, (v1, v2) in enumerate(zip(d1, d2)):
            if not compare_nested_dicts(v1, v2, path=f"{path}[{i}]", rtol=rtol, atol=atol, verbose=verbose):
                return False
        return True

    # -----------------------------
    # numpy arrays
    # -----------------------------
    if isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray):
        if d1.shape != d2.shape:
            if verbose:
                print(f"[ARRAY SHAPE MISMATCH] at {path}: {d1.shape} != {d2.shape}")
            return False

        if d1.dtype != d2.dtype:
            if verbose:
                print(f"[ARRAY DTYPE MISMATCH] at {path}: {d1.dtype} != {d2.dtype}")
            return False

        # float arrays
        if np.issubdtype(d1.dtype, np.floating) or np.issubdtype(d1.dtype, np.complexfloating):
            equal = np.allclose(d1, d2, rtol=rtol, atol=atol, equal_nan=True)
        else:
            equal = np.array_equal(d1, d2)

        if not equal:
            if verbose:
                print(f"[ARRAY VALUE MISMATCH] at {path}")
                diff_idx = np.argwhere(~np.isclose(d1, d2, rtol=rtol, atol=atol, equal_nan=True)) \
                    if np.issubdtype(d1.dtype, np.floating) else np.argwhere(d1 != d2)
                if diff_idx.size > 0:
                    idx = tuple(diff_idx[0])
                    print(f"First mismatch at index {idx}: {d1[idx]} != {d2[idx]}")
            return False

        return True

    # -----------------------------
    # one array, one not
    # -----------------------------
    if isinstance(d1, np.ndarray) != isinstance(d2, np.ndarray):
        if verbose:
            print(f"[TYPE MISMATCH] at {path}: {type(d1)} != {type(d2)}")
        return False

    # -----------------------------
    # float scalars
    # -----------------------------
    if isinstance(d1, float) and isinstance(d2, float):
        if not np.isclose(d1, d2, rtol=rtol, atol=atol, equal_nan=True):
            if verbose:
                print(f"[FLOAT MISMATCH] at {path}: {d1} != {d2}")
            return False
        return True

    # -----------------------------
    # everything else
    # -----------------------------
    if d1 != d2:
        if verbose:
            print(f"[VALUE MISMATCH] at {path}: {d1!r} != {d2!r}")
        return False

    return True

