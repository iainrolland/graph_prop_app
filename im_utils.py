import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rio


def netcdf_to_tif(cdf_path, drop_bands=None):
    nc_file = xr.open_dataset(cdf_path)
    if drop_bands is None:
        drop_bands = []
    data = nc_file[[variable for variable in nc_file.keys() if variable not in drop_bands]]
    data.rio.set_spatial_dims(x_dim="x", y_dim="y")
    data.rio.to_raster(f"{cdf_path.split('.')[0]}.tif")


def imageify(matrix, mask, shape, fill_value=np.nan):
    """
    if you have pixels/nodes in an image which are removed (i.e. like pixels representing sea in SMAP data) before
    completion then you have fewer samples in matrix than you need to put back into image of shape `shape'

    matrix: a flattened set of samples (i.e. of shape (#samples, #features))
    mask: ndim 2 matrix of shape (#rows, #cols)
    shape: a tuple (#rows, #cols, #feats)
    fill_value: the value which will appear in output where mask was False
    """
    output = fill_value * np.ones(shape)
    output[mask] = matrix
    return output


def slc_off_mask(shape):
    """
    Creates a boolean mask replicating what might be observed in a Landsat 7 image after the Scan Line Corrector failure
    """
    x_pixels = np.tile(np.arange(shape[1])[None, :], (shape[0], 1)) + 1e-6
    y_pixels = np.tile(np.arange(shape[0])[:, None], (1, shape[1])) + 1e-6
    slope_dist = np.cos(np.arctan(y_pixels / x_pixels) + (np.pi / 2 - np.arctan(0.179))) * (
            x_pixels ** 2 + y_pixels ** 2) ** 0.5
    return ((slope_dist % 31.4) / 31.4) > (4 / 9.5)


def rolling_stripes_mask(shape, gap_npixels, repeat_npixels, roll_npixels, horizontal=True):
    """
    shape: (rows, columns, #bands)
    returns boolean mask with same shape as `shape' with stripes (either horizontal/vertical) with False
    each band has its False stripes shifted by roll_npixels such that no single pixel has all its bands masked
    gap_npixels is the thickness of the missing stripes
    repeat_npixels is how frequently we insert a missing stripe (in pixels)
    """
    if not len(shape) == 3:
        raise ValueError(f"input `shape' must be of length 3 but was given as {shape}")
    if repeat_npixels <= gap_npixels:
        raise ValueError("".join([
            "`repeat_npixels' must be greater than `gap_npixels' but were ",
            f"{repeat_npixels} and {gap_npixels} respectively"
        ]))
    if horizontal:
        pixels = np.tile(np.arange(shape[0])[:, None], (1, shape[1]))
    else:
        pixels = np.tile(np.arange(shape[1])[None, :], (shape[0], 1))
    return np.stack([((pixels + roll_npixels * band) % repeat_npixels) >= gap_npixels for band in range(shape[-1])],
                    axis=-1)


def demo_rolling_stripes_mask():
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
    mask = rolling_stripes_mask(
        shape=(50, 50, 3), gap_npixels=2, repeat_npixels=10, roll_npixels=4, horizontal=True
    )
    for band in range(mask.shape[-1]):
        ax[0, band].matshow(mask[..., band])
    mask = rolling_stripes_mask(
        shape=(50, 50, 3), gap_npixels=2, repeat_npixels=10, roll_npixels=4, horizontal=False
    )
    for band in range(mask.shape[-1]):
        ax[1, band].matshow(mask[..., band])
    plt.show()


def demo_slc_off():
    fig, ax = plt.subplots()
    mask = slc_off_mask((400, 400))
    ax.matshow(mask)
    plt.show()


def reverse_never_observed(features, reversing_mask):
    if not features.ndim == 2:
        raise ValueError("Expected 2-dimensional features. Input should be (#nodes, #features)")
    output = np.zeros((reversing_mask.shape[0], features.shape[1]), dtype=features.dtype)
    output[reversing_mask] = features
    return output


if __name__ == '__main__':
    # demo_rolling_stripes_mask()
    demo_slc_off()
