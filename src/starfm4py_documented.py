# STARFM Spatiotemporal Fusion Script with Line-by-Line Documentation
# Author: Nikolina Mileva (Modified and documented by Jacob Nesslage for use with Planet and UAV imagery)

import zarr                      # For efficient array storage on disk
import numpy as np               # For numerical computations and array manipulation
import dask.array as da          # For parallel and lazy computation on large arrays
from dask.diagnostics import ProgressBar  # Visual progress indicator for Dask
from parameters import (windowSize, logWeight, temp, mid_idx, numberClass, spatImp, 
                        specUncertainty, tempUncertainty, path)  # Parameters defined in an external file


# === Function: Flatten and Save Image Blocks ===
def block2row(array, row, folder, block_id=None):
    if array.shape[0] == windowSize:
        name_string = str(block_id[0] + 1)  # Convert block ID to string (1-based)
        m, n = array.shape  # Dimensions of input block
        u = m + 1 - windowSize  # Number of vertical windows
        v = n + 1 - windowSize  # Number of horizontal windows

        start_idx = np.arange(u)[:, None] * n + np.arange(v)  # Top-left indices of windows
        offset_idx = np.arange(windowSize)[:, None] * n + np.arange(windowSize)  # Pixel offsets within each window

        flat_array = np.take(array, start_idx.ravel()[:, None] + offset_idx.ravel())  # Flattened windows

        file_name = path + folder + name_string + 'r' + row + '.zarr'  # Output filename
        zarr.save(file_name, flat_array)  # Save flattened windows as Zarr array

    return array  # Return required by Dask (even if unused)


# === Function: Partition Image into Overlapping Blocks ===
def partition(image, folder):
    image_da = da.from_array(image, chunks=(windowSize, image.shape[1]))  # Convert to Dask array with vertical chunks
    image_pad = da.pad(image_da, windowSize // 2, mode='constant')  # Pad image to handle edge effects

    for i in range(0, windowSize):  # Loop over rows of sliding windows
        row = str(i)  # Row identifier for output
        block_i = image_pad[i:, :]  # Shifted version of the image
        block_i_da = da.rechunk(block_i, chunks=(windowSize, image_pad.shape[1]))  # Rechunk to fit window

        # Apply block2row to each chunk, compute result
        block_i_da.map_blocks(block2row, dtype=int, row=row, folder=folder).compute()


# === Function: Stack Zarr Blocks into a Dask Array ===
def da_stack(folder, shape):
    da_list = []  # List to hold block arrays
    full_path = path + folder  # Absolute path to folder
    max_blocks = shape[0] // windowSize + 1  # Number of vertical blocks

    for block in range(1, max_blocks + 1):  # Loop through blocks
        for row in range(0, windowSize):  # Loop through rows
            name = str(block) + 'r' + str(row)  # Zarr filename
            full_name = full_path + name + '.zarr'
            try:
                da_array = da.from_zarr(full_name)  # Load Zarr block
                da_list.append(da_array)  # Append to list
            except Exception:
                continue  # Skip if file is missing

    return da.rechunk(da.concatenate(da_list, axis=0), chunks=(shape[1], windowSize**2))  # Stack and rechunk


# === Function: Spectral Distance ===
def spectral_distance(fine_image_t0, coarse_image_t0):
    spec_diff = fine_image_t0 - coarse_image_t0  # Difference between fine and coarse reflectance
    spec_dist = 1 / (abs(spec_diff) + 1.0)  # Inverse distance metric
    print("Done spectral distance!", spec_dist)
    return spec_diff, spec_dist


# === Function: Temporal Distance ===
def temporal_distance(coarse_image_t0, coarse_image_t1):
    temp_diff = coarse_image_t1 - coarse_image_t0  # Change over time
    temp_dist = 1 / (abs(temp_diff) + 1.0)  # Inverse distance metric
    print("Done temporal distance!", temp_dist)
    return temp_diff, temp_dist


# === Function: Spatial Distance from Center Pixel ===
def spatial_distance(array):
    coord = np.sqrt((np.mgrid[0:windowSize, 0:windowSize] - windowSize // 2)**2)  # Coordinate grid
    spat_dist = np.sqrt(((0 - coord[0])**2 + (0 - coord[1])**2))  # Euclidean distance
    rel_spat_dist = spat_dist / spatImp + 1.0  # Normalize
    rev_spat_dist = 1 / rel_spat_dist  # Reverse so closer = higher
    flat_spat_dist = np.ravel(rev_spat_dist)  # Flatten to 1D
    spat_dist_da = da.from_array(flat_spat_dist, chunks=flat_spat_dist.shape)  # Convert to Dask array
    print("Done spatial distance!", spat_dist_da)
    return spat_dist_da


# === Function: Threshold for Similarity Detection ===
def similarity_threshold(fine_image_t0):
    fine_image_t0 = da.where(fine_image_t0 == 0, np.nan, fine_image_t0)  # Ignore zeros
    st_dev = da.nanstd(fine_image_t0, axis=1)  # Row-wise std dev
    sim_threshold = st_dev * 2 / numberClass  # Dynamic threshold
    print("Done similarity threshold!", sim_threshold)
    return sim_threshold


# === Function: Identify Spectrally Similar Pixels ===
def similarity_pixels(fine_image_t0):
    sim_threshold = similarity_threshold(fine_image_t0)
    similar_pixels = da.where(
        abs(fine_image_t0 - fine_image_t0[:, mid_idx][:, None]) <= sim_threshold[:, None], 1, 0
    )  # Compare to center pixel
    print("Done similarity pixels!", similar_pixels)
    return similar_pixels


# === Function: Filter Spectrally and Temporally Similar Pixels ===
def filtering(fine_image_t0, spec_dist, temp_dist, spec_diff, temp_diff):
    similar_pixels = similarity_pixels(fine_image_t0)
    max_spec_dist = abs(spec_diff)[:, mid_idx][:, None] + specUncertainty + 1
    max_temp_dist = abs(temp_diff)[:, mid_idx][:, None] + tempUncertainty + 1
    spec_filter = da.where(spec_dist > 1.0 / max_spec_dist, 1, 0)  # Spectral filter

    st_filter = spec_filter  # Default to spectral only
    if temp is True:
        temp_filter = da.where(temp_dist > 1.0 / max_temp_dist, 1, 0)  # Temporal filter
        st_filter = spec_filter * temp_filter  # Combine filters

    similar_pixels_filtered = similar_pixels * st_filter  # Mask
    print("Done filtering!", similar_pixels_filtered)
    return similar_pixels_filtered


# === Function: Combine Distances ===
def comb_distance(spec_dist, temp_dist, spat_dist):
    if logWeight:
        spec_dist = da.log(spec_dist + 1)  # Optional log transform
        temp_dist = da.log(temp_dist + 1)
    comb_dist = da.rechunk(spec_dist * temp_dist * spat_dist, chunks=spec_dist.chunksize)  # Combine all
    print("Done comb distance!", comb_dist)
    return comb_dist


# === Function: Compute Pixel Weights ===
def weighting(spec_dist, temp_dist, comb_dist, similar_pixels_filtered):
    zero_spec_dist = da.where(spec_dist[:, mid_idx][:, None] == 1, 1, 0)  # Zero-distance pixel
    zero_temp_dist = da.where(temp_dist[:, mid_idx][:, None] == 1, 1, 0)
    zero_dist_mid = da.where(zero_spec_dist == 1, zero_spec_dist, zero_temp_dist)

    shape = da.subtract(spec_dist.shape, (0, 1))
    zero_dist = da.zeros(shape, chunks=(spec_dist.shape[0], shape[1]))
    zero_dist = da.insert(zero_dist, [mid_idx], zero_dist_mid, axis=1)

    weights = da.where(da.sum(zero_dist, 1)[:, None] == 1, zero_dist, comb_dist)  # Use fallback if applicable
    weights_filt = weights * similar_pixels_filtered  # Mask with similarity

    norm_weights = da.rechunk(weights_filt / (da.sum(weights_filt, 1)[:, None]),
                              chunks=spec_dist.chunksize)  # Normalize
    print("Done weighting!", norm_weights)
    return norm_weights


# === Function: Predict Fine-Resolution Reflectance ===
def predict(fine_image_t0, coarse_image_t0, coarse_image_t1, shape):
    spec_diff, spec_dist = spectral_distance(fine_image_t0, coarse_image_t0)
    temp_diff, temp_dist = temporal_distance(coarse_image_t0, coarse_image_t1)
    spat_dist = spatial_distance(fine_image_t0)
    comb_dist = comb_distance(spec_dist, temp_dist, spat_dist)
    similar_pixels = filtering(fine_image_t0, spec_dist, temp_dist, spec_diff, temp_diff)
    weights = weighting(spec_dist, temp_dist, comb_dist, similar_pixels)

    pred_refl = fine_image_t0 + temp_diff  # Temporal shift
    weighted_pred_refl = da.sum(pred_refl * weights, axis=1)  # Weighted average
    prediction = da.reshape(weighted_pred_refl, shape)  # Back to image
    print("Done prediction!")
    return prediction


# === Function: Full STARFM Execution with Output ===
def starfm(fine_image_t0, coarse_image_t0, coarse_image_t1, profile, shape):
    print('Processing...')
    prediction_da = predict(fine_image_t0, coarse_image_t0, coarse_image_t1, shape)
    with ProgressBar():  # Show progress bar
        prediction = prediction_da.compute()  # Run Dask pipeline
    return prediction  # Final NumPy array
