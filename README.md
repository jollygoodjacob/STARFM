# Spatiotemporal Fusion (STF) Project
Spatiotemporal fusion methods, compiled and modified for the IoT4Ag Thrust 3 Data Fusion project at UCM

## License

    Spatiotemporal fusion methods, compiled and modified for the IoT4Ag Thrust 3 Data Fusion project at UCM in Python
    Copyright (C) 2025 Jacob Nesslage (for UBDF, FSDAR, and Fit-FC)
    Copyright (C) 2025 Nikolina Mileva, modified by Jacob Nesslage (for STARFM)
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
## STARFM

The STARFM (Spatial and Temporal Adaptive Reflectance Fusion Model) algorithm is widely used to generate high-resolution, high-frequency remote sensing data by fusing low-resolution, high-temporal and high-resolution, low-temporal imagery to produce a synthetic image.

Let’s break down how STARFM works, with specific reference to the Python implementation by Mileva (2018), available here:

### Intuition - STARFM
STARFM predicts what a fine-resolution image would look like on a date where only coarse-resolution data is available by using:

-  One fine-resolution image at a base date (e.g., UAV or Planet)

-  Two coarse-resolution images at base and target dates (e.g., MODIS or Sentinel)

It assumes that spectral change in coarse resolution over time also applies locally at finer scale.

### Processing Steps
1. Input data:

- F_t1: fine-resolution image at time t1 

- C_t1: coarse-resolution image at time t1 
  
- C_t2: coarse-resolution image at time t2 (prediction date)

2. Calculate change in coarse resolution
3. Loop over fine-resolution pixels: For each fine-resolution pixel, the algorithm:

- Finds a window of nearby pixels in the fine-resolution image

- Checks spectral similarity between fine and coarse images at t1

- Computes weights based on:

  - Spatial distance

  - Spectral similarity (Euclidean difference)

  - Temporal change magnitude

4. Prediction:
- Apply weighted average of reflectance changes

- Add that change to the fine-resolution pixel at t1:
6. Output:
- A synthetic high-resolution image at time t2: F_t2

### Credit / References
The STARFM algorithm used herein was modified from scripts developed by Mileva et al. 2018 for use in Python. The STARFM algorithm, first written in C, was published by Guo et al. 2006.

F. Gao, J. Masek, M. Schwaller, F. Hall. On the blending of the Landsat and MODIS surface reflectance : Predicting daily Landsat surface reflectance. IEEE Transactions on Geoscience and Remote Sensing, 44 (8) (2006), pp. 2207-2218

Mileva, N., Mecklenburg, S. & Gascon, F. (2018). New tool for spatiotemporal image fusion in remote sensing - a case study approach using Sentinel-2 and Sentinel-3 data. In Bruzzone, L. & Bovolo, F. (Eds.), SPIE Proceedings Vol. 10789: Image and Signal Processing for Remote Sensing XXIV. Berlin, Germany: International Society for Optics and Photonics. doi: 10.1117/12.2327091; https://doi.org/10.1117/12.2327091
