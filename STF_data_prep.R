
# Load the terra package for raster handling
require(terra)

# -----------------------------------------------------------
# 📥 Load input rasters
# -----------------------------------------------------------

# Load PlanetScope imagery for two dates (8-band harmonized, surface reflectance)
planet_t0 <- rast("Z:/Projects/IoT4Ag/Data/rasters/Planet/IoT4Ag_Pistachio_psscene_analytic_8b_sr_udm2/files/20220802_182705_60_247e_3B_AnalyticMS_SR_8b_harmonized_clip.tif")
planet_t1 <- rast("Z:/Projects/IoT4Ag/Data/rasters/Planet/IoT4Ag_Pistachio_psscene_analytic_8b_sr_udm2/files/20220812_182402_83_2480_3B_AnalyticMS_SR_8b_harmonized_clip.tif")

# Load UAV orthomosaic
uav_t0 <- rast("Z:/Projects/IoT4Ag/Outputs/SfM/Pistachio-farm-8-2-2022-all/odm_orthophoto/odm_orthophoto_8.tif")

# Print raster summaries (dimensions, CRS, resolution, etc.)
planet_t0
planet_t1
uav_t0

# -----------------------------------------------------------
# 🔄 Reproject PlanetScope data to match UAV CRS and resolution
# -----------------------------------------------------------

# Ensure all rasters share the same coordinate reference system and resolution
# This enables direct comparison and fusion
planet_t0 <- project(planet_t0, uav_t0)
planet_t1 <- project(planet_t1, uav_t0)

# Check alignment after projection
planet_t0
planet_t1
uav_t0

# -----------------------------------------------------------
# 🧪 Preprocess bands and scale values
# -----------------------------------------------------------

# For UAV image:
# Use first 3 bands (RGB), normalize from 0–255 to 0–1 (common for deep learning or display)
uav_t0 <- uav_t0[[1:3]] / 255

# For PlanetScope images:
# Extract RGB bands: 2 (Blue), 4 (Green), 6 (Red) — based on harmonized 8-band schema
# Scale reflectance from 0–10000 to 0–1
planet_t0 <- planet_t0[[c(2, 4, 6)]] / 10000
planet_t1 <- planet_t1[[c(2, 4, 6)]] / 10000

# -----------------------------------------------------------
# 💾 Save processed outputs
# -----------------------------------------------------------

# Write the UAV RGB composite
writeRaster(
  uav_t0,
  "Z:/Projects/IoT4Ag/Outputs/STF_example_data/20220802_RGB_UAV.tif",
  overwrite = TRUE
)

# Write the PlanetScope RGB composites for both timepoints
writeRaster(
  planet_t0,
  "Z:/Projects/IoT4Ag/Outputs/STF_example_data/20220802_RGB_Planet.tif",
  overwrite = TRUE
)

writeRaster(
  planet_t1,
  "Z:/Projects/IoT4Ag/Outputs/STF_example_data/20220812_RGB_Planet.tif",
  overwrite = TRUE
)

# -----------------------------------------------------------
# ✅ Outputs:
# - UAV RGB composite scaled to [0, 1]
# - Planet RGB composites for two dates, scaled to [0, 1]
# All outputs are reprojected and aligned in the UAV coordinate system
# -----------------------------------------------------------
