# Rail and Address Line-of-Sight (LOS) Analysis

This workflow identifies addresses near rail lines and scores their line-of-sight (LOS) to the rail, considering building obstructions. All outputs are saved in the `output/` directory.

**Before running your own analysis, delete any example files in the `output/` directory and the example `data/addresses.geojson` file.**
This ensures you are working with your own data and results.

## Installation

All required Python packages are listed in `requirements.txt`.

Install them with:
```sh
pip install -r requirements.txt
```

## How it works
- **Rail lines** are fetched from the FRA MainLine MapServer for your area of interest.
- **Addresses** must be downloaded by the user from OpenAddresses and saved as `data/addresses.geojson`.
- **Building footprints** are automatically fetched from OpenStreetMap (OSM) for your area and saved as `output/building_footprints.geojson`.
- The script computes which addresses are within a user-defined buffer of the rail, and assigns a LOS score (1 = clear sight, 0 = blocked by a building).
- Results are saved as a CSV and a map in the `output/` directory.

## What you need to do

### 1. **Download addresses for your area**
- Go to [OpenAddresses Data](https://batch.openaddresses.io/data#map=0/0/0)
- Use Ctrl+F to search for your city, county, or region.
- Download the dataset (preferably as GeoJSON, or convert to GeoJSON).
- Save the file as `data/addresses.geojson` in your project directory.
- **Delete any example `addresses.geojson` file before adding your own.**

### 2. **Set your analysis parameters**
Edit the top of `src/address_los_score.py` to set:
- `center_lat` and `center_lon`: the center of your area of interest (in decimal degrees)
- `radius_km`: the radius (in kilometers) around the center to analyze
- `rail_buffer_m`: the buffer distance (in meters) from the rail line to consider addresses "nearby"

Example:
```python
center_lat, center_lon = 41.87825, -87.62975
radius_km = 1.0
rail_buffer_m = 100
```

### 3. **Run the script**
```sh
python src/address_los_score.py
```
- The script will automatically fetch building footprints for your area from OSM if not already present in `output/building_footprints.geojson`.
- All outputs will be saved in the `output/` directory:
  - `output/building_footprints.geojson`: Building footprints for your area
  - `output/address_los_scores.csv`: Addresses near rail with LOS scores
  - `output/address_los_scores.png`: Map visualization
- **Delete any example files in the `output/` directory before running your own analysis.**

## Notes
- The workflow is fully automated except for downloading the address file from OpenAddresses.
- You can change the AOI, buffer, and other parameters as needed.
- For best results, ensure all data uses the WGS84 (EPSG:4326) coordinate system. # address-line-of-sight-to-railway
