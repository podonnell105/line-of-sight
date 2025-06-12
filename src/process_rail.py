import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
from shapely.geometry import Point, box
import requests
from io import BytesIO

# FRA MainLine MapServer REST endpoint
BASE_URL = "https://fragis.fra.dot.gov/arcgis/rest/services/FRA/MainLine/MapServer/0/query"
DATA_DIR = 'data'


BOX_SIZE_DEGREES = 0.5  # Base size of bounding box in degrees
OVERLAP_FACTOR = 0.3  # 30% overlap between boxes

def calculate_bounding_boxes(lat, lon, base_size, num_boxes=10):
    """Calculate multiple overlapping bounding boxes in a grid pattern around the target point."""
    boxes = []
    
    # Calculate grid dimensions (3x4 grid with some overlap)
    grid_cols = 3
    grid_rows = 4
    
    # Calculate step size with overlap
    step_size = base_size * (1 - OVERLAP_FACTOR)
    
    # Calculate starting point (top-left of grid)
    start_lon = lon - (step_size * (grid_cols - 1) / 2)
    start_lat = lat + (step_size * (grid_rows - 1) / 2)
    
    # Generate boxes in a grid pattern
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Calculate box center
            center_lon = start_lon + (col * step_size)
            center_lat = start_lat - (row * step_size)
            
            # Calculate box bounds
            half_size = base_size / 2
            box_dict = {
                'xmin': center_lon - half_size,
                'ymin': center_lat - half_size,
                'xmax': center_lon + half_size,
                'ymax': center_lat + half_size
            }
            boxes.append(box_dict)
    
    return boxes

def fetch_rail_lines_in_bbox(bbox):
    """Fetch rail lines within the specified bounding box."""
    url = (
        f"{BASE_URL}"
        f"?where=CLASS='1'"  # Filter for Class 1 railroads
        f"&outFields=*"
        f"&f=geojson"
        f"&geometry={bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']}"
        f"&geometryType=esriGeometryEnvelope"
        f"&inSR=4326"
        f"&spatialRel=esriSpatialRelIntersects"
    )
    print(f"Fetching Class 1 rail lines within bounding box: {bbox}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        print("Response headers:", response.headers)
        print("Response content (first 500 chars):", response.content[:500])
        try:
            gdf = gpd.read_file(BytesIO(response.content))
            print(f"Fetched {len(gdf)} Class 1 rail line segments")
            return gdf
        except Exception as e:
            print(f"GeoPandas could not read from BytesIO: {e}")
            # Save to file for manual inspection
            debug_path = "debug_rail.json"
            with open(debug_path, "wb") as f:
                f.write(response.content)
            print(f"Saved response to {debug_path} for inspection.")
            # Try reading from file
            try:
                gdf = gpd.read_file(debug_path)
                print(f"Fetched {len(gdf)} Class 1 rail line segments (from file)")
                return gdf
            except Exception as e2:
                print(f"GeoPandas could not read from file: {e2}")
                return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def combine_and_deduplicate(gdfs):
    """Combine multiple GeoDataFrames and remove duplicates."""
    if not gdfs:
        return None
    
    # Combine all GeoDataFrames
    combined = pd.concat(gdfs, ignore_index=True)
    
    # Remove any rows where geometry is missing
    combined = combined.dropna(subset=['geometry'])
    
    # Ensure CRS is consistent
    if combined.crs is None:
        combined.set_crs(epsg=4326, inplace=True)
    
    # Remove exact duplicates
    combined = combined.drop_duplicates(subset=['geometry'])
    
    print(f"\nCombined and deduplicated results:")
    print(f"Total unique rail segments: {len(combined)}")
    
    return combined
