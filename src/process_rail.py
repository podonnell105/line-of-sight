import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
from shapely.geometry import Point, box, LineString
import requests
from io import BytesIO
import urllib.parse
import json

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
    """Fetch rail lines within the specified bounding box and filter for Class 1 railroads by owner."""
    # Build query parameters (no where filter)
    params = {
        'outFields': '*',
        'f': 'geojson',
        'geometryType': 'esriGeometryEnvelope',
        'spatialRel': 'esriSpatialRelIntersects',
        'geometry': f"{bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']}"
    }
    print(f"Query parameters: {params}")
    try:
        response = requests.get(BASE_URL, params=params)
        print(f"Request URL: {response.url}")
        response.raise_for_status()
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {response.headers}")
        print(f"Response content: {response.text[:1000]}")
        try:
            gdf = gpd.read_file(BytesIO(response.content))
            print(f"Fetched {len(gdf)} rail line segments (all classes)")
        except Exception as e:
            print(f"GeoPandas could not read from BytesIO: {e}")
            debug_path = "debug_rail.json"
            with open(debug_path, "w") as f:
                f.write(response.text)
            print(f"Saved response to {debug_path} for inspection.")
            try:
                gdf = gpd.read_file(debug_path)
                print(f"Fetched {len(gdf)} rail line segments (from file)")
            except Exception as e2:
                print(f"GeoPandas could not read from file: {e2}")
                return None
        # Filter for Class 1 by RROWNER1
        class1_owners = ["BNSF", "UP", "NS", "CSXT", "KCS", "CN", "CPRS"]
        class1_gdf = gdf[gdf["RROWNER1"].isin(class1_owners)]
        print(f"Filtered to {len(class1_gdf)} Class 1 rail line segments")
        return class1_gdf
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print("Response text:", e.response.text)
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

def calculate_los_score_buildings(address_point, rail_point, buildings_gdf):
    """
    Calculate line-of-sight score between an address and rail point using only building footprints.
    Returns 1 if line of sight is clear, 0 if blocked by buildings.
    """
    try:
        # Create a line between address and rail point
        sightline = LineString([address_point, rail_point])
        
        # Check if sightline crosses any building
        for _, bldg in buildings_gdf.iterrows():
            # If the address is inside this building, skip it
            if bldg.geometry.contains(Point(address_point)):
                continue
            if bldg.geometry.crosses(sightline) or bldg.geometry.intersects(sightline):
                return 0  # Blocked by building
        
        return 1  # Clear line of sight
        
    except Exception as e:
        print(f"Error calculating LOS score: {e}")
        return 0
