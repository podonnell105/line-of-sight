import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import logging
import os
from tqdm import tqdm
import shutil
import time
import json
from shapely.geometry import box
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('address_fetch.log'),
        logging.StreamHandler()
    ]
)

def get_state_bounds(state_abbr):
    """
    Get the bounding box for a state using OpenStreetMap's Nominatim API
    """
    url = f"https://nominatim.openstreetmap.org/search"
    params = {
        'q': f"{state_abbr}, USA",
        'format': 'json',
        'limit': 1
    }
    headers = {
        'User-Agent': 'LineOfSightAnalysis/1.0'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if data:
            # Get the bounding box
            bounds = data[0]['boundingbox']
            return {
                'min_lat': float(bounds[0]),
                'max_lat': float(bounds[1]),
                'min_lon': float(bounds[2]),
                'max_lon': float(bounds[3])
            }
        return None
    except Exception as e:
        logging.error(f"Error getting state bounds: {str(e)}")
        return None

def split_bbox(bbox, grid_size=0.5):
    """
    Split a bounding box into smaller grid cells
    """
    min_lat, max_lat = bbox['min_lat'], bbox['max_lat']
    min_lon, max_lon = bbox['min_lon'], bbox['max_lon']
    
    lat_steps = np.arange(min_lat, max_lat, grid_size)
    lon_steps = np.arange(min_lon, max_lon, grid_size)
    
    cells = []
    for i in range(len(lat_steps)-1):
        for j in range(len(lon_steps)-1):
            cells.append({
                'min_lat': lat_steps[i],
                'max_lat': lat_steps[i+1],
                'min_lon': lon_steps[j],
                'max_lon': lon_steps[j+1]
            })
    return cells

def fetch_addresses_from_osm(bbox):
    """
    Fetch addresses from OpenStreetMap for a given bounding box
    """
    # Overpass API endpoint
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Query to get all nodes and ways with address tags
    query = f"""
    [out:json][timeout:300];
    (
      node["addr:housenumber"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
      way["addr:housenumber"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
    );
    out body;
    >;
    out skel qt;
    """
    
    try:
        response = requests.post(overpass_url, data=query)
        response.raise_for_status()
        data = response.json()
        
        addresses = []
        for element in data.get('elements', []):
            # Skip if no tags
            if 'tags' not in element:
                continue
                
            # Skip if no house number
            if 'addr:housenumber' not in element['tags']:
                continue
            
            # Get coordinates
            lat = None
            lon = None
            
            if element['type'] == 'node':
                lat = element.get('lat')
                lon = element.get('lon')
            elif element['type'] == 'way':
                # For ways, use the first node's coordinates
                if 'nodes' in element and element['nodes']:
                    # Find the first node in the elements list
                    for node in data['elements']:
                        if node['type'] == 'node' and node['id'] == element['nodes'][0]:
                            lat = node.get('lat')
                            lon = node.get('lon')
                            break
            
            if lat is not None and lon is not None:
                address = {
                    'address': f"{element['tags'].get('addr:housenumber', '')} {element['tags'].get('addr:street', '')}",
                    'type': 'residential',
                    'geometry': Point(lon, lat)
                }
                addresses.append(address)
        
        logging.info(f"Found {len(addresses)} addresses in bbox")
        return addresses
    except Exception as e:
        logging.error(f"Error fetching addresses from OSM: {str(e)}")
        return []

def process_and_save_addresses(state_abbr, output_dir='web_data'):
    """
    Process address data and save to file in the correct format
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove existing uploaded_addresses.geojson if it exists
    existing_file = os.path.join(output_dir, 'uploaded_addresses.geojson')
    if os.path.exists(existing_file):
        os.remove(existing_file)
        logging.info(f"Removed existing {existing_file}")
    
    # Get state bounds
    state_bounds = get_state_bounds(state_abbr)
    if not state_bounds:
        logging.error(f"Could not get bounds for state {state_abbr}")
        return None
    
    # Split state into smaller grid cells
    grid_cells = split_bbox(state_bounds, grid_size=0.25)  # Reduced grid size for better results
    logging.info(f"Split state into {len(grid_cells)} grid cells")
    
    # Fetch addresses for each grid cell
    all_addresses = []
    for i, cell in enumerate(tqdm(grid_cells, desc="Processing grid cells")):
        addresses = fetch_addresses_from_osm(cell)
        all_addresses.extend(addresses)
        
        # Add a small delay to avoid overwhelming the API
        time.sleep(1)
        
        # Log progress
        if (i + 1) % 10 == 0:
            logging.info(f"Processed {i + 1}/{len(grid_cells)} grid cells, found {len(all_addresses)} addresses so far")
    
    if all_addresses:
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(all_addresses, crs='EPSG:4326')
        
        # Save to file
        output_path = os.path.join(output_dir, 'uploaded_addresses.geojson')
        gdf.to_file(output_path, driver='GeoJSON')
        logging.info(f"Saved {len(gdf)} address records to {output_path}")
        
        return output_path
    return None

if __name__ == "__main__":
    # Example usage
    state_abbr = 'MO'  # Example: Missouri
    output_path = process_and_save_addresses(state_abbr) 