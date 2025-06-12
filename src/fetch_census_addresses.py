import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import logging
import os
from tqdm import tqdm
import shutil
import time
import numpy as np
from process_rail import fetch_rail_lines_in_bbox

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('address_fetch.log'),
        logging.StreamHandler()
    ]
)

def fetch_addresses_from_osm(bbox):
    """
    Fetch addresses from OpenStreetMap for a given bounding box
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
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
            if 'tags' not in element or 'addr:housenumber' not in element['tags']:
                continue
            lat = None
            lon = None
            if element['type'] == 'node':
                lat = element.get('lat')
                lon = element.get('lon')
            elif element['type'] == 'way':
                if 'nodes' in element and element['nodes']:
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

def process_and_save_addresses(state_abbr, output_dir='web_data', buffer_m=500, max_buffers=None):
    """
    Fetch Class 1 rail lines, buffer by 500m, fetch addresses only near rails, and save to file.
    Optionally limit to max_buffers buffers for testing.
    """
    os.makedirs(output_dir, exist_ok=True)
    existing_file = os.path.join(output_dir, 'uploaded_addresses.geojson')
    if os.path.exists(existing_file):
        os.remove(existing_file)
        logging.info(f"Removed existing {existing_file}")

    # 1. Get state bounds using Nominatim
    url = f"https://nominatim.openstreetmap.org/search"
    params = {
        'q': f"{state_abbr}, USA",
        'format': 'json',
        'limit': 1
    }
    headers = {'User-Agent': 'LineOfSightAnalysis/1.0'}
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        if not data:
            logging.error(f"Could not get bounds for state {state_abbr}")
            return None
        bounds = data[0]['boundingbox']
        state_bbox = {
            'xmin': float(bounds[2]),
            'ymin': float(bounds[0]),
            'xmax': float(bounds[3]),
            'ymax': float(bounds[1])
        }
    except Exception as e:
        logging.error(f"Error getting state bounds: {str(e)}")
        return None

    # 2. Fetch Class 1 rail lines in the state
    logging.info(f"Fetching Class 1 rail lines in {state_abbr}...")
    rail_gdf = fetch_rail_lines_in_bbox(state_bbox)
    if rail_gdf is None or rail_gdf.empty:
        logging.error(f"No Class 1 rail lines found in {state_abbr}")
        return None
    logging.info(f"Fetched {len(rail_gdf)} Class 1 rail line segments.")

    # 3. Buffer rail lines by 500m (convert to metric CRS first)
    rail_gdf = rail_gdf.to_crs(epsg=3857)
    rail_gdf['buffer'] = rail_gdf.geometry.buffer(buffer_m)
    buffers = rail_gdf['buffer'].to_crs(epsg=4326)

    # Optionally limit number of buffers for testing
    if max_buffers is not None:
        buffers = buffers[:max_buffers]

    # 4. For each buffer, get its bounding box and fetch addresses
    all_addresses = []
    for i, geom in tqdm(list(enumerate(buffers)), desc="Processing rail buffers"):
        if geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        bbox = {'min_lat': miny, 'min_lon': minx, 'max_lat': maxy, 'max_lon': maxx}
        addresses = fetch_addresses_from_osm(bbox)
        all_addresses.extend(addresses)
        time.sleep(1)  # Be nice to Overpass API
        if (i + 1) % 10 == 0:
            logging.info(f"Processed {i + 1}/{len(buffers)} buffers, found {len(all_addresses)} addresses so far")

    if all_addresses:
        gdf = gpd.GeoDataFrame(all_addresses, crs='EPSG:4326')
        gdf = gdf.drop_duplicates(subset=['geometry'])
        output_path = os.path.join(output_dir, 'uploaded_addresses.geojson')
        gdf.to_file(output_path, driver='GeoJSON')
        logging.info(f"Saved {len(gdf)} address records to {output_path}")
        return output_path
    else:
        logging.error("No addresses found near rail lines.")
        return None

if __name__ == "__main__":
    state_abbr = 'IL'  # Example: Illinois
    output_path = process_and_save_addresses(state_abbr) 