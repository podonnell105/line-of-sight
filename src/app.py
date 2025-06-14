import os
from dotenv import load_dotenv
import geopandas as gpd
from process_rail import (
    fetch_rail_lines_in_bbox,
    calculate_los_score_buildings
)
import tempfile
import json
from shapely.geometry import Point, box, shape
from shapely.ops import nearest_points
import pandas as pd
import logging
import urllib3
import warnings
import shutil
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
import contextily as ctx
from fetch_census_addresses import process_and_save_addresses
import requests
import math
import time
from sklearn.cluster import DBSCAN
import numpy as np
import gc

# Load environment variables from .env file
load_dotenv(override=True)  # Force reload of environment variables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)

# State names mapping
STATES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}

# Create output directories
OUTPUT_DIR = 'output'
TEMP_DIR = os.path.join(OUTPUT_DIR, 'temp')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def cleanup_temp_files():
    """Clean up only temporary files in the temp directory"""
    if os.path.exists(TEMP_DIR):
        for filename in os.listdir(TEMP_DIR):
            if filename.startswith('buildings_') and filename.endswith('.geojson'):
                try:
                    os.remove(os.path.join(TEMP_DIR, filename))
                    logging.info(f"Cleaned up temporary file: {filename}")
                except Exception as e:
                    logging.warning(f"Failed to remove temporary file {filename}: {e}")

def fetch_buildings_osm(minx, miny, maxx, maxy, output_fn):
    # Add delay between requests
    time.sleep(1)  # 1 second delay
    
    # Define AOI polygon (closed)
    aoi_geom = {
        "coordinates": [[
            [minx, maxy],
            [minx, miny],
            [maxx, miny],
            [maxx, maxy],
            [minx, maxy],
        ]],
        "type": "Polygon",
    }
    aoi_shape = shape(aoi_geom)
    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:300];
    (
      way["building"]({miny},{minx},{maxy},{maxx});
      relation["building"]({miny},{minx},{maxy},{maxx});
    );
    out body;
    >;
    out skel qt;
    """
    print("Fetching buildings from OpenStreetMap via Overpass API...")
    response = requests.get(overpass_url, params={'data': overpass_query})
    if response.status_code != 200:
        print(f"Overpass API error: {response.status_code} {response.text}")
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    try:
        data = response.json()
    except Exception as e:
        print(f"Error parsing Overpass response as JSON: {e}")
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    if 'elements' not in data or not data['elements']:
        print("No elements in Overpass response or rate limited.")
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    print("Processing OSM data...")
    features = []
    nodes = {}
    for element in data['elements']:
        if element['type'] == 'node':
            nodes[element['id']] = [element['lon'], element['lat']]
    for element in data['elements']:
        if element['type'] == 'way' and 'nodes' in element:
            coords = []
            for node_id in element['nodes']:
                if node_id in nodes:
                    coords.append(nodes[node_id])
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])  # Close the polygon
            if len(coords) >= 4:
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [coords]
                    },
                    'properties': {
                        'id': element['id'],
                        'building': element.get('tags', {}).get('building', 'yes')
                    }
                }
                features.append(feature)
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }
    gdf = gpd.GeoDataFrame.from_features(geojson, crs='EPSG:4326')
    gdf = gdf[gdf.geometry.within(aoi_shape)]
    if output_fn is not None:
        print(f"Saving {len(gdf)} building footprints to {output_fn} ...")
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
        gdf.to_file(output_fn, driver='GeoJSON')
        print(f"Saved building footprints to {output_fn}")
    return gdf

def cluster_addresses_and_create_bboxes(gdf, eps_m=2000, min_samples=1, buffer_m=250):
    """
    Cluster addresses using DBSCAN and create non-overlapping bounding boxes for each cluster.
    eps_m: max distance between addresses in a cluster (meters)
    buffer_m: buffer to add around each cluster's bounding box (meters)
    Returns: list of bounding boxes [minx, miny, maxx, maxy]
    """
    coords = np.array([(geom.x, geom.y) for geom in gdf.geometry])
    db = DBSCAN(eps=eps_m, min_samples=min_samples, metric='euclidean').fit(coords)
    gdf['cluster'] = db.labels_
    bboxes = []
    for cluster_id in sorted(gdf['cluster'].unique()):
        cluster_gdf = gdf[gdf['cluster'] == cluster_id]
        if cluster_gdf.empty:
            continue
        minx, miny, maxx, maxy = cluster_gdf.total_bounds
        # Add buffer
        minx -= buffer_m
        miny -= buffer_m
        maxx += buffer_m
        maxy += buffer_m
        bboxes.append([minx, miny, maxx, maxy])
    # Merge overlapping boxes
    boxes_gdf = gpd.GeoDataFrame(geometry=[box(*b) for b in bboxes], crs=gdf.crs)
    merged = boxes_gdf.union_all()
    if merged.geom_type == 'Polygon':
        merged_boxes = [merged.bounds]
    else:
        merged_boxes = [geom.bounds for geom in merged.geoms]
    return merged_boxes

def tile_buffered_corridor(buffered_union, tile_size=1000):
    # buffered_union: shapely Polygon or MultiPolygon in metric CRS (e.g., EPSG:3857)
    minx, miny, maxx, maxy = buffered_union.bounds
    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tile = box(x, y, x + tile_size, y + tile_size)
            if buffered_union.intersects(tile):
                tiles.append(tile)
            y += tile_size
        x += tile_size
    return tiles

def process_state(state_abbr):
    """Process a single state and return statistics"""
    try:
        logging.info(f"Starting analysis for state: {state_abbr}")
        
        # Create output files for this state
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(OUTPUT_DIR, f'address_los_scores_{state_abbr}_{timestamp}.csv')
        clear_output_file = os.path.join(OUTPUT_DIR, f'clear_los_addresses_{state_abbr}_{timestamp}.csv')
        
        # Write headers to CSV
        with open(output_file, 'w') as f:
            f.write('address,coordinates,state,los_score\n')
        with open(clear_output_file, 'w') as f:
            f.write('address,coordinates,state,los_score\n')

        # Get state bounds using Nominatim
        url = f"https://nominatim.openstreetmap.org/search"
        params = {
            'q': f"{state_abbr}, USA",
            'format': 'json',
            'limit': 1
        }
        headers = {'User-Agent': 'LineOfSightAnalysis/1.0'}
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

        # Fetch rail lines for the state
        logging.info("Fetching rail lines...")
        rail_gdf = fetch_rail_lines_in_bbox(state_bbox)
        if rail_gdf is None or rail_gdf.empty:
            logging.error(f"No rail lines found in {state_abbr}")
            return None

        # Buffer rail lines by 200m in metric CRS
        rail_gdf = rail_gdf.to_crs(epsg=3857)
        buffer_dist = 200
        rail_buffer = rail_gdf.buffer(buffer_dist)
        corridor_union = rail_buffer.union_all()

        # Tile the buffered corridor with 1km x 1km boxes
        tiles = tile_buffered_corridor(corridor_union, tile_size=1000)
        logging.info(f"Tiled corridor into {len(tiles)} bounding boxes.")

        # Initialize statistics
        total_addresses = 0
        clear_los = 0
        blocked_los = 0

        # For each tile, fetch addresses, filter, fetch buildings, analyze LOS
        for i, tile in enumerate(tiles):
            logging.info(f"Processing tile {i+1}/{len(tiles)} for {state_abbr}")
            tile_wgs84 = gpd.GeoSeries([tile], crs='EPSG:3857').to_crs('EPSG:4326')[0]
            min_lon, min_lat, max_lon, max_lat = tile_wgs84.bounds
            
            # Fetch addresses in this tile (from OSM or file)
            # Here, we use fetch_addresses_from_osm for demonstration; replace with your preferred method
            from fetch_census_addresses import fetch_addresses_from_osm
            bbox = {'min_lat': min_lat, 'min_lon': min_lon, 'max_lat': max_lat, 'max_lon': max_lon}
            addresses = fetch_addresses_from_osm(bbox, state_abbr)
            if not addresses:
                logging.info(f"No addresses found in tile {i+1}")
                continue
            addr_gdf = gpd.GeoDataFrame(addresses, crs='EPSG:4326')
            addr_gdf = addr_gdf.to_crs(epsg=3857)
            
            # Filter addresses to those within 150m of any rail line in the tile
            tile_rail = rail_gdf[rail_gdf.intersects(tile)]
            if tile_rail.empty:
                logging.info(f"No rail lines in tile {i+1}")
                continue
            rail_union = tile_rail.unary_union
            addr_gdf['distance_to_rail_m'] = addr_gdf.geometry.apply(lambda pt: pt.distance(rail_union))
            close_addr_gdf = addr_gdf[addr_gdf['distance_to_rail_m'] <= 150].copy()
            if close_addr_gdf.empty:
                logging.info(f"No addresses within 150m of rail in tile {i+1}")
                continue
            
            # Fetch buildings in this tile
            buildings_path = os.path.join(TEMP_DIR, f'buildings_{state_abbr}_tile{i}.geojson')
            buildings_gdf = fetch_buildings_osm(min_lon, min_lat, max_lon, max_lat, buildings_path)
            if buildings_gdf.crs is None:
                buildings_gdf.set_crs(epsg=4326, inplace=True)
            buildings_gdf = buildings_gdf.to_crs(epsg=3857)
            
            # Analyze LOS for each address
            for idx, addr in close_addr_gdf.iterrows():
                try:
                    nearest_rail_pt = nearest_points(addr.geometry, rail_union)[1]
                    score = calculate_los_score_buildings(
                        addr.geometry,
                        nearest_rail_pt,
                        buildings_gdf
                    )
                    # Format address from available fields
                    addr_parts = []
                    address_fields = ['address', 'full_address', 'street_address', 'street', 'location']
                    for field in address_fields:
                        if field in addr and pd.notnull(addr[field]):
                            addr_parts.append(str(addr[field]))
                            break
                    if state_abbr not in ' '.join(addr_parts):
                        addr_parts.append(state_abbr)
                    if not addr_parts:
                        addr_wgs84 = gpd.GeoSeries([addr.geometry], crs='EPSG:3857').to_crs('EPSG:4326')[0]
                        addr_parts.append(f"Location at {addr_wgs84.y:.6f}, {addr_wgs84.x:.6f}")
                    formatted_addr = ', '.join(addr_parts)
                    addr_wgs84 = gpd.GeoSeries([addr.geometry], crs='EPSG:3857').to_crs('EPSG:4326')[0]
                    coords = f"{addr_wgs84.y:.6f}, {addr_wgs84.x:.6f}"
                    with open(output_file, 'a') as f:
                        f.write(f'"{formatted_addr}","{coords}","{state_abbr}",{score}\n')
                    if score == 1:
                        with open(clear_output_file, 'a') as f:
                            f.write(f'"{formatted_addr}","{coords}","{state_abbr}",{score}\n')
                    total_addresses += 1
                    if score == 1:
                        clear_los += 1
                    else:
                        blocked_los += 1
                except Exception as e:
                    logging.error(f"Error processing address: {str(e)}")
                    continue
            # Clean up temporary files
            if os.path.exists(buildings_path):
                try:
                    os.unlink(buildings_path)
                except Exception as e:
                    logging.warning(f"Failed to remove temporary file {buildings_path}: {str(e)}")
            del buildings_gdf
            del close_addr_gdf
            del addr_gdf
            del tile_rail
            gc.collect()
        if total_addresses == 0:
            logging.error(f"No results generated for {state_abbr}")
            return None
        statistics = {
            'state': state_abbr,
            'total_addresses': total_addresses,
            'clear_los': clear_los,
            'blocked_los': blocked_los,
            'clear_percentage': float(clear_los) / max(1, total_addresses) * 100,
            'output_file': output_file,
            'clear_output_file': clear_output_file
        }
        logging.info(f"Completed analysis for {state_abbr}")
        return statistics
    except Exception as e:
        logging.error(f"Error processing state {state_abbr}: {str(e)}")
        return None

def main():
    # List of top 10 states by grain rail traffic
    states_to_process = ['IL', 'WA', 'ND', 'MN', 'TX', 'KS', 'IA', 'NE', 'IN', 'MO']
    
    # Create summary file
    summary_file = os.path.join(OUTPUT_DIR, 'analysis_summary.csv')
    with open(summary_file, 'w') as f:
        f.write('state,total_addresses,clear_los,blocked_los,clear_percentage,output_file,clear_output_file\n')
    
    # Process each state
    for state_abbr in states_to_process:
        logging.info(f"Starting processing for state: {state_abbr}")
        
        # Process the state
        stats = process_state(state_abbr)
        
        if stats:
            # Write to summary file
            with open(summary_file, 'a') as f:
                f.write(f"{stats['state']},{stats['total_addresses']},{stats['clear_los']},"
                       f"{stats['blocked_los']},{stats['clear_percentage']:.2f},"
                       f"{os.path.basename(stats['output_file'])},"
                       f"{os.path.basename(stats['clear_output_file'])}\n")
        
        # Clean up temporary files
        cleanup_temp_files()
        
        # Add delay between states to avoid rate limiting
        time.sleep(5)
    
    logging.info("Completed processing all states")

if __name__ == '__main__':
    main() 