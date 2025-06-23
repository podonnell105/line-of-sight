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
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('address_fetch.log'),
        logging.StreamHandler()
    ]
)

def fetch_addresses_from_osm(bbox, state_abbr):
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
                # Get all address components
                housenumber = element['tags'].get('addr:housenumber', '')
                street = element['tags'].get('addr:street', '')
                city = element['tags'].get('addr:city', '')
                county = element['tags'].get('addr:county', '')
                state = element['tags'].get('addr:state', '')
                postcode = element['tags'].get('addr:postcode', '')
                
                # Only include addresses from the specified state
                if state.upper() != state_abbr.upper():
                    continue
                
                # Format full address
                full_address = f"{housenumber}, {street}, {city}, {county} County, {state}, {postcode}, United States"
                
                address = {
                    'address': full_address,
                    'type': 'residential',
                    'geometry': Point(lon, lat)
                }
                addresses.append(address)
        logging.info(f"Found {len(addresses)} addresses in bbox")
        return addresses
    except Exception as e:
        logging.error(f"Error fetching addresses from OSM: {str(e)}")
        return []

def process_and_save_addresses(state_abbr, output_dir='web_data', buffer_m=100, max_buffers = 50, chunk_size=1000):
    """
    Fetch Class 1 rail lines, buffer by specified distance, fetch addresses only near rails,
    process in chunks, and save to file.
    """
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, 'temp')
    
    # Ensure temp directory is clean
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    logging.info(f"Created clean temp directory at {temp_dir}")
    
    # Clear existing files
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

    # 3. Buffer rail lines
    rail_gdf = rail_gdf.to_crs(epsg=3857)  # Convert to metric CRS for accurate buffer
    rail_gdf['buffer'] = rail_gdf.geometry.buffer(buffer_m)
    buffers = rail_gdf['buffer'].to_crs(epsg=4326)

    # Limit buffers before processing
    if max_buffers is not None:
        buffers = buffers.iloc[:max_buffers]  # Use iloc to properly slice the GeoSeries
        logging.info(f"Limited to {len(buffers)} buffers")

    # 4. Process buffers in chunks
    all_addresses = []
    chunk_num = 0
    
    for i, geom in tqdm(list(enumerate(buffers)), desc="Processing rail buffers"):
        if geom.is_empty:
            continue
            
        # Get the rail lines that intersect with this buffer
        buffer_3857 = gpd.GeoDataFrame(geometry=[geom], crs='EPSG:4326').to_crs('EPSG:3857')
        intersecting_rails = rail_gdf[rail_gdf.geometry.intersects(buffer_3857.geometry[0])]
        
        if intersecting_rails.empty:
            continue
            
        # Create a single buffer for all intersecting rail lines
        rail_buffer = intersecting_rails.geometry.unary_union.buffer(buffer_m)
        
        # Get bbox for OSM query
        minx, miny, maxx, maxy = geom.bounds
        bbox = {'min_lat': miny, 'min_lon': minx, 'max_lat': maxy, 'max_lon': maxx}
        addresses = fetch_addresses_from_osm(bbox, state_abbr)
        
        # Filter addresses to ensure they are within 100m of rail lines
        if addresses:
            # Convert addresses to GeoDataFrame for spatial filtering
            addr_gdf = gpd.GeoDataFrame(addresses, crs='EPSG:4326')
            addr_gdf = addr_gdf.to_crs(epsg=3857)  # Convert to metric CRS
            
            # Filter addresses that are within the buffer
            addr_gdf = addr_gdf[addr_gdf.geometry.within(rail_buffer)]
            
            if not addr_gdf.empty:
                # Convert back to list of addresses
                filtered_addresses = addr_gdf.to_crs(epsg=4326).to_dict('records')
                all_addresses.extend(filtered_addresses)
                logging.info(f"Found {len(filtered_addresses)} addresses within {buffer_m}m of rail lines in bbox")
            else:
                logging.info(f"No addresses within {buffer_m}m of rail lines in bbox")
        
        # Save chunk if it reaches chunk_size
        if len(all_addresses) >= chunk_size:
            chunk_file = os.path.join(temp_dir, f'chunk_{chunk_num}.geojson')
            gdf = gpd.GeoDataFrame(all_addresses, crs='EPSG:4326')
            gdf = gdf.drop_duplicates(subset=['geometry'])
            gdf.to_file(chunk_file, driver='GeoJSON')
            logging.info(f"Saved chunk {chunk_num} with {len(gdf)} addresses to {chunk_file}")
            all_addresses = []
            chunk_num += 1
        
        time.sleep(1)  # Be nice to Overpass API
        if (i + 1) % 10 == 0:
            logging.info(f"Processed {i + 1}/{len(buffers)} buffers, current chunk size: {len(all_addresses)}")

    # Save remaining addresses
    if all_addresses:
        chunk_file = os.path.join(temp_dir, f'chunk_{chunk_num}.geojson')
        gdf = gpd.GeoDataFrame(all_addresses, crs='EPSG:4326')
        gdf = gdf.drop_duplicates(subset=['geometry'])
        gdf.to_file(chunk_file, driver='GeoJSON')
        logging.info(f"Saved final chunk {chunk_num} with {len(gdf)} addresses to {chunk_file}")

    # 5. Combine all chunks into final file
    chunk_files = sorted(glob.glob(os.path.join(temp_dir, 'chunk_*.geojson')))
    if chunk_files:
        combined_gdf = pd.concat([gpd.read_file(f) for f in chunk_files])
        combined_gdf = combined_gdf.drop_duplicates(subset=['geometry'])
        combined_gdf.to_file(existing_file, driver='GeoJSON')
        logging.info(f"Combined {len(chunk_files)} chunks into final file with {len(combined_gdf)} addresses")
        
        # Clean up temp files
        for f in chunk_files:
            os.remove(f)
        os.rmdir(temp_dir)
        
        return existing_file
    else:
        logging.error("No addresses found near rail lines.")
        return None

