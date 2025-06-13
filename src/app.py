from flask import Flask, render_template, request, jsonify, send_file
import os
from dotenv import load_dotenv
import geopandas as gpd
from address_los_score_lidar import (
    bbox_from_point_radius,
    fetch_rail_lines_in_bbox,
    fetch_buildings_osm,
    get_lidar_data,
    process_and_analyze_lidar_data,
    calculate_los_score
)
import tempfile
import json
from shapely.geometry import Point, box, shape
from shapely.ops import nearest_points
import folium
from folium.plugins import Draw
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
import rasterio
from rasterio.merge import merge as raster_merge
import gc

# Load environment variables from .env file
load_dotenv(override=True)  # Force reload of environment variables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Log environment variable status
api_key = os.getenv('GOOGLE_MAPS_API_KEY')
if api_key:
    logging.info(f"Google Maps API key loaded: {api_key[:4]}...{api_key[-4:]}")
else:
    logging.error("Google Maps API key not found in environment variables")
    # Try to load directly from .env file
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('GOOGLE_MAPS_API_KEY='):
                    api_key = line.strip().split('=')[1]
                    os.environ['GOOGLE_MAPS_API_KEY'] = api_key
                    logging.info(f"Manually loaded API key: {api_key[:4]}...{api_key[-4:]}")
                    break
    except Exception as e:
        logging.error(f"Error reading .env file: {str(e)}")

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)

# Create web-specific directories
WEB_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web_data')
WEB_OUTPUT_DIR = os.path.join(WEB_DATA_DIR, 'output')
WEB_TEMP_DIR = os.path.join(WEB_DATA_DIR, 'temp')

for directory in [WEB_DATA_DIR, WEB_OUTPUT_DIR, WEB_TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)

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

def cleanup_old_files():
    """Clean up files older than 1 hour in the temp and output directories"""
    current_time = datetime.now()
    for directory in [WEB_TEMP_DIR, WEB_OUTPUT_DIR]:
        if not os.path.exists(directory):
            continue
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.getmtime(filepath) < (current_time.timestamp() - 3600):
                try:
                    os.remove(filepath)
                    logging.info(f"Cleaned up old file: {filepath}")
                except Exception as e:
                    logging.warning(f"Failed to remove old file {filepath}: {e}")

def save_analysis_plot(close_addr_gdf, rail_gdf, buildings_gdf, min_lat, min_lon, max_lat, max_lon, output_path):
    """Save analysis plot with the full bounding box extent"""
    try:
        plt.ioff()  # Turn off interactive mode
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # Convert to WGS84 for plotting
        close_addr_gdf = close_addr_gdf.to_crs(epsg=4326)
        rail_gdf = rail_gdf.to_crs(epsg=4326)
        buildings_gdf = buildings_gdf.to_crs(epsg=4326)
        
        # Plot rail lines
        rail_gdf.plot(ax=ax, color='yellow', linewidth=2, label='Rail Lines', zorder=3)
        
        # Plot buildings
        if not buildings_gdf.empty:
            buildings_gdf.plot(ax=ax, color='gray', alpha=0.3, edgecolor='k', linewidth=0.2, label='Buildings', zorder=2)
        
        # Plot addresses with LOS scores
        blocked = close_addr_gdf[close_addr_gdf['los_score'] == 0]
        clear = close_addr_gdf[close_addr_gdf['los_score'] == 1]
        
        # Create scatter plots instead of using plot() for better legend handling
        if not blocked.empty:
            ax.scatter(
                blocked.geometry.x,
                blocked.geometry.y,
            color='red', 
                s=100,
            alpha=0.7, 
            label='LOS=0 (Blocked)',
            zorder=4
        )
        
        if not clear.empty:
            ax.scatter(
                clear.geometry.x,
                clear.geometry.y,
            color='lime', 
                s=100,
            alpha=0.7, 
            label='LOS=1 (Clear)',
            zorder=5
        )
        
        # Set bounds to the full bounding box
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        
        # Add satellite basemap
        try:
            ctx.add_basemap(
                ax, 
                crs='EPSG:4326',
                source=ctx.providers.Esri.WorldImagery,
                attribution=False,
                attribution_size=8
            )
        except Exception as e:
            logging.warning(f"Could not add satellite basemap: {e}")
            try:
                ctx.add_basemap(
                    ax, 
                    crs='EPSG:4326',
                    source=ctx.providers.Stamen.Terrain,
                    attribution=False,
                    attribution_size=8
                )
            except Exception as e:
                logging.warning(f"Could not add terrain basemap: {e}")
        
        # Add title and labels
        ax.set_title('Address Line-of-Sight (LOS) to Rail Lines\nwith Satellite Imagery', 
                     fontsize=16, pad=20)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Add legend with explicit handles
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(
            handles=handles,
            labels=labels,
            loc='upper right',
            fontsize=12,
            framealpha=0.8,
            edgecolor='black'
        )
        
        # Add statistics text box
        stats_text = f"""
        Total Addresses: {len(close_addr_gdf)}\nClear LOS (Score=1): {len(clear)}\nBlocked LOS (Score=0): {len(blocked)}\nClear Percentage: {len(clear)/len(close_addr_gdf)*100:.1f}%
        """
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
        ax.text(
            0.02, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            family='monospace',
            verticalalignment='bottom',
            bbox=props
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plt.ion()  # Turn interactive mode back on
    except Exception as e:
        logging.error(f"Error creating plot: {str(e)}")
        raise

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
    merged = boxes_gdf.union_all()  # Updated from unary_union to union_all
    if merged.geom_type == 'Polygon':
        merged_boxes = [merged.bounds]
    else:
        merged_boxes = [geom.bounds for geom in merged.geoms]
    return merged_boxes

def get_lidar_data_with_timeout(bbox, timeout=30):
    """
    Get LiDAR data with timeout using Google Elevation API.
    
    Args:
        bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax)
        timeout (int): Timeout in seconds
        
    Returns:
        str: Path to the saved GeoTIFF file or None if failed
    """
    start_time = time.time()
    try:
        # Get LiDAR data using Google Elevation API
        tif_file = get_lidar_data(bbox)
        duration = time.time() - start_time
        logging.info(f"LiDAR data received in {duration:.1f} seconds")
        return tif_file
    except Exception as e:
        duration = time.time() - start_time
        logging.error(f"Error getting LiDAR data after {duration:.1f} seconds: {str(e)}")
        return None

def ensure_valid_geodf(gdf, crs='EPSG:4326'):
    if not isinstance(gdf, gpd.GeoDataFrame) or 'geometry' not in gdf.columns:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    gdf = gdf[~gdf['geometry'].isnull()]
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    return gdf

@app.route('/')
def index():
    return render_template('index.html', states=STATES)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not file.filename.endswith('.geojson'):
            return jsonify({'error': 'File must be a GeoJSON file'}), 400
        
        # Save the uploaded file
        file_path = os.path.join(WEB_DATA_DIR, 'uploaded_addresses.geojson')
        file.save(file_path)
        
        # Verify the file is valid GeoJSON
        try:
            gdf = gpd.read_file(file_path)
            if gdf.empty:
                return jsonify({'error': 'The uploaded file contains no data'}), 400
            return jsonify({'message': 'File uploaded successfully'})
        except Exception as e:
            logging.error(f"Error reading GeoJSON file: {str(e)}")
            return jsonify({'error': 'Invalid GeoJSON file'}), 400
            
    except Exception as e:
        logging.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': 'Server error during file upload'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Clean up old files before starting new analysis
        cleanup_old_files()
        
        # Get state from request
        state = request.json.get('state', 'IL')
        logging.info(f"Starting analysis for state: {state}")

        # Create fresh directories
        for directory in [WEB_TEMP_DIR, WEB_OUTPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
            logging.info(f"Created fresh directory: {directory}")

        # Create output CSV file with headers
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(WEB_OUTPUT_DIR, f'address_los_scores_{timestamp}.csv')
        clear_output_file = os.path.join(WEB_OUTPUT_DIR, f'clear_los_addresses_{STATES[state]}_{timestamp}.csv')
        
        # Write headers to CSV
        with open(output_file, 'w') as f:
            f.write('address,coordinates,state,los_score\n')
        with open(clear_output_file, 'w') as f:
            f.write('address,coordinates,state,los_score\n')

        

        # 2. Load all addresses for the state
        addr_path = os.path.join(WEB_DATA_DIR, 'uploaded_addresses.geojson')
        gdf = gpd.read_file(addr_path)
        logging.info(f"Total addresses loaded from file: {len(gdf)}")

        # Ensure CRS is set for all geometries
        if rail_gdf.crs is None:
            rail_gdf.set_crs(epsg=4326, inplace=True)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)

        # Convert to metric CRS for accurate distance calculations
        rail_gdf = rail_gdf.to_crs(epsg=3857)
        gdf = gdf.to_crs(epsg=3857)
        
       

        if gdf.empty:
            logging.error("No addresses within 100m of rail lines. Aborting analysis.")
            return jsonify({'error': 'No addresses within 100m of rail lines.'}), 400
        
        # Cluster addresses and create non-overlapping bounding boxes
        merged_boxes = cluster_addresses_and_create_bboxes(gdf)
        logging.info(f"Created {len(merged_boxes)} non-overlapping bounding boxes for LiDAR requests.")

        # Initialize statistics
        total_addresses = 0
        clear_los = 0
        blocked_los = 0

        # Process each box
        for i, box_coords in enumerate(merged_boxes):
            minx, miny, maxx, maxy = box_coords
            logging.info(f"Processing box {i+1}/{len(merged_boxes)}: {box_coords}")
            
            # Convert box coordinates back to WGS84 for API calls
            box_wgs84 = gpd.GeoDataFrame(
                geometry=[box(minx, miny, maxx, maxy)],
                crs='EPSG:3857'
            ).to_crs('EPSG:4326')
            min_lon, min_lat, max_lon, max_lat = box_wgs84.total_bounds
            
            # Get LiDAR data
            bbox = [min_lon, min_lat, max_lon, max_lat]
            tif_file = get_lidar_data(bbox)
            if not tif_file:
                logging.error(f"Failed to get LiDAR data for box {i+1}")
                continue
                
            # Process LiDAR data
            processed_file, elevation, slope, variance, tree_mask, shrub_mask, building_mask, stats = process_and_analyze_lidar_data(tif_file)
            if not processed_file:
                logging.error(f"Failed to process LiDAR data for box {i+1}")
                continue
                
            # Get buildings in this area
            buildings_path = os.path.join(WEB_TEMP_DIR, f'buildings_{i}.geojson')
            try:
                buildings_gdf = fetch_buildings_osm(min_lon, min_lat, max_lon, max_lat, buildings_path)
                
                # Ensure buildings have correct CRS
                if buildings_gdf.crs is None:
                    buildings_gdf.set_crs(epsg=4326, inplace=True)
                buildings_gdf = buildings_gdf.to_crs(epsg=3857)
                
                # Get addresses in this box
                box_geom = box(minx, miny, maxx, maxy)
                box_addr_gdf = gdf[gdf.geometry.within(box_geom)].copy()
                
                if box_addr_gdf.empty:
                    logging.info(f"No addresses in box {i+1}")
                    continue
                    
                # Process addresses in chunks
                chunk_size = 50  # Process 50 addresses at a time
                for chunk_start in range(0, len(box_addr_gdf), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(box_addr_gdf))
                    chunk = box_addr_gdf.iloc[chunk_start:chunk_end]
                    
                    # Process each address in the chunk
                    for idx, addr in chunk.iterrows():
                        try:
                            # Find nearest rail point
                            nearest_rail_pt = nearest_points(addr.geometry, rail_gdf.union_all())[1]
                            
                            # Convert points to WGS84 for elevation analysis
                            addr_wgs84 = gpd.GeoSeries([addr.geometry], crs='EPSG:3857').to_crs('EPSG:4326')[0]
                            rail_wgs84 = gpd.GeoSeries([nearest_rail_pt], crs='EPSG:3857').to_crs('EPSG:4326')[0]
                            
                            # Calculate LOS score
                            score = calculate_los_score(
                                bbox,
                                (addr_wgs84.x, addr_wgs84.y),
                                (rail_wgs84.x, rail_wgs84.y),
                                elevation, tree_mask, shrub_mask, building_mask
                            )
                            
                            # Format address from available fields
                            addr_parts = []
                            address_fields = ['address', 'full_address', 'street_address', 'street', 'location']
                            for field in address_fields:
                                if field in addr and pd.notnull(addr[field]):
                                    addr_parts.append(str(addr[field]))
                                    break
                            
                            # Add state from request if not in address
                            if state not in ' '.join(addr_parts):
                                addr_parts.append(state)
                            
                            # If no address parts found, use coordinates as identifier
                            if not addr_parts:
                                addr_parts.append(f"Location at {addr_wgs84.y:.6f}, {addr_wgs84.x:.6f}")
                            
                            formatted_addr = ', '.join(addr_parts)
                            
                            # Format coordinates as a single string
                            coords = f"{addr_wgs84.y:.6f}, {addr_wgs84.x:.6f}"
                            
                            # Write to CSV with proper escaping and los_score
                            with open(output_file, 'a') as f:
                                f.write(f'"{formatted_addr}","{coords}","{state}",{score}\n')
                            
                            # Write to clear LOS file if score is 1
                            if score == 1:
                                with open(clear_output_file, 'a') as f:
                                    f.write(f'"{formatted_addr}","{coords}","{state}",{score}\n')
                            
                            # Update statistics
                            total_addresses += 1
                            if score == 1:
                                clear_los += 1
                            else:
                                blocked_los += 1
                                
                        except Exception as e:
                            logging.error(f"Error processing address: {str(e)}")
                            continue
                    
                    # Clear memory after each chunk
                    del chunk
                    gc.collect()  # Force garbage collection
                
                # Clean up temporary files
                for file_path in [tif_file, processed_file, buildings_path]:
                    if os.path.exists(file_path):
                        try:
                            os.unlink(file_path)
                        except Exception as e:
                            logging.warning(f"Failed to remove temporary file {file_path}: {str(e)}")
                
                # Clear memory
                del buildings_gdf
                del box_addr_gdf
                gc.collect()  # Force garbage collection
                
            except Exception as e:
                logging.error(f"Error processing box {i+1}: {str(e)}")
                continue
        
        if total_addresses == 0:
            return jsonify({'error': 'No results generated'}), 400
            
        # Calculate final statistics
        statistics = {
            'total_addresses': total_addresses,
            'clear_los': clear_los,
            'blocked_los': blocked_los,
            'clear_percentage': float(clear_los) / max(1, total_addresses) * 100
        }
        
        return jsonify({
            'success': True,
            'message': 'Analysis complete',
            'output_file': os.path.basename(output_file),
            'clear_output_file': os.path.basename(clear_output_file),
            'download_url': f'/results/{os.path.basename(output_file)}',
            'clear_download_url': f'/results/{os.path.basename(clear_output_file)}',
            'statistics': statistics
        })
        
    except Exception as e:
        logging.error(f"Error in analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def get_results(filename):
    try:
        # Validate filename to prevent directory traversal
        if not filename.startswith('address_los_scores_') or not filename.endswith('.csv'):
            return jsonify({'error': 'Invalid file requested'}), 403
            
        file_path = os.path.join(WEB_OUTPUT_DIR, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        # Set proper headers for CSV download
        return send_file(
            file_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logging.error(f"Error serving results file: {str(e)}")
        return jsonify({'error': 'Failed to serve results file'}), 500

@app.route('/select_state', methods=['POST'])
def select_state():
    try:
        state_abbr = request.form.get('state')
        logging.info(f"Received state selection: {state_abbr}")
        
        if not state_abbr:
            logging.error("No state provided in request")
            return jsonify({'error': 'No state selected'}), 400
            
        if state_abbr not in STATES:
            logging.error(f"Invalid state abbreviation: {state_abbr}")
            return jsonify({'error': f'Invalid state selection: {state_abbr}'}), 400

        logging.info(f"Processing addresses for state: {state_abbr}")
        # Process addresses for the selected state with 100m buffer
        output_path = process_and_save_addresses(
            state_abbr=state_abbr,
            output_dir='web_data',
            buffer_m=100,  # 100m buffer around rail lines
            max_buffers=None,
            chunk_size=1000
        )

        if output_path:
            logging.info(f"Successfully processed addresses for {state_abbr}")
            return jsonify({
                'success': True,
                'message': f'Successfully processed addresses for {STATES[state_abbr]}',
                'file_path': output_path
            })
        else:
            logging.error(f"Failed to process addresses for {state_abbr}")
            return jsonify({
                'error': f'Failed to process addresses for {STATES[state_abbr]}'
            }), 500

    except Exception as e:
        logging.error(f"Error in select_state: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
    gdf = ensure_valid_geodf(gdf)
    return gdf

@app.route('/map_viewer')
def map_viewer():
    return render_template('map_viewer.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 