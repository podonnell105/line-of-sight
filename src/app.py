from flask import Flask, render_template, request, jsonify, send_file
import os
import geopandas as gpd
from address_los_score_lidar import (
    bbox_from_point_radius,
    fetch_rail_lines_in_bbox,
    fetch_buildings_osm,
    get_opentopography_lidar,
    process_lidar_data,
    analyze_vegetation,
    calculate_los_score
)
import tempfile
import json
from shapely.geometry import Point, box
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

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

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
    """Clean up files older than 1 hour in the temp directory"""
    current_time = datetime.now()
    for filename in os.listdir(WEB_TEMP_DIR):
        filepath = os.path.join(WEB_TEMP_DIR, filename)
        if os.path.getmtime(filepath) < (current_time.timestamp() - 3600):
            try:
                os.remove(filepath)
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
        close_addr_gdf[close_addr_gdf['los_score'] == 0].plot(
            ax=ax, 
            color='red', 
            markersize=100, 
            alpha=0.7, 
            label='LOS=0 (Blocked)',
            zorder=4
        )
        
        close_addr_gdf[close_addr_gdf['los_score'] == 1].plot(
            ax=ax, 
            color='lime', 
            markersize=100, 
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
        
        # Add legend
        legend = ax.legend(
            loc='upper right',
            fontsize=12,
            framealpha=0.8,
            edgecolor='black'
        )
        
        # Add statistics text box
        stats_text = f"""
        Total Addresses: {len(close_addr_gdf)}\nClear LOS (Score=1): {len(close_addr_gdf[close_addr_gdf['los_score'] == 1])}\nBlocked LOS (Score=0): {len(close_addr_gdf[close_addr_gdf['los_score'] == 0])}\nClear Percentage: {len(close_addr_gdf[close_addr_gdf['los_score'] == 1])/len(close_addr_gdf)*100:.1f}%
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
    merged = boxes_gdf.unary_union
    if merged.geom_type == 'Polygon':
        merged_boxes = [merged.bounds]
    else:
        merged_boxes = [geom.bounds for geom in merged.geoms]
    return merged_boxes

def get_opentopography_lidar_with_timeout(bbox, timeout=30):
    """Get LiDAR data with timeout"""
    try:
        logging.info(f"Requesting LiDAR data for bbox: {bbox}")
        start_time = time.time()
        tif_file = get_opentopography_lidar(bbox)
        elapsed = time.time() - start_time
        logging.info(f"LiDAR data received in {elapsed:.1f} seconds")
        return tif_file
    except Exception as e:
        logging.error(f"Error getting LiDAR data: {str(e)}")
        return None

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
        # Clean up all files in output and temp directories before starting new analysis
        for dir_path in [WEB_OUTPUT_DIR, WEB_TEMP_DIR]:
            for filename in os.listdir(dir_path):
                if filename == '.gitignore':
                    continue
                file_path = os.path.join(dir_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logging.warning(f"Failed to remove {file_path}: {e}")
        
        data = request.json
        state_abbr = data.get('state')
        shrub_height_threshold = float(data.get('shrub_height_threshold', 0.5))
        if not state_abbr or state_abbr not in STATES:
            return jsonify({'error': 'Invalid or missing state'}), 400
        logging.info(f"Starting analysis for state: {state_abbr}")
        
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
        data_json = response.json()
        if not data_json:
            return jsonify({'error': 'Could not get bounds for state'}), 400
        bounds = data_json[0]['boundingbox']
        min_lat = float(bounds[0])
        max_lat = float(bounds[1])
        min_lon = float(bounds[2])
        max_lon = float(bounds[3])
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        radius_km = max(
            abs(max_lat - min_lat) * 111.32,
            abs(max_lon - min_lon) * 111.32
        ) / 2

        # 1. Fetch rail lines for the state
        logging.info("Fetching rail lines...")
        rail_gdf = fetch_rail_lines_in_bbox({
            'xmin': min_lon,
            'ymin': min_lat,
            'xmax': max_lon,
            'ymax': max_lat
        })
        if rail_gdf is None or rail_gdf.empty:
            return jsonify({'error': 'No rail lines found in the area'}), 400

        # 2. Load all addresses for the state
        addr_path = os.path.join(WEB_DATA_DIR, 'uploaded_addresses.geojson')
        gdf = gpd.read_file(addr_path)
        logging.info(f"Total addresses loaded from file: {len(gdf)}")

        # 3. Create larger bounding boxes that efficiently cover rail segments and addresses
        # Ensure CRS is set for all geometries
        if rail_gdf.crs is None:
            rail_gdf.set_crs(epsg=4326, inplace=True)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)

        # Convert to metric CRS for accurate distance calculations
        rail_gdf = rail_gdf.to_crs(epsg=3857)
        gdf = gdf.to_crs(epsg=3857)
        
        # Cluster addresses and create non-overlapping bounding boxes
        merged_boxes = cluster_addresses_and_create_bboxes(gdf)
        logging.info(f"Created {len(merged_boxes)} non-overlapping bounding boxes for LiDAR requests.")

        # Fetch LiDAR for each bounding box and merge
        tif_files = []
        for i, bbox in enumerate(merged_boxes):
            # Convert bbox from EPSG:3857 to EPSG:4326
            minx, miny, maxx, maxy = bbox
            bbox_geom = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=3857).to_crs(epsg=4326)[0]
            lon_min, lat_min, lon_max, lat_max = bbox_geom.bounds
            bbox_wgs84 = [lon_min, lat_min, lon_max, lat_max]
            logging.info(f"Fetching LiDAR for box {i+1}/{len(merged_boxes)}: {bbox_wgs84}")
            tif_file = get_opentopography_lidar_with_timeout(bbox_wgs84)
            if tif_file:
                tif_files.append(tif_file)
            else:
                logging.warning(f"No LiDAR for box {i+1}, skipping.")
        if not tif_files:
            return jsonify({'error': 'No LiDAR data could be fetched for any area.'}), 400
        # Merge all LiDAR tiles into one
        srcs = [rasterio.open(f) for f in tif_files]
        mosaic, out_trans = raster_merge(srcs)
        merged_tif = os.path.join(WEB_TEMP_DIR, 'merged_lidar.tif')
        out_meta = srcs[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
        with rasterio.open(merged_tif, "w", **out_meta) as dest:
            dest.write(mosaic)
        for src in srcs:
            src.close()
        for f in tif_files:
            os.unlink(f)
        logging.info(f"Merged LiDAR data saved to {merged_tif}")
        # Now use merged_tif for further processing (process_lidar_data, analyze_vegetation, LOS, etc.)
        processed_file = process_lidar_data(merged_tif)
        if not processed_file:
            return jsonify({'error': 'Failed to process merged LiDAR data.'}), 400
        temp_processed = os.path.join(WEB_TEMP_DIR, 'merged_processed.tif')
        shutil.copy2(processed_file, temp_processed)
        os.unlink(processed_file)
        elevation, slope, variance, tree_mask, shrub_mask, building_mask, stats = analyze_vegetation(temp_processed)
        if elevation is None:
            return jsonify({'error': 'Failed to analyze vegetation for merged LiDAR.'}), 400
        # LOS analysis for all addresses
        gdf['distance_to_rail_m'] = gdf.geometry.apply(
            lambda pt: min(pt.distance(rail) for rail in rail_gdf.geometry)
        )
        close_addr_gdf = gdf[gdf['distance_to_rail_m'] <= 500].copy()
        if not close_addr_gdf.empty:
            los_scores = []
            for idx, addr in close_addr_gdf.iterrows():
                nearest_rail = min(rail_gdf.geometry, key=lambda x: addr.geometry.distance(x))
                nearest_rail_pt = gpd.GeoSeries([nearest_rail], crs=3857).to_crs(epsg=4326)[0]
                score = calculate_los_score(
                    merged_boxes[0],  # Use the merged bounding box for LOS
                    (addr.geometry.x, addr.geometry.y),
                    (nearest_rail_pt.x, nearest_rail_pt.y),
                    elevation,
                    tree_mask,
                    shrub_mask,
                    building_mask
                )
                los_scores.append(score)
            close_addr_gdf['los_score'] = los_scores
            all_results = [close_addr_gdf.to_crs(epsg=4326)]
        else:
            return jsonify({'error': 'No addresses within 500m of rail.'}), 400
        
        # Aggregate all results
        if not all_results:
            return jsonify({'error': 'No addresses with LOS results found.'}), 400
            
        final_gdf = gpd.GeoDataFrame(pd.concat(all_results, ignore_index=True), crs='EPSG:4326')
        plot_file = os.path.join(WEB_OUTPUT_DIR, 'address_los_scores.png')
        
        # For plotting, use the full state bounds
        buildings_gdf = fetch_buildings_osm(min_lon, min_lat, max_lon, max_lat, None)
        save_analysis_plot(final_gdf, rail_gdf.to_crs(epsg=4326), buildings_gdf, min_lat, min_lon, max_lat, max_lon, plot_file)
        
        clear_los_gdf = final_gdf[final_gdf['los_score'] == 1].copy()
        clear_los_file = os.path.join(WEB_OUTPUT_DIR, 'clear_los_addresses.csv')
        clear_los_gdf.to_csv(clear_los_file, index=False)
        
        statistics = {
            'total_addresses': int(len(final_gdf)),
            'clear_los': int(len(clear_los_gdf)),
            'blocked_los': int((final_gdf['los_score'] == 0).sum()),
            'clear_percentage': float(len(clear_los_gdf)) / max(1, len(final_gdf)) * 100
        }
        
        return jsonify({
            'success': True,
            'plot_file': os.path.basename(plot_file),
            'clear_los_file': os.path.basename(clear_los_file),
            'statistics': statistics,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'radius_km': radius_km
        })
        
    except Exception as e:
        logging.error(f"Error in analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def get_results(filename):
    try:
        allowed = filename.startswith('address_los_scores_lidar_') and filename.endswith('.png') or \
                  filename.startswith('clear_los_addresses_') and filename.endswith('.csv')
        if not allowed:
            return jsonify({'error': 'File not available for download'}), 403
        return send_file(os.path.join(WEB_OUTPUT_DIR, filename))
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
        # For initial test, limit to 100 buffers. Change max_buffers as needed.
        output_path = process_and_save_addresses(state_abbr, max_buffers=25)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 