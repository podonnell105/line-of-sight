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
                logger.warning(f"Failed to remove old file {filepath}: {e}")

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
            logger.warning(f"Could not add satellite basemap: {e}")
            try:
                ctx.add_basemap(
                    ax, 
                    crs='EPSG:4326',
                    source=ctx.providers.Stamen.Terrain,
                    attribution=False,
                    attribution_size=8
                )
            except Exception as e:
                logger.warning(f"Could not add terrain basemap: {e}")
        
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
        logger.error(f"Error creating plot: {str(e)}")
        raise

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
            logger.error(f"Error reading GeoJSON file: {str(e)}")
            return jsonify({'error': 'Invalid GeoJSON file'}), 400
            
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
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
                    logger.warning(f"Failed to remove {file_path}: {e}")
        
        data = request.json
        center_lat = float(data['center_lat'])
        center_lon = float(data['center_lon'])
        radius_km = float(data['radius_km'])
        rail_buffer_m = float(data['rail_buffer_m'])
        shrub_height_threshold = float(data['shrub_height_threshold'])
        
        logger.info(f"Starting analysis for center: {center_lat}, {center_lon}")
        
        # Calculate bounding box
        min_lat, min_lon, max_lat, max_lon = bbox_from_point_radius(
            center_lat, center_lon, radius_km
        )
        
        # 1. Fetch rail lines from shapefiles
        logger.info("Fetching rail lines...")
        rail_gdf = fetch_rail_lines_in_bbox({
            'xmin': min_lon,
            'ymin': min_lat,
            'xmax': max_lon,
            'ymax': max_lat
        })
        if rail_gdf is None or rail_gdf.empty:
            return jsonify({'error': 'No rail lines found in the area'}), 400
        
        # 2. Load and filter addresses
        logger.info("Loading and filtering addresses...")
        addr_path = os.path.join(WEB_DATA_DIR, 'uploaded_addresses.geojson')
        gdf = gpd.read_file(addr_path)
        logger.info(f"Total addresses loaded from file: {len(gdf)}")
        
        # Ensure number and street fields exist
        if 'properties' in gdf.columns:
            # Try to extract number and street from properties if they exist
            gdf['number'] = gdf['properties'].apply(lambda x: x.get('number', '') if isinstance(x, dict) else '')
            gdf['street'] = gdf['properties'].apply(lambda x: x.get('street', '') if isinstance(x, dict) else '')
        
        bbox_geom = box(min_lon, min_lat, max_lon, max_lat)
        logger.info(f"Bounding box coordinates: min_lon={min_lon}, min_lat={min_lat}, max_lon={max_lon}, max_lat={max_lat}")
        
        # Check if geometries are valid
        invalid_geoms = ~gdf.geometry.is_valid
        if invalid_geoms.any():
            logger.warning(f"Found {invalid_geoms.sum()} invalid geometries in the address data")
            # Try to fix invalid geometries
            gdf.geometry = gdf.geometry.buffer(0)
        
        addr_gdf = gdf[gdf.geometry.within(bbox_geom)].copy()
        logger.info(f"Addresses within bounding box: {len(addr_gdf)}")
        
        if addr_gdf.empty:
            # Try a more lenient check using intersects instead of within
            addr_gdf = gdf[gdf.geometry.intersects(bbox_geom)].copy()
            logger.info(f"Addresses intersecting bounding box: {len(addr_gdf)}")
            
            if addr_gdf.empty:
                return jsonify({'error': 'No addresses found in the area'}), 400
        
        # 3. Get elevation data
        logger.info("Getting elevation data...")
        bbox = [min_lon, min_lat, max_lon, max_lat]
        tif_file = get_opentopography_lidar(bbox)
        if not tif_file:
            return jsonify({'error': 'Failed to get elevation data'}), 400
        
        # Copy the elevation data to our temp directory
        temp_tif = os.path.join(WEB_TEMP_DIR, os.path.basename(tif_file))
        shutil.copy2(tif_file, temp_tif)
        os.unlink(tif_file)  # Remove original temp file
        
        # 4. Process elevation and analyze vegetation
        logger.info("Processing elevation data...")
        processed_file = process_lidar_data(temp_tif)
        if not processed_file:
            return jsonify({'error': 'Failed to process elevation data'}), 400
        
        # Copy processed file to temp directory
        temp_processed = os.path.join(WEB_TEMP_DIR, os.path.basename(processed_file))
        shutil.copy2(processed_file, temp_processed)
        os.unlink(processed_file)  # Remove original processed file
        
        elevation, slope, variance, tree_mask, shrub_mask, building_mask, stats = analyze_vegetation(
            temp_processed
        )
        
        if elevation is None:
            return jsonify({'error': 'Failed to analyze vegetation'}), 400
        
        # 5. Calculate LOS scores
        logger.info("Calculating LOS scores...")
        metric_crs = 3857
        addr_gdf = addr_gdf.to_crs(epsg=metric_crs)
        rail_gdf = rail_gdf.to_crs(epsg=metric_crs)
        
        rail_union = rail_gdf.union_all()
        addr_gdf['distance_to_rail_m'] = addr_gdf.geometry.apply(
            lambda pt: pt.distance(rail_union)
        )
        close_addr_gdf = addr_gdf[addr_gdf['distance_to_rail_m'] <= rail_buffer_m].copy()
        
        los_scores = []
        for idx, addr in close_addr_gdf.iterrows():
            nearest_rail_pt = gpd.GeoSeries(
                [addr.geometry], crs=metric_crs
            ).to_crs(epsg=4326)[0]
            
            score = calculate_los_score(
                [min_lon, min_lat, max_lon, max_lat],
                (addr.geometry.x, addr.geometry.y),
                (nearest_rail_pt.x, nearest_rail_pt.y),
                elevation,
                tree_mask,
                shrub_mask,
                building_mask
            )
            los_scores.append(score)
        
        close_addr_gdf['los_score'] = los_scores
        
        # Fetch building footprints (do not save to file, just get the GeoDataFrame)
        buildings_gdf = fetch_buildings_osm(min_lon, min_lat, max_lon, max_lat, None)
        
        # Save all outputs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save clear LOS addresses CSV
        clear_los_path = os.path.join(WEB_OUTPUT_DIR, f'clear_los_addresses_{timestamp}.csv')
        clear_los_gdf = close_addr_gdf[close_addr_gdf['los_score'] == 1].copy()
        # Ensure number and street fields are included in the output
        if 'number' in clear_los_gdf.columns and 'street' in clear_los_gdf.columns:
            clear_los_gdf = clear_los_gdf[['number', 'street', 'geometry', 'los_score', 'distance_to_rail_m'] + 
                                        [col for col in clear_los_gdf.columns if col not in ['number', 'street', 'geometry', 'los_score', 'distance_to_rail_m']]]
        clear_los_gdf.to_csv(clear_los_path, index=False)
        
        # Save analysis plot
        plot_path = os.path.join(WEB_OUTPUT_DIR, f'address_los_scores_lidar_{timestamp}.png')
        save_analysis_plot(close_addr_gdf, rail_gdf, buildings_gdf, min_lat, min_lon, max_lat, max_lon, plot_path)
        
        # Calculate statistics
        total_addresses = len(close_addr_gdf)
        clear_los = len(close_addr_gdf[close_addr_gdf['los_score'] == 1])
        blocked_los = len(close_addr_gdf[close_addr_gdf['los_score'] == 0])
        clear_percentage = (clear_los / total_addresses * 100) if total_addresses > 0 else 0
        
        # Clean up old files
        cleanup_old_files()
        
        logger.info("Analysis complete")
        return jsonify({
            'message': 'Analysis complete',
            'statistics': {
                'total_addresses': total_addresses,
                'clear_los': clear_los,
                'blocked_los': blocked_los,
                'clear_percentage': clear_percentage
            },
            'plot_file': os.path.basename(plot_path),
            'clear_los_file': os.path.basename(clear_los_path)
        })
    except Exception as e:
        logger.error(f"Error in analyze: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/results/<filename>')
def get_results(filename):
    try:
        allowed = filename.startswith('address_los_scores_lidar_') and filename.endswith('.png') or \
                  filename.startswith('clear_los_addresses_') and filename.endswith('.csv')
        if not allowed:
            return jsonify({'error': 'File not available for download'}), 403
        return send_file(os.path.join(WEB_OUTPUT_DIR, filename))
    except Exception as e:
        logger.error(f"Error serving results file: {str(e)}")
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
        # Process addresses for the selected state
        output_path = process_and_save_addresses(state_abbr)

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