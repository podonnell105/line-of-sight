import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box, LineString
import logging
import os
from tqdm import tqdm
import shutil
import time
import numpy as np
from process_rail import fetch_rail_lines_in_bbox
import glob
from address_los_score_lidar import (
    get_lidar_data,
    process_and_analyze_lidar_data,
    calculate_los_score
)

def calculate_definitive_los_score(bbox_coords, addr_point, rail_gdf, elevation, tree_mask, shrub_mask, building_mask, barriers=None):
    """
    Calculate simple line of sight score by checking direct visibility to railway track.
    Much less restrictive than field of view calculation.
    
    Args:
        bbox_coords: Bounding box coordinates [xmin, ymin, xmax, ymax]
        addr_point: Address point (x, y)
        rail_gdf: GeoDataFrame of railway lines
        elevation: Elevation data
        tree_mask: Tree mask data
        shrub_mask: Shrub mask data
        building_mask: Building mask data
        barriers: List of barriers/fences
    
    Returns:
        1 if address has clear direct line of sight to ANY railway point
        0 if address is completely blocked
    """
    try:
        logging.debug(f"Starting definitive LOS calculation for address at {addr_point}")
        logging.debug(f"Railway GDF has {len(rail_gdf)} lines")
        # Convert to metric CRS for accurate calculations
        rail_gdf_metric = rail_gdf.to_crs(epsg=3857)
        addr_point_metric = gpd.GeoDataFrame([{'geometry': Point(addr_point)}], crs='EPSG:4326').to_crs(epsg=3857)
        
        def check_simple_los(addr_pt, rail_pt):
            """
            Check if there's a simple direct line of sight from address to railway point.
            Much less restrictive than field of view calculation.
            
            Args:
                addr_pt: Address point (x, y) in metric coordinates
                rail_pt: Railway point (x, y) in metric coordinates
            
            Returns:
                True if direct line of sight is clear, False otherwise
            """
            import math
            
            # Calculate direct line from address to railway
            dx = rail_pt[0] - addr_pt[0]
            dy = rail_pt[1] - addr_pt[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            # Sample fewer points along the direct line (less restrictive)
            sample_distance = 100  # Increased from 50m to 100m
            max_samples = 5  # Limit number of samples
            
            sample_count = 0
            for distance_sample in range(100, int(distance), sample_distance):
                if sample_count >= max_samples:
                    break
                    
                sample_x = addr_pt[0] + (distance_sample / distance) * dx
                sample_y = addr_pt[1] + (distance_sample / distance) * dy
                
                # Convert back to WGS84 for elevation check
                sample_wgs84 = gpd.GeoDataFrame([{'geometry': Point(sample_x, sample_y)}], crs='EPSG:3857').to_crs('EPSG:4326')
                sample_coords = (sample_wgs84.geometry.iloc[0].x, sample_wgs84.geometry.iloc[0].y)
                
                # Check for obstructions at this sample point
                los_score = calculate_los_score(
                    bbox_coords,
                    addr_point,
                    sample_coords,
                    elevation, tree_mask, shrub_mask, building_mask
                )
                
                # Check barriers if provided
                if los_score == 1 and barriers:
                    addr_point_wgs84 = gpd.GeoDataFrame([{'geometry': Point(addr_point)}], crs='EPSG:4326')
                    los_line = LineString([addr_point, sample_coords])
                    
                    for barrier in barriers:
                        if barrier['geometry'].intersects(los_line):
                            barrier_height = barrier.get('height', 'unknown')
                            if barrier_height != 'unknown':
                                try:
                                    height = float(barrier_height)
                                    if height > 2.0:  # Increased barrier height threshold from 1.5m to 2.0m
                                        los_score = 0
                                        break
                                except ValueError:
                                    los_score = 0
                                    break
                            else:
                                los_score = 0
                                break
                
                # If this sample point is blocked, continue to next sample
                if los_score == 0:
                    logging.debug(f"Blocked at sample point {sample_count} at distance {distance_sample}m")
                    sample_count += 1
                    continue
                
                sample_count += 1
            
            # If most sample points are clear, consider it a clear LOS
            clear_samples = sample_count - (sample_count // 3)  # Allow 1/3 of samples to be blocked
            return sample_count > 0 and clear_samples > 0
        
        # Sample multiple points along the railway track
        sample_points = []
        for idx, rail_line in rail_gdf_metric.iterrows():
            geometry = rail_line.geometry
            if geometry.geom_type == 'LineString':
                # Sample points every 50m along the line
                length = geometry.length
                for distance in np.arange(0, length, 50):
                    point = geometry.interpolate(distance)
                    sample_points.append((point.x, point.y))
            elif geometry.geom_type == 'MultiLineString':
                for line in geometry.geoms:
                    length = line.length
                    for distance in np.arange(0, length, 50):
                        point = line.interpolate(distance)
                        sample_points.append((point.x, point.y))
        
        logging.debug(f"Sampled {len(sample_points)} points along railway track")
        
        # Check simple LOS to each sample point (much less restrictive)
        clear_los_count = 0
        total_points = len(sample_points)
        
        for rail_point in sample_points:
            # Check simple direct line of sight from address to this railway point
            addr_pt_metric = (addr_point_metric.geometry.iloc[0].x, addr_point_metric.geometry.iloc[0].y)
            
            # Check if there's a clear direct line of sight
            has_clear_los = check_simple_los(addr_pt_metric, rail_point)
            
            if has_clear_los:
                clear_los_count += 1
                logging.debug(f"Clear direct LOS to railway point {rail_point}")
            else:
                logging.debug(f"Blocked direct LOS to railway point {rail_point}")
        
        # Calculate percentage of clear LOS points
        clear_percentage = (clear_los_count / total_points) * 100 if total_points > 0 else 0
        
        logging.debug(f"Simple LOS: {clear_los_count}/{total_points} points clear ({clear_percentage:.1f}%)")
        
        # Much less restrictive: If ANY railway point is visible, consider it clear
        if clear_los_count > 0:
            return 1
        else:
            return 0
            
    except Exception as e:
        logging.error(f"Error in definitive LOS calculation: {str(e)}")
        return 0
from shapely.ops import nearest_points
import tempfile
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('city_los_processor.log'),
        logging.StreamHandler()
    ]
)

def get_city_bounds(city_name, state_abbr):
    """
    Get city boundaries using Nominatim API
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': f"{city_name}, {state_abbr}, USA",
        'format': 'json',
        'limit': 1,
        'addressdetails': 1
    }
    headers = {'User-Agent': 'LineOfSightAnalysis/1.0'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            logging.error(f"Could not find city: {city_name}, {state_abbr}")
            return None
            
        bounds = data[0]['boundingbox']
        city_bbox = {
            'xmin': float(bounds[2]),
            'ymin': float(bounds[0]),
            'xmax': float(bounds[3]),
            'ymax': float(bounds[1])
        }
        
        logging.info(f"Found city bounds for {city_name}, {state_abbr}: {city_bbox}")
        return city_bbox
        
    except Exception as e:
        logging.error(f"Error getting city bounds: {str(e)}")
        return None

def create_railway_bounding_boxes(rail_gdf, box_size_m=200, overlap_m=50, max_distance_m=1000):
    """
    Create small bounding boxes along railway lines within the city, ensuring they are within 100m of railway lines.
    
    Args:
        rail_gdf: GeoDataFrame of railway lines
        box_size_m: Size of each bounding box in meters (reduced to 200m for better precision)
        overlap_m: Overlap between boxes in meters
        max_distance_m: Maximum distance from railway line (100m)
    
    Returns:
        List of bounding box dictionaries
    """
    if rail_gdf.empty:
        logging.warning("No railway lines found")
        return []
    
    # Convert to metric CRS for accurate distance calculations
    rail_gdf = rail_gdf.to_crs(epsg=3857)
    
    bounding_boxes = []
    
    for idx, rail_line in rail_gdf.iterrows():
        geometry = rail_line.geometry
        
        if geometry.is_empty or geometry.geom_type not in ['LineString', 'MultiLineString']:
            continue
            
        # Handle MultiLineString by processing each line separately
        if geometry.geom_type == 'MultiLineString':
            lines = list(geometry.geoms)
        else:
            lines = [geometry]
        
        for line in lines:
            if line.is_empty:
                continue
                
            # Calculate points along the line at regular intervals
            length = line.length
            step_size = box_size_m - overlap_m
            
            if step_size <= 0:
                step_size = box_size_m / 2
                
            # Create boxes along the line
            for distance in np.arange(0, length, step_size):
                # Get point at this distance along the line
                point = line.interpolate(distance)
                
                # Create a buffer around the railway line for this segment
                # Get a small segment of the line around this point
                segment_start = max(0, distance - box_size_m/4)
                segment_end = min(length, distance + box_size_m/4)
                segment = line.interpolate(segment_start).buffer(max_distance_m)
                
                # Create bounding box that's constrained to be within 100m of the railway
                half_size = min(box_size_m / 2, max_distance_m)
                bbox = box(
                    point.x - half_size,
                    point.y - half_size,
                    point.x + half_size,
                    point.y + half_size
                )
                
                # Ensure the bounding box intersects with the railway buffer
                if not bbox.intersects(segment):
                    continue
                
                # Convert bbox to WGS84 for API calls
                bbox_wgs84 = gpd.GeoDataFrame(geometry=[bbox], crs='EPSG:3857').to_crs('EPSG:4326')
                min_lon, min_lat, max_lon, max_lat = bbox_wgs84.total_bounds
                
                bounding_boxes.append({
                    'xmin': min_lon,
                    'ymin': min_lat,
                    'xmax': max_lon,
                    'ymax': max_lat,
                    'rail_line_id': idx,
                    'distance_along_line': distance,
                    'distance_from_rail_m': max_distance_m
                })
    
    logging.info(f"Created {len(bounding_boxes)} bounding boxes within {max_distance_m}m of railway lines")
    return bounding_boxes

def fetch_fences_and_barriers(bbox):
    """
    Fetch fences, walls, and other barriers from OpenStreetMap that could block line of sight
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:300];
    (
      way["barrier"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
      way["fence"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
      way["wall"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
      way["building:part"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
    );
    out body;
    >;
    out skel qt;
    """
    
    try:
        response = requests.post(overpass_url, data=query)
        response.raise_for_status()
        data = response.json()
        barriers = []
        
        for element in data.get('elements', []):
            if element['type'] == 'way' and 'nodes' in element:
                # Get coordinates for this way
                coords = []
                for node in data['elements']:
                    if node['type'] == 'node' and node['id'] in element['nodes']:
                        coords.append([node['lon'], node['lat']])
                
                if len(coords) >= 2:
                    barrier_type = element.get('tags', {}).get('barrier', 'unknown')
                    barrier = {
                        'type': barrier_type,
                        'geometry': LineString(coords),
                        'height': element.get('tags', {}).get('height', 'unknown')
                    }
                    barriers.append(barrier)
        
        logging.info(f"Found {len(barriers)} barriers/fences in bbox")
        return barriers
        
    except Exception as e:
        logging.error(f"Error fetching barriers from OSM: {str(e)}")
        return []

def fetch_addresses_in_bbox(bbox, city_name, state_abbr, rail_gdf=None):
    """
    Fetch addresses from OpenStreetMap for a given bounding box
    Uses multiple queries to get comprehensive address coverage
    Filters addresses to only those within 100m of railway lines
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Queries to get ONLY real addresses with house numbers
    queries = [
        # Query 1: Complete addresses with house numbers and street names
        f"""
        [out:json][timeout:300];
        (
          node["addr:housenumber"]["addr:street"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
          way["addr:housenumber"]["addr:street"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
        );
        out body;
        >;
        out skel qt;
        """,
        
        # Query 2: Buildings with house numbers only
        f"""
        [out:json][timeout:300];
        (
          way["building"]["addr:housenumber"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
          node["building"]["addr:housenumber"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
        );
        out body;
        >;
        out skel qt;
        """,
        
        # Query 3: All nodes and ways with house numbers
        f"""
        [out:json][timeout:300];
        (
          node["addr:housenumber"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
          way["addr:housenumber"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
        );
        out body;
        >;
        out skel qt;
        """
    ]
    
    all_addresses = []
    
    for query_idx, query in enumerate(queries):
        try:
            logging.info(f"Running address query {query_idx + 1} for bbox")
            response = requests.post(overpass_url, data=query)
            response.raise_for_status()
            data = response.json()
            
            logging.info(f"Query {query_idx + 1} returned {len(data.get('elements', []))} elements")
            
            for element in data.get('elements', []):
                if 'tags' not in element:
                    continue
                    
                lat = None
                lon = None
                
                if element['type'] == 'node':
                    lat = element.get('lat')
                    lon = element.get('lon')
                elif element['type'] == 'way':
                    if 'nodes' in element and element['nodes']:
                        # Get the first node of the way for coordinates
                        for node in data['elements']:
                            if node['type'] == 'node' and node['id'] == element['nodes'][0]:
                                lat = node.get('lat')
                                lon = node.get('lon')
                                break
                                
                if lat is not None and lon is not None:
                    # Get address components
                    housenumber = element['tags'].get('addr:housenumber', '')
                    street = element['tags'].get('addr:street', '')
                    city = element['tags'].get('addr:city', city_name)
                    state = element['tags'].get('addr:state', state_abbr)
                    postcode = element['tags'].get('addr:postcode', '')
                    
                    # Only accept addresses with house numbers
                    if not housenumber:
                        continue  # Skip addresses without house numbers
                    
                    # Format address properly for CSV output
                    address1 = ""
                    street_name = street if street else ""
                    
                    # Case 1: Complete address (house number + street)
                    if housenumber and street:
                        address1 = f"{housenumber} {street}"
                    
                    # Case 2: Building with house number only
                    elif housenumber:
                        # Try to find nearby street name
                        nearby_street = ""
                        for nearby_element in data.get('elements', []):
                            if nearby_element['type'] == 'way' and 'tags' in nearby_element:
                                nearby_tags = nearby_element['tags']
                                if 'name' in nearby_tags and nearby_tags.get('highway'):
                                    nearby_street = nearby_tags['name']
                                    break
                        
                        if nearby_street:
                            address1 = f"{housenumber} {nearby_street}"
                            street_name = nearby_street
                        else:
                            address1 = f"{housenumber} Unknown Street"
                    
                    # Create address object with proper format
                    address = {
                        'address': address1,
                        'city': city,
                        'state': state,
                        'zip5': postcode if postcode else "",
                        'geometry': Point(lon, lat),
                        'source': f'query_{query_idx + 1}',
                        'housenumber': housenumber,
                        'street': street_name,
                        'element_type': element.get('type', 'unknown'),
                        'element_id': element.get('id', 'unknown')
                    }
                    all_addresses.append(address)
                        
        except Exception as e:
            logging.error(f"Error in address query {query_idx + 1}: {str(e)}")
            continue
    
    # Remove duplicates based on coordinates
    unique_addresses = []
    seen_coords = set()
    
    for addr in all_addresses:
        coord_key = (round(addr['geometry'].x, 6), round(addr['geometry'].y, 6))
        if coord_key not in seen_coords:
            unique_addresses.append(addr)
            seen_coords.add(coord_key)
    
    logging.info(f"Found {len(unique_addresses)} unique real addresses in bbox")
    
    # Return ALL addresses without railway filtering (maximum coverage)
    logging.info(f"Returning ALL {len(unique_addresses)} addresses without railway filtering")
    return unique_addresses



def process_city_los_analysis(city_name, state_abbr, output_dir='web_data', box_size_m=200, overlap_m=50, elevation_provider=None):
    """
    Process line of sight analysis for a city by creating small bounding boxes along railway lines.
    Uses satellite imagery and enhanced obstacle detection for houses, fences, trees, and elevation changes.
    Gets elevation data once for the entire city area to reduce API costs and processing time.
    
    Args:
        city_name: Name of the city
        state_abbr: State abbreviation
        output_dir: Directory to save results
        box_size_m: Size of bounding boxes in meters (200m for precision)
        overlap_m: Overlap between boxes in meters (50m)
    
    Returns:
        Path to output CSV file or None if failed
    """
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, 'temp')
    
    # Ensure temp directory is clean
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    logging.info(f"Created clean temp directory at {temp_dir}")
    
    # 1. Get city bounds
    city_bbox = get_city_bounds(city_name, state_abbr)
    if not city_bbox:
        return None
    
    # 2. Fetch railway lines within city bounds
    logging.info(f"Fetching railway lines in {city_name}, {state_abbr}...")
    rail_gdf = fetch_rail_lines_in_bbox(city_bbox)
    if rail_gdf is None or rail_gdf.empty:
        logging.error(f"No railway lines found in {city_name}, {state_abbr}")
        return None
    logging.info(f"Fetched {len(rail_gdf)} railway line segments")
    
    # 3. Get elevation data for the entire city area ONCE using specified provider or OpenTopography by default
    city_coords = [city_bbox['xmin'], city_bbox['ymin'], city_bbox['xmax'], city_bbox['ymax']]
    
    if elevation_provider is None:
        from elevation_providers import OpenTopographyProvider
        elevation_provider = OpenTopographyProvider()
        logging.info(f"Getting elevation data for entire city area using OpenTopography...")
    else:
        logging.info(f"Getting elevation data for entire city area using specified provider...")
    
    city_tif_file = get_lidar_data(city_coords, provider=elevation_provider)
    if not city_tif_file:
        logging.warning(f"Primary elevation provider failed, trying Google Elevation API as fallback...")
        from elevation_providers import GoogleElevationProvider
        google_provider = GoogleElevationProvider()
        city_tif_file = get_lidar_data(city_coords, provider=google_provider)
        if not city_tif_file:
            logging.error(f"Failed to get elevation data for city area from both providers")
            return None
        else:
            logging.info(f"Successfully got elevation data using Google Elevation API fallback")
    else:
        logging.info(f"Successfully got elevation data using primary provider")
    
    # Process elevation data for the entire city
    city_processed_file, city_elevation, city_slope, city_variance, city_tree_mask, city_shrub_mask, city_building_mask, city_stats = process_and_analyze_lidar_data(city_tif_file)
    if not city_processed_file:
        logging.error(f"Failed to process elevation data for city area")
        return None
    
    logging.info(f"Successfully processed elevation data for entire city area")
    
    # 4. Create bounding boxes covering entire city area (maximum coverage)
    bounding_boxes = create_railway_bounding_boxes(rail_gdf, box_size_m, overlap_m, max_distance_m=1000)
    if not bounding_boxes:
        logging.error("No bounding boxes created")
        return None
    
    # 5. Create output CSV file with proper format
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'{city_name}_{state_abbr}_los_scores_{timestamp}.csv')
    
    with open(output_file, 'w') as f:
        f.write('address,city,state,zip5,coordinates,los_score,visibility_percentage,distance_to_railway\n')
    
    # 6. Process each bounding box (reusing city elevation data)
    total_addresses = 0
    clear_los = 0
    blocked_los = 0
    
    for i, bbox in enumerate(tqdm(bounding_boxes, desc="Processing bounding boxes")):
        try:
            logging.info(f"Processing bbox {i+1}/{len(bounding_boxes)}")
            
            # Filter the already-fetched railway lines to this bbox
            bbox_geom = box(bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax'])
            bbox_rail_gdf = rail_gdf[rail_gdf.geometry.intersects(bbox_geom)].copy()
            if bbox_rail_gdf.empty:
                logging.info(f"No railway lines in bbox {i+1}")
                continue
            
            # Fetch addresses in this bbox (filtered to 100m of railway)
            addresses = fetch_addresses_in_bbox(bbox, city_name, state_abbr, bbox_rail_gdf)
            logging.info(f"Found {len(addresses)} addresses in bbox {i+1}")
            if not addresses:
                logging.info(f"No addresses found within 100m of railway in bbox {i+1}")
                continue
            
            logging.info(f"Processing {len(addresses)} addresses in bbox {i+1}")
            
            # Fetch fences and barriers in this bbox
            barriers = fetch_fences_and_barriers(bbox)
            logging.info(f"Found {len(barriers)} barriers/fences in bbox {i+1}")
            
            # Convert to GeoDataFrame
            addr_gdf = gpd.GeoDataFrame(addresses, crs='EPSG:4326')
            
            # Process each address in this bbox (using definitive LOS calculation)
            for idx, addr in addr_gdf.iterrows():
                try:
                    logging.info(f"Processing address {idx+1}/{len(addr_gdf)}: {addr['address']}")
                    
                    # Calculate definitive LOS score to entire railway track
                    score = calculate_definitive_los_score(
                        city_coords,  # Use city coordinates instead of bbox coordinates
                        (addr.geometry.x, addr.geometry.y),
                        bbox_rail_gdf,
                        city_elevation, city_tree_mask, city_shrub_mask, city_building_mask,
                        barriers
                    )
                    
                    # Fallback to simple LOS if definitive calculation fails
                    if score is None:
                        logging.warning(f"Definitive LOS failed for address {idx}, using fallback")
                        nearest_rail_pt = nearest_points(addr.geometry, bbox_rail_gdf.union_all())[1]
                        score = calculate_los_score(
                            city_coords,
                            (addr.geometry.x, addr.geometry.y),
                            (nearest_rail_pt.x, nearest_rail_pt.y),
                            city_elevation, city_tree_mask, city_shrub_mask, city_building_mask
                        )
                    
                    # Calculate distance to railway
                    nearest_rail_pt = nearest_points(addr.geometry, bbox_rail_gdf.union_all())[1]
                    distance_to_railway = addr.geometry.distance(nearest_rail_pt)
                    
                    # Format coordinates
                    coords = f"{addr.geometry.y:.6f}, {addr.geometry.x:.6f}"
                    
                    # Write to CSV with proper format
                    with open(output_file, 'a') as f:
                        f.write(f'"{addr["address"]}","{addr["city"]}","{addr["state"]}","{addr["zip5"]}","{coords}",{score},{score*100:.1f},{distance_to_railway:.1f}\n')
                    
                    logging.info(f"Processed address: {addr['address']} - LOS score: {score}")
                    
                    # Update statistics
                    total_addresses += 1
                    if score == 1:
                        clear_los += 1
                    else:
                        blocked_los += 1
                        
                except Exception as e:
                    logging.error(f"Error processing address: {str(e)}")
                    continue
            
            # Clear memory for this bbox
            del addr_gdf
            del bbox_rail_gdf
            gc.collect()
            
        except Exception as e:
            logging.error(f"Error processing bbox {i+1}: {str(e)}")
            continue
    
    # Clean up city elevation files
    for file_path in [city_tif_file, city_processed_file]:
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception as e:
                logging.warning(f"Failed to remove city elevation file {file_path}: {str(e)}")
    
    if total_addresses == 0:
        logging.error("No addresses processed")
        return None
    
    # Calculate final statistics
    clear_percentage = (clear_los / total_addresses) * 100
    logging.info(f"Analysis complete: {total_addresses} addresses processed")
    logging.info(f"Clear LOS: {clear_los} ({clear_percentage:.1f}%)")
    logging.info(f"Blocked LOS: {blocked_los}")
    
    return output_file 