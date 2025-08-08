#!/usr/bin/env python3
"""
Coordinate-based Railway Line of Sight Processor
Takes a list of coordinates and creates bounding boxes along railway lines within 1km of each coordinate.
Each bounding box covers a 100m buffer from the railway line and fetches all addresses within.
"""

import os
import sys
import time
import logging
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box, LineString
from shapely.ops import unary_union
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import our existing modules
from process_rail import fetch_rail_lines_in_bbox, combine_and_deduplicate
from city_los_processor import fetch_addresses_in_bbox, calculate_definitive_los_score
from fetch_lidar import get_lidar_data
from elevation_providers import OpenTopographyProvider, GoogleElevationProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coordinate_processing.log'),
        logging.StreamHandler()
    ]
)

# Railway yard names to search for on OpenStreetMap
RAILWAY_YARD_NAMES = [
    "CSX Howell Yard, Evansville, IN",
    "CSX South Anderson Yard, Anderson, IN", 
    "NS Piqua Yard, Fort Wayne, IN",
    "Piqua Yard, Fort Wayne, IN"  # Alternative search for NS Piqua Yard
]

def search_osm_coordinates(location_name):
    """
    Search OpenStreetMap Nominatim API for coordinates of a railway yard by name.
    
    Args:
        location_name (str): Name of the railway yard to search for
    
    Returns:
        tuple: (lat, lon, display_name) or None if not found
    """
    logging.info(f"Searching OSM for: {location_name}")
    
    # Nominatim API endpoint
    url = "https://nominatim.openstreetmap.org/search"
    
    # Search parameters
    params = {
        'q': location_name,
        'format': 'json',
        'limit': 5,
        'countrycodes': 'us',
        'addressdetails': 1
    }
    
    headers = {
        'User-Agent': 'LineOfSight-CropSafe/1.0 (railway_analysis)'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        results = response.json()
        
        if not results:
            logging.warning(f"No results found for: {location_name}")
            return None
        
        # Look for railway-related results first
        for result in results:
            display_name = result.get('display_name', '')
            class_type = result.get('class', '')
            type_name = result.get('type', '')
            
            # Prefer railway-related results
            if any(keyword in display_name.lower() for keyword in ['rail', 'yard', 'station']) or \
               class_type == 'railway' or type_name in ['yard', 'station']:
                lat = float(result['lat'])
                lon = float(result['lon'])
                logging.info(f"Found railway location: {display_name} at ({lat}, {lon})")
                return (lat, lon, display_name)
        
        # If no railway-specific results, use the first result
        result = results[0]
        lat = float(result['lat'])
        lon = float(result['lon'])
        display_name = result.get('display_name', location_name)
        logging.info(f"Found general location: {display_name} at ({lat}, {lon})")
        return (lat, lon, display_name)
        
    except Exception as e:
        logging.error(f"Error searching for {location_name}: {e}")
        return None

def get_updated_coordinates():
    """
    Search OpenStreetMap for all railway yard coordinates and return updated list.
    
    Returns:
        list: Updated coordinates with (lat, lon, name) tuples
    """
    logging.info("Searching OpenStreetMap for railway yard coordinates...")
    
    updated_coordinates = []
    
    for yard_name in RAILWAY_YARD_NAMES:
        try:
            result = search_osm_coordinates(yard_name)
            if result:
                lat, lon, display_name = result
                updated_coordinates.append((lat, lon, display_name))
                # Rate limiting for Nominatim
                time.sleep(1)
            else:
                logging.warning(f"Could not find coordinates for: {yard_name}")
        except Exception as e:
            logging.error(f"Error processing {yard_name}: {e}")
            continue
    
    logging.info(f"Found {len(updated_coordinates)} railway yard locations")
    return updated_coordinates

# Coordinate list will be populated from OpenStreetMap search
COORDINATES = []

def create_coordinate_bounding_box(lat, lon, radius_km=1.0):
    """
    Create a bounding box around a coordinate point.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        radius_km (float): Radius in kilometers (default 1.0)
    
    Returns:
        dict: Bounding box with xmin, ymin, xmax, ymax
    """
    # Convert km to degrees (approximate)
    # 1 degree lat ≈ 111 km, 1 degree lon ≈ 111 km * cos(lat)
    lat_offset = radius_km / 111.0
    lon_offset = radius_km / (111.0 * np.cos(np.radians(lat)))
    
    return {
        'xmin': lon - lon_offset,
        'ymin': lat - lat_offset,
        'xmax': lon + lon_offset,
        'ymax': lat + lat_offset
    }

def fetch_railway_from_osm(bbox):
    """
    Fetch railway lines from OpenStreetMap Overpass API within bounding box.
    
    Args:
        bbox (dict): Bounding box with xmin, ymin, xmax, ymax
    
    Returns:
        GeoDataFrame: Railway lines within the area
    """
    logging.info(f"Fetching railway lines from OSM")
    
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Overpass query for railway lines
    query = f"""
    [out:json][timeout:60];
    (
      way["railway"~"^(rail|main_line)$"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
      relation["railway"~"^(rail|main_line)$"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
    );
    out geom;
    """
    
    headers = {
        'User-Agent': 'LineOfSight-CropSafe/1.0 (railway_analysis)'
    }
    
    try:
        response = requests.post(overpass_url, data=query, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('elements'):
            logging.warning(f"No railway data found in OSM")
            return gpd.GeoDataFrame()
        
        # Convert to GeoDataFrame
        geometries = []
        properties = []
        
        for element in data['elements']:
            if element['type'] == 'way' and 'geometry' in element:
                coords = [(node['lon'], node['lat']) for node in element['geometry']]
                if len(coords) >= 2:
                    line = LineString(coords)
                    geometries.append(line)
                    properties.append({
                        'osm_id': element.get('id'),
                        'railway': element.get('tags', {}).get('railway', ''),
                        'name': element.get('tags', {}).get('name', ''),
                        'operator': element.get('tags', {}).get('operator', '')
                    })
        
        if not geometries:
            logging.warning(f"No valid railway geometries found")
            return gpd.GeoDataFrame()
        
        gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs='EPSG:4326')
        logging.info(f"Found {len(gdf)} railway segments from OSM")
        return gdf
        
    except Exception as e:
        logging.error(f"Error fetching railway data from OSM: {e}")
        return gpd.GeoDataFrame()

def fetch_railway_near_coordinate(lat, lon, radius_km=1.0):
    """
    Fetch railway lines within radius_km of the given coordinate using OpenStreetMap.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        radius_km (float): Search radius in kilometers
    
    Returns:
        GeoDataFrame: Railway lines within the area
    """
    logging.info(f"Fetching railway lines within {radius_km}km of ({lat}, {lon})")
    
    bbox = create_coordinate_bounding_box(lat, lon, radius_km)
    
    # Use OpenStreetMap instead of FRA API
    rail_gdf = fetch_railway_from_osm(bbox)
    
    if rail_gdf.empty:
        logging.warning(f"No railway lines found near coordinate ({lat}, {lon})")
        return gpd.GeoDataFrame()
    
    logging.info(f"Found {len(rail_gdf)} railway segments near ({lat}, {lon})")
    return rail_gdf

def create_railway_bounding_boxes_for_coordinate(rail_gdf, bbox_size_m=1000, buffer_m=100):
    """
    Create large bounding boxes along railway lines, then filter addresses to 100m from railway.
    
    Args:
        rail_gdf (GeoDataFrame): Railway lines
        bbox_size_m (int): Size of each bounding box in meters (1000m = 1km)
        buffer_m (int): Buffer distance from railway in meters (100m)
    
    Returns:
        list: List of bounding box dictionaries
    """
    if rail_gdf.empty:
        return []
    
    # Convert to meter-based CRS for accurate distance calculations
    rail_gdf_m = rail_gdf.to_crs('EPSG:3857')
    
    # Combine all railway geometries
    combined_rail = unary_union(rail_gdf_m.geometry)
    
    bounding_boxes = []
    
    if hasattr(combined_rail, 'geoms'):
        # MultiLineString
        geometries = list(combined_rail.geoms)
    else:
        # Single LineString
        geometries = [combined_rail]
    
    for i, geom in enumerate(geometries):
        if hasattr(geom, 'length'):
            # Create points along the railway line every bbox_size_m meters
            distance = 0
            bbox_id = 0
            
            while distance < geom.length:
                try:
                    # Get point along the line
                    point = geom.interpolate(distance)
                    
                    # Create bounding box around this point
                    bbox_geom = box(
                        point.x - bbox_size_m/2,
                        point.y - bbox_size_m/2,
                        point.x + bbox_size_m/2,
                        point.y + bbox_size_m/2
                    )
                    
                    # Convert back to lat/lon
                    bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs='EPSG:3857')
                    bbox_gdf_latlon = bbox_gdf.to_crs('EPSG:4326')
                    bbox_bounds = bbox_gdf_latlon.total_bounds
                    
                    bbox_dict = {
                        'id': f"railway_{i}_bbox_{bbox_id}",
                        'xmin': bbox_bounds[0],
                        'ymin': bbox_bounds[1],
                        'xmax': bbox_bounds[2],
                        'ymax': bbox_bounds[3],
                        'center_x': point.x,
                        'center_y': point.y
                    }
                    
                    bounding_boxes.append(bbox_dict)
                    bbox_id += 1
                    
                except Exception as e:
                    logging.warning(f"Error creating bbox at distance {distance}: {e}")
                
                distance += bbox_size_m
    
    logging.info(f"Created {len(bounding_boxes)} bounding boxes along railway lines")
    return bounding_boxes

def extract_city_state_from_location(location_name):
    """
    Extract city and state from location name like:
    'CSX Howell Yard, Dixie Flyer Road, Evansville, Vanderburgh County, Indiana, 47712, United States'
    """
    try:
        # Split by commas and clean up
        parts = [part.strip() for part in location_name.split(',')]
        
        # State mapping
        state_mapping = {
            'Indiana': 'IN',
            'Illinois': 'IL',
            'Ohio': 'OH',
            'Michigan': 'MI',
            'Kentucky': 'KY',
            'Tennessee': 'TN'
        }
        
        state_abbr = 'IN'  # Default to Indiana
        city_name = 'Unknown'  # Default city
        
        # Find state first
        for part in parts:
            if part in state_mapping:
                state_abbr = state_mapping[part]
                break
        
        # Look for city name - typically comes before the county
        for i, part in enumerate(parts):
            # Skip yard names, roads, and infrastructure
            if any(keyword in part.lower() for keyword in ['yard', 'road', 'drive', 'street', 'csx', 'ns', 'trace', 'marine']):
                continue
            # Skip ZIP codes
            if part.isdigit():
                continue
            # Skip "United States"
            if part == 'United States':
                continue
            # If we find a county, the previous part was likely the city
            if part.endswith('County') and i > 0:
                city_name = parts[i-1]
                break
            # If this part doesn't contain any excluded keywords and isn't infrastructure, it might be a city
            elif (not part.endswith('County') and 
                  not part.isdigit() and 
                  part != 'United States' and
                  not any(keyword in part.lower() for keyword in ['yard', 'road', 'drive', 'street', 'csx', 'ns', 'trace', 'marine'])):
                # Check if next part is a county or state
                if i + 1 < len(parts):
                    next_part = parts[i + 1]
                    if next_part.endswith('County') or next_part in state_mapping:
                        city_name = part
                        break
        
        logging.info(f"Extracted city: {city_name}, state: {state_abbr} from {location_name}")
        return city_name, state_abbr
        
    except Exception as e:
        logging.warning(f"Error extracting city/state from {location_name}: {e}. Using defaults.")
        return 'Unknown', 'IN'

def process_coordinate_location(lat, lon, location_name, elevation_provider):
    """
    Process a single coordinate location to find addresses with line-of-sight to railway.
    
    Args:
        lat (float): Latitude of the railway yard
        lon (float): Longitude of the railway yard
        location_name (str): Name of the location
        elevation_provider: Elevation data provider
    
    Returns:
        list: List of address results
    """
    logging.info(f"Processing location: {location_name} at ({lat}, {lon})")
    
    # Extract city and state from location name
    city_name, state_abbr = extract_city_state_from_location(location_name)
    
    # Fetch railway lines within 3km of the coordinate (increased from 1km)
    radius_km = 3.0  # Increased to reach residential areas around industrial railway yards
    logging.info(f"Fetching railway lines within {radius_km}km of ({lat}, {lon})")
    rail_gdf = fetch_railway_near_coordinate(lat, lon, radius_km)
    
    if rail_gdf.empty:
        logging.warning(f"No railway lines found near {location_name}")
        return []
    
    # 2. Create bounding boxes along railway lines
    bounding_boxes = create_railway_bounding_boxes_for_coordinate(rail_gdf)
    
    if not bounding_boxes:
        logging.warning(f"No bounding boxes created for {location_name}")
        return []
    
    # 3. Fetch elevation data for the area (1km radius around coordinate)
    coord_bbox = create_coordinate_bounding_box(lat, lon, radius_km=1.0)
    logging.info(f"Fetching elevation data for {location_name}")
    
    try:
        # Convert bbox dict to tuple format expected by get_lidar_data
        bbox_tuple = (coord_bbox['xmin'], coord_bbox['ymin'], coord_bbox['xmax'], coord_bbox['ymax'])
        elevation_data = get_lidar_data(bbox_tuple, provider=elevation_provider)
        if elevation_data is None:
            logging.error(f"Failed to fetch elevation data for {location_name}")
            return []
    except Exception as e:
        logging.error(f"Error fetching elevation data for {location_name}: {e}")
        return []
    
    # 4. Process each bounding box
    all_addresses = []
    total_boxes = len(bounding_boxes)
    
    for i, bbox in enumerate(bounding_boxes):
        logging.info(f"Processing bbox {i+1}/{total_boxes} for {location_name}")
        
        try:
            # Fetch addresses in this bounding box with city and state info
            addresses = fetch_addresses_in_bbox(bbox, city_name, state_abbr, rail_gdf)
            
            if not addresses:
                logging.info(f"No addresses found in bbox {i+1}")
                continue
            
            # Filter railway data for this bbox
            bbox_geom = box(bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax'])
            bbox_gdf = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs='EPSG:4326')
            bbox_rail_gdf = rail_gdf[rail_gdf.intersects(bbox_geom)]
            
            if bbox_rail_gdf.empty:
                logging.warning(f"No railway in bbox {i+1}, skipping LOS calculation")
                continue
            
            # Filter addresses to only those within 100m of railway lines
            filtered_addresses = []
            for addr in addresses:
                try:
                    # Get the point from the address geometry
                    addr_point = addr['geometry']  # This is already a Point object from fetch_addresses_in_bbox
                    
                    # Calculate distance to railway
                    addr_gdf = gpd.GeoDataFrame([{'geometry': addr_point}], crs='EPSG:4326')
                    addr_gdf_m = addr_gdf.to_crs('EPSG:3857')
                    rail_gdf_m = bbox_rail_gdf.to_crs('EPSG:3857')
                    
                    distances = addr_gdf_m.geometry.iloc[0].distance(rail_gdf_m.geometry)
                    distance_to_railway = distances.min()
                    
                    # Only include addresses within 100m of railway
                    if distance_to_railway <= 100:
                        # Calculate LOS score - pass parameters in correct order
                        los_score = calculate_definitive_los_score(
                            bbox,  # bbox_coords
                            (addr_point.x, addr_point.y),  # addr_point as tuple
                            bbox_rail_gdf,  # rail_gdf
                            elevation_data,  # elevation
                            None,  # tree_mask (not available)
                            None,  # shrub_mask (not available) 
                            None,  # building_mask (not available)
                            None   # barriers (not available)
                        )
                        
                        # Add location info and results to address
                        addr_result = {
                            'location_name': location_name,
                            'location_lat': lat,
                            'location_lon': lon,
                            'address': addr['address'],
                            'city': addr['city'],
                            'state': addr['state'],
                            'zip5': addr['zip5'],
                            'coordinates': f"{addr_point.y},{addr_point.x}",
                            'los_score': los_score,
                            'visibility_percentage': los_score * 100,
                            'distance_to_railway': distance_to_railway
                        }
                        
                        filtered_addresses.append(addr_result)
                        all_addresses.append(addr_result)
                    
                except Exception as e:
                    logging.error(f"Error processing address in {location_name}: {e}")
                    continue
            
            logging.info(f"Found {len(addresses)} addresses in bbox, filtered to {len(filtered_addresses)} within 100m of railway")
        
        except Exception as e:
            logging.error(f"Error processing bbox {i+1} for {location_name}: {e}")
            continue
        
        # Rate limiting
        time.sleep(0.5)
    
    logging.info(f"Found {len(all_addresses)} addresses for {location_name}")
    return all_addresses

def main():
    """
    Main function to process all coordinates and generate CSV output.
    """
    logging.info("Starting coordinate-based railway processing")
    
    # First, get updated coordinates from OpenStreetMap
    coordinates = get_updated_coordinates()
    
    if not coordinates:
        logging.error("No valid coordinates found from OpenStreetMap search")
        return
    
    # Set up elevation provider (try OpenTopography first, fallback to Google)
    elevation_provider = None
    try:
        elevation_provider = OpenTopographyProvider()
        logging.info("Using OpenTopography for elevation data")
    except Exception as e:
        logging.warning(f"OpenTopography failed, falling back to Google: {e}")
        try:
            elevation_provider = GoogleElevationProvider()
            logging.info("Using Google Elevation API for elevation data")
        except Exception as e:
            logging.error(f"Both elevation providers failed: {e}")
            return
    
    # Create output directory
    output_dir = "coordinate_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each coordinate
    all_results = []
    
    for lat, lon, location_name in coordinates:
        try:
            logging.info(f"Processing: {location_name}")
            addresses = process_coordinate_location(lat, lon, location_name, elevation_provider)
            all_results.extend(addresses)
            
            # Save individual location results
            if addresses:
                location_filename = location_name.replace(" ", "_").replace(",", "").replace("—", "-")
                location_filename = "".join(c for c in location_filename if c.isalnum() or c in "._-")
                location_file = os.path.join(output_dir, f"{location_filename}_addresses.csv")
                
                df_location = pd.DataFrame(addresses)
                df_location.to_csv(location_file, index=False)
                logging.info(f"Saved {len(addresses)} addresses to {location_file}")
            
            # Delay between locations
            time.sleep(2)
            
        except Exception as e:
            logging.error(f"Error processing {location_name}: {e}")
            continue
    
    # Save combined results
    if all_results:
        combined_file = os.path.join(output_dir, "all_coordinate_locations_addresses.csv")
        df_combined = pd.DataFrame(all_results)
        df_combined.to_csv(combined_file, index=False)
        logging.info(f"Saved {len(all_results)} total addresses to {combined_file}")
        
        # Summary statistics
        logging.info("\n=== PROCESSING SUMMARY ===")
        logging.info(f"Total locations processed: {len(coordinates)}")
        logging.info(f"Total addresses found: {len(all_results)}")
        
        if len(all_results) > 0:
            df_summary = df_combined.groupby('location_name').agg({
                'address': 'count',
                'los_score': 'mean',
                'distance_to_railway': 'mean'
            }).round(3)
            logging.info(f"\nSummary by location:")
            logging.info(df_summary)
    else:
        logging.warning("No addresses found for any coordinates")

if __name__ == "__main__":
    main() 