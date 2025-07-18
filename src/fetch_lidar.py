import os
import requests
import geopandas as gpd
from shapely.geometry import box
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
import rasterio
from rasterio.plot import show
import contextily as ctx
from scipy.spatial import cKDTree
from dotenv import load_dotenv
import time
import json
from rasterio.transform import from_origin
from elevation_providers import GoogleElevationProvider

# Load environment variables
load_dotenv(override=True)

def get_google_elevation(bbox):
    """
    Query Google Elevation API for high-resolution elevation data
    """
    try:
        # Get API key from environment
        api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables")

        # Create a grid of points within the bounding box
        min_lon, min_lat, max_lon, max_lat = bbox
        grid_size = 100  # Number of points in each direction
        lons = np.linspace(min_lon, max_lon, grid_size)
        lats = np.linspace(min_lat, max_lat, grid_size)
        
        # Create points for API request
        points = []
        for lat in lats:
            for lon in lons:
                points.append(f"{lat},{lon}")
        
        # Split points into chunks to avoid URL length limits
        chunk_size = 100
        elevation_data = []
        
        for i in range(0, len(points), chunk_size):
            chunk = points[i:i + chunk_size]
            locations = '|'.join(chunk)
            
            # Make request to Google Elevation API
            url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={locations}&key={api_key}"
            response = requests.get(url)
            
            if response.status_code != 200:
                raise Exception(f"Google Elevation API error: {response.status_code} {response.text}")
            
            data = response.json()
            if data['status'] != 'OK':
                raise Exception(f"Google Elevation API error: {data['status']}")
            
            # Extract elevation values
            chunk_elevations = [result['elevation'] for result in data['results']]
            elevation_data.extend(chunk_elevations)
            
            # Respect rate limits
            time.sleep(0.1)  # 100ms delay between requests
        
        # Reshape elevation data into a grid
        elevation_grid = np.array(elevation_data).reshape(grid_size, grid_size)
        
        # Create a temporary file for the elevation data
        temp_file = 'temp_elevation.tif'
        
        # Calculate the transform
        x_res = (max_lon - min_lon) / grid_size
        y_res = (max_lat - min_lat) / grid_size
        transform = from_origin(min_lon, max_lat, x_res, y_res)
        
        # Save as GeoTIFF
        with rasterio.open(
            temp_file,
            'w',
            driver='GTiff',
            height=grid_size,
            width=grid_size,
            count=1,
            dtype=elevation_grid.dtype,
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(elevation_grid, 1)
        
        return temp_file

    except Exception as e:
        print(f"Error getting elevation data: {e}")
        return None

def get_lidar_data(bbox, provider=None):
    """
    Get elevation data for a bounding box using the specified provider.
    Args:
        bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax)
        provider (ElevationProvider): Optional. If None, use GoogleElevationProvider.
    Returns:
        str: Path to the saved GeoTIFF file
    """
    if provider is None:
        provider = GoogleElevationProvider()
    return provider.get_elevation_tif(bbox)

def process_and_analyze_lidar_data(tif_file, output_dir='data', tile_size=250):
    """
    Process downloaded elevation data and analyze vegetation in a single pass
    Using memory-efficient processing with small tiles and float32
    Returns paths to saved files and statistics
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the GeoTIFF file
        with rasterio.open(tif_file) as src:
            # Get the dimensions
            height = src.height
            width = src.width
            
            # Calculate number of tiles
            n_tiles_h = (height + tile_size - 1) // tile_size
            n_tiles_w = (width + tile_size - 1) // tile_size
            
            # Initialize output files
            elevation_file = os.path.join(output_dir, 'elevation_data.tif')
            slope_file = os.path.join(output_dir, 'slope_data.tif')
            variance_file = os.path.join(output_dir, 'variance_data.tif')
            tree_mask_file = os.path.join(output_dir, 'tree_mask.tif')
            shrub_mask_file = os.path.join(output_dir, 'shrub_mask.tif')
            building_mask_file = os.path.join(output_dir, 'building_mask.tif')
            
            # Create output files
            with rasterio.open(elevation_file, 'w',
                             driver='GTiff',
                             height=height,
                             width=width,
                             count=1,
                             dtype=np.float32,
                             crs=src.crs,
                             transform=src.transform) as dst_elev, \
                 rasterio.open(slope_file, 'w',
                             driver='GTiff',
                             height=height,
                             width=width,
                             count=1,
                             dtype=np.float32,
                             crs=src.crs,
                             transform=src.transform) as dst_slope, \
                 rasterio.open(variance_file, 'w',
                             driver='GTiff',
                             height=height,
                             width=width,
                             count=1,
                             dtype=np.float32,
                             crs=src.crs,
                             transform=src.transform) as dst_var, \
                 rasterio.open(tree_mask_file, 'w',
                             driver='GTiff',
                             height=height,
                             width=width,
                             count=1,
                             dtype=np.uint8,
                             crs=src.crs,
                             transform=src.transform) as dst_tree, \
                 rasterio.open(shrub_mask_file, 'w',
                             driver='GTiff',
                             height=height,
                             width=width,
                             count=1,
                             dtype=np.uint8,
                             crs=src.crs,
                             transform=src.transform) as dst_shrub, \
                 rasterio.open(building_mask_file, 'w',
                             driver='GTiff',
                             height=height,
                             width=width,
                             count=1,
                             dtype=np.uint8,
                             crs=src.crs,
                             transform=src.transform) as dst_building:
                
                # Initialize statistics
                stats = {
                    'max_elevation': float('-inf'),
                    'min_elevation': float('inf'),
                    'elevation_sum': 0,
                    'elevation_count': 0,
                    'max_slope': float('-inf'),
                    'slope_sum': 0,
                    'slope_count': 0
                }
                
                # Initialize arrays for the full dataset
                elevation = np.zeros((height, width), dtype=np.float32)
                slope = np.zeros((height, width), dtype=np.float32)
                variance = np.zeros((height, width), dtype=np.float32)
                tree_mask = np.zeros((height, width), dtype=np.uint8)
                shrub_mask = np.zeros((height, width), dtype=np.uint8)
                building_mask = np.zeros((height, width), dtype=np.uint8)
                
                # Process in tiles
                for i in range(n_tiles_h):
                    for j in range(n_tiles_w):
                        # Calculate window
                        window = rasterio.windows.Window(
                            j * tile_size,
                            i * tile_size,
                            min(tile_size, width - j * tile_size),
                            min(tile_size, height - i * tile_size)
                        )
                        
                        # Read the tile
                        tile_elevation = src.read(1, window=window).astype(np.float32)
                        
                        # Calculate slope and aspect for the tile using vectorized operations
                        dy, dx = np.gradient(tile_elevation)
                        tile_slope_magnitude = np.sqrt(dx**2 + dy**2)
                        
                        # Calculate local variance using a rolling window approach
                        window_size = 3
                        kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size * window_size)
                        mean = np.convolve(tile_elevation.ravel(), kernel.ravel(), mode='same').reshape(tile_elevation.shape)
                        mean_sq = np.convolve((tile_elevation**2).ravel(), kernel.ravel(), mode='same').reshape(tile_elevation.shape)
                        tile_local_variance = mean_sq - mean**2
                        
                        # Calculate thresholds once per tile
                        var_75 = np.percentile(tile_local_variance, 75)
                        var_50 = np.percentile(tile_local_variance, 50)
                        var_90 = np.percentile(tile_local_variance, 90)
                        slope_50 = np.percentile(tile_slope_magnitude, 50)
                        slope_25 = np.percentile(tile_slope_magnitude, 25)
                        slope_90 = np.percentile(tile_slope_magnitude, 90)
                        
                        # Identify features in the tile using vectorized operations
                        tile_tree_mask = (tile_local_variance > var_75) & \
                                       (tile_slope_magnitude > slope_50)
                        
                        tile_shrub_mask = (tile_local_variance > var_50) & \
                                        (tile_local_variance <= var_75) & \
                                        (tile_slope_magnitude > slope_25)
                        
                        tile_building_mask = (tile_local_variance > var_90) & \
                                           (tile_slope_magnitude > slope_90)
                        
                        # Update statistics
                        stats['max_elevation'] = max(stats['max_elevation'], np.max(tile_elevation))
                        stats['min_elevation'] = min(stats['min_elevation'], np.min(tile_elevation))
                        stats['elevation_sum'] += np.sum(tile_elevation)
                        stats['elevation_count'] += tile_elevation.size
                        stats['max_slope'] = max(stats['max_slope'], np.max(tile_slope_magnitude))
                        stats['slope_sum'] += np.sum(tile_slope_magnitude)
                        stats['slope_count'] += tile_slope_magnitude.size
                        
                        # Write all data to their respective files
                        dst_elev.write(tile_elevation, 1, window=window)
                        dst_slope.write(tile_slope_magnitude, 1, window=window)
                        dst_var.write(tile_local_variance, 1, window=window)
                        dst_tree.write(tile_tree_mask.astype(np.uint8), 1, window=window)
                        dst_shrub.write(tile_shrub_mask.astype(np.uint8), 1, window=window)
                        dst_building.write(tile_building_mask.astype(np.uint8), 1, window=window)
                        
                        # Update the full arrays
                        y_start = i * tile_size
                        y_end = min(y_start + tile_size, height)
                        x_start = j * tile_size
                        x_end = min(x_start + tile_size, width)
                        
                        elevation[y_start:y_end, x_start:x_end] = tile_elevation
                        slope[y_start:y_end, x_start:x_end] = tile_slope_magnitude
                        variance[y_start:y_end, x_start:x_end] = tile_local_variance
                        tree_mask[y_start:y_end, x_start:x_end] = tile_tree_mask
                        shrub_mask[y_start:y_end, x_start:x_end] = tile_shrub_mask
                        building_mask[y_start:y_end, x_start:x_end] = tile_building_mask
                        
                        # Clean up temporary arrays
                        del tile_elevation, dy, dx, tile_slope_magnitude
                        del mean, mean_sq, tile_local_variance
                        del tile_tree_mask, tile_shrub_mask, tile_building_mask
                    
                    print(f"Processed tile row {i+1}/{n_tiles_h}")
            
            # Calculate final statistics
            stats['mean_elevation'] = stats['elevation_sum'] / stats['elevation_count']
            stats['mean_slope'] = stats['slope_sum'] / stats['slope_count']
            
            # Clean up the temporary file
            if os.path.exists(tif_file):
                os.unlink(tif_file)
            
            # Return all expected values
            return (
                elevation_file,  # elevation file path
                elevation,      # elevation array
                slope,         # slope array
                variance,      # variance array
                tree_mask,     # tree mask array
                shrub_mask,    # shrub mask array
                building_mask, # building mask array
                stats         # statistics dictionary
            )
            
    except Exception as e:
        print(f"Error processing and analyzing elevation data: {e}")
        if os.path.exists(tif_file):
            os.unlink(tif_file)
        return None, None, None, None, None, None, None, None

def load_processed_data(file_paths):
    """
    Load processed data from saved files
    """
    try:
        with rasterio.open(file_paths['elevation']) as src:
            elevation = src.read(1)
        with rasterio.open(file_paths['slope']) as src:
            slope = src.read(1)
        with rasterio.open(file_paths['variance']) as src:
            variance = src.read(1)
        with rasterio.open(file_paths['tree_mask']) as src:
            tree_mask = src.read(1).astype(bool)
        with rasterio.open(file_paths['shrub_mask']) as src:
            shrub_mask = src.read(1).astype(bool)
        with rasterio.open(file_paths['building_mask']) as src:
            building_mask = src.read(1).astype(bool)
            
        return elevation, slope, variance, tree_mask, shrub_mask, building_mask
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None, None, None, None, None, None

def visualize_vegetation(bbox, tif_file):
    """
    Create a visualization of the vegetation and obstruction analysis
    """
    try:
        # Process and analyze the data
        file_paths = process_and_analyze_lidar_data(tif_file)
        if not file_paths:
            return
            
        # Load the processed data
        elevation, slope, variance, tree_mask, shrub_mask, building_mask = load_processed_data(file_paths)
        if elevation is None:
            return
            
        # Calculate statistics
        stats = {
            'total_points': elevation.size,
            'tree_points': np.sum(tree_mask),
            'shrub_points': np.sum(shrub_mask),
            'building_points': np.sum(building_mask),
            'max_elevation': np.max(elevation),
            'min_elevation': np.min(elevation),
            'mean_elevation': np.mean(elevation),
            'max_slope': np.max(slope),
            'mean_slope': np.mean(slope)
        }
            
        # Create figure with four subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
        
        # Plot 1: Elevation
        im1 = ax1.imshow(elevation, cmap='terrain')
        plt.colorbar(im1, ax=ax1, label='Elevation (m)')
        ax1.set_title('Elevation Profile')
        
        # Plot 2: Slope
        im2 = ax2.imshow(slope, cmap='viridis')
        plt.colorbar(im2, ax=ax2, label='Slope (degrees)')
        ax2.set_title('Slope Analysis')
        
        # Plot 3: Local Variance
        im3 = ax3.imshow(variance, cmap='plasma')
        plt.colorbar(im3, ax=ax3, label='Local Variance')
        ax3.set_title('Local Variance (Potential Vegetation)')
        
        # Plot 4: Classification
        classification = np.zeros_like(elevation)
        classification[tree_mask] = 1
        classification[shrub_mask] = 2
        classification[building_mask] = 3
        
        im4 = ax4.imshow(classification, cmap='tab10', vmin=0, vmax=3)
        plt.colorbar(im4, ax=ax4, label='Classification')
        ax4.set_title('Vegetation and Obstruction Classification')
        
        # Add statistics text
        stats_text = f"""
        Total Points: {stats['total_points']:,}
        Tree Points: {stats['tree_points']:,} ({stats['tree_points']/stats['total_points']*100:.1f}%)
        Shrub Points: {stats['shrub_points']:,} ({stats['shrub_points']/stats['total_points']*100:.1f}%)
        Building Points: {stats['building_points']:,} ({stats['building_points']/stats['total_points']*100:.1f}%)
        Elevation Range: {stats['min_elevation']:.1f}m - {stats['max_elevation']:.1f}m
        Mean Elevation: {stats['mean_elevation']:.1f}m
        Mean Slope: {stats['mean_slope']:.1f}°
        """
        fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace')
        
        plt.tight_layout()
        
        # Save the visualization
        output_file = 'output/vegetation_analysis.png'
        os.makedirs('output', exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
        plt.close(fig)
        
        # Clean up loaded data
        del elevation, slope, variance, tree_mask, shrub_mask, building_mask
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

def calculate_los_score(bbox, address_point, rail_point, elevation, tree_mask, shrub_mask, building_mask):
    """
    Calculate line-of-sight score between an address and rail point
    """
    try:
        # Convert points to raster coordinates
        x_min, y_min, x_max, y_max = bbox
        x_res = (x_max - x_min) / elevation.shape[1]
        y_res = (y_max - y_min) / elevation.shape[0]
        
        # Convert address and rail points to raster coordinates
        addr_x = int((address_point[0] - x_min) / x_res)
        addr_y = int((address_point[1] - y_min) / y_res)
        rail_x = int((rail_point[0] - x_min) / x_res)
        rail_y = int((rail_point[1] - y_min) / y_res)
        
        # Create line of sight points
        num_points = 100
        x = np.linspace(addr_x, rail_x, num_points)
        y = np.linspace(addr_y, rail_y, num_points)
        
        # Check each point along the line of sight
        for i in range(num_points):
            x_idx = int(x[i])
            y_idx = int(y[i])
            
            # Skip if out of bounds
            if not (0 <= x_idx < elevation.shape[1] and 0 <= y_idx < elevation.shape[0]):
                continue
                
            # Check for buildings
            if building_mask[y_idx, x_idx]:
                return 0  # Blocked by building
                
            # Check for trees
            if tree_mask[y_idx, x_idx]:
                return 0  # Blocked by tree
                
            # Check for shrubs (if they're tall enough)
            if shrub_mask[y_idx, x_idx] and elevation[y_idx, x_idx] > elevation[addr_y, addr_x] + 2:  # 2m threshold
                return 0  # Blocked by tall shrub
                
            # Check for elevation changes
            if i > 0:
                prev_x = int(x[i-1])
                prev_y = int(y[i-1])
                if (0 <= prev_x < elevation.shape[1] and 0 <= prev_y < elevation.shape[0]):
                    elevation_change = abs(elevation[y_idx, x_idx] - elevation[prev_y, prev_x])
                    if elevation_change > 5:  # 5m threshold for significant elevation change
                        return 0  # Blocked by elevation change
        
        return 1  # Clear line of sight
        
    except Exception as e:
        print(f"Error calculating LOS score: {e}")
        return 0

def analyze_los(bbox, file_paths, addresses, rail_points):
    """
    Analyze line of sight for multiple addresses
    """
    try:
        # Load the processed data
        elevation, slope, variance, tree_mask, shrub_mask, building_mask = load_processed_data(file_paths)
        if elevation is None:
            return None
            
        los_scores = []
        for addr_point, rail_point in zip(addresses, rail_points):
            score = calculate_los_score(bbox, addr_point, rail_point, 
                                      elevation, tree_mask, shrub_mask, building_mask)
            los_scores.append(score)
            
        # Clean up loaded data
        del elevation, slope, variance, tree_mask, shrub_mask, building_mask
            
        return los_scores
        
    except Exception as e:
        print(f"Error analyzing LOS: {e}")
        return None

def visualize_los(bbox, file_paths, addresses, rail_points, los_scores):
    """
    Create a visualization of the line-of-sight analysis
    """
    try:
        # Load the processed data
        elevation, slope, variance, tree_mask, shrub_mask, building_mask = load_processed_data(file_paths)
        if elevation is None:
            return
            
        # Create figure with four subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
        
        # Plot 1: Elevation with addresses and rail points
        im1 = ax1.imshow(elevation, cmap='terrain')
        plt.colorbar(im1, ax=ax1, label='Elevation (m)')
        
        # Plot addresses and rail points
        for addr, rail, score in zip(addresses, rail_points, los_scores):
            x_min, y_min, x_max, y_max = bbox
            x_res = (x_max - x_min) / elevation.shape[1]
            y_res = (y_max - y_min) / elevation.shape[0]
            
            addr_x = (addr[0] - x_min) / x_res
            addr_y = (addr[1] - y_min) / y_res
            rail_x = (rail[0] - x_min) / x_res
            rail_y = (rail[1] - y_min) / y_res
            
            color = 'green' if score == 1 else 'red'
            ax1.plot([addr_x, rail_x], [addr_y, rail_y], color=color, alpha=0.5)
            ax1.scatter(addr_x, addr_y, color=color, s=50)
            ax1.scatter(rail_x, rail_y, color='blue', s=50)
        
        ax1.set_title('Elevation with Line-of-Sight Analysis')
        
        # Plot 2: Vegetation Classification
        classification = np.zeros_like(elevation)
        classification[tree_mask] = 1
        classification[shrub_mask] = 2
        classification[building_mask] = 3
        
        im2 = ax2.imshow(classification, cmap='tab10', vmin=0, vmax=3)
        plt.colorbar(im2, ax=ax2, label='Classification')
        ax2.set_title('Vegetation and Obstruction Classification')
        
        # Plot 3: Slope
        im3 = ax3.imshow(slope, cmap='viridis')
        plt.colorbar(im3, ax=ax3, label='Slope (degrees)')
        ax3.set_title('Slope Analysis')
        
        # Plot 4: LOS Score Distribution
        ax4.hist(los_scores, bins=[0, 1], color=['red', 'green'])
        ax4.set_title('Line-of-Sight Score Distribution')
        ax4.set_xlabel('LOS Score (0=Blocked, 1=Clear)')
        ax4.set_ylabel('Count')
        
        # Add statistics text
        stats_text = f"""
        Total Addresses: {len(addresses)}
        Clear LOS (Score=1): {sum(los_scores)}
        Blocked LOS (Score=0): {len(los_scores) - sum(los_scores)}
        Clear Percentage: {sum(los_scores)/len(los_scores)*100:.1f}%
        """
        fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace')
        
        plt.tight_layout()
        
        # Save the visualization
        output_file = 'output/los_analysis.png'
        os.makedirs('output', exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
        plt.close(fig)
        
        # Clean up loaded data
        del elevation, slope, variance, tree_mask, shrub_mask, building_mask
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Test coordinates for Chicago area
    bbox = [-87.650, 41.870, -87.620, 41.890]  # Downtown Chicago area
    
    # Get elevation data
    tif_file = get_lidar_data(bbox)
    if tif_file:
        # Process the downloaded data
        file_paths = process_and_analyze_lidar_data(tif_file)
        if file_paths:
            print(f"Successfully processed and analyzed elevation data")
            
            # Example addresses and rail points (you would get these from your address data)
            # These are just example points - replace with your actual data
            addresses = [
                (bbox[0] + 0.01, bbox[1] + 0.01),  # Example address 1
                (bbox[0] + 0.02, bbox[1] + 0.02),  # Example address 2
                (bbox[0] + 0.03, bbox[1] + 0.03),  # Example address 3
            ]
            
            rail_points = [
                (bbox[2] - 0.01, bbox[3] - 0.01),  # Example rail point 1
                (bbox[2] - 0.02, bbox[3] - 0.02),  # Example rail point 2
                (bbox[2] - 0.03, bbox[3] - 0.03),  # Example rail point 3
            ]
            
            # Calculate LOS scores
            los_scores = analyze_los(bbox, file_paths, addresses, rail_points)
            
            # Create visualization
            visualize_los(bbox, file_paths, addresses, rail_points, los_scores)
        else:
            print("Failed to process and analyze elevation data")
    else:
        print("Failed to download elevation data")

if __name__ == "__main__":
    main()
