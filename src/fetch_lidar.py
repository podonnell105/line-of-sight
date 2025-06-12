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

def get_opentopography_lidar(bbox):
    """
    Query OpenTopography API for available LiDAR point cloud data in the given bounding box
    """
    # OpenTopography API endpoint for USGS 3DEP data
    base_url = "https://portal.opentopography.org/API/usgsdem"
    
    # Your API key
    api_key = os.getenv('OPENTOPOGRAPHY_API_KEY')
    if not api_key:
        print("WARNING: OPENTOPOGRAPHY_API_KEY environment variable is not set")
        return None
    print(f"Using API key: {api_key[:4]}...{api_key[-4:]}")  # Only show first/last 4 chars for security
    
    # Convert bbox to the format expected by the API
    params = {
        'datasetName': 'USGS10m',  # Using 10m resolution data
        'south': bbox[1],
        'north': bbox[3],
        'west': bbox[0],
        'east': bbox[2],
        'outputFormat': 'GTiff',
        'API_Key': api_key
    }
    
    print("Querying OpenTopography API for available LiDAR data...")
    try:
        # Get the data - no need for JSON headers since we're downloading a GeoTIFF
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
            tmp_path = tmp.name
            
        print(f"Downloaded data to {tmp_path}")
        return tmp_path
        
    except requests.exceptions.RequestException as e:
        print(f"Error: API request failed: {e}")
        if hasattr(e.response, 'text'):
            print("Response text:", e.response.text)
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def process_lidar_data(tif_file, output_dir='data'):
    """
    Process downloaded elevation data and extract information
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the GeoTIFF file
        with rasterio.open(tif_file) as src:
            # Read the elevation data
            elevation = src.read(1)
            
            # Get the transform
            transform = src.transform
            
            # Create output file
            output_file = os.path.join(output_dir, 'elevation_data.tif')
            
            # Save the processed data
            with rasterio.open(output_file, 'w', 
                             driver='GTiff',
                             height=elevation.shape[0],
                             width=elevation.shape[1],
                             count=1,
                             dtype=elevation.dtype,
                             crs=src.crs,
                             transform=transform) as dst:
                dst.write(elevation, 1)
                
        print(f"Processed elevation data to {output_file}")
        
        # Clean up the temporary file
        os.unlink(tif_file)
        
        return output_file
        
    except Exception as e:
        print(f"Error processing elevation data: {e}")
        if os.path.exists(tif_file):
            os.unlink(tif_file)
        return None

def analyze_vegetation(tif_file):
    """
    Analyze elevation data to identify trees, shrubs, and other obstructions
    """
    try:
        # Read the GeoTIFF file
        with rasterio.open(tif_file) as src:
            elevation = src.read(1)
            
            # Calculate slope and aspect
            slope = np.gradient(elevation)
            slope_magnitude = np.sqrt(slope[0]**2 + slope[1]**2)
            aspect = np.arctan2(slope[1], slope[0])
            
            # Calculate local variance to detect sudden elevation changes (like trees)
            window_size = 3
            local_variance = np.zeros_like(elevation)
            for i in range(window_size, elevation.shape[0] - window_size):
                for j in range(window_size, elevation.shape[1] - window_size):
                    window = elevation[i-window_size:i+window_size+1, 
                                    j-window_size:j+window_size+1]
                    local_variance[i,j] = np.var(window)
            
            # Identify potential vegetation and obstructions
            # Trees typically have high local variance and moderate to high slope
            tree_mask = (local_variance > np.percentile(local_variance, 75)) & \
                       (slope_magnitude > np.percentile(slope_magnitude, 50))
            
            # Shrubs typically have moderate local variance and moderate slope
            shrub_mask = (local_variance > np.percentile(local_variance, 50)) & \
                        (local_variance <= np.percentile(local_variance, 75)) & \
                        (slope_magnitude > np.percentile(slope_magnitude, 25))
            
            # Buildings typically have high local variance and very high slope
            building_mask = (local_variance > np.percentile(local_variance, 90)) & \
                          (slope_magnitude > np.percentile(slope_magnitude, 90))
            
            # Calculate statistics
            stats = {
                'total_points': elevation.size,
                'tree_points': np.sum(tree_mask),
                'shrub_points': np.sum(shrub_mask),
                'building_points': np.sum(building_mask),
                'max_elevation': np.max(elevation),
                'min_elevation': np.min(elevation),
                'mean_elevation': np.mean(elevation),
                'max_slope': np.max(slope_magnitude),
                'mean_slope': np.mean(slope_magnitude)
            }
            
            return elevation, slope_magnitude, local_variance, tree_mask, shrub_mask, building_mask, stats
            
    except Exception as e:
        print(f"Error analyzing vegetation: {e}")
        return None, None, None, None, None, None, None

def visualize_vegetation(bbox, tif_file):
    """
    Create a visualization of the vegetation and obstruction analysis
    """
    try:
        # Analyze the data
        elevation, slope, variance, tree_mask, shrub_mask, building_mask, stats = analyze_vegetation(tif_file)
        if elevation is None:
            return
            
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

def analyze_los(bbox, elevation, tree_mask, shrub_mask, building_mask, addresses, rail_points):
    """
    Analyze line of sight for multiple addresses
    """
    try:
        los_scores = []
        for addr_point, rail_point in zip(addresses, rail_points):
            score = calculate_los_score(bbox, addr_point, rail_point, 
                                      elevation, tree_mask, shrub_mask, building_mask)
            los_scores.append(score)
            
        return los_scores
        
    except Exception as e:
        print(f"Error analyzing LOS: {e}")
        return None

def visualize_los(bbox, elevation, tree_mask, shrub_mask, building_mask, addresses, rail_points, los_scores):
    """
    Create a visualization of the line-of-sight analysis
    """
    try:
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
        slope = np.gradient(elevation)
        slope_magnitude = np.sqrt(slope[0]**2 + slope[1]**2)
        im3 = ax3.imshow(slope_magnitude, cmap='viridis')
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
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Test coordinates for Chicago area
    bbox = [-87.650, 41.870, -87.620, 41.890]  # Downtown Chicago area
    
    # Get elevation data
    tif_file = get_opentopography_lidar(bbox)
    if tif_file:
        # Process the downloaded data
        processed_file = process_lidar_data(tif_file)
        if processed_file:
            print(f"Successfully processed elevation data to {processed_file}")
            
            # Analyze vegetation and obstructions
            elevation, slope, variance, tree_mask, shrub_mask, building_mask, stats = analyze_vegetation(processed_file)
            
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
            los_scores = analyze_los(bbox, elevation, tree_mask, shrub_mask, building_mask, 
                                   addresses, rail_points)
            
            # Create visualization
            visualize_los(bbox, elevation, tree_mask, shrub_mask, building_mask,
                         addresses, rail_points, los_scores)
        else:
            print("Failed to process elevation data")
    else:
        print("Failed to download elevation data")

if __name__ == "__main__":
    main()
