import sys
import os
import math
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
from shapely.geometry import box

# Import fetch_rail_lines_in_bbox from process_rail
sys.path.append(os.path.dirname(__file__))
from process_rail import fetch_rail_lines_in_bbox

def bbox_from_point_radius(lat, lon, radius_km):
    delta_lat = radius_km / 111.32
    delta_lon = radius_km / (111.32 * math.cos(math.radians(lat)))
    min_lat = lat - delta_lat
    max_lat = lat + delta_lat
    min_lon = lon - delta_lon
    max_lon = lon + delta_lon
    return min_lat, min_lon, max_lat, max_lon

def main():
    # Parameters
    center_lat, center_lon = 41.87825, -87.62975  # Downtown Chicago coordinates
    radius_km = 10
    min_lat, min_lon, max_lat, max_lon = bbox_from_point_radius(center_lat, center_lon, radius_km)
    bbox_geom = box(min_lon, min_lat, max_lon, max_lat)
    rail_buffer_m = 100  # meters

    # 1. Fetch rail lines within bounding box using process_rail
    print("Fetching rail lines in bounding box using process_rail...")
    rail_gdf = fetch_rail_lines_in_bbox({'xmin': min_lon, 'ymin': min_lat, 'xmax': max_lon, 'ymax': max_lat})
    if rail_gdf is None or rail_gdf.empty:
        print("No rail lines found in the area.")
        return
    print(f"Fetched {len(rail_gdf)} rail line segments.")

    # 2. Load address points from source.geojson and filter to bounding box
    print("Loading and filtering address points from 'addresses.geojson'...")
    rail_path = os.path.join(os.path.dirname(__file__), '../data/addresses.geojson')
    gdf = gpd.read_file(rail_path)
    addr_gdf = gdf[(gdf.geometry.type == 'Point') & (gdf.geometry.within(bbox_geom))].copy()
    print(f"Filtered to {len(addr_gdf)} address points in the area.")
    if addr_gdf.empty:
        print("No addresses found in the area.")
        return

    # 3. Find addresses within 100m of any rail line
    print(f"Finding addresses within {rail_buffer_m} meters of any rail line...")
    # Project to a metric CRS for accurate distance calculation
    addr_gdf = addr_gdf.to_crs(epsg=3857)
    rail_gdf = rail_gdf.to_crs(epsg=3857)
    # Create a single MultiLineString for all rails
    rail_union = rail_gdf.unary_union
    addr_gdf['distance_to_rail_m'] = addr_gdf.geometry.apply(lambda pt: pt.distance(rail_union))
    close_addr_gdf = addr_gdf[addr_gdf['distance_to_rail_m'] <= rail_buffer_m].copy()
    print(f"Found {len(close_addr_gdf)} addresses within {rail_buffer_m} meters of rail lines.")

    # 4. Save these addresses to CSV (in WGS84)
    close_addr_gdf = close_addr_gdf.to_crs(epsg=4326)
    out_csv = 'data/addresses_near_rail.csv'
    close_addr_gdf[['geometry'] + [col for col in close_addr_gdf.columns if col != 'geometry']].to_csv(out_csv, index=False)
    print(f"Saved addresses within {rail_buffer_m}m of rail to {out_csv}")

    # 5. Plot
    print("Plotting map...")
    fig, ax = plt.subplots(figsize=(12, 12))
    # Plot all rail lines
    rail_gdf = rail_gdf.to_crs(epsg=4326)
    rail_gdf.plot(ax=ax, color='black', linewidth=1, label='Rail Lines')
    # Plot all addresses
    addr_gdf = addr_gdf.to_crs(epsg=4326)
    addr_gdf.plot(ax=ax, color='gray', markersize=5, alpha=0.3, label='All Addresses')
    # Plot close addresses
    close_addr_gdf.plot(ax=ax, color='red', markersize=20, alpha=0.8, label=f'Addresses within {rail_buffer_m}m')
    # Set bounds
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    # Add basemap
    try:
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.Stamen.TerrainBackground)
    except Exception as e:
        print(f"Could not add basemap: {e}")
    ax.set_title(f'Addresses within {rail_buffer_m}m of Rail Lines')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    plt.tight_layout()
    os.makedirs('data', exist_ok=True)
    plt.savefig('data/addresses_near_rail_map.png')
    print("Saved map to data/addresses_near_rail_map.png")
    plt.close(fig)

if __name__ == "__main__":
    main() 