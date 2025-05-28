from fetch_lidar import get_opentopography_lidar, process_lidar_data, analyze_vegetation
from process_rail import fetch_rail_lines_in_bbox
import sys
import os
import math
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
from shapely.geometry import box, LineString, shape
from shapely.ops import nearest_points
import requests
import json
import laspy
import numpy as np
from scipy.spatial import cKDTree
from rasterio import features
import rasterio
from rasterio.transform import from_origin
import tempfile

# Import fetch_rail_lines_in_bbox from process_rail
sys.path.append(os.path.dirname(__file__))


def bbox_from_point_radius(lat, lon, radius_km):
    delta_lat = radius_km / 111.32
    delta_lon = radius_km / (111.32 * math.cos(math.radians(lat)))
    min_lat = lat - delta_lat
    max_lat = lat + delta_lat
    min_lon = lon - delta_lon
    max_lon = lon + delta_lon
    return min_lat, min_lon, max_lat, max_lon


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
                    elevation_change = abs(
                        elevation[y_idx, x_idx] - elevation[prev_y, prev_x])
                    if elevation_change > 5:  # 5m threshold for significant elevation change
                        return 0  # Blocked by elevation change

        return 1  # Clear line of sight

    except Exception as e:
        print(f"Error calculating LOS score: {e}")
        return 0


def fetch_buildings_osm(minx, miny, maxx, maxy, output_fn):
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
    data = response.json()
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


def main():
    # Parameters
    center_lat, center_lon = 39.120514926269486, -94.59273376669232
    radius_km = 2
    rail_buffer_m = 40

    min_lat, min_lon, max_lat, max_lon = bbox_from_point_radius(
        center_lat, center_lon, radius_km)
    bbox_geom = box(min_lon, min_lat, max_lon, max_lat)

    # 1. Fetch rail lines within bounding box
    print("Fetching rail lines in bounding box...")
    rail_gdf = fetch_rail_lines_in_bbox(
        {'xmin': min_lon, 'ymin': min_lat, 'xmax': max_lon, 'ymax': max_lat})
    if rail_gdf is None or rail_gdf.empty:
        print("No rail lines found in the area.")
        return
    print(f"Fetched {len(rail_gdf)} rail line segments.")

    # 2. Load address points
    print("Loading and filtering address points...")
    addr_path = os.path.join(os.path.dirname(
        __file__), '../data/addresses.geojson')
    gdf = gpd.read_file(addr_path)
    addr_gdf = gdf[(gdf.geometry.type == 'Point') & (
        gdf.geometry.within(bbox_geom))].copy()
    print(f"Filtered to {len(addr_gdf)} address points in the area.")
    if addr_gdf.empty:
        print("No addresses found in the area.")
        return

    # 3. Load or fetch building footprints
    print("Loading building footprints...")
    buildings_path = os.path.join(os.path.dirname(
        __file__), '../output/building_footprints.geojson')
    if not os.path.exists(buildings_path):
        buildings_gdf = fetch_buildings_osm(
            min_lon, min_lat, max_lon, max_lat, buildings_path)
    else:
        buildings_gdf = gpd.read_file(buildings_path)
        buildings_gdf = buildings_gdf[buildings_gdf.geometry.intersects(
            bbox_geom)].copy()
        if buildings_gdf.empty:
            print(
                "No building footprints found in file for this area. Fetching from OSM...")
            buildings_gdf = fetch_buildings_osm(
                min_lon, min_lat, max_lon, max_lat, buildings_path)
    print(f"Filtered to {len(buildings_gdf)} building footprints in the area.")

    # 4. Get elevation data from OpenTopography
    print("Getting elevation data from OpenTopography...")
    bbox = [min_lon, min_lat, max_lon, max_lat]
    tif_file = get_opentopography_lidar(bbox)
    if not tif_file:
        print("Failed to get elevation data")
        return

    # 5. Process elevation data and analyze vegetation
    print("Processing elevation data and analyzing vegetation...")
    processed_file = process_lidar_data(tif_file)
    if not processed_file:
        print("Failed to process elevation data")
        return

    elevation, slope, variance, tree_mask, shrub_mask, building_mask, stats = analyze_vegetation(
        processed_file)
    if elevation is None:
        print("Failed to analyze vegetation")
        return

    # 6. Project all to metric CRS for distance/LOS calculations
    metric_crs = 3857
    addr_gdf = addr_gdf.to_crs(epsg=metric_crs)
    rail_gdf = rail_gdf.to_crs(epsg=metric_crs)
    buildings_gdf = buildings_gdf.to_crs(epsg=metric_crs)

    # 7. Find addresses within buffer of rail
    print(
        f"Finding addresses within {rail_buffer_m} meters of any rail line...")
    rail_union = rail_gdf.union_all()  # Updated from unary_union to union_all()
    addr_gdf['distance_to_rail_m'] = addr_gdf.geometry.apply(
        lambda pt: pt.distance(rail_union))
    close_addr_gdf = addr_gdf[addr_gdf['distance_to_rail_m']
                              <= rail_buffer_m].copy()
    print(
        f"Found {len(close_addr_gdf)} addresses within {rail_buffer_m} meters of rail lines.")

    # 8. Compute line-of-sight (LOS) score for each address
    print("Computing line-of-sight (LOS) scores...")
    los_scores = []
    for idx, addr in close_addr_gdf.iterrows():
        nearest_rail_pt = nearest_points(addr.geometry, rail_union)[1]

        # Convert points to WGS84 for elevation analysis
        addr_wgs84 = gpd.GeoSeries(
            [addr.geometry], crs=metric_crs).to_crs(epsg=4326)[0]
        rail_wgs84 = gpd.GeoSeries(
            [nearest_rail_pt], crs=metric_crs).to_crs(epsg=4326)[0]

        # Calculate LOS score
        score = calculate_los_score(bbox,
                                    (addr_wgs84.x, addr_wgs84.y),
                                    (rail_wgs84.x, rail_wgs84.y),
                                    elevation, tree_mask, shrub_mask, building_mask)
        los_scores.append(score)

        if idx % 10 == 0 or idx == close_addr_gdf.index[-1]:
            print(
                f"Address {idx}: ({addr.geometry.x:.6f}, {addr.geometry.y:.6f}) - LOS: {'Blocked' if score == 0 else 'Clear'}")

    close_addr_gdf['los_score'] = los_scores

    # 9. Save results
    close_addr_gdf = close_addr_gdf.to_crs(epsg=4326)

    # Save all results
    out_csv = 'output/address_los_scores_lidar.csv'
    close_addr_gdf[['los_score', 'distance_to_rail_m', 'geometry'] + [col for col in close_addr_gdf.columns if col not in [
        'los_score', 'distance_to_rail_m', 'geometry']]].to_csv(out_csv, index=False)
    print(f"Saved all address LOS scores to {out_csv}")

    # Save only addresses with clear LOS (score = 1)
    clear_los_gdf = close_addr_gdf[close_addr_gdf['los_score'] == 1].copy()
    clear_los_csv = 'output/clear_los_addresses.csv'
    clear_los_gdf[['los_score', 'distance_to_rail_m', 'geometry'] + [col for col in clear_los_gdf.columns if col not in [
        'los_score', 'distance_to_rail_m', 'geometry']]].to_csv(clear_los_csv, index=False)
    print(
        f"Saved {len(clear_los_gdf)} addresses with clear LOS to {clear_los_csv}")
    print(
        f"Clear LOS percentage: {len(clear_los_gdf)/len(close_addr_gdf)*100:.1f}%")

    # 10. Plot
    print("Plotting map...")
    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot rail lines
    rail_gdf = rail_gdf.to_crs(epsg=4326)
    rail_gdf.plot(ax=ax, color='yellow', linewidth=2,
                  label='Rail Lines', zorder=3)

    # Plot buildings
    buildings_gdf = buildings_gdf.to_crs(epsg=4326)
    if not buildings_gdf.empty:
        buildings_gdf.plot(ax=ax, color='gray', alpha=0.3,
                           edgecolor='k', linewidth=0.2, label='Buildings', zorder=2)

    # Plot addresses with LOS scores
    # Plot blocked points first (so they appear behind clear points)
    close_addr_gdf[close_addr_gdf['los_score'] == 0].plot(
        ax=ax,
        color='red',
        markersize=100,
        alpha=0.7,
        label='LOS=0 (Blocked)',
        zorder=4
    )

    # Plot clear points on top
    close_addr_gdf[close_addr_gdf['los_score'] == 1].plot(
        ax=ax,
        color='lime',
        markersize=100,
        alpha=0.7,
        label='LOS=1 (Clear)',
        zorder=5
    )

    # Set bounds with a small buffer
    buffer = 0.002  # approximately 200 meters
    ax.set_xlim(min_lon - buffer, max_lon + buffer)
    ax.set_ylim(min_lat - buffer, max_lat + buffer)

    # Add satellite basemap
    try:
        ctx.add_basemap(
            ax,
            crs='EPSG:4326',
            source=ctx.providers.Esri.WorldImagery,  # Satellite imagery
            attribution=False,
            attribution_size=8
        )
    except Exception as e:
        print(f"Could not add satellite basemap: {e}")
        try:
            # Fallback to terrain basemap
            ctx.add_basemap(
                ax,
                crs='EPSG:4326',
                source=ctx.providers.Stamen.Terrain,
                attribution=False,
                attribution_size=8
            )
        except Exception as e:
            print(f"Could not add terrain basemap: {e}")

    # Add title and labels
    ax.set_title('Address Line-of-Sight (LOS) to Rail Lines\nwith Satellite Imagery',
                 fontsize=16, pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    # Add legend with larger font and better positioning
    legend = ax.legend(
        loc='upper right',
        fontsize=12,
        framealpha=0.8,
        edgecolor='black'
    )

    # Add statistics text box
    stats_text = f"""
    Total Addresses: {len(close_addr_gdf)}
    Clear LOS (Score=1): {len(close_addr_gdf[close_addr_gdf['los_score'] == 1])}
    Blocked LOS (Score=0): {len(close_addr_gdf[close_addr_gdf['los_score'] == 0])}
    Clear Percentage: {len(close_addr_gdf[close_addr_gdf['los_score'] == 1])/len(close_addr_gdf)*100:.1f}%
    """

    # Add text box with statistics
    props = dict(boxstyle='round', facecolor='white',
                 alpha=0.8, edgecolor='black')
    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        family='monospace',
        verticalalignment='bottom',
        bbox=props
    )

    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/address_los_scores_lidar.png',
                dpi=300, bbox_inches='tight')
    print("Saved map to output/address_los_scores_lidar.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
