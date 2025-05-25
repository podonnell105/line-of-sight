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
    print(f"Saving {len(gdf)} building footprints to {output_fn} ...")
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)
    gdf.to_file(output_fn, driver='GeoJSON')
    print(f"Saved building footprints to {output_fn}")
    return gdf

def main():
    # Parameters
    # Example Downtown Chicago coordinates - change these to your desired location
    center_lat, center_lon = 41.71281492427427, -87.62571402239901
    radius_km = 1 #Smaller radius than 1 will be faster to run


    min_lat, min_lon, max_lat, max_lon = bbox_from_point_radius(center_lat, center_lon, radius_km)
    bbox_geom = box(min_lon, min_lat, max_lon, max_lat)

    # Change this to the buffer distance you want to use for the rail lines
    rail_buffer_m = 100  # meters

    # 1. Fetch rail lines within bounding box using process_rail
    print("Fetching rail lines in bounding box using process_rail...")
    rail_gdf = fetch_rail_lines_in_bbox({'xmin': min_lon, 'ymin': min_lat, 'xmax': max_lon, 'ymax': max_lat})
    if rail_gdf is None or rail_gdf.empty:
        print("No rail lines found in the area.")
        return
    print(f"Fetched {len(rail_gdf)} rail line segments.")

    # 2. Load address points from addresses.geojson and filter to bounding box
    print("Loading and filtering address points from 'addresses.geojson'...")
    addr_path = os.path.join(os.path.dirname(__file__), '../data/addresses.geojson')
    gdf = gpd.read_file(addr_path)
    addr_gdf = gdf[(gdf.geometry.type == 'Point') & (gdf.geometry.within(bbox_geom))].copy()
    print(f"Filtered to {len(addr_gdf)} address points in the area.")
    if addr_gdf.empty:
        print("No addresses found in the area.")
        return

    # 3. Load or fetch building footprints and filter to bounding box
    print("Loading building footprints from 'output/building_footprints.geojson'...")
    buildings_path = os.path.join(os.path.dirname(__file__), '../output/building_footprints.geojson')
    if not os.path.exists(buildings_path):
        buildings_gdf = fetch_buildings_osm(min_lon, min_lat, max_lon, max_lat, buildings_path)
    else:
        buildings_gdf = gpd.read_file(buildings_path)
        buildings_gdf = buildings_gdf[buildings_gdf.geometry.intersects(bbox_geom)].copy()
        if buildings_gdf.empty:
            print("No building footprints found in file for this area. Fetching from OSM...")
            buildings_gdf = fetch_buildings_osm(min_lon, min_lat, max_lon, max_lat, buildings_path)
    print(f"Filtered to {len(buildings_gdf)} building footprints in the area.")

    # 4. Project all to metric CRS for distance/LOS calculations
    metric_crs = 3857
    addr_gdf = addr_gdf.to_crs(epsg=metric_crs)
    rail_gdf = rail_gdf.to_crs(epsg=metric_crs)
    buildings_gdf = buildings_gdf.to_crs(epsg=metric_crs)

    # 5. Find addresses within 100m of any rail line
    print(f"Finding addresses within {rail_buffer_m} meters of any rail line...")
    rail_union = rail_gdf.unary_union
    addr_gdf['distance_to_rail_m'] = addr_gdf.geometry.apply(lambda pt: pt.distance(rail_union))
    close_addr_gdf = addr_gdf[addr_gdf['distance_to_rail_m'] <= rail_buffer_m].copy()
    print(f"Found {len(close_addr_gdf)} addresses within {rail_buffer_m} meters of rail lines.")

    # 6. Compute line-of-sight (LOS) score for each address
    print("Computing line-of-sight (LOS) scores...")
    los_scores = []
    for idx, addr in close_addr_gdf.iterrows():
        # Find nearest point on rail
        nearest_rail_pt = nearest_points(addr.geometry, rail_union)[1]
        sightline = LineString([addr.geometry, nearest_rail_pt])
        # Check if sightline crosses any building (excluding the building containing the address, if any)
        blocked = False
        for _, bldg in buildings_gdf.iterrows():
            # If the address is inside this building, skip it
            if bldg.geometry.contains(addr.geometry):
                continue
            if bldg.geometry.crosses(sightline) or bldg.geometry.intersects(sightline):
                blocked = True
                break
        los_score = 0 if blocked else 1
        los_scores.append(los_score)
        # Console log for each address
        if idx % 10 == 0 or idx == close_addr_gdf.index[-1]:
            print(f"Address {idx}: ({addr.geometry.x:.6f}, {addr.geometry.y:.6f}) - LOS: {'Blocked' if blocked else 'Clear'}")
    close_addr_gdf['los_score'] = los_scores

    # 7. Save all address data with LOS score to CSV (in WGS84)
    close_addr_gdf = close_addr_gdf.to_crs(epsg=4326)
    out_csv = 'output/address_los_scores.csv'
    close_addr_gdf[['los_score', 'distance_to_rail_m', 'geometry'] + [col for col in close_addr_gdf.columns if col not in ['los_score', 'distance_to_rail_m', 'geometry']]].to_csv(out_csv, index=False)
    print(f"Saved address LOS scores to {out_csv}")

    # 8. Plot
    print("Plotting map...")
    fig, ax = plt.subplots(figsize=(12, 12))
    # Plot all rail lines
    rail_gdf = rail_gdf.to_crs(epsg=4326)
    rail_gdf.plot(ax=ax, color='black', linewidth=1, label='Rail Lines')
    # Plot all buildings
    buildings_gdf = buildings_gdf.to_crs(epsg=4326)
    if not buildings_gdf.empty:
        buildings_gdf.plot(ax=ax, color='gray', alpha=0.5, edgecolor='k', linewidth=0.2, label='Buildings')
    else:
        print('No building footprints to plot in this area.')
    # Plot addresses with LOS=1 (clear sight)
    close_addr_gdf[close_addr_gdf['los_score'] == 1].plot(ax=ax, color='green', markersize=30, alpha=0.8, label='LOS=1 (Clear)')
    # Plot addresses with LOS=0 (blocked)
    close_addr_gdf[close_addr_gdf['los_score'] == 0].plot(ax=ax, color='red', markersize=30, alpha=0.8, label='LOS=0 (Blocked)')
    # Set bounds
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    # Add basemap
    try:
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.Stamen.TerrainBackground)
    except Exception as e:
        print(f"Could not add basemap: {e}")
    ax.set_title('Address Line-of-Sight (LOS) to Rail Lines')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    plt.tight_layout()
    os.makedirs('data', exist_ok=True)
    plt.savefig('output/address_los_scores.png')
    print("Saved map to output/address_los_scores.png")
    plt.close(fig)

if __name__ == "__main__":
    main() 