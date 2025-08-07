#!/usr/bin/env python3
"""
Diagnostic script to check what buildings and addresses exist around railway yard coordinates.
This will help us understand why no addresses are being found.
"""

import requests
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Test coordinates
TEST_COORDS = [
    (37.9602389, -87.6067297, "CSX Howell Yard, Evansville"),
    (40.0832513, -85.688746, "CSX South Anderson Yard, Anderson"),
    (41.0644145, -85.1058342, "NS Piqua Yard, Fort Wayne")
]

def test_overpass_queries(lat, lon, location_name, radius_km=2.0):
    """Test different Overpass queries to see what's available in the area."""
    
    # Create a larger bounding box around the coordinate
    lat_offset = radius_km / 111.0  # Approximate km to degrees
    lon_offset = radius_km / (111.0 * abs(lat))  # Adjust for latitude
    
    bbox = {
        'ymin': lat - lat_offset,
        'ymax': lat + lat_offset,
        'xmin': lon - lon_offset,
        'xmax': lon + lon_offset
    }
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    print(f"\n{'='*50}")
    print(f"Testing area around: {location_name}")
    print(f"Coordinates: ({lat}, {lon})")
    print(f"Search radius: {radius_km}km")
    print(f"Bounding box: {bbox}")
    print(f"{'='*50}")
    
    # Test 1: Any buildings at all
    query1 = f"""
    [out:json][timeout:60];
    (
      way["building"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
      relation["building"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
    );
    out body;
    >;
    out skel qt;
    """
    
    print("\n1. Testing for ANY buildings...")
    try:
        response = requests.post(overpass_url, data=query1, timeout=60)
        if response.status_code == 200:
            data = response.json()
            buildings = data.get('elements', [])
            print(f"   Found {len(buildings)} buildings total")
            
            # Analyze building types
            building_types = {}
            for building in buildings:
                if 'tags' in building:
                    btype = building['tags'].get('building', 'unknown')
                    building_types[btype] = building_types.get(btype, 0) + 1
            
            print(f"   Building types: {building_types}")
        else:
            print(f"   Error: {response.status_code}")
    except Exception as e:
        print(f"   Exception: {e}")
    
    time.sleep(2)
    
    # Test 2: Any addresses with house numbers
    query2 = f"""
    [out:json][timeout:60];
    (
      node["addr:housenumber"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
      way["addr:housenumber"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
    );
    out body;
    >;
    out skel qt;
    """
    
    print("\n2. Testing for addresses with house numbers...")
    try:
        response = requests.post(overpass_url, data=query2, timeout=60)
        if response.status_code == 200:
            data = response.json()
            addresses = data.get('elements', [])
            print(f"   Found {len(addresses)} addresses with house numbers")
            
            # Show first few addresses
            for i, addr in enumerate(addresses[:3]):
                if 'tags' in addr:
                    housenumber = addr['tags'].get('addr:housenumber', 'N/A')
                    street = addr['tags'].get('addr:street', 'N/A')
                    city = addr['tags'].get('addr:city', 'N/A')
                    print(f"   Address {i+1}: {housenumber} {street}, {city}")
        else:
            print(f"   Error: {response.status_code}")
    except Exception as e:
        print(f"   Exception: {e}")
    
    time.sleep(2)
    
    # Test 3: Residential buildings specifically
    query3 = f"""
    [out:json][timeout:60];
    (
      way["building"~"^(house|residential|detached|apartments|yes)$"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
      relation["building"~"^(house|residential|detached|apartments|yes)$"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
    );
    out body;
    >;
    out skel qt;
    """
    
    print("\n3. Testing for residential buildings...")
    try:
        response = requests.post(overpass_url, data=query3, timeout=60)
        if response.status_code == 200:
            data = response.json()
            residential = data.get('elements', [])
            print(f"   Found {len(residential)} residential buildings")
        else:
            print(f"   Error: {response.status_code}")
    except Exception as e:
        print(f"   Exception: {e}")
    
    time.sleep(2)
    
    # Test 4: Check population density with amenities
    query4 = f"""
    [out:json][timeout:60];
    (
      node["amenity"~"^(school|hospital|restaurant|cafe|shop|store)$"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
      way["amenity"~"^(school|hospital|restaurant|cafe|shop|store)$"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
    );
    out body;
    >;
    out skel qt;
    """
    
    print("\n4. Testing for population indicators (amenities)...")
    try:
        response = requests.post(overpass_url, data=query4, timeout=60)
        if response.status_code == 200:
            data = response.json()
            amenities = data.get('elements', [])
            print(f"   Found {len(amenities)} amenities")
            
            # Count amenity types
            amenity_types = {}
            for amenity in amenities:
                if 'tags' in amenity:
                    atype = amenity['tags'].get('amenity', 'unknown')
                    amenity_types[atype] = amenity_types.get(atype, 0) + 1
            
            print(f"   Amenity types: {amenity_types}")
        else:
            print(f"   Error: {response.status_code}")
    except Exception as e:
        print(f"   Exception: {e}")

def main():
    """Run diagnostic tests on all railway yard coordinates."""
    print("Railway Yard Area Diagnostic Tool")
    print("=================================")
    
    for lat, lon, location_name in TEST_COORDS:
        test_overpass_queries(lat, lon, location_name)
        print(f"\n{'='*50}")
        time.sleep(3)  # Be nice to the API
    
    print("\nDiagnostic complete!")
    print("\nRecommendations:")
    print("1. If very few buildings found: Railway yards are in industrial areas - expand search radius")
    print("2. If buildings but no addresses: OSM data incomplete - try different query strategies")
    print("3. If no residential buildings: Need to search further from industrial railway areas")

if __name__ == "__main__":
    main() 