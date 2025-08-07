#!/usr/bin/env python3
"""
Test script to check if we can find any addresses in the railway yard areas
"""

import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_address_finding():
    """Test if we can find any addresses in the railway yard areas"""
    
    # Test coordinates (the ones we found)
    test_locations = [
        (37.9602389, -87.6067297, "CSX Howell Yard, Evansville"),
        (40.0832513, -85.688746, "CSX South Anderson Yard, Anderson"),
        (41.0644145, -85.1058342, "NS Piqua Yard, Fort Wayne")
    ]
    
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    for lat, lon, name in test_locations:
        logging.info(f"\n=== Testing {name} ===")
        
        # Create a larger bounding box (2km radius)
        bbox = {
            'xmin': lon - 0.02,  # ~2km
            'ymin': lat - 0.02,
            'xmax': lon + 0.02,
            'ymax': lat + 0.02
        }
        
        # Simple query to find ANY addresses
        query = f"""
        [out:json][timeout:60];
        (
          node["addr:housenumber"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
          way["addr:housenumber"]({bbox['ymin']},{bbox['xmin']},{bbox['ymax']},{bbox['xmax']});
        );
        out body;
        >;
        out skel qt;
        """
        
        try:
            response = requests.post(overpass_url, data=query)
            response.raise_for_status()
            data = response.json()
            
            addresses = []
            for element in data.get('elements', []):
                if 'tags' in element and 'addr:housenumber' in element['tags']:
                    housenumber = element['tags'].get('addr:housenumber', '')
                    street = element['tags'].get('addr:street', '')
                    city = element['tags'].get('addr:city', '')
                    state = element['tags'].get('addr:state', '')
                    
                    if element['type'] == 'node':
                        lat = element.get('lat')
                        lon = element.get('lon')
                    else:
                        # For ways, we'd need to get coordinates from nodes
                        lat = lon = None
                    
                    addresses.append({
                        'housenumber': housenumber,
                        'street': street,
                        'city': city,
                        'state': state,
                        'lat': lat,
                        'lon': lon
                    })
            
            logging.info(f"Found {len(addresses)} addresses in {name}")
            for i, addr in enumerate(addresses[:5]):  # Show first 5
                logging.info(f"  {i+1}. {addr['housenumber']} {addr['street']}, {addr['city']}, {addr['state']}")
            if len(addresses) > 5:
                logging.info(f"  ... and {len(addresses) - 5} more")
                
        except Exception as e:
            logging.error(f"Error testing {name}: {e}")

if __name__ == "__main__":
    test_address_finding() 