import abc

class ElevationProvider(abc.ABC):
    @abc.abstractmethod
    def get_elevation_tif(self, bbox):
        """
        Given bbox = (min_lon, min_lat, max_lon, max_lat), return path to GeoTIFF file.
        """
        pass

class GoogleElevationProvider(ElevationProvider):
    def get_elevation_tif(self, bbox):
        from fetch_lidar import get_google_elevation
        return get_google_elevation(bbox)

class OpenTopographyProvider(ElevationProvider):
    def get_elevation_tif(self, bbox):
        """
        Query OpenTopography API for elevation data and save as GeoTIFF.
        """
        import requests, tempfile, os
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv(override=True)
        
        # Get API key from environment
        api_key = os.getenv('OPENTOPOGRAPHY_API_KEY')
        if not api_key:
            raise Exception("OPENTOPOGRAPHY_API_KEY not found in environment variables")
        
        min_lon, min_lat, max_lon, max_lat = bbox
        # Use SRTMGL1 (30m) global DEM with API key
        url = (
            f"https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1"
            f"&south={min_lat}&north={max_lat}&west={min_lon}&east={max_lon}&outputFormat=GTiff"
            f"&API_Key={api_key}"
        )
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"OpenTopography API error: {response.status_code} {response.text}")
        temp_file = tempfile.mktemp(suffix=".tif")
        with open(temp_file, "wb") as f:
            f.write(response.content)
        return temp_file 