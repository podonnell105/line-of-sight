# Line of Sight Analysis Tool

This tool analyzes line of sight between addresses and rail lines using elevation data and building footprints.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd line-of-sight
   ```

2. **Create and activate a virtual environment**
   ```bash
   # For macOS/Linux
   python -m venv venv
   source venv/bin/activate

   # For Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Google Maps API Key**
   - Go to https://console.cloud.google.com/
   - Create a new project or select an existing one
   - Enable the "Elevation API" for your project
   - Create credentials (API key) for the Elevation API

5. **Configure Environment Variables**
   Create a `.env` file in the project root with:
   ```
   GOOGLE_MAPS_API_KEY=your_api_key_here
   ```

## Data Sources
- **Rail Lines**: The rail line data (`data/shapefiles/tl_2022_us_rails.*`) comes from the U.S. Census Bureau's TIGER/Line Shapefiles (2022). These files contain the national rail network data and are used as the base layer for rail line analysis. (`https://fragis.fra.dot.gov/arcgis/rest/services/FRA/MainLine/MapServer`)
- **Addresses**: User-provided data from OpenAddresses (`https://batch.openaddresses.io/data#map=0/0/0`), saved as `data/addresses.geojson`
- **Building Footprints**: Automatically fetched from OpenStreetMap (OSM)
- **LiDAR Data**: 
  - Source: OpenTopography API (USGS 3DEP 10m resolution data)
  - Used to analyze terrain, vegetation, and buildings that might obstruct line of sight
  - The tool automatically:
    1. Fetches LiDAR data for your analysis area
    2. Processes elevation data to identify:
       - Trees (high local variance, moderate slope)
       - Shrubs (moderate local variance, moderate slope)
       - Buildings (high local variance, very high slope)
    3. Uses this information to calculate line-of-sight scores between addresses and rail lines
  - Note: LiDAR coverage may vary by region. The tool will notify you if data is unavailable for your area.

## Features

- Line of sight analysis between addresses and rail lines
- Building footprint detection
- Elevation data processing
- Vegetation and obstruction analysis
- Source: Google Maps Elevation API (high-resolution elevation data)

## Usage

1. Start the application:
   ```bash
   python src/app.py
   ```

2. Open your browser to `http://localhost:5001`

3. Upload a GeoJSON file containing addresses

4. Select a state for analysis

5. View results and download reports

## Running the Application

1. **Start the Flask server**
   ```bash
   python src/app.py
   ```

2. **Access the web interface**
   - Open your web browser
   - Navigate to `http://localhost:5000`

## Using the Web Interface

1. **Upload Address Data**
   - Prepare a GeoJSON file containing address points
   - Click "Choose File" and select your GeoJSON file
   - Click "Upload"

2. **Configure Analysis**
   - Enter the center coordinates (latitude and longitude)
   - Set the analysis radius (in kilometers)
   - Configure rail buffer distance (in meters)
   - Set shrub height threshold (in meters)
   - Click "Analyze"

3. **View Results**
   - The analysis will show:
     - Clear LOS addresses (green markers)
     - Blocked LOS addresses (red markers)
     - Rail lines (yellow lines)
     - Buildings (gray polygons)
   - Download the results:
     - PNG plot of the analysis
     - CSV file of clear LOS addresses

## Running the Analysis Manually

If you prefer to run the analysis directly using Python instead of the web interface:

1. **Prepare your data**
   ```bash
   # Create necessary directories
   mkdir -p data/shapefiles
   mkdir -p output
   ```

2. **Set up your analysis parameters**
   Edit `src/address_los_score_lidar.py` and modify these variables at the top:
   ```python
   # Analysis area center point
   center_lat = 41.87825  # Example: Chicago
   center_lon = -87.62975
   
   # Analysis parameters
   radius_km = 1.0        # Radius around center point to analyze
   rail_buffer_m = 100    # Distance from rail to consider addresses "nearby"
   shrub_height_threshold = 2.0  # Minimum height (meters) for shrubs to block LOS
   ```

3. **Run the analysis**
   ```bash
   python src/address_los_score_lidar.py
   ```

4. **View the results**
   The script will generate:
   - `output/address_los_scores_lidar_[timestamp].png`: Map visualization
   - `output/clear_los_addresses_[timestamp].csv`: List of addresses with clear LOS
   - `output/building_footprints.geojson`: Building footprints for the area

5. **Customize the analysis**
   You can modify these key functions in `src/address_los_score_lidar.py`:
   ```python
   def calculate_los_score(bbox, address_point, rail_point, elevation, tree_mask, shrub_mask, building_mask):
       # Adjust these thresholds to change sensitivity
       SHRUB_HEIGHT_THRESHOLD = 2.0  # meters
       ELEVATION_CHANGE_THRESHOLD = 5.0  # meters
       
       # Your custom logic here
       ...

   def analyze_vegetation(tif_file):
       # Modify vegetation detection parameters
       window_size = 3  # Size of window for local variance calculation
       tree_threshold = 75  # Percentile for tree detection
       shrub_threshold = 50  # Percentile for shrub detection
       ...

   def process_lidar_data(tif_file):
       # Change how LiDAR data is processed
       ...
   ```

6. **Troubleshooting manual runs**
   - Ensure your Google Maps API key is set in the environment
   - Verify the API key is set correctly: `echo $GOOGLE_MAPS_API_KEY`
   - Check Google Cloud Console for API usage and quotas

## Troubleshooting

1. **API Key Issues**
   - Verify the API key is set correctly: `echo $GOOGLE_MAPS_API_KEY`
   - Check Google Cloud Console for API usage and quotas

2. **Package Installation Issues**
   - Update pip: `python -m pip install --upgrade pip`
   - Try installing packages individually if batch install fails
   - Check Python version compatibility

3. **Analysis Errors**
   - Verify input GeoJSON format
   - Check coordinate system (should be WGS84)
   - Ensure analysis area is within available LiDAR coverage

