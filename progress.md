# Line-of-Sight CropSafe Project Progress

## Latest Development: Coordinate-Based Railway Processing Script

### New Script: `coordinate_railway_processor.py`
**Purpose**: Process specific railway yard coordinates to find addresses with line-of-sight to railway lines within 1km radius.

**Key Features**:
- Takes list of specific coordinates (railway yards and locations)
- Creates 1km search radius around each coordinate
- Finds railway lines within the area
- Creates 200m x 200m bounding boxes along railways with 100m buffer
- Fetches real addresses with house numbers from OpenStreetMap
- Calculates line-of-sight scores using elevation data
- Generates individual and combined CSV outputs

**Coordinates Processed**:
1. CSX Howell Yard — Evansville, IN (37.95833, -87.61417)
2. CSX South Anderson Yard — Anderson, IN (40.07401, -85.68021)
3. NS Piqua Yard — Fort Wayne, IN (41.06444, -85.10722)
4. 2 mi E of Piqua Yard toward Montpelier, OH (41.06444, -85.06900)

**Technical Implementation**:
- Uses existing modular architecture (elevation_providers, city_los_processor functions)
- OpenTopography API primary, Google Elevation API fallback
- Rate limiting for OpenStreetMap API compliance
- Robust error handling and logging
- Creates `coordinate_output/` directory with individual and combined CSV files

**Output Format**: 
`location_name,location_lat,location_lon,address,city,state,zip5,coordinates,los_score,visibility_percentage,distance_to_railway`

---

## Previous Major Development: City-Based Line-of-Sight Analysis System

### Core Architecture Changes
The project was transformed from state-based to city-based analysis with the following key modules:

#### 1. **Main Application** (`src/app.py`)
- Flask web application with `/analyze_city` route
- Handles city/state input and initiates processing
- Directory cleanup preserves previous output files in `web_data/output/`
- Integration with modular elevation providers

#### 2. **City Processing Engine** (`src/city_los_processor.py`)
- **Core Function**: `process_city_los_analysis(city, state, elevation_provider)`
- Fetches city boundaries using Nominatim API
- Creates railway bounding boxes within city limits
- Processes addresses in batches with line-of-sight calculations
- Optimized to fetch elevation data once per city (not per bounding box)
- Optimized to fetch railway data once per city, then filter per bounding box

#### 3. **Elevation Data Providers** (`src/elevation_providers.py`)
- **Abstract Base Class**: `ElevationProvider`
- **OpenTopographyProvider**: High-resolution LiDAR data (primary)
- **GoogleElevationProvider**: Google Elevation API (fallback)
- Modular design allows easy swapping of providers

#### 4. **Address Fetching System**
- **3 Focused OSM Queries**: Only real addresses with house numbers
- **Rate Limiting**: 1-second delays between queries, retry logic for 429/503 errors
- **Address Format**: `address,city,state,zip5,coordinates,los_score,visibility_percentage,distance_to_railway`
- **Filtering**: Removes synthetic data, only accepts addresses with `addr:housenumber`

#### 5. **Line-of-Sight Calculation**
- **Simplified Algorithm**: Direct line-of-sight (was 40-degree field of view)
- **Success Criteria**: Any railway point visible (was 50% of sampled points)
- **Parameters**: 100m sample distance, max 5 samples, 2.0m barrier height
- **Terrain Analysis**: Uses LiDAR elevation, slope, and variance for obstacle detection

### Batch Processing System

#### **Batch Script** (`run_batch_processing.py`)
- Processes 14 predefined cities across NC, SC, GA, VA
- OpenTopography primary, Google fallback per city
- 30-second delays between cities for API rate limiting
- Individual CSV generation: `{City}_{State}_los_scores_{timestamp}.csv`
- Comprehensive error handling and progress logging

#### **Cities Processed**:
Fayetteville NC, Selma NC, Rocky Mount NC, Morehead City NC, Salisbury NC, Pembroke NC, Wilmington NC, Spartanburg SC, Columbia SC, Denmark SC, Kingstree SC, Florence SC, Savannah GA, Norfolk VA

### Data Quality and Management

#### **Duplicate Removal** (`output/dupe.py`)
- Removes duplicates based on `address`, `city`, `state` columns
- Processes all CSV files in directory
- Overwrites original files with cleaned versions
- **Results**: Removed 855+ duplicates across 10 files (31.5% reduction rate)

#### **Output Management**
- **Temp Directory**: `web_data/temp/` (cleaned on each run)
- **Output Directory**: `web_data/output/` (preserved across runs)
- **Coordinate Output**: `coordinate_output/` (new for coordinate processing)

### Technical Optimizations

#### **Cost/Performance Improvements**:
1. **Elevation Data**: Fetch once per city vs. per bounding box (major cost savings)
2. **Railway Data**: Fetch once per city, filter per bounding box (eliminates redundancy)
3. **Address Queries**: Reduced from 7 to 3 focused queries (faster, more accurate)
4. **LOS Calculation**: Simplified from 40-degree field to direct line-of-sight (faster processing)

#### **Import System Fix**:
- Converted all relative imports to absolute imports
- Added `sys.path.insert` in runner scripts
- Created `run_app.py` for deployment compatibility
- Resolved `ImportError: attempted relative import with no known parent package`

#### **API Integration**:
- **OpenTopography**: Requires API key (OPENTOPOGRAPHY_API_KEY in .env)
- **Google Elevation**: Fallback when OpenTopography fails
- **OpenStreetMap**: Rate limited queries with retry logic
- **Nominatim**: City boundary geocoding

### Current System Capabilities

✅ **City-based analysis** with complete railway coverage within city boundaries
✅ **Coordinate-based analysis** for specific railway yard locations  
✅ **Modular elevation providers** (OpenTopography + Google fallback)
✅ **Real address validation** with house numbers only
✅ **Accurate line-of-sight calculations** using high-resolution LiDAR
✅ **Batch processing** for multiple cities with API rate management
✅ **Data deduplication** and quality control
✅ **Comprehensive logging** and error handling
✅ **Deployment-ready** with absolute imports and runner scripts

### Latest Files and Their Functions

| File | Purpose | Key Features |
|------|---------|--------------|
| `coordinate_railway_processor.py` | Process specific coordinates | 1km radius search, railway bounding boxes, address collection |
| `src/city_los_processor.py` | City-wide LOS analysis | Optimized elevation/railway fetching, real address validation |
| `src/elevation_providers.py` | Modular elevation data | OpenTopography + Google providers, easy swapping |
| `run_batch_processing.py` | Automated multi-city processing | 14 cities, API management, individual CSVs |
| `output/dupe.py` | Data quality control | Remove duplicates across all CSV files |
| `run_app.py` | Flask app deployment | Import path fixes for production |

The system now supports both city-wide analysis and specific coordinate-based processing, with robust data collection, accurate LOS calculations, and comprehensive output management. 

### Update: Coordinate Processor fixes (OSM geocoding + larger boxes + 100m filter)
- Files updated: `coordinate_railway_processor.py`
  - Added OSM Nominatim search to resolve yard names to precise coordinates via `search_osm_coordinates` and `get_updated_coordinates`.
  - Switched railway source from FRA (SSL issue) to OpenStreetMap Overpass for rail geometries.
  - Fixed OpenTopography bbox format by passing `(xmin, ymin, xmax, ymax)` tuple.
  - Adjusted processing to fetch railway lines within 1km of each coordinate.
  - Bounding boxes along rail set to 1km x 1km for broad capture, then filtered addresses to those within 100m of rail for relevance.
  - Extracted `city_name` and `state_abbr` from OSM display names to satisfy `fetch_addresses_in_bbox` signature.
- New file: `test_address_finding.py`
  - Simple Overpass probe confirming addresses exist around the target areas; validated 11 (Evansville), 8 (Anderson), and 8,271 (Fort Wayne) addresses within a ~2km window.

Reasoning: Initial 200m boxes around industrial yards missed nearby residential streets. Larger capture (1km) combined with a 100m distance filter to the railway ensures we gather sufficient candidates and keep only near-track addresses. Also resolved elevation bbox tuple format for OpenTopography. 