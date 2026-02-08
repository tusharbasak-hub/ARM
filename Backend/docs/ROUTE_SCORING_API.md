# Route Scoring API

## Overview

The Route Scoring API provides intelligent route recommendations based on both road quality data and distance. It fetches routes from Mapbox Directions API and scores them using aggregated road quality observations from your existing system.

## Endpoints

### 1. Get Scored Routes

**Endpoint:** `GET /api/routes/score`

**Description:** Get up to 3 scored routes between source and destination with quality-based recommendations.

**Query Parameters:**
- `sourceLat` (required): Source latitude (-90 to 90)
- `sourceLng` (required): Source longitude (-180 to 180)
- `destinationLat` (required): Destination latitude (-90 to 90)
- `destinationLng` (required): Destination longitude (-180 to 180)
- `maxRoutes` (optional): Maximum routes to return (1-5, default: 3)

**Example Request:**
```
GET /api/routes/score?sourceLat=40.7128&sourceLng=-74.0060&destinationLat=40.7589&destinationLng=-73.9851&maxRoutes=3
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "source": {
      "latitude": 40.7128,
      "longitude": -74.006
    },
    "destination": {
      "latitude": 40.7589,
      "longitude": -73.9851
    },
    "routes": [
      {
        "rank": 1,
        "distance": 8542,
        "distanceKm": 8.54,
        "duration": 1234,
        "durationMinutes": 21,
        "roadQualityScore": 1.23,
        "qualityRating": "Good",
        "finalScore": 0.5512,
        "geometry": "encoded_polyline_string",
        "segmentCount": 45,
        "segmentsWithData": 38,
        "dataCompleteness": 84.4
      },
      {
        "rank": 2,
        "distance": 9120,
        "distanceKm": 9.12,
        "duration": 1356,
        "durationMinutes": 23,
        "roadQualityScore": 0.85,
        "qualityRating": "Excellent",
        "finalScore": 0.5896,
        "geometry": "encoded_polyline_string",
        "segmentCount": 48,
        "segmentsWithData": 40,
        "dataCompleteness": 83.3
      }
    ],
    "bestRoute": {
      "rank": 1,
      "distance": 8542,
      "distanceKm": 8.54,
      "duration": 1234,
      "durationMinutes": 21,
      "roadQualityScore": 1.23,
      "qualityRating": "Good",
      "finalScore": 0.5512,
      "geometry": "encoded_polyline_string",
      "segmentCount": 45,
      "segmentsWithData": 38,
      "dataCompleteness": 84.4,
      "reason": "Optimal balance of shortest distance and best road quality (Good)."
    },
    "timestamp": "2026-01-31T10:30:00.000Z"
  }
}
```

### 2. Evaluate Custom Route

**Endpoint:** `POST /api/routes/evaluate`

**Description:** Evaluate road quality for a custom route geometry (useful for re-evaluating existing routes).

**Request Body:**
```json
{
  "geometry": "polyline6_encoded_string"
}
```

**Example Response:**
```json
{
  "success": true,
  "data": {
    "roadQualityScore": 1.45,
    "qualityRating": "Good",
    "segmentCount": 52,
    "segmentsWithData": 44,
    "dataCompleteness": 84.6
  }
}
```

## Response Fields Explained

### Route Object
- **rank**: Route ranking (1 = best)
- **distance**: Route distance in meters
- **distanceKm**: Route distance in kilometers
- **duration**: Estimated duration in seconds
- **durationMinutes**: Estimated duration in minutes (rounded up)
- **roadQualityScore**: Average road quality score (0-3 scale, lower is better)
  - 0.0-0.5: Excellent
  - 0.5-1.5: Good
  - 1.5-2.5: Fair
  - 2.5-3.0: Poor
- **qualityRating**: Human-readable quality rating
- **finalScore**: Combined score considering both quality and distance (lower is better)
- **geometry**: Mapbox polyline6 encoded route geometry
- **segmentCount**: Total number of road segments in route
- **segmentsWithData**: Number of segments with actual observation data
- **dataCompleteness**: Percentage of segments with observation data

### Best Route
The `bestRoute` object includes all route fields plus:
- **reason**: Explanation of why this route is recommended

## Scoring Logic

### Road Quality Score
- Calculated as the average quality score of all road segments in the route
- Uses aggregated quality scores from existing observation system
- Segments without data default to score 1.0 ("Good" assumption)

### Final Score Calculation
```
finalScore = (normalizedQuality × 0.6) + (normalizedDistance × 0.4)
```

Where:
- `normalizedQuality = roadQualityScore / 3` (converts 0-3 scale to 0-1)
- `normalizedDistance = distanceKm / 10` (normalized relative to 10km reference)
- Weights: 60% quality, 40% distance (configurable via env vars)

**Lower final score = Better route**

## Caching

Routes are cached for 1 hour (configurable) with smart invalidation:
- Cache automatically invalidates if segment quality scores change significantly (>0.3 difference)
- Cache keys based on source/destination coordinates (rounded to 4 decimal places)
- Redis-backed caching with graceful degradation if Redis unavailable

## Error Responses

### 400 Bad Request
```json
{
  "success": false,
  "message": "Validation failed",
  "errors": [
    {
      "field": "sourceLat",
      "message": "\"sourceLat\" is required"
    }
  ]
}
```

### 404 Not Found
```json
{
  "success": false,
  "message": "No routes found between the specified locations",
  "error": "..."
}
```

### 503 Service Unavailable
```json
{
  "success": false,
  "message": "Route service temporarily unavailable",
  "error": "Mapbox API error: ..."
}
```

## Configuration

Add to your `.env` file:

```env
# Mapbox Configuration
MAPBOX_ACCESS_TOKEN=your_mapbox_access_token_here

# Route Scoring Weights (must sum to 1.0)
ROUTE_QUALITY_WEIGHT=0.6    # Preference for road quality
ROUTE_DISTANCE_WEIGHT=0.4   # Preference for shorter distance

# Cache Settings
ROUTE_CACHE_TTL=3600         # Cache duration in seconds
```

## Integration with Existing Features

✅ **Preserves all existing functionality:**
- Live road observations continue working exactly as before
- Real-time quality score aggregation unchanged
- Socket.io broadcasts remain functional
- Existing API endpoints untouched

✅ **Reuses existing infrastructure:**
- Uses same RoadSegment model and aggregated scores
- Leverages existing Redis cache configuration
- Integrates with existing authentication middleware
- Follows same validation patterns

## Frontend Integration Example

```javascript
// Fetch scored routes
const response = await fetch(
  `/api/routes/score?` +
  `sourceLat=40.7128&sourceLng=-74.0060&` +
  `destinationLat=40.7589&destinationLng=-73.9851&` +
  `maxRoutes=3`
);

const { data } = await response.json();

// Display best route
console.log(`Best route: ${data.bestRoute.distanceKm}km`);
console.log(`Quality: ${data.bestRoute.qualityRating}`);
console.log(`Reason: ${data.bestRoute.reason}`);

// Render all routes on map
data.routes.forEach(route => {
  // Decode polyline and render on map
  const coordinates = decodePolyline(route.geometry);
  renderRoute(coordinates, {
    color: route.rank === 1 ? 'green' : 'gray',
    quality: route.qualityRating
  });
});
```

## Performance Considerations

- **First request**: ~2-3 seconds (Mapbox API + database queries)
- **Cached requests**: ~50-100ms
- **Cache invalidation**: Automatic based on segment score changes
- **Segment matching**: Uses MongoDB geospatial indexes for fast lookups
- **Scalability**: Can handle thousands of requests/min with Redis caching

## Future ML Integration

The architecture is designed for easy ML enhancement:

```javascript
// routeScoring.js - calculateFinalScore method
// Replace this method to plug in ML-based scoring
calculateFinalScore(roadQualityScore, distanceKm, mapboxRoute) {
  // Call ML model here
  return mlModel.predict({
    quality: roadQualityScore,
    distance: distanceKm,
    duration: mapboxRoute.duration,
    // Add more features...
  });
}
```
