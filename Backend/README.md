# Road Quality Monitoring Backend

A scalable crowd-sourced road quality monitoring system backend built with Node.js, Express, Socket.IO, and MongoDB.

## Features

- **Real-time Updates**: Socket.IO-based regional broadcasting
- **Map Matching**: OpenStreetMap-based GPS point snapping
- **Spatial Segmentation**: Geohash-based regional partitioning
- **Data Aggregation**: Weighted average with time decay
- **Authentication**: JWT-based auth with anonymous support
- **Scalability**: Horizontal scaling ready with Socket.IO rooms

## Architecture

```
Mobile App (TinyML) → REST API → Map Matching → MongoDB
                    ↓
                Socket.IO Rooms (by region)
                    ↓
            Real-time Broadcasts (regional)
```

## Data Flow

1. Mobile app sends processed data: `{userId, latitude, longitude, roadQuality, timestamp, speed}`
2. Backend performs map matching (snap to road segment)
3. Data stored in MongoDB with geohash-based regionId
4. Aggregation logic computes weighted quality score
5. Socket.IO broadcasts updates to users in same region

## Installation

```bash
npm install
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

## Running

Development:
```bash
npm run dev
```

Production:
```bash
npm start
```

## Cloud Deployment (Render)

This backend is ready to be deployed directly as a **Web Service** on [Render](https://render.com/).

### Deployment Configuration:
- **Root Directory:** `Backend`
- **Build Command:** `npm install`
- **Start Command:** `node src/server.js`

### Environment Variables:
Add the following keys in Render's **Environment** settings:
- `NODE_ENV`: `production`
- `MONGODB_URI`: *Your MongoDB Atlas connection string*
- `JWT_SECRET`: *A secure random string for signing session tokens*
- `JWT_EXPIRY`: `30d`
- `CORS_ORIGIN`: `*`
- `ROUTE_QUALITY_WEIGHT`: `0.7`
- `ROUTE_DISTANCE_WEIGHT`: `0.3`
- `ROUTE_CACHE_TTL`: `900`
- `TIME_DECAY_HOURS`: `24`
- `MIN_OBSERVATIONS_FOR_AGGREGATION`: `1`

## Dynamic Map Matching & Seeding

The system features a self-populating road database:
1. When a client submits coordinates, the backend searches for a matching `RoadSegment` within 500 meters.
2. If no segment is found, it automatically queries the public **OSRM API** (`/nearest/` endpoint) to resolve the nearest real-world road name and snapped coordinate.
3. It creates and saves a new `RoadSegment` document in MongoDB with a generated `LineString` geometry and immediately broadcasts it to all connected apps via Socket.IO.
4. Future observations near this road will automatically link to this segment.

## API Endpoints

### REST API

- `POST /api/auth/register` - Register user
- `POST /api/auth/login` - Login user
- `POST /api/auth/anonymous` - Get anonymous device token
- `POST /api/observations` - Submit road quality observation
- `GET /api/roads/region/:regionId` - Get road segments in region
- `GET /api/roads/nearby?lat=&lng=&radius=` - Get nearby road segments

### WebSocket Events

**Client → Server:**
- `authenticate` - Authenticate socket connection
- `join-region` - Join a regional room
- `leave-region` - Leave a regional room

**Server → Client:**
- `road-quality-update` - Road quality update in region
- `error` - Error message

## MongoDB Collections

- `users` - User accounts
- `roadSegments` - Road segments with aggregated quality scores
- `observations` - Raw road quality observations
- `sessions` - Active user sessions (optional)

## Spatial Segmentation

Uses **Geohash** (precision 6) for spatial partitioning:
- Each geohash covers ~1.2km × 0.61km area
- Users in same geohash join same Socket.IO room
- Efficient regional broadcasting

## Scaling Considerations

- Stateless API design
- Socket.IO with Redis adapter for multi-server
- MongoDB sharding by regionId
- Load balancing with sticky sessions
- CDN for static map tiles

## License

MIT
