const axios = require('axios');

/**
 * Map matching service using OSRM (OpenStreetMap Routing Machine)
 */
class MapMatchingService {
    constructor() {
        this.baseUrl = process.env.MAP_MATCHING_API_URL || 'https://routing.openstreetmap.de/routed-car/match/v1/driving';
    }

    /**
     * Helper to snap a coordinate to the nearest road point
     */
    async snapCoordinate(latitude, longitude) {
        try {
            const nearestUrl = this.baseUrl.replace('/match/', '/nearest/');
            const url = `${nearestUrl}/${longitude},${latitude}?number=1`;
            const response = await axios.get(url, { timeout: 3000 });
            if (response.data.code === 'Ok' && response.data.waypoints && response.data.waypoints.length > 0) {
                const loc = response.data.waypoints[0].location; // [longitude, latitude]
                return { lng: loc[0], lat: loc[1] };
            }
        } catch (err) {
            // Quietly catch errors to return null
        }
        return null;
    }

    /**
     * Match a single GPS point to nearest road
     * @param {number} latitude 
     * @param {number} longitude 
     * @param {Array} fallbackDirection
     * @returns {Object} Matched road segment info
     */
    async matchPoint(latitude, longitude, fallbackDirection = null) {
        try {
            // For single point, use nearest service instead of match
            const nearestUrl = this.baseUrl.replace('/match/', '/nearest/');
            const url = `${nearestUrl}/${longitude},${latitude}?number=1`;

            const response = await axios.get(url, { timeout: 5000 });

            if (response.data.code !== 'Ok' || !response.data.waypoints || response.data.waypoints.length === 0) {
                return null;
            }

            const waypoint = response.data.waypoints[0];
            const matchedLng = waypoint.location[0];
            const matchedLat = waypoint.location[1];

            // Determine road direction by offsetting and snapping
            let direction = fallbackDirection || [1, 0]; // Default or fallback direction

            // Try East offset (+0.0003 deg longitude) and North offset (+0.0003 deg latitude)
            const eastSnap = await this.snapCoordinate(matchedLat, matchedLng + 0.0003);
            const northSnap = await this.snapCoordinate(matchedLat + 0.0003, matchedLng);

            if (eastSnap && northSnap) {
                const distEast = Math.hypot(eastSnap.lng - matchedLng, eastSnap.lat - matchedLat);
                const distNorth = Math.hypot(northSnap.lng - matchedLng, northSnap.lat - matchedLat);

                let bestSnap = null;
                // Whichever offset snaps further away has projection along the road.
                if (distEast > distNorth && distEast > 1e-7) {
                    bestSnap = eastSnap;
                } else if (distNorth > 1e-7) {
                    bestSnap = northSnap;
                }

                if (bestSnap) {
                    const dx = bestSnap.lng - matchedLng;
                    const dy = bestSnap.lat - matchedLat;
                    const len = Math.hypot(dx, dy);
                    if (len > 1e-7) {
                        direction = [dx / len, dy / len];
                    }
                }
            }

            return {
                roadSegmentId: this.generateRoadSegmentId(waypoint),
                matchedLatitude: matchedLat,
                matchedLongitude: matchedLng,
                distance: waypoint.distance || 0, // Distance from original point to matched point
                confidence: this.calculateConfidence(waypoint.distance),
                roadName: waypoint.name || 'Unknown Road',
                direction
            };
        } catch (error) {
            console.error('Map matching error:', error.message);
            // Fallback: return original point with low confidence and default/fallback direction
            return {
                roadSegmentId: this.generateFallbackSegmentId(latitude, longitude),
                matchedLatitude: latitude,
                matchedLongitude: longitude,
                distance: 0,
                confidence: 0.3,
                roadName: 'Unknown Road',
                direction: fallbackDirection || [1, 0]
            };
        }
    }

    /**
     * Match multiple GPS points to a route
     * @param {Array} points Array of {latitude, longitude, timestamp}
     * @returns {Object} Matched route info
     */
    async matchRoute(points) {
        try {
            if (!points || points.length < 2) {
                return null;
            }

            // Format coordinates for OSRM (longitude,latitude pairs)
            const coordinates = points.map(p => `${p.longitude},${p.latitude}`).join(';');

            // Add timestamps if available
            const timestamps = points.map(p =>
                p.timestamp ? Math.floor(new Date(p.timestamp).getTime() / 1000) : 0
            ).join(';');

            const url = `${this.baseUrl}/${coordinates}?overview=full&timestamps=${timestamps}&geometries=geojson`;

            const response = await axios.get(url, { timeout: 10000 });

            if (response.data.code !== 'Ok' || !response.data.matchings) {
                return null;
            }

            const matching = response.data.matchings[0];

            return {
                geometry: matching.geometry,
                confidence: matching.confidence || 0.5,
                distance: matching.distance,
                duration: matching.duration
            };
        } catch (error) {
            console.error('Route matching error:', error.message);
            return null;
        }
    }

    /**
     * Generate a unique road segment ID from waypoint
     */
    generateRoadSegmentId(waypoint) {
        // Use location hash as segment ID
        const lat = Math.round(waypoint.location[1] * 10000);
        const lon = Math.round(waypoint.location[0] * 10000);
        return `seg_${lat}_${lon}`;
    }

    /**
     * Generate fallback segment ID when map matching fails
     */
    generateFallbackSegmentId(latitude, longitude) {
        const lat = Math.round(latitude * 10000);
        const lon = Math.round(longitude * 10000);
        return `fallback_${lat}_${lon}`;
    }

    /**
     * Calculate confidence based on matching distance
     * @param {number} distance Distance in meters
     * @returns {number} Confidence score 0-1
     */
    calculateConfidence(distance) {
        if (!distance) return 0.9;
        if (distance < 10) return 0.9;
        if (distance < 30) return 0.7;
        if (distance < 50) return 0.5;
        return 0.3;
    }
}

module.exports = new MapMatchingService();
