const routeScoringService = require('../services/routeScoring');

/**
 * Get scored routes between source and destination
 */
exports.getScoredRoutes = async (req, res, next) => {
    try {
        const {
            sourceLat,
            sourceLng,
            destinationLat,
            destinationLng,
            maxRoutes = 3
        } = req.validatedQuery;

        const source = {
            latitude: sourceLat,
            longitude: sourceLng
        };

        const destination = {
            latitude: destinationLat,
            longitude: destinationLng
        };

        // Get and score routes
        const result = await routeScoringService.getAndScoreRoutes(
            source,
            destination,
            parseInt(maxRoutes)
        );

        res.json({
            success: true,
            data: result
        });
    } catch (error) {
        // Handle specific errors
        if (error.message.includes('Mapbox')) {
            return res.status(503).json({
                success: false,
                message: 'Route service temporarily unavailable',
                error: error.message
            });
        }

        if (error.message.includes('No routes found')) {
            return res.status(404).json({
                success: false,
                message: 'No routes found between the specified locations',
                error: error.message
            });
        }

        next(error);
    }
};

/**
 * Get route quality preview for a specific route geometry
 * Useful for custom routes or re-evaluation
 */
exports.evaluateCustomRoute = async (req, res, next) => {
    try {
        const { geometry } = req.body;

        if (!geometry || typeof geometry !== 'string') {
            return res.status(400).json({
                success: false,
                message: 'Route geometry (polyline6 encoded) is required'
            });
        }

        // Create a mock Mapbox route object
        const mockRoute = {
            geometry,
            distance: 0, // Will be calculated if needed
            duration: 0
        };

        const scoredRoute = await routeScoringService.scoreRoute(mockRoute);

        res.json({
            success: true,
            data: {
                roadQualityScore: parseFloat(scoredRoute.roadQualityScore.toFixed(2)),
                qualityRating: routeScoringService.getQualityRating(scoredRoute.roadQualityScore),
                segmentCount: scoredRoute.segmentCount,
                segmentsWithData: scoredRoute.segmentsWithData,
                dataCompleteness: parseFloat(
                    (scoredRoute.segmentsWithData / scoredRoute.segmentCount * 100).toFixed(1)
                )
            }
        });
    } catch (error) {
        next(error);
    }
};
