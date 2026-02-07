const express = require('express');
const router = express.Router();
const routeController = require('../controllers/routeController');
const { validateQuery } = require('../middleware/validation');
const optionalAuthenticateFirebase = require('../middleware/optionalFirebaseAuth');

/**
 * @route   GET /api/routes/score
 * @desc    Get scored routes between source and destination
 * @access  Public
 * @query   sourceLat, sourceLng, destinationLat, destinationLng, maxRoutes (optional)
 */
router.get(
    '/score',
    optionalAuthenticateFirebase,
    validateQuery('routeQuery'),
    routeController.getScoredRoutes
);

/**
 * @route   POST /api/routes/evaluate
 * @desc    Evaluate road quality for a custom route geometry
 * @access  Public
 * @body    { geometry: "polyline6_encoded_string" }
 */
router.post(
    '/evaluate',
    optionalAuthenticateFirebase,
    routeController.evaluateCustomRoute
);

module.exports = router;
