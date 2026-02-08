const express = require('express');
const router = express.Router();
const roadController = require('../controllers/roadController');
const { validateQuery } = require('../middleware/validation');
const optionalAuthenticateFirebase = require('../middleware/optionalFirebaseAuth');

/**
 * @route   GET /api/roads/region/:regionId
 * @desc    Get road segments in a region
 * @access  Public
 */
router.get('/region/:regionId', optionalAuthenticateFirebase, roadController.getRoadSegmentsByRegion);

/**
 * @route   GET /api/roads/map
 * @desc    Get all road segments for map visualization (green/yellow/red)
 * @access  Public
 */
router.get('/map', validateQuery('mapSegments'), roadController.getMapSegments);

/**
 * @route   GET /api/roads/nearby
 * @desc    Get nearby road segments
 * @access  Public
 */
router.get('/nearby', optionalAuthenticateFirebase, validateQuery('nearbyQuery'), roadController.getNearbyRoadSegments);

/**
 * @route   GET /api/roads/segment/:segmentId
 * @desc    Get specific road segment details
 * @access  Public
 */
router.get('/segment/:segmentId', optionalAuthenticateFirebase, roadController.getRoadSegmentDetails);

/**
 * @route   GET /api/roads/region/:regionId/stats
 * @desc    Get road quality statistics for a region
 * @access  Public
 */
router.get('/region/:regionId/stats', optionalAuthenticateFirebase, roadController.getRegionStatistics);

module.exports = router;
