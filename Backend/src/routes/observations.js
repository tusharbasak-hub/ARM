const express = require('express');
const router = express.Router();
const observationController = require('../controllers/observationController');
const { validate } = require('../middleware/validation');
const authenticateFirebase = require('../middleware/firebaseAuth');

/**
 * @route   POST /api/observations
 * @desc    Submit road quality observation
 * @access  Private
 */
router.post('/', authenticateFirebase, validate('observation'), observationController.submitObservation);

/**
 * @route   POST /api/observations/patch
 * @desc    Submit road quality patch observation (continuous bad road stretch)
 * @access  Private
 */
router.post('/patch', authenticateFirebase, validate('patch'), observationController.submitPatch);

/**
 * @route   GET /api/observations/history
 * @desc    Get user's observation history
 * @access  Private
 */
router.get('/history', authenticateFirebase, observationController.getObservationHistory);

/**
 * @route   GET /api/observations/recent
 * @desc    Get recent bad road observations for map markers (browsing mode)
 * @access  Public (no auth required for viewing alerts)
 */
router.get('/recent', require('../middleware/validation').validateQuery('recentAlerts'), observationController.getRecentAlerts);

module.exports = router;
