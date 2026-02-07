const express = require('express');
const router = express.Router();

// Import route modules
const authRoutes = require('./auth');
const observationRoutes = require('./observations');
const roadRoutes = require('./roads');
const routeRoutes = require('./routes');

// Mount routes
router.use('/auth', authRoutes);
router.use('/observations', observationRoutes);
router.use('/roads', roadRoutes);
router.use('/routes', routeRoutes);

// Health check endpoint
router.get('/health', (req, res) => {
    res.json({
        success: true,
        message: 'API is running',
        timestamp: new Date().toISOString()
    });
});

module.exports = router;
