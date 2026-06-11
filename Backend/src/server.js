// ==========================================
// FILE: Backend/src/app.js (or server.js)
// ==========================================

require('dotenv').config();

const express = require('express');
const http = require('http');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');

const connectDB = require('./config/database');
const { initializeRedis, closeRedis, getRedisClient } = require('./config/redis');
const socketModule = require('./socket'); // Clean extraction strategy applied
const routes = require('./routes');
const { errorHandler, notFound } = require('./middleware/errorHandler');

const app = express();
const server = http.createServer(app);

// Initialize DB and Redis
connectDB();
initializeRedis().catch(err => {
    console.warn('Redis initialization failed, continuing without Redis:', err.message);
});

// Socket.IO Server initialization fix
const io = socketModule.init(server);
app.set('io', io);

// Security
app.use(helmet());
app.use(cors({ origin: process.env.CORS_ORIGIN || '*', credentials: true }));
app.use(compression());

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Logging
app.use(morgan(process.env.NODE_ENV === 'development' ? 'dev' : 'combined'));

// Rate limiting
app.use('/api', rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 100
}));

// Routes
app.use('/api', routes);

// Health Check Endpoints
app.get('/', (req, res) => {
    res.json({ status: 'API running ' });
});

app.get('/health', async (req, res) => {
    const redisClient = getRedisClient();
    let redisStatus = 'disconnected';

    if (redisClient) {
        try {
            await redisClient.ping();
            redisStatus = 'connected';
        } catch (err) {
            redisStatus = 'error';
        }
    }

    res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        services: {
            mongodb: 'connected',
            redis: redisStatus,
            socketio: 'active'
        }
    });
});

app.use(notFound);
app.use(errorHandler);

const PORT = process.env.PORT || 5000;

server.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on port ${PORT}`);
});

const shutdown = async () => {
    console.log('\n Shutting down gracefully...');
    try {
        await closeRedis();
        server.close(() => {
            console.log('Server closed');
            process.exit(0);
        });

        setTimeout(() => {
            console.error('Forced shutdown');
            process.exit(1);
        }, 10000);
    } catch (error) {
        console.error('Shutdown error:', error);
        process.exit(1);
    }
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
process.on('unhandledRejection', console.error);
process.on('uncaughtException', console.error);

module.exports = { app, server, io };