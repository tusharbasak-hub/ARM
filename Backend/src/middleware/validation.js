const Joi = require('joi');

/**
 * Validation schemas
 */
const schemas = {
    register: Joi.object({
        email: Joi.string().email().required(),
        password: Joi.string().min(6).required(),
        name: Joi.string().min(2).max(50).required(),
        deviceId: Joi.string().optional()
    }),

    login: Joi.object({
        email: Joi.string().email().required(),
        password: Joi.string().required(),
        deviceId: Joi.string().optional()
    }),

    anonymous: Joi.object({
        deviceId: Joi.string().required()
    }),

    observation: Joi.object({
        latitude:          Joi.number().min(-90).max(90).required(),
        longitude:         Joi.number().min(-180).max(180).required(),
        iriScore:          Joi.number().min(0).required(),
        hasPothole:        Joi.boolean().default(false),
        potholeConfidence: Joi.number().min(0).max(1).default(0),
        roadSegmentId:     Joi.string().optional(),
        deviceId:          Joi.string().optional(),
        sessionId:         Joi.string().optional(),
        recordedAt:        Joi.date().iso().optional(),
        speed:             Joi.number().min(0).optional(),
    }),

    nearbyQuery: Joi.object({
        lat: Joi.number().min(-90).max(90).required(),
        lng: Joi.number().min(-180).max(180).required(),
        radius: Joi.number().min(100).max(50000).default(5000) // meters
    }),

    routeQuery: Joi.object({
        sourceLat: Joi.number().min(-90).max(90).required(),
        sourceLng: Joi.number().min(-180).max(180).required(),
        destinationLat: Joi.number().min(-90).max(90).required(),
        destinationLng: Joi.number().min(-180).max(180).required(),
        maxRoutes: Joi.number().integer().min(1).max(5).default(3)
    }),

    patch: Joi.object({
        observations: Joi.array().items(Joi.object({
            latitude:          Joi.number().min(-90).max(90).required(),
            longitude:         Joi.number().min(-180).max(180).required(),
            iriScore:          Joi.number().min(0).required(),
            hasPothole:        Joi.boolean().default(false),
            potholeConfidence: Joi.number().min(0).max(1).default(0),
            roadSegmentId:     Joi.string().optional(),
            deviceId:          Joi.string().optional(),
            sessionId:         Joi.string().optional(),
            recordedAt:        Joi.date().iso().optional(),
        })).min(1).required(),
    }),

    recentAlerts: Joi.object({
        limit:    Joi.number().integer().min(1).max(500).default(100),
        hoursBack: Joi.number().integer().min(1).max(168).default(72),
    }),

    mapSegments: Joi.object({
        regionIds: Joi.array().items(Joi.string()).min(1).max(25).required() // Viewport + neighbors
    })
};

/**
 * Middleware factory for request validation
 */
const validate = (schemaName) => {
    return (req, res, next) => {
        const schema = schemas[schemaName];

        if (!schema) {
            return next(new Error(`Schema '${schemaName}' not found`));
        }

        const { error, value } = schema.validate(req.body, {
            abortEarly: false,
            stripUnknown: true
        });

        if (error) {
            const errors = error.details.map(detail => ({
                field: detail.path.join('.'),
                message: detail.message
            }));

            return res.status(400).json({
                success: false,
                message: 'Validation failed',
                errors
            });
        }

        req.validatedData = value;
        next();
    };
};

/**
 * Query parameter validation
 */
const validateQuery = (schemaName) => {
    return (req, res, next) => {
        const schema = schemas[schemaName];

        if (!schema) {
            return next(new Error(`Schema '${schemaName}' not found`));
        }

        const { error, value } = schema.validate(req.query, {
            abortEarly: false,
            stripUnknown: true
        });

        if (error) {
            const errors = error.details.map(detail => ({
                field: detail.path.join('.'),
                message: detail.message
            }));

            return res.status(400).json({
                success: false,
                message: 'Validation failed',
                errors
            });
        }

        req.validatedQuery = value;
        next();
    };
};

module.exports = {
    validate,
    validateQuery
};
