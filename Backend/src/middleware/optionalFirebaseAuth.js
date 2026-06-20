'use strict';

const jwt = require('jsonwebtoken');
const User = require('../models/User');

/**
 * Optional JWT auth — doesn't block the request if no/invalid token.
 * Attaches req.user if a valid token is present.
 */
const optionalAuth = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) return next();

    const token = authHeader.substring(7);
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    const user = await User.findById(decoded.userId);

    if (user) {
      user.lastActive = new Date();
      await user.save();
      req.user = user;
      req.userId = user._id;
    }
  } catch (_) {
    // silently ignore invalid / expired tokens on optional routes
  }
  next();
};

module.exports = optionalAuth;
