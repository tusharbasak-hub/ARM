'use strict';
// Firebase auth has been replaced by JWT.
// This file re-exports the JWT middleware so existing route imports still work.
const authenticateJWT = require('./auth');
module.exports = authenticateJWT;
