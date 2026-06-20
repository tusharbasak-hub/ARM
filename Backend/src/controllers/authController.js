'use strict';

const bcrypt   = require('bcryptjs');
const jwt      = require('jsonwebtoken');
const User     = require('../models/User');

// ── Helper ────────────────────────────────────────────────────────────────
function signToken(userId) {
  return jwt.sign({ userId }, process.env.JWT_SECRET, {
    expiresIn: process.env.JWT_EXPIRY || '30d',
  });
}

function userPayload(user) {
  return {
    id:          user._id,
    email:       user.email,
    name:        user.name,
    deviceId:    user.deviceId,
    isAnonymous: user.isAnonymous,
    createdAt:   user.createdAt,
  };
}

// ── POST /api/auth/register ───────────────────────────────────────────────
exports.register = async (req, res, next) => {
  try {
    const { email, password, name, deviceId } = req.validatedData;

    // Check if email already exists
    const emailExists = await User.findOne({ email });
    if (emailExists) {
      return res.status(400).json({ success: false, message: 'Email already registered' });
    }

    let user;
    if (deviceId) {
      const deviceUser = await User.findOne({ deviceId });
      if (deviceUser) {
        if (deviceUser.isAnonymous) {
          // Upgrade existing anonymous user on this device to registered user
          const hashedPassword = await bcrypt.hash(password, 10);
          deviceUser.email = email;
          deviceUser.password = hashedPassword;
          deviceUser.name = name;
          deviceUser.isAnonymous = false;
          deviceUser.lastActive = new Date();
          user = await deviceUser.save();
        } else {
          // If a registered user already owns this deviceId, disassociate it from them
          // so this new user can register with it.
          deviceUser.deviceId = undefined;
          await deviceUser.save();
        }
      }
    }

    if (!user) {
      const hashedPassword = await bcrypt.hash(password, 10);
      user = await User.create({ email, password: hashedPassword, name, deviceId, isAnonymous: false });
    }

    const token = signToken(user._id);
    res.status(201).json({ success: true, token, data: { user: userPayload(user) } });
  } catch (error) {
    next(error);
  }
};

// ── POST /api/auth/login ──────────────────────────────────────────────────
exports.login = async (req, res, next) => {
  try {
    const { email, password, deviceId } = req.validatedData;

    const user = await User.findOne({ email }).select('+password');
    if (!user) {
      return res.status(401).json({ success: false, message: 'Invalid email or password' });
    }

    const valid = await bcrypt.compare(password, user.password);
    if (!valid) {
      return res.status(401).json({ success: false, message: 'Invalid email or password' });
    }

    // Update deviceId if the user is logging in from a different device
    if (deviceId && user.deviceId !== deviceId) {
      user.deviceId = deviceId;
    }
    user.lastActive = new Date();
    await user.save();

    const token = signToken(user._id);
    res.json({ success: true, token, data: { user: userPayload(user) } });
  } catch (error) {
    next(error);
  }
};

// ── POST /api/auth/anonymous ──────────────────────────────────────────────
exports.anonymous = async (req, res, next) => {
  try {
    const { deviceId } = req.validatedData;

    let user = await User.findOne({ deviceId, isAnonymous: true });
    if (!user) {
      user = await User.create({
        deviceId,
        isAnonymous: true,
        name: `Rider_${deviceId.substring(0, 6)}`,
      });
    }

    const token = signToken(user._id);
    res.json({ success: true, token, data: { user: userPayload(user) } });
  } catch (error) {
    next(error);
  }
};

// ── GET /api/auth/profile ─────────────────────────────────────────────────
exports.getProfile = async (req, res, next) => {
  try {
    const user = await User.findById(req.userId);
    if (!user) return res.status(404).json({ success: false, message: 'User not found' });

    const observationCount = await require('../models/Observation')
      .countDocuments({ deviceId: user.deviceId });

    res.json({
      success: true,
      data: {
        user: { ...userPayload(user), observationCount, lastActive: user.lastActive },
      },
    });
  } catch (error) {
    next(error);
  }
};
