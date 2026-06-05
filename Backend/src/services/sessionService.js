// ==========================================
// FILE: Backend/src/services/sessionService.js
// ==========================================

const { getRedisClient } = require('../config/redis');

const SESSION_TTL = 300; 
const HEARTBEAT_TTL = 300; 

const createSession = async (socketId, userId, metadata = {}) => {
    const client = getRedisClient();
    if (!client) return false;

    try {
        const sessionKey = `session:${socketId}`;
        const sessionData = {
            socketId,
            userId: userId.toString(),
            connectedAt: new Date().toISOString(),
            ...metadata
        };

        await client.setEx(sessionKey, SESSION_TTL, JSON.stringify(sessionData));

        const userSessionsKey = `user:${userId}:sessions`;
        await client.sAdd(userSessionsKey, socketId);
        await client.expire(userSessionsKey, SESSION_TTL);

        return true;
    } catch (error) {
        console.error('Redis createSession error:', error.message);
        return false;
    }
};

const updateSession = async (socketId, updates = {}) => {
    const client = getRedisClient();
    if (!client) return false;

    try {
        const sessionKey = `session:${socketId}`;
        const existing = await client.get(sessionKey);

        if (!existing) return false;

        const sessionData = {
            ...JSON.parse(existing),
            ...updates,
            lastActivityAt: new Date().toISOString()
        };

        await client.setEx(sessionKey, SESSION_TTL, JSON.stringify(sessionData));
        return true;
    } catch (error) {
        console.error('Redis updateSession error:', error.message);
        return false;
    }
};

const getSession = async (socketId) => {
    const client = getRedisClient();
    if (!client) return null;

    try {
        const data = await client.get(`session:${socketId}`);
        return data ? JSON.parse(data) : null;
    } catch (error) {
        return null;
    }
};

const leaveRegion = async (socketId, regionId) => {
    const client = getRedisClient();
    if (!client) return false;

    try {
        await client.sRem(`region:${regionId}:members`, socketId);
        return true;
    } catch (error) {
        console.error('Redis leaveRegion error:', error.message);
        return false;
    }
};

const deleteSession = async (socketId) => {
    const client = getRedisClient();
    if (!client) return false;

    try {
        const session = await getSession(socketId);
        if (!session) return false;

        await client.del(`session:${socketId}`);

        if (session.userId) {
            await client.sRem(`user:${session.userId}:sessions`, socketId);
        }

        if (session.currentRegionId) {
            await leaveRegion(socketId, session.currentRegionId);
        }

        return true;
    } catch (error) {
        console.error('Redis deleteSession error:', error.message);
        return false;
    }
};

const joinRegion = async (socketId, regionId, location = {}) => {
    const client = getRedisClient();
    if (!client) return false;

    try {
        const regionKey = `region:${regionId}:members`;
        await client.sAdd(regionKey, socketId);
        await client.expire(regionKey, SESSION_TTL);

        await updateSession(socketId, { currentRegionId: regionId, lastLocation: location });
        return true;
    } catch (error) {
        console.error('Redis joinRegion error:', error.message);
        return false;
    }
};

const getRegionMembers = async (regionId) => {
    const client = getRedisClient();
    if (!client) return [];
    try {
        return await client.sMembers(`region:${regionId}:members`);
    } catch (error) {
        return [];
    }
};

const recordHeartbeat = async (socketId) => {
    const client = getRedisClient();
    if (!client) return false;

    try {
        await client.setEx(`heartbeat:${socketId}`, HEARTBEAT_TTL, new Date().toISOString());
        await updateSession(socketId, {});
        return true;
    } catch (error) {
        return false;
    }
};

const getUserSessions = async (userId) => {
    const client = getRedisClient();
    if (!client) return [];

    try {
        const socketIds = await client.sMembers(`user:${userId}:sessions`);
        const sessions = [];
        for (const socketId of socketIds) {
            const session = await getSession(socketId);
            if (session) sessions.push(session);
        }
        return sessions;
    } catch (error) {
        return [];
    }
};

const cleanupExpiredSessions = async () => {
    const client = getRedisClient();
    if (!client) return 0;

    try {
        const pattern = 'session:*';
        let cursor = 0;
        let cleaned = 0;

        do {
            const result = await client.scan(cursor, { MATCH: pattern, COUNT: 100 });
            cursor = result.cursor;
            const keys = result.keys;

            for (const key of keys) {
                const ttl = await client.ttl(key);
                if (ttl === -1) {
                    await client.del(key);
                    cleaned++;
                }
            }
        } while (cursor !== 0);

        return cleaned;
    } catch (error) {
        return 0;
    }
};

module.exports = {
    createSession,
    updateSession,
    getSession,
    deleteSession,
    joinRegion,
    leaveRegion,
    getRegionMembers,
    recordHeartbeat,
    getUserSessions,
    cleanupExpiredSessions
};