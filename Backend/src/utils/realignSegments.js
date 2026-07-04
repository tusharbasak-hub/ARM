// ==========================================
// FILE: Backend/src/utils/realignSegments.js
// ==========================================

require('dotenv').config();
const mongoose = require('mongoose');
const connectDB = require('../config/database');
const RoadSegment = require('../models/RoadSegment');
const mapMatching = require('../services/mapMatching');

async function run() {
    // Connect to database
    await connectDB();

    console.log('Fetching all road segments...');
    const segments = await RoadSegment.find({});
    console.log(`Found ${segments.length} segments in the database.`);

    let updatedCount = 0;
    let failedCount = 0;

    for (let i = 0; i < segments.length; i++) {
        const seg = segments[i];
        const roadSegmentId = seg.roadSegmentId;
        const name = seg.name || 'Unknown Road';

        let lat = 0;
        let lng = 0;

        if (seg.centerPoint && seg.centerPoint.length === 2) {
            lng = seg.centerPoint[0];
            lat = seg.centerPoint[1];
        } else if (seg.geometry && seg.geometry.coordinates && seg.geometry.coordinates.length > 0) {
            // Fallback: take first coordinate
            lng = seg.geometry.coordinates[0][0];
            lat = seg.geometry.coordinates[0][1];
        } else {
            console.warn(`[${i+1}/${segments.length}] Skipping segment ${roadSegmentId} (${name}): No location coordinates found`);
            failedCount++;
            continue;
        }

        console.log(`[${i+1}/${segments.length}] Snapping segment ${roadSegmentId} (${name}) at [${lng}, ${lat}]...`);

        try {
            const matched = await mapMatching.matchPoint(lat, lng);
            if (matched && matched.direction) {
                // Update coordinates along the matched direction
                const dx = matched.direction[0];
                const dy = matched.direction[1];

                const coordinates = [
                    [matched.matchedLongitude - 0.00045 * dx, matched.matchedLatitude - 0.00045 * dy],
                    [matched.matchedLongitude + 0.00045 * dx, matched.matchedLatitude + 0.00045 * dy]
                ];

                await RoadSegment.findByIdAndUpdate(seg._id, {
                    $set: {
                        'geometry.coordinates': coordinates,
                        centerPoint: [matched.matchedLongitude, matched.matchedLatitude]
                    }
                });

                console.log(` -> Successfully updated segment direction to [${dx.toFixed(4)}, ${dy.toFixed(4)}]`);
                updatedCount++;
            } else {
                console.warn(` -> Map matching failed to return direction for segment ${roadSegmentId}`);
                failedCount++;
            }
        } catch (err) {
            console.error(` -> Error updating segment ${roadSegmentId}:`, err.message);
            failedCount++;
        }

        // Small delay to be polite to public OSRM API
        await new Promise(resolve => setTimeout(resolve, 300));
    }

    console.log(`\nMigration complete. Updated: ${updatedCount}, Failed/Skipped: ${failedCount}`);
    await mongoose.connection.close();
    process.exit(0);
}

run().catch(err => {
    console.error('Migration failed:', err);
    process.exit(1);
});
