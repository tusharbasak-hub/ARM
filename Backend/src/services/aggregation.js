const RoadSegment = require('../models/RoadSegment');
const Observation = require('../models/Observation');

/**
 * Service for aggregating road quality data
 */
class AggregationService {
    constructor() {
        this.TIME_DECAY_HOURS = parseInt(process.env.TIME_DECAY_HOURS) || 24;
        this.MIN_OBSERVATIONS = parseInt(process.env.MIN_OBSERVATIONS_FOR_AGGREGATION) || 3;
    }

    /**
     * Aggregate observations for a road segment
     * @param {string} roadSegmentId 
     * @returns {Object} Aggregated quality data
     */
    async aggregateRoadSegment(roadSegmentId) {
        try {
            // Get road segment first to check for patch data
            const roadSegment = await RoadSegment.findOne({ roadSegmentId });

            if (!roadSegment) {
                return null;
            }

            // Check if segment has patch-based data
            const totalPatchLength = (roadSegment.len1 || 0) + (roadSegment.len2 || 0) + (roadSegment.len3 || 0);
            const hasPatchData = totalPatchLength >= 5; // At least 5m of patch data

            let aggregatedScore;
            let confidenceScore;
            let distribution = null;
            let scoringMethod;

            if (hasPatchData) {
                // Use patch-based scoring
                const segmentLength = roadSegment.segmentLength || 100;
                const len1 = roadSegment.len1 || 0;
                const len2 = roadSegment.len2 || 0;
                const len3 = roadSegment.len3 || 0;

                // Calculate patch-based score: weighted average of severity
                const patchScore = (len1 * 1 + len2 * 2 + len3 * 3) / segmentLength;
                aggregatedScore = Math.min(Math.max(patchScore, 0), 3); // Clamp to 0-3

                // Confidence based on coverage ratio and observation count
                const coverageRatio = Math.min(totalPatchLength / segmentLength, 1.0);
                const observationConfidence = Math.min((roadSegment.observationCount || 0) / 10, 1.0);
                confidenceScore = coverageRatio * 0.6 + observationConfidence * 0.4;

                scoringMethod = 'patch-based';
            } else {
                // Fallback to observation-based scoring
                const cutoffTime = new Date(Date.now() - this.TIME_DECAY_HOURS * 60 * 60 * 1000);

                const observations = await Observation.find({
                    roadSegmentId,
                    timestamp: { $gte: cutoffTime },
                    matchingConfidence: { $gte: 0.5 }
                }).sort({ timestamp: -1 });

                if (observations.length < this.MIN_OBSERVATIONS) {
                    return null; // Not enough data
                }

                aggregatedScore = this.calculateWeightedScore(observations);
                confidenceScore = this.calculateConfidenceScore(observations);
                distribution = this.calculateDistribution(observations);
                scoringMethod = 'observation-based';
            }

            // Update road segment
            roadSegment.aggregatedQualityScore = aggregatedScore;
            roadSegment.confidenceScore = confidenceScore;
            if (distribution) {
                roadSegment.qualityDistribution = distribution;
            }
            roadSegment.lastUpdated = new Date();

            await roadSegment.save();

            return {
                roadSegmentId,
                aggregatedQualityScore: aggregatedScore,
                confidenceScore,
                observationCount: roadSegment.observationCount || 0,
                distribution,
                scoringMethod
            };
        } catch (error) {
            console.error('Aggregation error:', error);
            throw error;
        }
    }

    /**
     * Calculate weighted quality score with time decay
     */
    calculateWeightedScore(observations) {
        const now = Date.now();
        let weightedSum = 0;
        let totalWeight = 0;

        observations.forEach(obs => {
            const ageMs = now - new Date(obs.timestamp).getTime();
            const ageHours = ageMs / (60 * 60 * 1000);

            // Exponential time decay
            const timeWeight = Math.exp(-ageHours / this.TIME_DECAY_HOURS);

            // Speed-based weight (lower speed = more accurate observation)
            let speedWeight = 1.0;
            if (obs.speed < 5) speedWeight = 1.2;
            else if (obs.speed > 20) speedWeight = 0.7;

            // Matching confidence weight
            const matchWeight = obs.matchingConfidence || 0.8;

            const weight = timeWeight * speedWeight * matchWeight;

            weightedSum += obs.roadQuality * weight;
            totalWeight += weight;
        });

        return totalWeight > 0 ? weightedSum / totalWeight : 0;
    }

    /**
     * Calculate confidence score based on data quality
     */
    calculateConfidenceScore(observations) {
        const n = observations.length;

        // More observations = higher confidence
        let sampleConfidence = Math.min(n / 20, 1.0);

        // Calculate variance (lower variance = higher confidence)
        const mean = observations.reduce((sum, obs) => sum + obs.roadQuality, 0) / n;
        const variance = observations.reduce((sum, obs) =>
            sum + Math.pow(obs.roadQuality - mean, 2), 0) / n;
        const varianceConfidence = Math.max(0, 1 - variance / 2);

        // Recency (more recent data = higher confidence)
        const avgAge = observations.reduce((sum, obs) =>
            sum + (Date.now() - new Date(obs.timestamp).getTime()), 0) / n;
        const avgAgeHours = avgAge / (60 * 60 * 1000);
        const recencyConfidence = Math.exp(-avgAgeHours / (this.TIME_DECAY_HOURS * 2));

        // Combined confidence
        return (sampleConfidence * 0.4 + varianceConfidence * 0.3 + recencyConfidence * 0.3);
    }

    /**
     * Calculate quality distribution
     */
    calculateDistribution(observations) {
        const dist = { excellent: 0, good: 0, bad: 0, worst: 0 };

        observations.forEach(obs => {
            switch (obs.roadQuality) {
                case 0: dist.excellent++; break;
                case 1: dist.good++; break;
                case 2: dist.bad++; break;
                case 3: dist.worst++; break;
            }
        });

        return dist;
    }

    /**
     * Check if aggregation should trigger update broadcast
     */
    shouldBroadcastUpdate(oldScore, newScore, confidenceScore) {
        // Only broadcast if:
        // 1. Confidence is high enough
        if (confidenceScore < 0.5) return false;

        // 2. Score changed significantly (> 0.3 on 0-3 scale)
        if (oldScore !== null && Math.abs(newScore - oldScore) < 0.3) return false;

        return true;
    }
}

module.exports = new AggregationService();
