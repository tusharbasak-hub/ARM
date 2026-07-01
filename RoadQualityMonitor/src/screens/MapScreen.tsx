import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  ActivityIndicator, Platform, PermissionsAndroid, StatusBar,
} from 'react-native';
import MapboxGL from '@rnmapbox/maps';
import BackgroundActions from 'react-native-background-actions';
import { ENV, COLORS, IRI_COLORS, IRI_LABELS, QUALITY_COLORS, QUALITY_LABELS, RoadQuality, IriCategory } from '../config/env';
import { socketService, RoadSegmentUpdate, MapPoint } from '../services/socketService';
import { sensorService } from '../services/sensorService';
import { windowManager } from '../services/windowService';
import { mlService } from '../services/mlService';
import { observationService } from '../services/observationService';
import { authService } from '../services/authService';

MapboxGL.setAccessToken(ENV.MAPBOX_TOKEN);

// ─── Types ────────────────────────────────────────────────────────────────────
interface SegmentMap { [id: string]: RoadSegmentUpdate }

// ─── GeoJSON helpers ──────────────────────────────────────────────────────────
function buildSegmentGeoJSON(segments: SegmentMap) {
  const features = Object.values(segments)
    .filter(s => s.polyline?.coordinates?.length >= 2)
    .map(s => ({
      type: 'Feature' as const,
      id:   s.roadSegmentId,
      properties: {
        iriCategory: s.iriCategory,
        color:       IRI_COLORS[s.iriCategory as IriCategory] ?? '#9E9E9E',
        iriScore:    s.averageIri,
        name:        s.name ?? '',
      },
      geometry: {
        type:        'LineString' as const,
        coordinates: s.polyline.coordinates,
      },
    }));

  return { type: 'FeatureCollection' as const, features };
}

function buildPotholeGeoJSON(points: MapPoint[]) {
  const features = points.map((p, i) => ({
    type: 'Feature' as const,
    id:   `pothole_${i}`,
    properties: { iriScore: p.iriScore },
    geometry: {
      type:        'Point' as const,
      coordinates: [p.location.lng, p.location.lat],
    },
  }));
  return { type: 'FeatureCollection' as const, features };
}

// ─── Component ────────────────────────────────────────────────────────────────
export const MapScreen = ({ navigation }: any) => {
  const [segments,        setSegments]        = useState<SegmentMap>({});
  const [potholes,        setPotholes]        = useState<MapPoint[]>([]);
  const [monitoring,      setMonitoring]      = useState(false);
  const [currentQuality,  setCurrentQuality]  = useState<RoadQuality | null>(null);
  const [socketConnected, setSocketConnected] = useState(false);
  const [mlReady,         setMlReady]         = useState(false);
  const [loading,         setLoading]         = useState(true);

  const cameraRef = useRef<MapboxGL.Camera>(null);
  const sessionIdRef = useRef<string | null>(null);

  // ─── Permissions ────────────────────────────────────────────────────────────
  const requestPerms = useCallback(async () => {
    if (Platform.OS === 'android') {
      const permissions = [
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
        PermissionsAndroid.PERMISSIONS.ACCESS_COARSE_LOCATION,
      ];

      // Request notification permission on Android 13+ (API 33+)
      if (Platform.Version >= 33) {
        permissions.push(PermissionsAndroid.PERMISSIONS.POST_NOTIFICATIONS);
      }

      const granted = await PermissionsAndroid.requestMultiple(permissions);

      const fineGranted = granted[PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION] === PermissionsAndroid.RESULTS.GRANTED;
      const coarseGranted = granted[PermissionsAndroid.PERMISSIONS.ACCESS_COARSE_LOCATION] === PermissionsAndroid.RESULTS.GRANTED;

      // Ask for background permission on Android 10+ (API 29+)
      if ((fineGranted || coarseGranted) && Platform.Version >= 29) {
        try {
          await PermissionsAndroid.request(
            PermissionsAndroid.PERMISSIONS.ACCESS_BACKGROUND_LOCATION,
            {
              title: 'Background Location Permission',
              message:
                'This app collects location data to monitor road quality even when the app is in the background or screen is off.',
              buttonNeutral: 'Ask Me Later',
              buttonNegative: 'Cancel',
              buttonPositive: 'OK',
            }
          );
        } catch (err) {
          console.warn('[Permissions] Background location permission request error:', err);
        }
      }
    }
  }, []);

  // ─── Init ────────────────────────────────────────────────────────────────────
  useEffect(() => {
    let mounted = true;

    const init = async () => {
      await requestPerms();

      // 1. Get token and connect socket
      const token = await authService.getToken();
      socketService.connect(ENV.API.BASE_URL, token ?? undefined);

      socketService.onConnect(() => {
        if (mounted) setSocketConnected(true);
      });
      socketService.onDisconnect(() => {
        if (mounted) setSocketConnected(false);
      });

      // 2. Load all segments when socket first connects
      socketService.onInitialSegments((segs) => {
        if (!mounted) return;
        const map: SegmentMap = {};
        segs.forEach(s => { map[s.roadSegmentId] = s; });
        setSegments(map);
        setLoading(false);
      });

      // 3. Live segment updates
      socketService.onSegmentUpdate((seg) => {
        if (!mounted) return;
        setSegments(prev => ({ ...prev, [seg.roadSegmentId]: seg }));
      });

      // 4. Pothole pins
      socketService.onMapPoint((pt) => {
        if (!mounted) return;
        setPotholes(prev => [pt, ...prev].slice(0, 200));
      });

      // 5. Init ML model
      try {
        await mlService.initialize();
        if (mounted) setMlReady(true);
      } catch (e) {
        console.warn('[ML] init failed, using mock mode:', e);
        if (mounted) setMlReady(true); // allow monitoring with mock
      }

      // Fallback: stop loading spinner after 8 seconds even if no segments
      setTimeout(() => { if (mounted) setLoading(false); }, 8000);
    };

    init();

    return () => {
      mounted = false;
      socketService.removeAll();
      sensorService.stopCollection();
      BackgroundActions.stop().catch(() => {});
    };
  }, []);

  // ─── Window → ML → Backend pipeline ─────────────────────────────────────────
  const startMonitoring = useCallback(async () => {
    // Generate a unique session ID
    sessionIdRef.current = 'sess_' + Date.now() + '_' + Math.random().toString(36).substring(2, 11);

    // 1. Ensure background location permission is requested/granted on Android 10+
    if (Platform.OS === 'android' && Platform.Version >= 29) {
      const hasBackgroundPerm = await PermissionsAndroid.check(
        PermissionsAndroid.PERMISSIONS.ACCESS_BACKGROUND_LOCATION
      );
      if (!hasBackgroundPerm) {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.ACCESS_BACKGROUND_LOCATION,
          {
            title: 'Background Location Permission',
            message:
              'To record road quality when the screen is off, please select "Allow all the time" in location settings.',
            buttonPositive: 'OK',
          }
        );
        if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
          console.warn('Background location permission not granted, tracking may fail when screen is off.');
        }
      }
    }

    const options = {
      taskName: 'RoadQualityMonitoring',
      taskTitle: 'Monitoring Road Quality',
      taskDesc: 'Sensing and analyzing road vibrations in the background...',
      taskIcon: {
        name: 'ic_launcher',
        type: 'mipmap',
      },
      color: COLORS.primary,
      parameters: {
        delay: 1000,
      },
    };

    const monitorBgTask = async (taskData?: any) => {
      await new Promise<void>(async (resolve) => {
        windowManager.init(async (windowData) => {
          // Run ML inference
          const prediction = await mlService.predict(windowData);
          const quality = (prediction ?? 0) as RoadQuality;
          setCurrentQuality(quality);

          const last = windowData[windowData.length - 1];
          const { latitude, longitude } = last.location;
          if (latitude === 0 && longitude === 0) return;

          // Map quality class to IRI score (rough mapping for backend)
          const iriScore = [0.8, 1.8, 3.0, 5.0][quality];
          const hasPothole = quality >= 3;

          await observationService.submit({
            latitude,
            longitude,
            iriScore,
            hasPothole,
            potholeConfidence: hasPothole ? 0.9 : 0,
            speed: last.speed,
            heading: last.location.heading !== -1 ? last.location.heading : undefined,
            sessionId: sessionIdRef.current ?? undefined,
          });
        });

        sensorService.startCollection((reading) => {
          windowManager.add(reading);
        });

        // Loop to keep background actions alive
        while (BackgroundActions.isRunning()) {
          await new Promise((r) => setTimeout(r, 1000));
        }

        resolve();
      });
    };

    try {
      await BackgroundActions.start(monitorBgTask, options);
      setMonitoring(true);
    } catch (e) {
      console.error('[BackgroundService] Failed to start:', e);
      // Fallback to normal foreground monitoring if background actions fails
      windowManager.init(async (windowData) => {
        const prediction = await mlService.predict(windowData);
        const quality = (prediction ?? 0) as RoadQuality;
        setCurrentQuality(quality);

        const last = windowData[windowData.length - 1];
        const { latitude, longitude } = last.location;
        if (latitude === 0 && longitude === 0) return;

        const iriScore = [0.8, 1.8, 3.0, 5.0][quality];
        const hasPothole = quality >= 3;

        await observationService.submit({
          latitude,
          longitude,
          iriScore,
          hasPothole,
          potholeConfidence: hasPothole ? 0.9 : 0,
          speed: last.speed,
          heading: last.location.heading !== -1 ? last.location.heading : undefined,
          sessionId: sessionIdRef.current ?? undefined,
        });
      });

      sensorService.startCollection((reading) => {
        windowManager.add(reading);
      });
      setMonitoring(true);
    }
  }, []);

  const stopMonitoring = useCallback(async () => {
    sensorService.stopCollection();
    windowManager.reset();
    try {
      await BackgroundActions.stop();
    } catch (e) {
      console.warn('[BackgroundService] Failed to stop:', e);
    }
    setMonitoring(false);
    setCurrentQuality(null);
    sessionIdRef.current = null;
  }, []);

  const toggleMonitoring = () => {
    monitoring ? stopMonitoring() : startMonitoring();
  };

  // ─── GeoJSON sources ──────────────────────────────────────────────────────────
  const segmentGeoJSON  = buildSegmentGeoJSON(segments);
  const potholeGeoJSON  = buildPotholeGeoJSON(potholes);
  const segmentCount    = Object.keys(segments).length;

  // ─── Render ──────────────────────────────────────────────────────────────────
  return (
    <View style={styles.root}>
      <StatusBar barStyle="light-content" backgroundColor="transparent" translucent />

      {/* ── Map ── */}
      <MapboxGL.MapView
        style={styles.map}
        styleURL="mapbox://styles/mapbox/dark-v11"
        logoEnabled={false}
        attributionEnabled={false}
        compassEnabled
        compassViewPosition={3}
      >
        <MapboxGL.Camera
          ref={cameraRef}
          zoomLevel={ENV.MAP.DEFAULT_ZOOM}
          centerCoordinate={[ENV.MAP.DEFAULT_CENTER.longitude, ENV.MAP.DEFAULT_CENTER.latitude]}
          followUserLocation={monitoring}
          followZoomLevel={15}
          animationMode="flyTo"
          animationDuration={1000}
        />

        <MapboxGL.UserLocation visible androidRenderMode="compass" />

        {/* Road segment polylines */}
        {segmentCount > 0 && (
          <MapboxGL.ShapeSource id="segments" shape={segmentGeoJSON} tolerance={0.5}>
            {/* Glow / halo layer */}
            <MapboxGL.LineLayer
              id="segments-glow"
              style={{
                lineColor:   ['get', 'color'],
                lineWidth:   8,
                lineOpacity: 0.15,
                lineBlur:    4,
              }}
            />
            {/* Main coloured polyline */}
            <MapboxGL.LineLayer
              id="segments-line"
              style={{
                lineColor:    ['get', 'color'],
                lineWidth:    4,
                lineOpacity:  0.9,
                lineCap:      'round',
                lineJoin:     'round',
              }}
            />
          </MapboxGL.ShapeSource>
        )}

        {/* Pothole markers */}
        {potholes.length > 0 && (
          <MapboxGL.ShapeSource id="potholes" shape={potholeGeoJSON}>
            <MapboxGL.CircleLayer
              id="pothole-circles"
              style={{
                circleColor:       '#F44336',
                circleRadius:      8,
                circleOpacity:     0.9,
                circleStrokeColor: '#FFFFFF',
                circleStrokeWidth: 2,
              }}
            />
          </MapboxGL.ShapeSource>
        )}
      </MapboxGL.MapView>

      {/* ── Top status bar ── */}
      <View style={styles.topBar}>
        <View style={styles.topBarLeft}>
          <View style={[styles.dot, { backgroundColor: socketConnected ? COLORS.primary : COLORS.danger }]} />
          <Text style={styles.topBarText}>
            {socketConnected ? `${segmentCount} segments` : 'Connecting…'}
          </Text>
        </View>
        <TouchableOpacity
          style={styles.logoutBtn}
          onPress={async () => {
            stopMonitoring();
            await authService.logout();
            navigation.replace('Login');
          }}
        >
          <Text style={styles.logoutText}>Exit</Text>
        </TouchableOpacity>
      </View>

      {/* ── Loading overlay ── */}
      {loading && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color={COLORS.primary} />
          <Text style={styles.loadingText}>Loading road data…</Text>
        </View>
      )}

      {/* ── Legend ── */}
      <View style={styles.legend}>
        {(['green', 'yellow', 'orange'] as IriCategory[]).map(cat => (
          <View key={cat} style={styles.legendRow}>
            <View style={[styles.legendDot, { backgroundColor: IRI_COLORS[cat] }]} />
            <Text style={styles.legendText}>{IRI_LABELS[cat]}</Text>
          </View>
        ))}
        <View style={styles.legendRow}>
          <View style={[styles.legendDot, { backgroundColor: '#F44336' }]} />
          <Text style={styles.legendText}>Pothole</Text>
        </View>
      </View>

      {/* ── Bottom panel ── */}
      <View style={styles.bottomPanel}>

        {/* Current quality badge — visible while monitoring */}
        {monitoring && currentQuality !== null && (
          <View style={[styles.qualityBadge, { borderColor: QUALITY_COLORS[currentQuality] }]}>
            <View style={[styles.qualityDot, { backgroundColor: QUALITY_COLORS[currentQuality] }]} />
            <Text style={styles.qualityLabel}>
              {QUALITY_LABELS[currentQuality]}
            </Text>
          </View>
        )}

        {/* ML / sensor status pill */}
        {monitoring && (
          <View style={styles.statusPill}>
            <ActivityIndicator size="small" color={COLORS.primary} />
            <Text style={styles.statusText}>Sensing road quality…</Text>
          </View>
        )}

        {/* Main monitoring toggle */}
        <TouchableOpacity
          style={[
            styles.monitorBtn,
            monitoring ? styles.monitorBtnStop : styles.monitorBtnStart,
            !mlReady && !monitoring && styles.btnDisabled,
          ]}
          onPress={toggleMonitoring}
          disabled={!mlReady && !monitoring}
          activeOpacity={0.85}
        >
          <Text style={styles.monitorBtnIcon}>{monitoring ? '⏹' : '▶'}</Text>
          <Text style={styles.monitorBtnText}>
            {monitoring ? 'Stop Monitoring' : 'Start Monitoring'}
          </Text>
        </TouchableOpacity>

      </View>
    </View>
  );
};

// ─── Styles ───────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: COLORS.bg },
  map:  { ...StyleSheet.absoluteFillObject },

  // Top bar
  topBar: {
    position:        'absolute',
    top:             48,
    left:            16,
    right:           16,
    flexDirection:   'row',
    justifyContent:  'space-between',
    alignItems:      'center',
    backgroundColor: 'rgba(13,13,13,0.85)',
    borderRadius:    12,
    paddingHorizontal: 14,
    paddingVertical:   10,
    borderWidth:     1,
    borderColor:     COLORS.border,
  },
  topBarLeft:  { flexDirection: 'row', alignItems: 'center' },
  dot:         { width: 8, height: 8, borderRadius: 4, marginRight: 8 },
  topBarText:  { color: COLORS.textSecondary, fontSize: 13, fontWeight: '600' },
  logoutBtn:   { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 8, backgroundColor: COLORS.surfaceLight },
  logoutText:  { color: COLORS.textSecondary, fontSize: 13, fontWeight: '600' },

  // Loading overlay
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(13,13,13,0.7)',
    justifyContent:  'center',
    alignItems:      'center',
  },
  loadingText: { color: COLORS.text, marginTop: 12, fontSize: 14 },

  // Legend
  legend: {
    position:        'absolute',
    top:             110,
    right:           16,
    backgroundColor: 'rgba(13,13,13,0.85)',
    borderRadius:    12,
    padding:         12,
    borderWidth:     1,
    borderColor:     COLORS.border,
    gap:             8,
  },
  legendRow: { flexDirection: 'row', alignItems: 'center' },
  legendDot: { width: 10, height: 10, borderRadius: 5, marginRight: 8 },
  legendText:{ color: COLORS.textSecondary, fontSize: 12, fontWeight: '500' },

  // Bottom panel
  bottomPanel: {
    position:      'absolute',
    bottom:        0,
    left:          0,
    right:         0,
    backgroundColor: 'rgba(13,13,13,0.92)',
    borderTopWidth:  1,
    borderTopColor:  COLORS.border,
    padding:         20,
    paddingBottom:   32,
    gap:             12,
  },

  // Quality badge
  qualityBadge: {
    flexDirection:    'row',
    alignItems:       'center',
    alignSelf:        'center',
    paddingHorizontal: 16,
    paddingVertical:   10,
    borderRadius:      24,
    borderWidth:       1.5,
    backgroundColor:   COLORS.surfaceLight,
  },
  qualityDot:   { width: 10, height: 10, borderRadius: 5, marginRight: 8 },
  qualityLabel: { color: COLORS.text, fontWeight: '700', fontSize: 15 },

  // Status pill
  statusPill: {
    flexDirection:    'row',
    alignItems:       'center',
    alignSelf:        'center',
    gap:              8,
    paddingHorizontal: 14,
    paddingVertical:   6,
    backgroundColor:   COLORS.surfaceLight,
    borderRadius:      20,
  },
  statusText: { color: COLORS.textSecondary, fontSize: 12 },

  // Monitor button
  monitorBtn: {
    flexDirection:  'row',
    alignItems:     'center',
    justifyContent: 'center',
    borderRadius:   14,
    padding:        16,
    gap:            10,
  },
  monitorBtnStart: { backgroundColor: COLORS.primary },
  monitorBtnStop:  { backgroundColor: COLORS.danger },
  monitorBtnIcon:  { fontSize: 18 },
  monitorBtnText:  { color: '#fff', fontWeight: '700', fontSize: 16 },
  btnDisabled:     { opacity: 0.45 },
});
