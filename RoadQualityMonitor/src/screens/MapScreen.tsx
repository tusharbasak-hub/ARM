import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  ActivityIndicator, Platform, PermissionsAndroid, StatusBar,
  Modal, useColorScheme,
} from 'react-native';
import MapboxGL from '@rnmapbox/maps';
import BackgroundActions from 'react-native-background-actions';
import { ENV, COLORS, IRI_COLORS, QUALITY_COLORS, QUALITY_LABELS, RoadQuality, IriCategory } from '../config/env';
import { socketService, RoadSegmentUpdate, MapPoint } from '../services/socketService';
import { sensorService } from '../services/sensorService';
import { windowManager, SensorReading } from '../services/windowService';
import { mlService } from '../services/mlService';
import { observationService } from '../services/observationService';
import { authService } from '../services/authService';

MapboxGL.setAccessToken(ENV.MAPBOX_TOKEN);

// ─── Types ────────────────────────────────────────────────────────────────────
interface SegmentMap { [id: string]: RoadSegmentUpdate }

// ─── GeoJSON builders ──────────────────────────────────────────────────────────
function buildIriGeoJSON(segments: SegmentMap, liveIriPoints: any[]) {
  const features: any[] = [];

  // 1. Historical segments represented as dots at their center points
  Object.values(segments).forEach(s => {
    const coords = s.centerPoint || (s.polyline?.coordinates?.length > 0 ? s.polyline.coordinates[0] : null);
    if (coords) {
      features.push({
        type: 'Feature' as const,
        id: `seg_${s.roadSegmentId}`,
        properties: {
          iriScore: s.averageIri,
        },
        geometry: {
          type: 'Point' as const,
          coordinates: coords,
        },
      });
    }
  });

  // 2. Live IRI predictions (every 100m)
  liveIriPoints.forEach((p, i) => {
    features.push({
      type: 'Feature' as const,
      id: `live_iri_${i}`,
      properties: {
        iriScore: p.iriScore,
      },
      geometry: {
        type: 'Point' as const,
        coordinates: [p.longitude, p.latitude],
      },
    });
  });

  return { type: 'FeatureCollection' as const, features };
}

function buildPotholeGeoJSON(potholes: MapPoint[], livePotholes: any[], isDarkMode: boolean) {
  const features: any[] = [];

  const getPotholeColor = (score: number) => {
    if (score >= 4.0) {
      return isDarkMode ? '#FF0033' : '#800000'; // Neon red vs Maroon
    } else if (score >= 2.5) {
      return '#FF9800'; // Orange (Med Pothole)
    } else {
      return '#FFC107'; // Yellow (Patches)
    }
  };

  // 1. Historical Potholes
  potholes.forEach((p, i) => {
    if (p.hasPothole) {
      features.push({
        type: 'Feature' as const,
        id: `pothole_${i}`,
        properties: {
          iriScore: p.iriScore,
          color: getPotholeColor(p.iriScore),
        },
        geometry: {
          type: 'Point' as const,
          coordinates: [p.location.lng, p.location.lat],
        },
      });
    }
  });

  // 2. Live Potholes
  livePotholes.forEach((p, i) => {
    features.push({
      type: 'Feature' as const,
      id: `live_pothole_${i}`,
      properties: {
        iriScore: p.iriScore,
        color: getPotholeColor(p.iriScore),
      },
      geometry: {
        type: 'Point' as const,
        coordinates: [p.longitude, p.latitude],
      },
    });
  });

  return { type: 'FeatureCollection' as const, features };
}

// ─── Component ────────────────────────────────────────────────────────────────
export const MapScreen = ({ navigation }: any) => {
  const systemTheme = useColorScheme();
  const isDarkMode = systemTheme === 'dark';

  const [segments,          setSegments]          = useState<SegmentMap>({});
  const [potholes,          setPotholes]          = useState<MapPoint[]>([]);
  const [liveIriPoints,     setLiveIriPoints]     = useState<any[]>([]);
  const [livePotholePoints, setLivePotholePoints] = useState<any[]>([]);

  const [monitoring,        setMonitoring]        = useState(false);
  const [currentQuality,    setCurrentQuality]    = useState<RoadQuality | null>(null);
  const [socketConnected,   setSocketConnected]   = useState(false);
  const [mlReady,           setMlReady]           = useState(false);
  const [loading,           setLoading]           = useState(true);

  // Vehicle selection modal state
  const [showVehicleModal,  setShowVehicleModal]  = useState(false);
  const [vehicleId,         setVehicleId]         = useState<number>(0); // SUV=0, Hatchback=1, Sedan=2

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

      // 4. Live map points (Potholes)
      socketService.onMapPoint((pt) => {
        if (!mounted) return;
        setPotholes(prev => [pt, ...prev].slice(0, 500));
      });

      // 5. Initial map points (Potholes history on load)
      socketService.onInitialMapPoints((pts) => {
        if (!mounted) return;
        setPotholes(pts);
      });

      // 6. Init ML model
      try {
        await mlService.initialize();
        if (mounted) setMlReady(true);
      } catch (e) {
        console.warn('[ML] init failed, using mock mode:', e);
        if (mounted) setMlReady(true);
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

  // ─── Dual-Model ML Inference Session Start ──────────────────────────────────
  const startMonitoring = useCallback(async (selectedVehId: number) => {
    // Generate a unique session ID
    sessionIdRef.current = 'sess_' + Date.now() + '_' + Math.random().toString(36).substring(2, 11);

    // Ensure background location permission is requested/granted on Android 10+
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

    // Shared callbacks for model prediction results
    const onIriReady = async (iriScore: number, lastReading: SensorReading) => {
      const { latitude, longitude, heading } = lastReading.location;
      if (latitude === 0 && longitude === 0) return;

      // Add to live IRI points (plotted in background layer)
      setLiveIriPoints(prev => [...prev, { latitude, longitude, iriScore }]);

      // Submit observation to backend
      await observationService.submit({
        latitude,
        longitude,
        iriScore,
        hasPothole: false,
        potholeConfidence: 0,
        speed: lastReading.speed,
        heading: heading !== -1 ? heading : undefined,
        sessionId: sessionIdRef.current ?? undefined,
      });
    };

    const onPotholeReady = async (potholeClass: number, lastReading: SensorReading) => {
      const { latitude, longitude, heading } = lastReading.location;
      if (latitude === 0 && longitude === 0) return;

      // Update current quality indicator on screen
      setCurrentQuality(potholeClass as RoadQuality);

      // Only overlay hazard classes on map (Patches, Med Pothole, Big Pothole)
      if (potholeClass >= 1) {
        const mappedIriScore = [0.8, 1.8, 3.0, 5.0][potholeClass];
        
        // Add to live potholes list
        setLivePotholePoints(prev => [...prev, {
          latitude,
          longitude,
          iriScore: mappedIriScore,
        }]);

        // Submit observation to backend
        await observationService.submit({
          latitude,
          longitude,
          iriScore: mappedIriScore,
          hasPothole: true,
          potholeConfidence: 0.9,
          speed: lastReading.speed,
          heading: heading !== -1 ? heading : undefined,
          sessionId: sessionIdRef.current ?? undefined,
        });
      }
    };

    const monitorBgTask = async (taskData?: any) => {
      await new Promise<void>(async (resolve) => {
        // Initialize window manager with callbacks and selected vehicle
        windowManager.init(onIriReady, onPotholeReady, selectedVehId);

        // Start 50Hz sensor collection
        sensorService.startCollection((reading) => {
          windowManager.add(reading);
        });

        // Loop to keep background actions alive
        while (BackgroundActions.isRunning()) {
          await new Promise<void>((r) => setTimeout(r, 1000));
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
      windowManager.init(onIriReady, onPotholeReady, selectedVehId);
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
    setLiveIriPoints([]);
    setLivePotholePoints([]);
  }, []);

  const toggleMonitoring = () => {
    if (monitoring) {
      stopMonitoring();
    } else {
      setShowVehicleModal(true); // Prompt vehicle selection first
    }
  };

  // ─── GeoJSON sources ──────────────────────────────────────────────────────────
  const iriGeoJSON = useMemo(() => buildIriGeoJSON(segments, liveIriPoints), [segments, liveIriPoints]);
  const potholeGeoJSON = useMemo(() => buildPotholeGeoJSON(potholes, livePotholePoints, isDarkMode), [potholes, livePotholePoints, isDarkMode]);

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

        {/* 1. Background IRI Layer (Large Translucent bubbles with smooth gradient) */}
        {iriGeoJSON.features.length > 0 && (
          <MapboxGL.ShapeSource id="iri-points-source" shape={iriGeoJSON}>
            <MapboxGL.CircleLayer
              id="iri-circles"
              style={{
                circleColor: [
                  'interpolate',
                  ['linear'],
                  ['get', 'iriScore'],
                  0, '#4CAF50',   // Green
                  4, '#8BC34A',   // Light Green
                  8, '#FFC107',   // Yellow
                  12, '#FF9800',  // Orange
                  15, '#F44336'   // Red (Saturates at red above 15)
                ],
                circleRadius: [
                  'interpolate',
                  ['exponential', 1.5],
                  ['zoom'],
                  10, 3,
                  14, 8,
                  17, 18,
                  20, 35
                ],
                circleOpacity: 0.35,
              }}
            />
          </MapboxGL.ShapeSource>
        )}

        {/* 2. Foreground Pothole Layer (Small Opaque circular pins layered above IRI) */}
        {potholeGeoJSON.features.length > 0 && (
          <MapboxGL.ShapeSource id="pothole-points-source" shape={potholeGeoJSON}>
            <MapboxGL.CircleLayer
              id="pothole-circles"
              style={{
                circleColor: ['get', 'color'],
                circleRadius: [
                  'interpolate',
                  ['exponential', 1.5],
                  ['zoom'],
                  10, 1.5,
                  14, 4,
                  17, 10,
                  20, 20
                ],
                circleOpacity: 1.0,
                circleStrokeColor: '#FFFFFF',
                circleStrokeWidth: 1.5,
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
            {socketConnected ? `${iriGeoJSON.features.length} points` : 'Connecting…'}
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
        <Text style={styles.legendHeader}>IRI Scale (Road Quality)</Text>
        <View style={styles.gradientBarWrapper}>
          <View style={styles.gradientBarWrapperRow}>
            <View style={[styles.gradientBarSection, { backgroundColor: '#4CAF50' }]} />
            <View style={[styles.gradientBarSection, { backgroundColor: '#8BC34A' }]} />
            <View style={[styles.gradientBarSection, { backgroundColor: '#FFC107' }]} />
            <View style={[styles.gradientBarSection, { backgroundColor: '#FF9800' }]} />
            <View style={[styles.gradientBarSection, { backgroundColor: '#F44336' }]} />
          </View>
          <View style={styles.gradientLabels}>
            <Text style={styles.gradientLabel}>0 (Good)</Text>
            <Text style={styles.gradientLabel}>8</Text>
            <Text style={styles.gradientLabel}>15+ (Bad)</Text>
          </View>
        </View>
        
        <Text style={[styles.legendHeader, { marginTop: 10 }]}>Hazards</Text>
        <View style={styles.legendRow}>
          <View style={[styles.legendDot, { backgroundColor: '#FFC107' }]} />
          <Text style={styles.legendText}>Patches</Text>
        </View>
        <View style={styles.legendRow}>
          <View style={[styles.legendDot, { backgroundColor: '#FF9800' }]} />
          <Text style={styles.legendText}>Med Pothole</Text>
        </View>
        <View style={styles.legendRow}>
          <View style={[styles.legendDot, { backgroundColor: isDarkMode ? '#FF0033' : '#800000' }]} />
          <Text style={styles.legendText}>Big Pothole</Text>
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

      {/* ── Vehicle Selection Modal ── */}
      <Modal
        visible={showVehicleModal}
        transparent
        animationType="fade"
        onRequestClose={() => setShowVehicleModal(false)}
      >
        <View style={styles.modalBackdrop}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Select Vehicle Type</Text>
            <Text style={styles.modalSubtitle}>
              The ML models require your vehicle category to calibrate suspension and vibration parameters.
            </Text>

            <View style={styles.vehicleOptions}>
              {[
                { id: 0, label: 'SUV / Multi Utility Vehicle', desc: 'Slower response, higher clearance' },
                { id: 1, label: 'Hatchback', desc: 'Stiffer suspension, standard clearance' },
                { id: 2, label: 'Sedan', desc: 'Softer suspension, lower clearance' }
              ].map(opt => (
                <TouchableOpacity
                  key={opt.id}
                  style={[
                    styles.vehicleCard,
                    vehicleId === opt.id && styles.vehicleCardSelected
                  ]}
                  onPress={() => {
                    setVehicleId(opt.id);
                    setShowVehicleModal(false);
                    // Start monitoring immediately after selection
                    startMonitoring(opt.id);
                  }}
                  activeOpacity={0.8}
                >
                  <View style={styles.vehicleCardHeader}>
                    <Text style={styles.vehicleCardLabel}>{opt.label}</Text>
                    {vehicleId === opt.id && <Text style={styles.checkIcon}>✓</Text>}
                  </View>
                  <Text style={styles.vehicleCardDesc}>{opt.desc}</Text>
                </TouchableOpacity>
              ))}
            </View>

            <TouchableOpacity
              style={styles.cancelBtn}
              onPress={() => setShowVehicleModal(false)}
            >
              <Text style={styles.cancelBtnText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
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
    width:           170,
  },
  legendHeader: {
    color: '#FFF',
    fontSize: 10,
    fontWeight: '700',
    marginBottom: 6,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  gradientBarWrapper: {
    marginBottom: 6,
  },
  gradientBarWrapperRow: {
    flexDirection: 'row',
    height: 8,
    borderRadius: 4,
    overflow: 'hidden',
  },
  gradientBarSection: {
    flex: 1,
  },
  gradientLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 4,
  },
  gradientLabel: {
    color: COLORS.textSecondary,
    fontSize: 9,
  },
  legendRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 4 },
  legendDot: { width: 10, height: 10, borderRadius: 5, marginRight: 8 },
  legendText:{ color: COLORS.textSecondary, fontSize: 11, fontWeight: '500' },

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

  // Modal Backdrop & Content
  modalBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.75)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20
  },
  modalContent: {
    backgroundColor: COLORS.surface,
    borderRadius: 16,
    padding: 24,
    width: '100%',
    maxWidth: 340,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  modalTitle: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 8,
    textAlign: 'center',
  },
  modalSubtitle: {
    color: COLORS.textSecondary,
    fontSize: 12,
    textAlign: 'center',
    marginBottom: 20,
    lineHeight: 18,
  },
  vehicleOptions: {
    gap: 12,
    marginBottom: 20,
  },
  vehicleCard: {
    backgroundColor: COLORS.surfaceLight,
    borderWidth: 1.5,
    borderColor: COLORS.border,
    borderRadius: 12,
    padding: 16,
  },
  vehicleCardSelected: {
    borderColor: COLORS.primary,
    backgroundColor: 'rgba(0, 200, 83, 0.05)',
  },
  vehicleCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  vehicleCardLabel: {
    color: '#FFF',
    fontSize: 14,
    fontWeight: '700',
  },
  checkIcon: {
    color: COLORS.primary,
    fontSize: 14,
    fontWeight: '700',
  },
  vehicleCardDesc: {
    color: COLORS.textSecondary,
    fontSize: 11,
  },
  cancelBtn: {
    paddingVertical: 12,
    alignItems: 'center',
  },
  cancelBtnText: {
    color: COLORS.textSecondary,
    fontSize: 14,
    fontWeight: '600',
  },
});
