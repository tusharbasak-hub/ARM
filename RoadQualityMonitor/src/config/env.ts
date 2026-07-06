// ─── Environment Configuration ────────────────────────────────────────────────
// For Android physical device, use your PC's local network IP.
// Run `ipconfig` in cmd and use the IPv4 address of your Wi-Fi adapter.
// e.g. http://192.168.1.42:5000
// DO NOT use localhost — it resolves to the phone itself on Android.

export const ENV = {
  API: {
    BASE_URL: 'https://road-quality-monitor-backend.onrender.com', // Production backend hosted on Render
    TIMEOUT: 10000,
  },

  MAP: {
    // Delhi NCR centre
    DEFAULT_CENTER: {
      latitude: 28.6139,
      longitude: 77.209,
    },
    DEFAULT_ZOOM: 13,
  },

  MAPBOX_TOKEN:
    'pk.eyJ1IjoidHVzaGFyYmFzYWsiLCJhIjoiY21rcmE4ZWV4MHdjYzNnczZxMXVyMWFmbiJ9.qMgF1d8iOC4XVP_iQetlPA',

  ML: {
    IRI_MODEL_FILE: require('../../assets/ml-model/iri_background_model.tflite'),
    POTHOLE_MODEL_FILE: require('../../assets/ml-model/road_vision_final_float16.tflite'),
    MODEL_FILE: require('../../assets/ml-model/iri_background_model.tflite'), // fallback
    SCALER_FILE: require('../../assets/ml-model/scaler_params.json'),
    WINDOW_SIZE: 20,
  },

  VISUALIZATION: {
    IRI_GRADIENT_STOPS: [
      { value: 0, color: '#4CAF50' },   // Green
      { value: 4, color: '#8BC34A' },   // Light Green
      { value: 8, color: '#FFC107' },   // Yellow
      { value: 12, color: '#FF9800' },  // Orange
      { value: 15, color: '#F44336' },  // Red (saturates above 15)
    ],
    POTHOLE_COLORS: {
      patches: '#FFC107',        // Yellow
      medPothole: '#FF9800',     // Orange
      bigPotholeLight: '#800000', // Maroon
      bigPotholeDark: '#FF0033',  // Neon Red
    }
  },
} as const;

// ─── IRI Category mapping (matches backend constants) ────────────────────────
export type IriCategory = 'green' | 'yellow' | 'orange';

export const IRI_COLORS: Record<IriCategory, string> = {
  green:  '#4CAF50',
  yellow: '#FFC107',
  orange: '#FF5722',
};

export const IRI_LABELS: Record<IriCategory, string> = {
  green:  'Good',
  yellow: 'Moderate',
  orange: 'Bad',
};

// ─── Road quality class mapping (ML model output 0-3) ────────────────────────
export type RoadQuality = 0 | 1 | 2 | 3;

export const QUALITY_COLORS: Record<RoadQuality, string> = {
  0: '#4CAF50', // Excellent → green
  1: '#FFC107', // Patches   → yellow
  2: '#FF9800', // Med Pothole → orange
  3: '#F44336', // Big Pothole → red
};

export const QUALITY_LABELS: Record<RoadQuality, string> = {
  0: 'Excellent',
  1: 'Patches',
  2: 'Med Pothole',
  3: 'Big Pothole',
};

// ─── App colour palette (dark theme) ─────────────────────────────────────────
export const COLORS = {
  bg:           '#0D0D0D',
  surface:      '#1A1A1A',
  surfaceLight: '#242424',
  border:       '#2C2C2C',
  primary:      '#00C853',   // green accent
  primaryDark:  '#009624',
  text:         '#FFFFFF',
  textSecondary:'#9E9E9E',
  textMuted:    '#616161',
  danger:       '#F44336',
  warning:      '#FFC107',
};
