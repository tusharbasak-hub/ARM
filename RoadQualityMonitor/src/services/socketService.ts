import { io, Socket } from 'socket.io-client';

export interface RoadSegmentUpdate {
  roadSegmentId: string;
  iriCategory:   'green' | 'yellow' | 'orange';
  averageIri:    number;
  sampleCount:   number;
  polyline:      { type: string; coordinates: number[][] };
  name?:         string;
  updatedAt?:    string;
}

export interface MapPoint {
  type:              string;
  location:          { lat: number; lng: number };
  iriScore:          number;
  hasPothole:        boolean;
  potholeConfidence: number;
  timestamp:         string;
}

type Listener<T> = (data: T) => void;

class SocketService {
  private socket: Socket | null = null;
  private _connected = false;

  // typed listener maps
  private segmentListeners:     Listener<RoadSegmentUpdate>[]  = [];
  private initialSegListeners:  Listener<RoadSegmentUpdate[]>[] = [];
  private mapPointListeners:    Listener<MapPoint>[]            = [];
  private connectListeners:     Listener<void>[]                = [];
  private disconnectListeners:  Listener<void>[]                = [];

  get connected() { return this._connected; }

  connect(url: string, token?: string) {
    if (this.socket) this.socket.disconnect();

    this.socket = io(url, {
      transports:       ['websocket', 'polling'],
      auth:             token ? { token } : {},
      reconnection:     true,
      reconnectionDelay: 2000,
      timeout:          10000,
    });

    this.socket.on('connect', () => {
      this._connected = true;
      console.log('[socket] connected:', this.socket?.id);
      this.connectListeners.forEach(fn => fn());
    });

    this.socket.on('disconnect', () => {
      this._connected = false;
      console.log('[socket] disconnected');
      this.disconnectListeners.forEach(fn => fn());
    });

    // Bulk initial state — sent once on connection
    this.socket.on('initial-segments', (data: { segments: RoadSegmentUpdate[] }) => {
      this.initialSegListeners.forEach(fn => fn(data.segments));
    });

    // Live per-segment update after aggregation fires
    this.socket.on('segment-polyline-update', (data: RoadSegmentUpdate) => {
      this.segmentListeners.forEach(fn => fn(data));
    });

    // Verified pothole pin
    this.socket.on('map-point-event', (data: MapPoint) => {
      this.mapPointListeners.forEach(fn => fn(data));
    });
  }

  disconnect() {
    this.socket?.disconnect();
    this.socket = null;
    this._connected = false;
  }

  // ─── Listener registration ─────────────────────────────────────────────────
  onConnect(fn: Listener<void>)                        { this.connectListeners.push(fn); }
  onDisconnect(fn: Listener<void>)                     { this.disconnectListeners.push(fn); }
  onInitialSegments(fn: Listener<RoadSegmentUpdate[]>) { this.initialSegListeners.push(fn); }
  onSegmentUpdate(fn: Listener<RoadSegmentUpdate>)     { this.segmentListeners.push(fn); }
  onMapPoint(fn: Listener<MapPoint>)                   { this.mapPointListeners.push(fn); }

  removeAll() {
    this.segmentListeners      = [];
    this.initialSegListeners   = [];
    this.mapPointListeners     = [];
    this.connectListeners      = [];
    this.disconnectListeners   = [];
  }
}

export const socketService = new SocketService();
