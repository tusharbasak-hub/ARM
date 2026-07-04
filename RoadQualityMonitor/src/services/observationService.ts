import { ENV } from '../config/env';
import { authService } from './authService';

export interface ObservationPayload {
  latitude:          number;
  longitude:         number;
  iriScore:          number;
  hasPothole:        boolean;
  potholeConfidence: number;
  deviceId?:         string;
  recordedAt?:       string;
  speed?:            number;
  heading?:          number;
  sessionId?:        string;
}

export const observationService = {
  async submit(payload: ObservationPayload): Promise<void> {
    const token    = await authService.getToken();
    const deviceId = await authService.getDeviceId();

    const body: ObservationPayload = {
      ...payload,
      deviceId:    deviceId,
      recordedAt:  new Date().toISOString(),
    };

    const res = await fetch(`${ENV.API.BASE_URL}/api/observations`, {
      method:  'POST',
      headers: {
        'Content-Type':  'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      // Don't throw — we don't want one failed observation to break the monitoring loop
      console.warn('[observation] submit failed:', err.error ?? res.status);
    }
  },

  async getHistory(): Promise<any[]> {
    const token = await authService.getToken();
    const res = await fetch(`${ENV.API.BASE_URL}/api/observations/history`, {
      headers: { 'Authorization': `Bearer ${token}` },
    });
    if (!res.ok) return [];
    const data = await res.json();
    return data.data ?? [];
  },
};
