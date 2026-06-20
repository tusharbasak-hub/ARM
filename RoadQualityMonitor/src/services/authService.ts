import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';
import { ENV } from '../config/env';

const TOKEN_KEY  = '@arm1_token';
const USER_KEY   = '@arm1_user';
const DEVICE_KEY = '@arm1_device_id';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function generateDeviceId(): string {
  return `device_${Platform.OS}_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

async function getDeviceId(): Promise<string> {
  let id = await AsyncStorage.getItem(DEVICE_KEY);
  if (!id) {
    id = generateDeviceId();
    await AsyncStorage.setItem(DEVICE_KEY, id);
  }
  return id;
}

async function post(path: string, body: object) {
  const res = await fetch(`${ENV.API.BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const json = await res.json();
  if (!res.ok) throw new Error(json.message || 'Request failed');
  return json;
}

// ─── Public API ───────────────────────────────────────────────────────────────

export const authService = {
  async register(email: string, password: string, name: string) {
    const deviceId = await getDeviceId();
    const data = await post('/api/auth/register', { email, password, name, deviceId });
    await this._persist(data.token, data.data.user);
    return data;
  },

  async login(email: string, password: string) {
    const deviceId = await getDeviceId();
    const data = await post('/api/auth/login', { email, password, deviceId });
    await this._persist(data.token, data.data.user);
    return data;
  },

  async guestLogin() {
    const deviceId = await getDeviceId();
    const data = await post('/api/auth/anonymous', { deviceId });
    await this._persist(data.token, data.data.user);
    return data;
  },

  async logout() {
    await AsyncStorage.multiRemove([TOKEN_KEY, USER_KEY]);
  },

  async getToken(): Promise<string | null> {
    return AsyncStorage.getItem(TOKEN_KEY);
  },

  async getUser(): Promise<any | null> {
    const raw = await AsyncStorage.getItem(USER_KEY);
    return raw ? JSON.parse(raw) : null;
  },

  async getDeviceId(): Promise<string> {
    return getDeviceId();
  },

  async _persist(token: string, user: any) {
    await AsyncStorage.multiSet([
      [TOKEN_KEY, token],
      [USER_KEY, JSON.stringify(user)],
    ]);
  },
};
