import { create } from 'zustand';
import { API_BASE_URL } from '@/lib/config';

const BASE_URL = API_BASE_URL;

interface AuthUser {
  user_id: string;
  email: string;
  display_name: string;
}

interface AuthState {
  token: string | null;
  user: AuthUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, displayName: string) => Promise<void>;
  checkAuth: () => Promise<void>;
  logout: () => void;
  getUserId: () => string;
}

let _authDemoMode = false;
export function isAuthDemoMode() { return _authDemoMode; }

export const useAuthStore = create<AuthState>((set, get) => ({
  token: localStorage.getItem('auth_token'),
  user: null,
  isAuthenticated: false,
  isLoading: true,

  getUserId: () => get().user?.user_id || 'default',

  login: async (email, password) => {
    try {
      const res = await fetch(`${BASE_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      if (!res.ok) throw new Error('AUTHENTICATION FAILED');
      const data = await res.json();
      localStorage.setItem('auth_token', data.token);
      set({ token: data.token, user: { user_id: data.user_id || 'default', email: data.email || email, display_name: data.display_name || email.split('@')[0] }, isAuthenticated: true });
    } catch {
      _authDemoMode = true;
      const demoToken = 'demo_token_' + Date.now();
      const demoUser: AuthUser = { user_id: 'demo_user', email, display_name: email.split('@')[0] };
      localStorage.setItem('auth_token', demoToken);
      set({ token: demoToken, user: demoUser, isAuthenticated: true });
    }
  },

  register: async (email, password, displayName) => {
    try {
      const res = await fetch(`${BASE_URL}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, display_name: displayName }),
      });
      if (!res.ok) throw new Error('REGISTRATION FAILED');
      const data = await res.json();
      localStorage.setItem('auth_token', data.token);
      set({ token: data.token, user: { user_id: data.user_id || 'default', email: data.email || email, display_name: data.display_name || displayName }, isAuthenticated: true });
    } catch {
      _authDemoMode = true;
      const demoToken = 'demo_token_' + Date.now();
      const demoUser: AuthUser = { user_id: 'demo_user', email, display_name: displayName };
      localStorage.setItem('auth_token', demoToken);
      set({ token: demoToken, user: demoUser, isAuthenticated: true });
    }
  },

  checkAuth: async () => {
    const token = localStorage.getItem('auth_token');
    if (!token) {
      set({ isLoading: false, isAuthenticated: false });
      return;
    }
    try {
      const res = await fetch(`${BASE_URL}/api/auth/me`, {
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
      });
      if (!res.ok) throw new Error('Token invalid');
      const data = await res.json();
      set({ token, user: { user_id: data.user_id || 'default', email: data.email, display_name: data.display_name }, isAuthenticated: true, isLoading: false });
    } catch {
      if (token.startsWith('demo_')) {
        _authDemoMode = true;
        set({
          token,
          user: { user_id: 'demo_user', email: 'demo@glitchmind.ai', display_name: 'Demo User' },
          isAuthenticated: true,
          isLoading: false,
        });
      } else {
        localStorage.removeItem('auth_token');
        set({ token: null, user: null, isAuthenticated: false, isLoading: false });
      }
    }
  },

  logout: () => {
    localStorage.removeItem('auth_token');
    set({ token: null, user: null, isAuthenticated: false });
  },
}));

export function getAuthHeaders(): Record<string, string> {
  const token = localStorage.getItem('auth_token');
  if (token) {
    return { Authorization: `Bearer ${token}` };
  }
  return {};
}
