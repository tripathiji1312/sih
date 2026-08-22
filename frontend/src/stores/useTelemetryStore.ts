import { create } from 'zustand';

interface TelemetryState {
  health: any;
  sensorTrust: Record<string, number>;
  alerts: any[];
  watchdog: any;
  measured: any;
  expected: any;
  connect: () => void;
}

export const useTelemetryStore = create<TelemetryState>((set) => ({
  health: { health_index: 100, confidence: 1 },
  sensorTrust: {},
  alerts: [],
  watchdog: { overall_status: 'HEALTHY', channels: {} },
  measured: null,
  expected: null,
  connect: () => {
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${wsProto}//${window.location.hostname}:8000/ws`;
    const ws = new WebSocket(url);
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'telemetry') {
          set({
            health: msg.health,
            sensorTrust: msg.sensor_trust,
            watchdog: msg.watchdog,
            measured: msg.measured,
            expected: msg.expected,
          });
          if (msg.alert) {
            set((s) => ({ alerts: [msg.alert, ...s.alerts].slice(0, 50) }));
          }
        }
      } catch {}
    };
    ws.onerror = () => console.warn('WS error, backend may be offline');
  },
}));
