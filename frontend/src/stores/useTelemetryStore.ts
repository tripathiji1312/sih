import { create } from 'zustand';
import {
  AlertEvent,
  FaultInjectionCommand,
  FaultType,
  HealthState,
  OverlayPoint,
  SensorTrust,
  TelemetryMessage,
  Watchdog,
} from '../types';

const MAX_ALERTS = 50;
const MAX_OVERLAY_POINTS = 120;

type ConnectionStatus = 'connecting' | 'live' | 'demo' | 'disconnected';
type ConnectFn = ((url: string) => void) & { __fallbackToDemo?: () => void };

interface TelemetryStore {
  connectionStatus: ConnectionStatus;
  health: HealthState;
  sensorTrust: SensorTrust;
  watchdog: Watchdog;
  alerts: AlertEvent[];
  overlayChannel: string;
  overlaySeries: OverlayPoint[];
  activeFault: FaultType | null;

  connect: (url: string) => void;
  disconnect: () => void;
  setOverlayChannel: (channel: string) => void;
  injectFault: (fault: FaultType, target?: string) => void;
  _ingest: (msg: TelemetryMessage) => void;
}

let socket: WebSocket | null = null;
let demoTimer: ReturnType<typeof setInterval> | null = null;

const initialHealth: HealthState = {
  overall_score: 0.92,
  confidence: 0.88,
  source: 'ml',
  subsystems: {
    cylinder_1: 0.94,
    cylinder_2: 0.93,
    cylinder_3: 0.91,
    cylinder_4: 0.9,
    lubrication: 0.95,
    cooling: 0.93,
    fuel_system: 0.96,
    electrical: 0.97,
  },
};

const initialWatchdog: Watchdog = {
  overall_status: 'HEALTHY',
  channels: {
    cht: { status: 'NOMINAL', staleness_s: 0.1 },
    egt: { status: 'NOMINAL', staleness_s: 0.1 },
    oil: { status: 'NOMINAL', staleness_s: 0.2 },
    fuel_flow: { status: 'NOMINAL', staleness_s: 0.1 },
    vibration: { status: 'NOMINAL', staleness_s: 0.3 },
  },
};

export const useTelemetryStore = create<TelemetryStore>((set, get) => ({
  connectionStatus: 'disconnected',
  health: initialHealth,
  sensorTrust: {
    cht_1: 0.98, cht_2: 0.97, cht_3: 0.96, cht_4: 0.98,
    egt_1: 0.95, egt_2: 0.95, egt_3: 0.94, egt_4: 0.96,
    oil_pressure: 0.99, oil_temp: 0.98, fuel_flow: 0.97, vibration: 0.9,
  },
  watchdog: initialWatchdog,
  alerts: [],
  overlayChannel: 'egt_2',
  overlaySeries: [],
  activeFault: null,

  connect: (url: string) => {
    set({ connectionStatus: 'connecting' });
    try {
      socket = new WebSocket(url);
      socket.onopen = () => set({ connectionStatus: 'live' });
      socket.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data) as TelemetryMessage;
          get()._ingest(msg);
        } catch {
          // malformed frame — ignore, watchdog will catch prolonged silence
        }
      };
      socket.onclose = () => {
        // Backend not reachable (common during a hackathon demo) — fall back
        // to a local simulator so the UI stays alive and demo-able.
        (get().connect as ConnectFn).__fallbackToDemo?.();
      };
      socket.onerror = () => socket?.close();
    } catch {
      startDemoLoop(set, get);
    }
  },

  disconnect: () => {
    socket?.close();
    socket = null;
    if (demoTimer) clearInterval(demoTimer);
    demoTimer = null;
    set({ connectionStatus: 'disconnected' });
  },

  setOverlayChannel: (channel: string) => set({ overlayChannel: channel, overlaySeries: [] }),

  injectFault: (fault: FaultType, target?: string) => {
    const cmd: FaultInjectionCommand = { type: 'fault_injection', fault, target };
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(cmd));
    }
    // Also reflect locally so the demo responds instantly even against
    // a real backend with network latency, and works standalone.
    set({ activeFault: fault === 'clear' ? null : fault });
    applyDemoFault(set, get, fault, target);
  },

  _ingest: (msg: TelemetryMessage) => {
    set((state) => {
      const alerts =
        msg.alerts && msg.alerts.length
          ? [...msg.alerts, ...state.alerts].slice(0, MAX_ALERTS)
          : state.alerts;
      const overlaySeries =
        msg.overlay && msg.overlay.channel === state.overlayChannel
          ? [...state.overlaySeries, msg.overlay.point].slice(-MAX_OVERLAY_POINTS)
          : state.overlaySeries;
      return {
        health: msg.health ?? state.health,
        sensorTrust: msg.sensorTrust ?? state.sensorTrust,
        watchdog: msg.watchdog ?? state.watchdog,
        alerts,
        overlaySeries,
      };
    });
  },
}));

// --- Demo/simulation fallback -------------------------------------------
// Lets the dashboard run and demo convincingly with zero backend, and lets
// mission-control fault buttons work even before FastAPI is wired up.

function startDemoLoop(
  set: (partial: Partial<TelemetryStore> | ((s: TelemetryStore) => Partial<TelemetryStore>)) => void,
  get: () => TelemetryStore
) {
  set({ connectionStatus: 'demo' });
  let t = 0;
  if (demoTimer) clearInterval(demoTimer);
  demoTimer = setInterval(() => {
    t += 1;
    const state = get();
    const fault = state.activeFault;
    const noise = () => (Math.random() - 0.5) * 0.02;

    const health: HealthState = {
      ...state.health,
      overall_score: clamp01(
        state.health.overall_score +
          (0.92 - state.health.overall_score) * 0.03 +
          noise() -
          (fault ? 0.015 : 0)
      ),
      confidence: clamp01(
        fault === 'sensor_drift'
          ? state.health.confidence - 0.02
          : state.health.confidence + (0.88 - state.health.confidence) * 0.03 + noise() * 0.2
      ),
      source: fault === 'sensor_drift' ? 'physics_fallback' : 'ml',
      subsystems: {
        ...state.health.subsystems,
        cylinder_2:
          fault === 'misfire'
            ? clamp01(state.health.subsystems.cylinder_2 - 0.03)
            : clamp01(state.health.subsystems.cylinder_2 + noise()),
        cooling:
          fault === 'overheat'
            ? clamp01(state.health.subsystems.cooling - 0.03)
            : clamp01(state.health.subsystems.cooling + noise()),
        lubrication:
          fault === 'oil_pressure_loss'
            ? clamp01(state.health.subsystems.lubrication - 0.04)
            : clamp01(state.health.subsystems.lubrication + noise()),
      },
    };

    const expected = 720 + 15 * Math.sin(t / 8);
    const measured =
      fault === 'overheat'
        ? expected + Math.min(60, t * 0.6)
        : expected + (Math.random() - 0.5) * 6;

    const overlaySeries =
      state.overlayChannel === 'egt_2'
        ? [...state.overlaySeries, { t: Date.now(), measured, expected }].slice(-MAX_OVERLAY_POINTS)
        : state.overlaySeries;

    let watchdog = state.watchdog;
    if (fault === 'data_loss') {
      watchdog = {
        overall_status: 'CRITICAL_DATA_LOSS',
        channels: Object.fromEntries(
          Object.entries(state.watchdog.channels).map(([k, v]) => [
            k,
            { status: 'LOST', staleness_s: v.staleness_s + 0.5 },
          ])
        ),
      };
    } else if (fault) {
      watchdog = {
        overall_status: 'DATA_DEGRADED',
        channels: Object.fromEntries(
          Object.entries(state.watchdog.channels).map(([k, v]) => [
            k,
            { status: k === 'vibration' ? 'SLOW' : v.status, staleness_s: v.staleness_s },
          ])
        ),
      };
    } else {
      watchdog = initialWatchdog;
    }

    set({ health, overlaySeries, watchdog });
  }, 800);
}

function applyDemoFault(
  set: (partial: Partial<TelemetryStore> | ((s: TelemetryStore) => Partial<TelemetryStore>)) => void,
  get: () => TelemetryStore,
  fault: FaultType,
  target?: string
) {
  if (get().connectionStatus !== 'demo') return; // live backend owns state
  const label: Record<FaultType, string> = {
    misfire: 'Misfire detected — cylinder 2',
    sensor_drift: 'Sensor drift suspected — model abstained, physics fallback active',
    overheat: 'Cylinder head temperature trending above expected envelope',
    oil_pressure_loss: 'Oil pressure trending below nominal band',
    data_loss: 'Telemetry channel loss — physics-only estimation active',
    clear: 'All injected faults cleared',
  };
  const severity =
    fault === 'clear' ? 'info' : fault === 'sensor_drift' || fault === 'data_loss' ? 'warning' : 'critical';
  const source: AlertEvent['source'] =
    fault === 'sensor_drift' || fault === 'data_loss' ? 'physics_fallback' : 'ml';
  const alert: AlertEvent = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    timestamp: Date.now(),
    severity: severity as AlertEvent['severity'],
    subsystem: target ?? fault,
    message: label[fault],
    confidence: fault === 'sensor_drift' || fault === 'data_loss' ? 0.42 : 0.91,
    source,
  };
  set((state) => ({ alerts: [alert, ...state.alerts].slice(0, MAX_ALERTS) }));
}

function clamp01(n: number) {
  return Math.max(0, Math.min(1, n));
}

// Attach fallback hook used by connect() above.
(useTelemetryStore.getState().connect as ConnectFn).__fallbackToDemo = () => {
  startDemoLoop(useTelemetryStore.setState, useTelemetryStore.getState);
};
