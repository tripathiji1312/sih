// Shared telemetry contract with the FastAPI backend.
// If the backend's exact WS payload differs, adjust this file only —
// every component reads through useTelemetryStore, not raw messages.

export type HealthBand = 'healthy' | 'caution' | 'warning' | 'critical' | 'unknown' | 'physics_fallback';

export type CylinderKey = 'cylinder_1' | 'cylinder_2' | 'cylinder_3' | 'cylinder_4';

export interface SubsystemHealth {
  cylinder_1: number; // 0..1, higher = healthier
  cylinder_2: number;
  cylinder_3: number;
  cylinder_4: number;
  lubrication: number;
  cooling: number;
  fuel_system: number;
  electrical: number;
}

export interface HealthState {
  overall_score: number; // 0..1
  confidence: number; // 0..1, model's confidence in overall_score
  source: 'ml' | 'physics_fallback'; // did the ONNX model produce this, or did FilterPy/physics take over
  subsystems: SubsystemHealth;
}

/** Per-channel sensor trust, keyed e.g. "cht_1", "egt_2", "oil_pressure" */
export type SensorTrust = Record<string, number>;

export type ChannelStatus = 'NOMINAL' | 'SLOW' | 'STALE' | 'LOST';

export interface WatchdogChannel {
  status: ChannelStatus;
  staleness_s: number;
}

export interface Watchdog {
  overall_status: 'HEALTHY' | 'DATA_DEGRADED' | 'CRITICAL_DATA_LOSS';
  channels: Record<string, WatchdogChannel>;
}

export type AlertSeverity = 'info' | 'caution' | 'warning' | 'critical';
export type AlertSource = 'ml' | 'physics_fallback' | 'threshold';

export interface AlertEvent {
  id: string;
  timestamp: number; // epoch ms
  severity: AlertSeverity;
  subsystem: string;
  message: string;
  confidence: number; // 0..1
  source: AlertSource;
}

export interface OverlayPoint {
  t: number; // epoch ms
  measured: number;
  expected: number;
}

export interface TelemetryMessage {
  type: 'telemetry';
  timestamp: number;
  health: HealthState;
  sensorTrust: SensorTrust;
  watchdog: Watchdog;
  alerts: AlertEvent[]; // new alerts since last message, store appends
  overlay: { channel: string; point: OverlayPoint };
}

export type FaultType =
  | 'misfire'
  | 'sensor_drift'
  | 'overheat'
  | 'oil_pressure_loss'
  | 'data_loss'
  | 'clear';

export interface FaultInjectionCommand {
  type: 'fault_injection';
  fault: FaultType;
  target?: CylinderKey | string;
}

export function bandFromScore(score: number, trusted: boolean): HealthBand {
  if (!trusted) return 'unknown';
  if (score < 0.3) return 'critical';
  if (score < 0.5) return 'warning';
  if (score < 0.7) return 'caution';
  return 'healthy';
}

export const BAND_COLOR: Record<HealthBand, string> = {
  healthy: '#22c55e',
  caution: '#eab308',
  warning: '#f97316',
  critical: '#ef4444',
  unknown: '#6b7280',
  physics_fallback: '#3b82f6',
};
