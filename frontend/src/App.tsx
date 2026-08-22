import { useEffect, useState } from 'react';
import { useTelemetryStore } from './stores/useTelemetryStore';
import HealthGauge from './components/HealthGauge';
import EngineSchematic from './components/EngineSchematic';
import AlertFeed from './components/AlertFeed';
import WatchdogBanner from './components/WatchdogBanner';
import MeasuredVsExpectedChart from './components/MeasuredVsExpectedChart';
import MissionControlPanel from './components/MissionControlPanel';

const WS_URL = import.meta.env.VITE_TELEMETRY_WS_URL ?? 'ws://localhost:8000/ws/telemetry';

const STATUS_COLOR: Record<string, string> = {
  live: 'var(--healthy)',
  demo: 'var(--fallback)',
  connecting: 'var(--caution)',
  disconnected: 'var(--text-muted)',
};

function useUtcClock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now.toISOString().slice(11, 19) + 'Z';
}

export default function App() {
  const connect = useTelemetryStore((s) => s.connect);
  const connectionStatus = useTelemetryStore((s) => s.connectionStatus);
  const clock = useUtcClock();

  useEffect(() => {
    connect(WS_URL);
  }, [connect]);

  const statusColor = STATUS_COLOR[connectionStatus];

  return (
    <div style={{ minHeight: '100%', padding: 20, display: 'flex', flexDirection: 'column', gap: 16 }}>
      <header
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          paddingBottom: 16,
          borderBottom: '1px solid var(--panel-border)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
          <h1 style={{ fontSize: 17, fontWeight: 700, margin: 0, letterSpacing: '-0.01em' }}>
            Aero piston engine — digital twin
          </h1>
          <span
            className="mono"
            style={{ fontSize: 11, color: 'var(--text-muted)' }}
          >
            MALE UAV ground control station · health monitor
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span className="pill">
            mission time <span style={{ color: 'var(--text-primary)' }}>{clock}</span>
          </span>
          <span className="pill" style={{ color: statusColor, borderColor: `${statusColor}55` }}>
            <span className="pill-dot" style={{ background: statusColor }} />
            {connectionStatus.toUpperCase()}
          </span>
        </div>
      </header>

      <WatchdogBanner />

      <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr 320px', gap: 16, alignItems: 'stretch' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <HealthGauge />
          <MissionControlPanel />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <EngineSchematic />
          <MeasuredVsExpectedChart />
        </div>

        <AlertFeed />
      </div>
    </div>
  );
}
