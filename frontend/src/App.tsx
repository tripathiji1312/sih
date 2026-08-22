import { useEffect } from 'react';
import { useTelemetryStore } from './stores/useTelemetryStore';
import HealthScore from './components/HealthScore';
import EngineSchematic from './components/EngineSchematic';
import AlertFeed from './components/AlertFeed';
import WatchdogBanner from './components/WatchdogBanner';
import DegradationChart from './components/DegradationChart';
import MissionControls from './components/MissionControls';

export default function App() {
  const connect = useTelemetryStore((s) => s.connect);
  useEffect(() => {
    connect();
  }, [connect]);

  return (
    <div style={{ fontFamily: 'system-ui', padding: 16, background: '#0f172a', color: '#e2e8f0', minHeight: '100vh' }}>
      <h1 style={{ margin: 0 }}>Digital Twin — Rotax-912 MALE UAV Engine</h1>
      <p style={{ opacity: 0.7 }}>Physics-anchored • Uncertainty-aware • Edge-ready</p>
      <WatchdogBanner />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
        <HealthScore />
        <EngineSchematic />
        <DegradationChart />
        <AlertFeed />
      </div>
      <MissionControls />
    </div>
  );
}
