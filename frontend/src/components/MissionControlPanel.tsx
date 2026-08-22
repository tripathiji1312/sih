import { useTelemetryStore } from '../stores/useTelemetryStore';
import { FaultType } from '../types';

const FAULT_BUTTONS: { fault: FaultType; label: string; target?: string; color: string }[] = [
  { fault: 'misfire', label: 'Inject misfire — C2', target: 'cylinder_2', color: 'var(--critical)' },
  { fault: 'overheat', label: 'Simulate overheat', color: 'var(--warning)' },
  { fault: 'oil_pressure_loss', label: 'Oil pressure loss', color: 'var(--warning)' },
  { fault: 'sensor_drift', label: 'Sensor drift — EGT2', target: 'egt_2', color: 'var(--fallback)' },
  { fault: 'data_loss', label: 'Simulate data loss', color: 'var(--critical)' },
];

export default function MissionControlPanel() {
  const injectFault = useTelemetryStore((s) => s.injectFault);
  const activeFault = useTelemetryStore((s) => s.activeFault);
  const connectionStatus = useTelemetryStore((s) => s.connectionStatus);

  return (
    <div className="panel">
      <div className="panel-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>Mission control — fault injection</span>
        {connectionStatus === 'demo' && (
          <span className="pill" style={{ color: 'var(--fallback)', borderColor: '#3b82f655', padding: '2px 8px', fontSize: 10 }}>
            demo mode
          </span>
        )}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 7, padding: 12 }}>
        {FAULT_BUTTONS.map((b) => {
          const active = activeFault === b.fault;
          return (
            <button
              key={b.fault}
              onClick={() => injectFault(b.fault, b.target)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 9,
                fontSize: 12,
                padding: '9px 12px',
                borderRadius: 5,
                border: `1px solid ${active ? b.color : 'var(--panel-border)'}`,
                background: active ? `${b.color}1a` : 'var(--bg-2)',
                color: active ? b.color : 'var(--text-primary)',
                textAlign: 'left',
              }}
            >
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: b.color, flexShrink: 0 }} />
              {b.label}
            </button>
          );
        })}
      </div>
      <div style={{ padding: '4px 12px 12px', borderTop: '1px solid var(--panel-border)', paddingTop: 12 }}>
        <button
          onClick={() => injectFault('clear')}
          disabled={!activeFault}
          style={{
            width: '100%',
            fontSize: 12,
            padding: '9px 12px',
            borderRadius: 5,
            border: '1px solid var(--panel-border)',
            background: 'transparent',
            color: activeFault ? 'var(--text-secondary)' : 'var(--text-muted)',
          }}
        >
          Clear faults
        </button>
      </div>
    </div>
  );
}
