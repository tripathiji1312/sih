import { useTelemetryStore } from '../stores/useTelemetryStore';
import { AlertEvent } from '../types';

const SEVERITY_COLOR: Record<AlertEvent['severity'], string> = {
  info: 'var(--text-secondary)',
  caution: 'var(--caution)',
  warning: 'var(--warning)',
  critical: 'var(--critical)',
};

function timeAgo(ts: number) {
  const s = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (s < 60) return `${s}s ago`;
  return `${Math.round(s / 60)}m ago`;
}

function AlertRow({ alert }: { alert: AlertEvent }) {
  const lowConfidence = alert.source === 'physics_fallback' || alert.confidence < 0.6;
  const color = SEVERITY_COLOR[alert.severity];

  return (
    <div
      style={{
        display: 'flex',
        gap: 10,
        padding: '9px 12px',
        borderBottom: '1px solid var(--panel-border)',
        borderLeft: `3px solid ${color}`,
        background: lowConfidence ? 'rgba(59,130,246,0.06)' : 'transparent',
        opacity: lowConfidence ? 0.85 : 1,
      }}
    >
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 13, color: 'var(--text-primary)' }}>{alert.message}</span>
          {lowConfidence && (
            <span
              className="mono"
              style={{
                fontSize: 9,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                color: 'var(--fallback)',
                border: '1px solid #3b82f655',
                background: '#3b82f61a',
                borderRadius: 4,
                padding: '1px 6px',
              }}
              title="Model abstained or confidence is low — physics-based estimate shown"
            >
              low confidence
            </span>
          )}
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }} className="mono">
          {alert.subsystem} · conf {(alert.confidence * 100).toFixed(0)}% · {alert.source} · {timeAgo(alert.timestamp)}
        </div>
      </div>
    </div>
  );
}

export default function AlertFeed() {
  const alerts = useTelemetryStore((s) => s.alerts);

  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="panel-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>Alert feed</span>
        <span
          className="mono"
          style={{
            fontSize: 10,
            padding: '1px 7px',
            borderRadius: 10,
            background: alerts.length ? 'var(--bg-2)' : 'transparent',
            border: alerts.length ? '1px solid var(--panel-border-strong)' : 'none',
            color: alerts.length ? 'var(--text-primary)' : 'var(--text-muted)',
          }}
        >
          {alerts.length}
        </span>
      </div>
      <div style={{ overflowY: 'auto', flex: 1 }}>
        {alerts.length === 0 && (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 8,
              padding: '36px 16px',
              color: 'var(--text-muted)',
            }}
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="9.5" stroke="var(--healthy)" strokeWidth="1.5" opacity={0.6} />
              <path d="M8 12.5l2.5 2.5L16 9.5" stroke="var(--healthy)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity={0.8} />
            </svg>
            <span style={{ fontSize: 12 }}>No alerts — nominal.</span>
          </div>
        )}
        {alerts.map((a) => (
          <AlertRow key={a.id} alert={a} />
        ))}
      </div>
    </div>
  );
}
