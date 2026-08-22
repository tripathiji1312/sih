import { useTelemetryStore } from '../stores/useTelemetryStore';
import { Watchdog } from '../types';

const STATUS_STYLE: Record<Watchdog['overall_status'], { color: string; label: string }> = {
  HEALTHY: { color: 'var(--healthy)', label: 'Data pipeline: nominal' },
  DATA_DEGRADED: { color: 'var(--caution)', label: 'Data pipeline: degraded — some channels slow' },
  CRITICAL_DATA_LOSS: { color: 'var(--critical)', label: 'Data loss — physics-only fallback active' },
};

const CHANNEL_COLOR = {
  NOMINAL: 'var(--healthy)',
  SLOW: 'var(--caution)',
  STALE: 'var(--warning)',
  LOST: 'var(--critical)',
} as const;

export default function WatchdogBanner() {
  const watchdog = useTelemetryStore((s) => s.watchdog);
  const style = STATUS_STYLE[watchdog.overall_status];

  return (
    <div
      role="status"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        flexWrap: 'wrap',
        padding: '12px 16px',
        borderRadius: 6,
        border: `1px solid ${style.color}40`,
        background: `${style.color}0d`,
      }}
    >
      <span style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: style.color,
            boxShadow: `0 0 0 3px ${style.color}22`,
          }}
        />
        <span className="mono" style={{ fontSize: 12, fontWeight: 600, color: style.color, whiteSpace: 'nowrap' }}>
          {style.label}
        </span>
      </span>

      <span style={{ width: 1, alignSelf: 'stretch', background: 'var(--panel-border)', flexShrink: 0 }} />

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', flex: 1 }}>
        {Object.entries(watchdog.channels).map(([name, ch]) => (
          <span
            key={name}
            className="mono"
            title={`${name}: ${ch.status.toLowerCase()}, ${ch.staleness_s.toFixed(1)}s since last sample`}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontSize: 11,
              padding: '3px 9px',
              borderRadius: 4,
              background: 'var(--bg-2)',
              border: '1px solid var(--panel-border)',
              color: 'var(--text-secondary)',
            }}
          >
            <span
              style={{ width: 5, height: 5, borderRadius: '50%', background: CHANNEL_COLOR[ch.status], flexShrink: 0 }}
            />
            {name}
            <span style={{ color: 'var(--text-muted)' }}>{ch.staleness_s.toFixed(1)}s</span>
          </span>
        ))}
      </div>
    </div>
  );
}
