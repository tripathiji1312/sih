import { useTelemetryStore } from '../stores/useTelemetryStore';
import { bandFromScore, BAND_COLOR, CylinderKey } from '../types';

// Horizontally-opposed 4-cylinder layout (Rotax 912-style), viewed from
// above: crankcase center, two cylinder banks left/right, oil sump below,
// fuel rail across the top. This is what makes the panel read as a twin
// of the physical engine, not a generic gauge cluster.

interface CylinderSpec {
  key: CylinderKey;
  label: string;
  x: number;
  y: number;
  trustKey: string;
}

const CYLINDERS: CylinderSpec[] = [
  { key: 'cylinder_1', label: 'C1', x: 70, y: 62, trustKey: 'cht_1' },
  { key: 'cylinder_2', label: 'C2', x: 70, y: 176, trustKey: 'cht_2' },
  { key: 'cylinder_3', label: 'C3', x: 290, y: 62, trustKey: 'cht_3' },
  { key: 'cylinder_4', label: 'C4', x: 290, y: 176, trustKey: 'cht_4' },
];

const CYL_W = 100;
const CYL_H = 50;

export default function EngineSchematic() {
  const health = useTelemetryStore((s) => s.health);
  const sensorTrust = useTelemetryStore((s) => s.sensorTrust);

  const colorFor = (score: number, trustKey?: string) => {
    const trusted = !trustKey || (sensorTrust[trustKey] ?? 1) > 0.5;
    const band = bandFromScore(score, trusted);
    return BAND_COLOR[band];
  };

  const oilColor = colorFor(health.subsystems.lubrication, 'oil_pressure');
  const coolingColor = colorFor(health.subsystems.cooling);
  const fuelColor = colorFor(health.subsystems.fuel_system, 'fuel_flow');
  const electricalColor = colorFor(health.subsystems.electrical);

  return (
    <div className="panel">
      <div className="panel-title">Engine schematic</div>
      <div style={{ padding: 14 }}>
      <svg viewBox="0 0 480 280" width="100%" role="img" aria-label="Engine schematic color-coded by subsystem health">
        <defs>
          <pattern id="instrument-grid" width="24" height="24" patternUnits="userSpaceOnUse">
            <path d="M 24 0 L 0 0 0 24" fill="none" stroke="rgba(255,255,255,0.025)" strokeWidth="1" />
          </pattern>
        </defs>
        <rect x={0} y={0} width={480} height={280} fill="url(#instrument-grid)" />

        {/* Fuel rail across the top, feeding all four cylinders */}
        <line x1={70} y1={40} x2={390} y2={40} stroke={fuelColor} strokeWidth={4} strokeLinecap="round" />
        <text x={230} y={28} textAnchor="middle" fontSize="10" fill="var(--text-secondary)">fuel rail</text>

        {/* Crankcase */}
        <rect x={190} y={100} width={100} height={90} rx={10} fill="#1a212b" stroke="var(--panel-border-strong)" strokeWidth={1.5} />
        <text x={240} y={140} textAnchor="middle" fontSize="10" fill="var(--text-secondary)">crankcase</text>
        <text x={240} y={154} textAnchor="middle" fontSize="9" fill={electricalColor} className="mono">
          alt {(health.subsystems.electrical * 100).toFixed(0)}%
        </text>

        {/* Oil sump */}
        <ellipse cx={240} cy={225} rx={55} ry={22} fill={oilColor} opacity={0.85} />
        <text x={240} y={229} textAnchor="middle" fontSize="10" fill="#0a0d12" fontWeight={600}>
          oil sump
        </text>

        {/* Cooling fin markers on the crankcase shoulders */}
        {[0, 1, 2, 3].map((i) => (
          <line key={i} x1={196 + i * 8} y1={104} x2={196 + i * 8} y2={112} stroke={coolingColor} strokeWidth={2} />
        ))}
        {[0, 1, 2, 3].map((i) => (
          <line key={`r${i}`} x1={256 + i * 8} y1={104} x2={256 + i * 8} y2={112} stroke={coolingColor} strokeWidth={2} />
        ))}

        {/* Injector connectors from fuel rail down to each cylinder */}
        {CYLINDERS.map((c) => {
          const cx = c.x < 240 ? c.x + CYL_W - 12 : c.x + 12;
          return <line key={c.key} x1={cx} y1={40} x2={cx} y2={c.y} stroke={fuelColor} strokeWidth={1.5} opacity={0.6} />;
        })}

        {/* Connecting rods from cylinders to crankcase */}
        {CYLINDERS.map((c) => {
          const isLeft = c.x < 240;
          const startX = isLeft ? c.x + CYL_W : c.x;
          const endX = isLeft ? 190 : 290;
          const y = c.y + CYL_H / 2;
          return <line key={`rod-${c.key}`} x1={startX} y1={y} x2={endX} y2={y} stroke="var(--panel-border-strong)" strokeWidth={3} />;
        })}

        {/* Cylinders */}
        {CYLINDERS.map((c) => {
          const score = health.subsystems[c.key];
          const trusted = (sensorTrust[c.trustKey] ?? 1) > 0.5;
          const band = bandFromScore(score, trusted);
          const color = BAND_COLOR[band];
          return (
            <g key={c.key}>
              <title>{`${c.label}: health ${(score * 100).toFixed(0)}%${trusted ? '' : ' — sensor untrusted'}`}</title>
              <rect x={c.x} y={c.y} width={CYL_W} height={CYL_H} rx={8} fill={color} opacity={0.85} />
              {/* cooling fins on cylinder body */}
              {[0, 1, 2, 3, 4].map((i) => (
                <line
                  key={i}
                  x1={c.x + 10 + i * 18}
                  y1={c.y - 4}
                  x2={c.x + 10 + i * 18}
                  y2={c.y}
                  stroke={color}
                  strokeWidth={2}
                />
              ))}
              <text x={c.x + CYL_W / 2} y={c.y + 22} textAnchor="middle" fontSize="14" fontWeight={600} fill="#0a0d12">
                {c.label}
              </text>
              <text x={c.x + CYL_W / 2} y={c.y + 38} textAnchor="middle" fontSize="10" fill="#0a0d12" className="mono">
                {trusted ? `${(score * 100).toFixed(0)}%` : 'sensor?'}
              </text>
            </g>
          );
        })}
      </svg>

      <div
        style={{
          display: 'flex',
          gap: 16,
          flexWrap: 'wrap',
          marginTop: 12,
          paddingTop: 12,
          borderTop: '1px solid var(--panel-border)',
          fontSize: 11,
          color: 'var(--text-secondary)',
        }}
      >
        {(['healthy', 'caution', 'warning', 'critical', 'unknown', 'physics_fallback'] as const).map((b) => (
          <span key={b} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 7, height: 7, borderRadius: 2, background: BAND_COLOR[b], display: 'inline-block' }} />
            {b.replace('_', ' ')}
          </span>
        ))}
      </div>
      </div>
    </div>
  );
}
