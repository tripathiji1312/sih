import { useTelemetryStore } from '../stores/useTelemetryStore';
import { bandFromScore, BAND_COLOR } from '../types';

// Semicircular gauge (180deg sweep). The needle shows overall_score.
// A second, thinner arc directly beneath the score arc shows confidence
// independently — a high score with low confidence should visually read
// as "uncertain", not "healthy".

const R = 90;
const CX = 110;
const CY = 110;

function polar(radius: number, angleDeg: number) {
  const rad = (Math.PI / 180) * angleDeg;
  return { x: CX + radius * Math.cos(rad), y: CY - radius * Math.sin(rad) };
}

function arcPath(radius: number, fromDeg: number, toDeg: number) {
  const start = polar(radius, fromDeg);
  const end = polar(radius, toDeg);
  // fromDeg is always >= toDeg here (we sweep left -> right, 180deg -> 0deg),
  // so the swept angle is always <= 180 and largeArc is always 0.
  const largeArc = fromDeg - toDeg > 180 ? 1 : 0;
  // sweep-flag must be 1: with two points on the SAME circle, sweep picks
  // which of the two possible arc *centers* SVG solves for. sweep=0 here
  // was solving for the mirrored center, producing a distorted arc instead
  // of the clean semicircle around (CX, CY).
  return `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} 1 ${end.x} ${end.y}`;
}

export default function HealthGauge() {
  const health = useTelemetryStore((s) => s.health);
  const score = health.overall_score;
  const confidence = health.confidence;
  const band = bandFromScore(score, true);
  const color = BAND_COLOR[health.source === 'physics_fallback' ? 'physics_fallback' : band];

  const scoreAngle = 180 - score * 180;
  const needle = polar(R - 14, scoreAngle);

  return (
    <div className="panel">
      <div className="panel-title">Engine health</div>
      <div style={{ padding: '18px 16px 16px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <svg viewBox="0 0 220 150" width="220" height="150" role="img" aria-label={`Overall engine health ${(score * 100).toFixed(0)} percent, confidence ${(confidence * 100).toFixed(0)} percent`}>
        <path d={arcPath(R, 180, 0)} stroke="#232b36" strokeWidth={14} fill="none" strokeLinecap="round" />
        <path
          d={arcPath(R, 180, 180 - score * 180)}
          stroke={color}
          strokeWidth={14}
          fill="none"
          strokeLinecap="round"
        />
        <path d={arcPath(R - 20, 180, 0)} stroke="#1a212b" strokeWidth={6} fill="none" strokeLinecap="round" />
        <path
          d={arcPath(R - 20, 180, 180 - confidence * 180)}
          stroke="var(--text-secondary)"
          strokeWidth={6}
          fill="none"
          strokeLinecap="round"
        />
        <line x1={CX} y1={CY} x2={needle.x} y2={needle.y} stroke="var(--text-primary)" strokeWidth={2} />
        <circle cx={CX} cy={CY} r={4} fill="var(--text-primary)" />
        <text x={CX} y={CY - 26} textAnchor="middle" className="mono" fontSize="26" fontWeight={600} fill="var(--text-primary)">
          {(score * 100).toFixed(0)}
        </text>
        <text x={CX} y={CY - 8} textAnchor="middle" fontSize="10" fill="var(--text-secondary)">
          health score
        </text>
      </svg>
      <div style={{ display: 'flex', gap: 8, marginTop: 6, alignItems: 'center' }}>
        <span className="pill">
          confidence <span style={{ color: 'var(--text-primary)' }}>{(confidence * 100).toFixed(0)}%</span>
        </span>
        <span className="pill" style={{ color, borderColor: `${color}55`, background: `${color}14` }}>
          <span className="pill-dot" style={{ background: color }} />
          {health.source === 'physics_fallback' ? 'physics fallback' : 'ml estimate'}
        </span>
      </div>
      </div>
    </div>
  );
}
