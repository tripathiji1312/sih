import Plot from 'react-plotly.js';
import { useTelemetryStore } from '../stores/useTelemetryStore';

const CHANNEL_OPTIONS = [
  { value: 'egt_2', label: 'EGT — cylinder 2' },
  { value: 'cht_2', label: 'CHT — cylinder 2' },
  { value: 'oil_pressure', label: 'Oil pressure' },
];

export default function MeasuredVsExpectedChart() {
  const series = useTelemetryStore((s) => s.overlaySeries);
  const overlayChannel = useTelemetryStore((s) => s.overlayChannel);
  const setOverlayChannel = useTelemetryStore((s) => s.setOverlayChannel);

  const t = series.map((p) => new Date(p.t));
  const measured = series.map((p) => p.measured);
  const expected = series.map((p) => p.expected);
  const residual = series.map((p) => Math.abs(p.measured - p.expected));

  return (
    <div className="panel">
      <div className="panel-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>Measured vs. expected</span>
        <select
          value={overlayChannel}
          onChange={(e) => setOverlayChannel(e.target.value)}
          style={{
            background: 'var(--bg-2)',
            color: 'var(--text-primary)',
            border: '1px solid var(--panel-border-strong)',
            borderRadius: 4,
            fontSize: 11,
            fontFamily: 'var(--font-mono)',
            padding: '3px 6px',
          }}
        >
          {CHANNEL_OPTIONS.map((c) => (
            <option key={c.value} value={c.value}>
              {c.label}
            </option>
          ))}
        </select>
      </div>
      <div style={{ padding: '8px 10px 4px' }}>
      <Plot
        data={[
          {
            x: t,
            y: expected,
            type: 'scatter',
            mode: 'lines',
            name: 'expected (physics model)',
            line: { color: '#5a6474', width: 1.5, dash: 'dot' },
          },
          {
            x: t,
            y: measured,
            type: 'scatter',
            mode: 'lines',
            name: 'measured',
            line: { color: '#3b8bd4', width: 2 },
          },
          {
            x: t,
            y: residual,
            type: 'scatter',
            mode: 'lines',
            name: 'residual',
            yaxis: 'y2',
            line: { color: '#f97316', width: 1 },
            opacity: 0.55,
          },
        ]}
        layout={{
          autosize: true,
          height: 260,
          margin: { l: 44, r: 44, t: 10, b: 32 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          font: { color: '#8c96a6', size: 10, family: 'IBM Plex Mono, monospace' },
          xaxis: { gridcolor: '#1c232d', showgrid: true },
          yaxis: { gridcolor: '#1c232d', title: { text: overlayChannel } },
          yaxis2: { overlaying: 'y', side: 'right', showgrid: false, title: { text: 'residual' } },
          legend: { orientation: 'h', y: -0.25, font: { size: 10 } },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
      </div>
    </div>
  );
}
