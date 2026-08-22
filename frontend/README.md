# Engine digital twin — operator dashboard

React + Vite + TypeScript frontend for the MALE UAV aero piston engine digital twin. Built to run standalone for demo purposes and to attach to the FastAPI WebSocket backend when it's ready.

## Run it

```bash
npm install
npm run dev
```

Open http://localhost:5173. With no backend running, the store automatically falls back to a **local demo simulator** (see status pill in the header: `DEMO`) — the schematic, gauge, chart and alerts animate on synthetic data so the UI is fully demoable with zero backend. This is a deliberate hackathon safety net: if FastAPI isn't wired up yet, or breaks mid-demo, judges still see a live-looking dashboard.

## Connecting the real backend

Set `VITE_TELEMETRY_WS_URL` (defaults to `ws://localhost:8000/ws/telemetry`) in a `.env.local` file. Once the FastAPI WS server accepts a connection, `connectionStatus` flips to `LIVE` and the store consumes real `TelemetryMessage` frames instead of the simulator.

The full WS contract (inbound telemetry + outbound fault-injection commands) is defined in `src/types.ts`. If the backend's actual payload shape differs from what's assumed there, **only that file needs to change** — every component reads exclusively through `useTelemetryStore`, never raw WS messages.

## Fault injection for the live demo

`MissionControlPanel` sends `{ type: 'fault_injection', fault, target }` over the socket (see `FaultInjectionCommand` in `types.ts`). In demo mode (no backend), the same click also mutates the local simulator directly, so the buttons work identically whether or not FastAPI is running — useful for rehearsing the demo script ahead of time.

## Components

| File | Deliverable it covers |
|---|---|
| `components/HealthGauge.tsx` | Health score gauge with an independent confidence arc — a high score with low confidence reads as "uncertain," not "healthy" |
| `components/EngineSchematic.tsx` | 2D SVG flat-4 (Rotax 912-style) engine schematic, color-coded per cylinder/oil/cooling/fuel/electrical by health + sensor trust |
| `components/AlertFeed.tsx` | Alert feed — physics-fallback / low-confidence alerts get a distinct blue tag and dimmed treatment vs. confident ML alerts |
| `components/WatchdogBanner.tsx` | Data pipeline health, deliberately separate from engine health — a `CRITICAL_DATA_LOSS` banner can be true even while engine health looks fine |
| `components/MeasuredVsExpectedChart.tsx` | Plotly measured-vs-expected overlay with a residual trace, channel selectable |
| `components/MissionControlPanel.tsx` | Fault injection buttons for the live demo |

## Design notes

Dark operator-console theme (near-black surfaces, hairline borders, IBM Plex Mono for all data readouts) rather than a generic SaaS dashboard look — intended to read as a GCS instrument panel. Semantic color bands (`healthy / caution / warning / critical / unknown / physics_fallback`) are centralized in `types.ts` (`BAND_COLOR`) so every component stays visually consistent, including the "unknown" (sensor untrusted) and "physics fallback" (model abstained) states that a threshold-only system wouldn't distinguish.
