# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run dev       # Start dev server with HMR
npm run build     # Production build
npm run lint      # Run ESLint
npm run preview   # Preview production build locally
```

No test framework is configured.

## Architecture

This is a React 19 + Vite app using D3.js (`d3@7`) to render an interactive **knowledge graph** for spaced-repetition learning.

The application is almost entirely contained in a single large file:

- `src/knowledge-graph-app.jsx` — all state management, D3 simulation, UI panels, and algorithm logic
- `src/App.jsx` — thin wrapper that renders `<KnowledgeGraphApp />`
- `src/main.jsx` — entry point, mounts App in StrictMode

### Key concepts inside `knowledge-graph-app.jsx`

The file is structured with labeled sections (`// ====== SECTION NAME ======`):

- **CONFIG** — tuning constants for the spaced-repetition algorithm (decay `lambda`, learning rate `eta`, priority weight `w`, Leitner box intervals, etc.)
- **Algorithm helpers** — `mastery()`, `updateConfidence()`, `priority()` implement the core SRS scoring logic
- **Storage** — `saveData` / `loadData` persist the entire graph state to `localStorage` under the key `kg-data`
- **Sample data** — `generateSampleData()` seeds a Chinese math knowledge tree (limits, derivatives, integrals, etc.) with pre-existing questions
- **D3 simulation** — a force-directed graph renders nodes (knowledge points) as colored circles; color encodes mastery level via `masteryColor()`
- **React state** — `nodes` and `questions` are plain objects keyed by ID; `state` holds `globalStep` and `recentQuestions`

### Data model

- **Node** (`knowledgePoint`): `{ id, name, parentId, confidence, totalChecks, totalErrors, lastCheckStep, forgetTendency }`
- **Question**: `{ id, content, knowledgePointIds[], difficulty, status, failedKnowledgePointIds[], history[], leitnerBox, lastReviewStep }`

### ESLint

Uses flat config (`eslint.config.js`). The `no-unused-vars` rule ignores variables matching `/^[A-Z_]/` (constants). Babel-based Fast Refresh via `@vitejs/plugin-react`.
