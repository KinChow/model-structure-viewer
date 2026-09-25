# ELK-only production switch and browser validation (2026-09-25)

## Scope

The viewer production diagram path now uses one ELK compound layout and one ELK
orthogonal routing pass. The previous Libavoid router and post-layout coordinate
mutation path were removed. Graph IR v2 remains the fact source; virtual lanes
exist only in Layout IR.

## Implementation

- `frontend/src/diagram/ReactFlowStructureDiagram.jsx` calls
  `layoutGraphWithElkOnly`.
- `frontend/src/diagram/elkOnlyLayout.js` builds input/main/auxiliary layout
  compounds, boundary ports, and ELK sections.
- `frontend/src/diagram/elkLayout.js` remains a compatibility re-export only.
- `@mr_mint/elkjs-libavoid` was removed from frontend dependencies.

## Verification

- Frontend targeted layout tests: **103/103 passed**.
- Frontend full unit suite before production cleanup: **619/619 passed**.
- Production build after cleanup: **passed**.
- Desktop Chrome architecture E2E: **3/3 passed**.
- All expanded built-in browser audit: **60/60 models**, **0 errors**, **0 page errors**,
  **0 potential edge occlusions**, **0 zero-tile models**.
- Representative Chrome deep-expansion smoke test:
  - DeepSeek-V4.1-Flash
  - DeepSeek-V4-Flash-0731
  - Qwen3.8-Flash-Next
  - GLM-5.2
  - all loaded and expanded with no page errors or routing alert.

Raw audit: `generated/elk-only-production-expanded-audit-2026-09-25.json`.

## Remaining validation

The production switch is complete, but the final gate still requires a fresh full
unit run after dependency removal and the mobile family E2E matrix. Existing
compare/layout baseline issues remain out of scope unless reproduced by the new
ELK-only path.
