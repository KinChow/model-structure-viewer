# Changelog

All notable changes to Model Structure Viewer will be documented in this file.

This project follows semantic versioning once releases are cut:

- `MAJOR`: incompatible API, CLI, or persisted cache format changes.
- `MINOR`: backward-compatible features, model support, UI improvements, or new diagnostics.
- `PATCH`: backward-compatible fixes, documentation updates, and test-only changes.

## [Unreleased]

### Added

- Started formal version tracking with this changelog.
- Added frontend status diagnostics that distinguish live meta introspection from config fallback paths.
- Added frontend unit tests for diagram viewport fitting and structure-status messaging.

### Fixed

- Added in-process structure response caching to avoid repeated expensive meta introspection for identical requests.
- Added a layered structure-generation path: config-first output, generic resource-budget gating, and isolated worker introspection with config fallback diagnostics when the worker fails or times out.
- Improved config fallback output for nested `text_config` blocks by exposing decoder layer counts under the nested text node.
- Centered and fit the Architecture diagram by default, while keeping zoom controls relative to the fitted view.
- Collapsed repeated layer patterns such as `A x3 + B + A x3 + B` into a single pattern group without hiding incomplete tails.
- Improved loading and diagnostics UI copy by showing `Generating...` with busy state and status chips for fallback reasons.

### Known Issues

- Unsupported or over-budget model structures may still fall back to config-derived views; the UI now exposes this as a warning status instead of hiding it.

## [0.1.0] - 2026-07-04

### Added

- Configuration-driven model structure viewer for local model folders and Hugging Face model ids.
- FastAPI service with endpoints for local model listing, Hugging Face lookup, structure generation, export, and settings.
- React frontend with source selection, local cache drawer, summary chips, Architecture, Layers, Export, and Raw Config tabs.
- Local model cache layout under `$MODEL_ROOT/<org>/<model>/config.json`.
- Hugging Face metadata caching for `config.json`, `README.md`, `configuration_*.py`, `modeling_*.py`, and `tokenization_*.py`, while excluding model weights.
- Meta-device model introspection with config-only fallback diagnostics.
- Repair strategies for known remote-code/config compatibility cases, including DeepSeek import compatibility and MiniMax-M3 config adaptation.
- Export support for JSON, Mermaid, and DOT.
- Test coverage for resolver behavior, API structure responses, repair strategies, folding, fallback construction, and exporters.

### Notes

- The tool is intended for structure inspection from model configuration and metadata. It does not download weights or run inference.
