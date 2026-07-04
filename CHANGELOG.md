# Changelog

All notable changes to Model Structure Viewer will be documented in this file.

This project follows semantic versioning once releases are cut:

- `MAJOR`: incompatible API, CLI, or persisted cache format changes.
- `MINOR`: backward-compatible features, model support, UI improvements, or new diagnostics.
- `PATCH`: backward-compatible fixes, documentation updates, and test-only changes.

## [Unreleased]

### Added

- Started formal version tracking with this changelog.

### Known Issues

- Large-model meta introspection can exceed local memory and terminate the API process, observed with repeated `zai-org/GLM-5.2` structure generation on a 16 GB machine.
- The Architecture view opens at the diagram origin instead of fitting or centering the model graph in the viewport.
- Layer folding only groups consecutive isomorphic layers; alternating repeated patterns such as `x3 + single + x3` remain visually fragmented.

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
