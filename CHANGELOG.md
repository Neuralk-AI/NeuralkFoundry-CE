# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Workflow caches now garbage-collect overlapping step outputs after each
  run. When a later step re-emits a field, the earlier step's copy on disk
  is removed and its ``_executed.json`` records a ``postponed`` entry. The
  step loader honors this marker so cached runs skip re-execution even
  though some on-disk artifacts are missing locally.

## [0.1.0] - 2025-07-18

### Added
- Initial public release.
- Core workflow engine
