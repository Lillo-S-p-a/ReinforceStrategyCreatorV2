# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-05-16

### Added
- Parallel training support using RLlib, Ray, and PyTorch
- Validation data evaluation during training
- Early stopping based on validation metrics
- Enhanced metrics tracking and reporting
- Database verification script (`verify_database.py`) for data integrity checks
- Database reset script (`reset_db.py`) for clean test runs
- Fix script for NULL initial portfolio values (`fix_null_initial_portfolio_values.py`)
- Comprehensive documentation in task files

### Changed
- Refactored training script (`train.py`) to support parallel training
- Improved error handling and logging in database callbacks
- Enhanced episode finalization to ensure proper database state

### Fixed
- Fixed SQLAlchemy func import in the finalize_incomplete_episodes method
- Fixed episode finalization on training termination
- Fixed trading operations logging functionality

## [1.1.0] - 2025-05-05

### Added
- Hyperparameter optimization capabilities
- Paper trading export functionality
- Improved reward function with risk management
- Enhanced technical indicators

### Changed
- Updated database schema for better metrics tracking
- Improved logging and monitoring

## [1.0.0] - 2025-04-30

### Added
- Initial release of the Reinforcement Learning Trading Strategy Creator
- Basic trading environment with DQN agent
- Database logging of training runs, episodes, steps, and trades
- Technical analysis indicators
- Basic visualization tools