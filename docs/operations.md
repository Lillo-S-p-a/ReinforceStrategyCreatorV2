# ReinforceStrategyCreatorV2: System Operations & Maintenance

This document outlines the procedures and best practices for monitoring, maintaining, and troubleshooting the ReinforceStrategyCreatorV2 system.

## Monitoring

### Dashboard

The system includes a comprehensive monitoring dashboard for visualizing training progress, model performance, and episode statistics:

- **Location**: Built from the `dashboard/` module
- **Execution**: Run with `python run_dashboard.py`
- **Features**:
  - Run summary with key performance metrics
  - Detailed episode analysis with visualizations
  - Model parameter visualization and comparison
  - Trade analysis charts
  - Decision-making pattern analysis
  - Production model management interface

### Key Metrics to Monitor

#### Training Metrics
- **Episode Rewards**: Found in `training_log.csv`
- **Profit and Loss (PnL)**: Overall profitability of trading episodes
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted return metric (values > 1 are good)
- **Max Drawdown**: Largest percentage drop from peak to trough
- **Sortino Ratio**: Similar to Sharpe but only penalizes downside volatility
- **Calmar Ratio**: Return divided by maximum drawdown

#### System Metrics
- **Episode Completion Rate**: Check with `python check_episodes.py`
- **Database Connection**: Monitor when initializing runs
- **Memory Usage**: Watch during replay buffer operations
- **Training Duration**: Monitor training times for optimization

### Analysis Scripts

Several custom scripts are available for ad-hoc monitoring and analysis:

- `analyze_latest_run.py`: Provides summary statistics for the latest completed training run
- `check_episodes.py`: Shows episode completion rates and identifies incomplete episodes
- `analyze_profile.py`: Analyzes performance bottlenecks (used with profiling data)
- `check_episode_details.py`: Extracts detailed information about specific episodes
- `get_episode_metrics.py`: Retrieves specific metrics for episode analysis
- `get_run_params.py`: Extracts training run parameters for reference

## Logging System

### Log Files and Locations

#### Primary Log Files

| File | Purpose | Format | Retention |
|------|---------|--------|-----------|
| `training_log.csv` | Records detailed step-by-step training data | CSV with headers: episode, step, action, reward, balance, shares_held, current_price, portfolio_value, current_position | Keep forever for historical analysis |
| `replay_buffer_debug.log` | Captures detailed debugging information from the replay buffer | Timestamped log entries with level (INFO/ERROR) and message | Rotate weekly |

#### Application Logs

- **Dashboard Application**: Logs to stdout, redirect to file if needed
- **Training Process**: 
  - Standard output (captured when running `train.py` or `train_debug.py`)
  - Detailed logs in `training_log.csv`

### Log Format

- **Replay Buffer Logs**: Standard Python logging format with timestamps
  ```
  2025-05-14 22:06:16,941 - INFO - Applied monkey patch to EpisodeReplayBuffer._sample_episodes
  ```

- **Training Log Format**: CSV with columns tracking each training step
  ```
  episode,step,action,reward,balance,shares_held,current_price,portfolio_value,current_position
  ```

### Log Analysis Tools

The system provides several tools for analyzing log data:

- `analyze_latest_run.py`: Processes database records to analyze the latest training run
  - Calculates average Sharpe Ratio across episodes
  - Computes non-HOLD operations per episode
  - Identifies trends in agent behavior

- Dashboard visualizations: Multiple charts for analyzing training behavior
  - Price and operations charts
  - Reward distribution analysis
  - Action transition analysis
  - Decision making pattern identification

## Backup and Recovery

### Database Backup

The system uses two PostgreSQL databases (see `docker-compose.yml`):
1. **TimescaleDB** (market data): Runs on port 5433
2. **PostgreSQL** (trading data): Runs on port 5434

#### Backup Procedures

1. **TimescaleDB Backup**:
   ```bash
   docker exec timescaledb_market_data pg_dump -U ${TIMESCALEDB_USER} ${TIMESCALEDB_DB} > marketdata_backup_$(date +\%Y\%m\%d).sql
   ```

2. **PostgreSQL Trading DB Backup**:
   ```bash
   docker exec postgres_trading_db pg_dump -U ${POSTGRES_USER} ${POSTGRES_DB} > trading_db_backup_$(date +\%Y\%m\%d).sql
   ```

3. **Volume Backup** (for containerized data):
   ```bash
   docker run --rm -v timescaledb_data:/source -v $(pwd)/backups:/target alpine tar -czf /target/timescaledb_data_$(date +\%Y\%m\%d).tar.gz /source
   docker run --rm -v postgres_data:/source -v $(pwd)/backups:/target alpine tar -czf /target/postgres_data_$(date +\%Y\%m\%d).tar.gz /source
   ```

#### Backup Schedule
- Full database backups: Daily
- Transaction log backups: Every 6 hours
- Store backups for 30 days minimum

### Model Backup

#### Production Models

Production models are stored in the `production_models/` directory as JSON files. Each file contains:
- Model parameters
- Performance metrics
- Run/episode identification
- Timestamp information

#### Backup Procedures

1. **Regular File Backup**:
   ```bash
   rsync -av production_models/ /path/to/backup/models/
   ```

2. **Cloud Backup** (optional):
   ```bash
   aws s3 sync production_models/ s3://your-bucket/models/
   # OR
   gsutil -m rsync -r production_models/ gs://your-bucket/models/
   ```

### Configuration Backup

Important configuration files to back up:
- `.env`: Contains environment variables and database credentials
- `pyproject.toml`: Package dependencies and settings
- `docker-compose.yml`: Container configuration

Backup these files whenever they change, and store copies with the database backups.

### Recovery Procedures

#### Database Recovery

1. **TimescaleDB Recovery**:
   ```bash
   # Create empty DB if needed
   docker exec timescaledb_market_data createdb -U ${TIMESCALEDB_USER} ${TIMESCALEDB_DB}
   
   # Restore from backup
   cat marketdata_backup_YYYYMMDD.sql | docker exec -i timescaledb_market_data psql -U ${TIMESCALEDB_USER} ${TIMESCALEDB_DB}
   ```

2. **PostgreSQL Trading DB Recovery**:
   ```bash
   # Create empty DB if needed
   docker exec postgres_trading_db createdb -U ${POSTGRES_USER} ${POSTGRES_DB}
   
   # Restore from backup
   cat trading_db_backup_YYYYMMDD.sql | docker exec -i postgres_trading_db psql -U ${POSTGRES_USER} ${POSTGRES_DB}
   ```

3. **Volume Recovery**:
   ```bash
   # Stop containers
   docker-compose down
   
   # Remove existing volumes
   docker volume rm timescaledb_data postgres_data
   
   # Create empty volumes
   docker volume create timescaledb_data
   docker volume create postgres_data
   
   # Restore volume data
   docker run --rm -v timescaledb_data:/target -v $(pwd)/backups:/source alpine tar -xzf /source/timescaledb_data_YYYYMMDD.tar.gz -C /
   docker run --rm -v postgres_data:/target -v $(pwd)/backups:/source alpine tar -xzf /source/postgres_data_YYYYMMDD.tar.gz -C /
   
   # Restart containers
   docker-compose up -d
   ```

#### Model Recovery

To restore models:
1. Place backed-up model JSON files in the `production_models/` directory
2. The system will automatically recognize and load them through the dashboard

#### Application Recovery

To restore the application after a failure:
1. Restore configuration files (`.env`, `docker-compose.yml`)
2. Verify PostgreSQL and TimescaleDB are running and accessible
3. Initialize database schema if needed: `python init_db.py`
4. Verify the system with diagnostic scripts: `python check_episodes.py`

## Troubleshooting

### Common Issues and Solutions

#### Training Not Starting / Errors During Training

**Issue**: Training script fails to start or crashes during execution.

**Solutions**:
1. **Check Environment Variables**:
   ```bash
   # Verify .env file exists and contains necessary credentials
   cat .env | grep DATABASE_URL
   ```

2. **Verify Database Connections**:
   ```bash
   # Check if containers are running
   docker ps | grep postgres
   
   # Test connection directly
   python -c "from reinforcestrategycreator.db_utils import get_db_session; print('DB Connection OK' if get_db_session() else 'DB Connection Failed')"
   ```

3. **Check Dependencies**:
   ```bash
   # Verify all dependencies are installed
   poetry install
   
   # Verify virtual environment is active
   poetry shell
   ```

4. **Check Log Files**:
   - Review `replay_buffer_debug.log` for any exceptions
   - Look for Python exceptions in terminal output

#### Dashboard Not Loading / Errors

**Issue**: Dashboard fails to load or display data.

**Solutions**:
1. **Check API Connectivity**:
   ```bash
   # Verify API service is running
   curl http://localhost:8000/health || echo "API not responding"
   ```
   
2. **Restart Dashboard Server**:
   ```bash
   # Kill and restart process
   pkill -f "streamlit run dashboard/main.py"
   python run_dashboard.py
   ```

3. **Clear Cache**:
   ```bash
   # Clear Streamlit cache
   rm -rf ~/.streamlit/
   ```

#### Database Connection Problems

**Issue**: Application fails to connect to databases.

**Solutions**:
1. **Check Docker Containers**:
   ```bash
   # Are containers running?
   docker ps | grep -E "timescaledb|postgres"
   
   # Check logs for database errors
   docker logs timescaledb_market_data
   docker logs postgres_trading_db
   ```
   
2. **Verify Environment Variables**:
   ```bash
   # Check DATABASE_URL format
   echo $DATABASE_URL
   
   # Should look like:
   # postgresql://user:password@localhost:5434/trading_db
   ```
   
3. **Restart Database Services**:
   ```bash
   # Restart database containers
   docker-compose restart timescaledb postgres
   ```

#### Model Loading Errors

**Issue**: System fails to load saved models.

**Solutions**:
1. **Check File Permissions**:
   ```bash
   # Verify production_models directory is readable
   ls -la production_models/
   ```
   
2. **Validate Model Files**:
   ```bash
   # Check if files are valid JSON
   for f in production_models/*.json; do python -m json.tool $f > /dev/null || echo "Invalid JSON: $f"; done
   ```
   
3. **Restore from Backup**:
   If models are corrupted, restore from your latest backup as described in the recovery procedures.

#### Replay Buffer Errors

**Issue**: Training crashes with replay buffer errors (common in `replay_buffer_debug.log`).

**Solutions**:
1. **Check Memory Usage**:
   ```bash
   # Monitor memory during training
   watch -n 1 "free -m"
   ```
   
2. **Reduce Batch Size**:
   Modify the `agent_batch_size` parameter in your training configuration to use less memory.
   
3. **Clear Cached Episodes**:
   ```python
   # Add this code at the start of training to clear any stale episodes
   from reinforcestrategycreator.rl_agent import reset_agent_memory
   reset_agent_memory()
   ```

#### Incomplete Episodes

**Issue**: Episodes are not being properly completed, as detected by `check_episodes.py`.

**Solutions**:
1. **Identify Incomplete Episodes**:
   ```bash
   python check_episodes.py
   ```
   
2. **Reset Stuck Episodes** (advanced):
   ```bash
   # Create a script to reset incomplete episodes
   python -c "from reinforcestrategycreator.db_utils import get_db_session; from reinforcestrategycreator.db_models import Episode; \
   with get_db_session() as db: \
     incomplete = db.query(Episode).filter(Episode.end_time.is_(None)).all(); \
     for ep in incomplete: print(f'Would reset episode {ep.episode_id}'); \
   "
   ```

### Diagnostic Tools

- `check_run_operations.py`: Dig into the details of a specific run
- `debug_replay_buffer.py`: Debug issues with the replay buffer component
- `analyze_profile.py`: Analyze performance bottlenecks using profiling data
- `verify_operations.py`: Validate the integrity of trading operations

### Log Analysis for Troubleshooting

When troubleshooting issues, examine these log patterns:

1. **Training Log Analysis**:
   ```bash
   # Look for episodes with negative rewards
   grep -E ",-[0-9]+\.[0-9]+" training_log.csv
   
   # Find episodes with large position changes
   awk -F, '$5 > 10000 || $5 < -10000' training_log.csv
   ```

2. **Replay Buffer Errors**:
   ```bash
   # Find error messages in replay buffer log
   grep ERROR replay_buffer_debug.log
   
   # Look for specific exceptions
   grep "Exception" replay_buffer_debug.log
   ```

3. **Performance Analysis**:
   ```bash
   # Find slow database queries
   docker exec postgres_trading_db psql -U ${POSTGRES_USER} ${POSTGRES_DB} -c "SELECT query, total_time FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"
   ```

## Maintenance Tasks

### Routine Maintenance

| Task | Frequency | Procedure |
|------|-----------|-----------|
| Database Backup | Daily | Run backup scripts described in Backup section |
| Log Rotation | Weekly | Archive and compress old logs (`*.log.1`, `*.log.2`, etc.) |
| Disk Space Check | Weekly | `du -h --max-depth=1 .` to identify large directories |
| Clean Old Models | Monthly | Archive models older than 6 months |
| Database Vacuum | Monthly | `docker exec postgres_trading_db psql -U ${POSTGRES_USER} ${POSTGRES_DB} -c "VACUUM ANALYZE;"` |
| Performance Check | Monthly | Run `analyze_profile.py` to identify bottlenecks |

### System Updates

When updating the system:

1. **Before Updates**:
   - Create full backups of databases and models
   - Document current performance metrics as baseline
   
2. **During Updates**:
   - Follow a staged rollout (development → staging → production)
   - Run comprehensive tests using `run_debug_train.sh`
   
3. **After Updates**:
   - Verify database schema compatibility
   - Run comparison analysis between new and old versions
   - Update documentation if operations procedures have changed