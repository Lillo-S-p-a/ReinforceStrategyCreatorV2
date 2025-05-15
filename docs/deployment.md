# ReinforceStrategyCreatorV2: System Deployment

This document provides detailed instructions for deploying the ReinforceStrategyCreatorV2 system across various environments (development, testing/staging, and production). It covers deployment processes, configurations, CI/CD pipelines, and rollback procedures.

## Deployment Overview

The ReinforceStrategyCreatorV2 system follows a containerized deployment strategy with the following key components:

- **Database Layer**: Multiple databases running in Docker containers:
  - TimescaleDB for market data with time-series extensions
  - PostgreSQL for trading data and application state
- **Application Components**:
  - Training Scripts (`run_train.sh`, `run_debug_train.sh`)
  - Analytics Dashboard (`run_dashboard.py`)
  - API Server (FastAPI)
- **Model Artifacts**:
  - Trained models stored in `models/` directory for development
  - Production-ready models stored in `production_models/` directory

The system uses environment variables for configuration across environments, allowing for consistent deployment across different setups.

## Development Environment Deployment

The development environment deployment focuses on local development and testing of features.

### Database Deployment

1. Ensure Docker and Docker Compose are installed and running.
2. Set environment variables in `.env` file:

```env
# TimescaleDB (marketdata)
TIMESCALEDB_USER=user
TIMESCALEDB_PASSWORD=password
TIMESCALEDB_DB=marketdata

# PostgreSQL (trading_db)
POSTGRES_USER=postgres
POSTGRES_PASSWORD=mysecretpassword
POSTGRES_DB=trading_db

# Connection string for application
DATABASE_URL=postgresql://postgres:mysecretpassword@localhost:5434/trading_db
API_KEY=test-key-123
```

3. Start the database services:

```bash
docker-compose up -d
```

4. Initialize the database schema:

```bash
poetry run python init_db.py
```

### Application Deployment

1. Start the training process with standard configuration:

```bash
./run_train.sh
```

This script:
- Sets environment variables to suppress warnings
- Disables deprecation warnings for Ray
- Runs the training script with the specified parameters

2. For debugging or development with enhanced logging:

```bash
./run_debug_train.sh
```

This script:
- Runs the debug version of the training script
- Outputs detailed logging to `replay_buffer_debug.log`

3. Deploy the dashboard for visualization and analysis:

```bash
poetry run python run_dashboard.py
```

The dashboard will be accessible at http://localhost:8501 by default.

4. Deploy the API service:

```bash
poetry run uvicorn reinforcestrategycreator.api.main:app --reload
```

The API will be accessible at http://localhost:8000 with documentation at http://localhost:8000/docs.

## Testing/Staging Environment Deployment

The testing/staging environment mirrors the production setup but with isolated resources for quality assurance.

### Database Deployment

1. Configure a separate `.env` file for testing/staging:

```env
# TimescaleDB (marketdata) - Test instance
TIMESCALEDB_USER=test_user
TIMESCALEDB_PASSWORD=test_password
TIMESCALEDB_DB=marketdata_test

# PostgreSQL (trading_db) - Test instance
POSTGRES_USER=postgres
POSTGRES_PASSWORD=test_password
POSTGRES_DB=trading_db_test

# Connection string for application
DATABASE_URL=postgresql://postgres:test_password@localhost:5434/trading_db_test
API_KEY=test-key-123
```

2. Start the database services with the test configuration:

```bash
docker-compose -f docker-compose.yml -f docker-compose.test.yml up -d
```

> Note: If a separate `docker-compose.test.yml` doesn't exist, you can use the base `docker-compose.yml` with the test-specific `.env` file.

### Application Deployment

1. Deploy the application components using the test configuration:

```bash
# Initialize test database
poetry run python init_db.py

# Run training with test parameters
./run_train.sh --test

# Deploy dashboard for test data
poetry run python run_dashboard.py --test

# Deploy API service in test mode
poetry run uvicorn reinforcestrategycreator.api.main:app --reload --env-file .env.test
```

## Production Environment Deployment

The production environment prioritizes stability, security, and performance for live trading operations.

### Pre-deployment Preparation

1. **Security Hardening**:
   - Replace default passwords with strong, unique credentials
   - Configure proper firewall rules
   - Set up SSL certificates for API endpoints
   - Review network security settings for containers

2. **Resource Allocation**:
   - Allocate appropriate CPU, memory, and storage resources
   - Consider dedicated hardware for databases
   - Configure swapping and memory limits for containers

### Database Deployment

1. Configure a production `.env` file:

```env
# TimescaleDB (marketdata) - Production
TIMESCALEDB_USER=prod_user
TIMESCALEDB_PASSWORD=<strong-password>
TIMESCALEDB_DB=marketdata_prod

# PostgreSQL (trading_db) - Production
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<strong-password>
POSTGRES_DB=trading_db_prod

# Connection string for application
DATABASE_URL=postgresql://postgres:<strong-password>@db-host:5432/trading_db_prod
API_KEY=<production-api-key>
```

2. Deploy the database containers with production configuration:

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

3. Set up automated backups for databases:

```bash
# Example cron job for daily backups
0 2 * * * docker exec postgres_trading_db pg_dump -U postgres trading_db_prod > /backup/trading_db_$(date +\%Y\%m\%d).sql
```

### Application Deployment

1. Deploy trained models to production:

```bash
# Copy a specific model to the production directory
cp models/model_ep<episode_id>_run<run_id>_<timestamp>.json production_models/
```

2. Deploy the API service with production settings:

```bash
# Using gunicorn for production deployment
poetry run gunicorn -w 4 -k uvicorn.workers.UvicornWorker reinforcestrategycreator.api.main:app
```

3. For dashboard deployment in production (if needed):

```bash
# Run with production flag
poetry run streamlit run dashboard/main.py --server.port=8501 --server.address=0.0.0.0 --env=prod
```

## CI/CD Pipeline

> Note: The project currently uses manual deployment processes. No automated CI/CD pipeline is currently implemented.

### Manual Deployment Process

The current deployment workflow follows these steps:

1. **Development Phase**:
   - Develop features locally
   - Test functionality in development environment
   - Push changes to version control

2. **Testing Phase**:
   - Pull changes in testing environment
   - Deploy databases and application services
   - Run validation tests
   - Review results

3. **Production Release**:
   - After successful testing, pull changes to production environment
   - Deploy database changes if needed
   - Start updated services
   - Monitor system behavior
   - Store trained models in `production_models/` directory

### Future CI/CD Recommendations

For future implementation, the following CI/CD pipeline is recommended:

1. Set up GitHub Actions or Jenkins for automated testing
2. Implement automated testing on pull requests
3. Create staging deployment on successful test completion
4. Set up approval workflows for production deployments
5. Automate Docker image building and publishing
6. Introduce semantic versioning for releases

## Rollback Procedures

### Development Environment Rollback

1. **Code Rollback**:

```bash
git checkout <previous-commit>
poetry install  # Reinstall dependencies if changed
```

2. **Database Rollback**:
   - The simplest approach is to recreate the development databases:

```bash
docker-compose down -v  # Remove containers and volumes
docker-compose up -d    # Recreate containers
poetry run python init_db.py  # Reinitialize schema
```

### Testing/Staging Environment Rollback

1. **Code Rollback**:

```bash
git checkout <previous-stable-tag>
poetry install
```

2. **Database Rollback**:
   - Restore from the most recent staging backup:

```bash
docker cp /backup/trading_db_<date>.sql postgres_trading_db:/tmp/
docker exec -it postgres_trading_db bash
psql -U postgres trading_db_test < /tmp/trading_db_<date>.sql
```

### Production Environment Rollback

1. **Model Rollback**:
   - Switch back to the previous stable model:

```bash
# Remove symlink to current model (if using symlinks)
rm production_models/current_model.json
# Link to previous stable model
ln -s production_models/model_ep<previous_episode>_run<previous_run>_<timestamp>.json production_models/current_model.json
```

2. **Code Rollback**:

```bash
git checkout <previous-production-tag>
poetry install
# Restart services
systemctl restart api-service
systemctl restart dashboard-service
```

3. **Database Rollback**:
   - For serious issues, restore from the most recent backup:

```bash
# Stop services that connect to the database
systemctl stop api-service
systemctl stop dashboard-service

# Restore database from backup
docker cp /backup/trading_db_prod_<date>.sql postgres_trading_db:/tmp/
docker exec -it postgres_trading_db bash
psql -U postgres trading_db_prod < /tmp/trading_db_prod_<date>.sql

# Restart services
systemctl start api-service
systemctl start dashboard-service
```

4. **Monitoring After Rollback**:
   - Monitor system logs for errors
   - Verify API endpoints are functioning correctly
   - Check dashboard visualizations and data integrity

## Deployment Checklist

Use this checklist for each deployment:

1. ☐ Back up relevant databases
2. ☐ Verify all tests pass in development environment
3. ☐ Deploy database changes (if any)
4. ☐ Deploy application code updates
5. ☐ Verify services start correctly
6. ☐ Run basic health checks on API endpoints
7. ☐ Verify dashboard functionality
8. ☐ Check logs for errors or warnings
9. ☐ Update documentation if procedures changed
10. ☐ Communicate deployment completion to stakeholders