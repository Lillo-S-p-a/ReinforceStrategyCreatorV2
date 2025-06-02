# Datadog Dashboard Templates

This directory contains JSON templates for Datadog dashboards designed to monitor the ML model pipeline in production.

## Available Dashboards

### 1. Production Monitoring Dashboard (`production_monitoring_dashboard.json`)
**Purpose**: Comprehensive overview of the entire production system.

**Key Widgets**:
- Model Performance Overview (Accuracy, Confidence)
- Data Drift Detection (PSI scores)
- Model Drift Detection (Performance metrics, F1 score)
- System Health Metrics (Latency, Error rates, Request volume)
- Alert Summary

**Use Case**: Primary dashboard for operations team to monitor overall system health.

### 2. Model Performance Dashboard (`model_performance_dashboard.json`)
**Purpose**: Real-time tracking of model performance and financial metrics.

**Key Widgets**:
- Real-time Model Performance (Accuracy, F1, Prediction Volume, Confidence)
- P&L Tracking (Cumulative and Daily P&L for trading models)
- Performance Metrics Over Time (Accuracy trends, Precision vs Recall)
- Trading Performance Metrics (Sharpe Ratio, Win Rate, Max Drawdown)

**Use Case**: For data scientists and traders to monitor model effectiveness and profitability.

### 3. Drift Detection Dashboard (`drift_detection_dashboard.json`)
**Purpose**: Detailed view of data and model drift indicators.

**Key Widgets**:
- Data Drift Overview (PSI and KS statistics by feature)
- Feature Distribution Changes (Top drifting features, mean/std changes)
- Model Drift Indicators (Performance degradation, confidence distribution)
- Drift Detection Alerts (Timeline and summary)
- Statistical Test Results (Chi-squared p-values)

**Use Case**: For ML engineers to investigate drift issues and determine when retraining is needed.

### 4. System Health Dashboard (`system_health_dashboard.json`)
**Purpose**: Infrastructure and system health monitoring.

**Key Widgets**:
- Service Health Overview (Uptime, Active versions, Request rate, Error rate)
- Latency Metrics (p50/p95/p99 distributions, latency by endpoint)
- Error Analysis (Errors by type, top error messages)
- Resource Utilization (CPU, Memory, Disk I/O)
- Model Version Deployment Status

**Use Case**: For DevOps/SRE teams to monitor infrastructure health and troubleshoot issues.

## Importing Dashboards to Datadog

1. **Via Datadog UI**:
   ```
   1. Log in to your Datadog account
   2. Navigate to Dashboards → New Dashboard
   3. Click the settings gear icon → Import Dashboard JSON
   4. Copy and paste the JSON content from the desired template
   5. Click Import
   ```

2. **Via Datadog API**:
   ```bash
   # Set your API and APP keys
   export DD_API_KEY="your_api_key"
   export DD_APP_KEY="your_app_key"
   
   # Import a dashboard
   curl -X POST "https://api.datadoghq.com/api/v1/dashboard" \
     -H "Content-Type: application/json" \
     -H "DD-API-KEY: ${DD_API_KEY}" \
     -H "DD-APPLICATION-KEY: ${DD_APP_KEY}" \
     -d @production_monitoring_dashboard.json
   ```

3. **Via Terraform** (if using Infrastructure as Code):
   ```hcl
   resource "datadog_dashboard_json" "production_monitoring" {
     dashboard = file("${path.module}/production_monitoring_dashboard.json")
   }
   ```

## Customization

### Template Variables
All dashboards include template variables that allow filtering:
- `$model_version`: Filter by specific model version or use `*` for all
- `$environment`: Filter by environment (production, staging, development)
- `$feature`: (Drift dashboard only) Filter by specific feature name

### Metric Naming Convention
The dashboards expect metrics with the following naming pattern:
- `model_pipeline.*`: All custom metrics from the pipeline
- `system.*`: System-level metrics (CPU, memory)
- `docker.*`: Container metrics (if using Docker)

### Alert Integration
The dashboards reference monitors with specific tags:
- `model_pipeline`: Tag for all pipeline-related monitors
- `production`: Environment tag
- Monitor IDs can be updated in the dashboard JSON if needed

## Required Metrics

Ensure your monitoring service is sending these metrics to Datadog:

### Model Performance Metrics
- `model_pipeline.model_accuracy`
- `model_pipeline.model_f1_score`
- `model_pipeline.model_precision`
- `model_pipeline.model_recall`
- `model_pipeline.prediction_confidence`
- `model_pipeline.predictions` (count)

### Drift Metrics
- `model_pipeline.data_drift_score`
- `model_pipeline.data_drift_psi`
- `model_pipeline.data_drift_ks`
- `model_pipeline.chi2_pvalue`
- `model_pipeline.model_drift_accuracy`
- `model_pipeline.model_drift_confidence`
- `model_pipeline.model_drift_performance_degradation`

### System Metrics
- `model_pipeline.request_latency`
- `model_pipeline.requests` (count)
- `model_pipeline.errors` (count with error_type tag)
- `model_pipeline.deployment_status`

### Trading Metrics (if applicable)
- `model_pipeline.trade_pnl`
- `model_pipeline.sharpe_ratio_30d`
- `model_pipeline.win_rate`
- `model_pipeline.max_drawdown`
- `model_pipeline.avg_trade_return`

## Events

The dashboards also display events:
- `data_drift_detected`: Triggered when data drift exceeds threshold
- `model_drift_detected`: Triggered when model drift is detected

## Best Practices

1. **Dashboard Organization**:
   - Pin the Production Monitoring Dashboard as the default view
   - Create dashboard lists for different teams (Ops, ML, Trading)
   - Use consistent color schemes across dashboards

2. **Alert Configuration**:
   - Set up monitors for critical metrics shown in dashboards
   - Configure alert channels (Slack, PagerDuty) for different severity levels
   - Use composite monitors to reduce alert fatigue

3. **Performance**:
   - Limit time ranges for heavy queries
   - Use dashboard caching for frequently accessed views
   - Consider creating simplified mobile views for on-call

4. **Access Control**:
   - Set appropriate permissions for each dashboard
   - Create read-only versions for stakeholders
   - Use teams to manage access at scale

## Troubleshooting

### Missing Metrics
If widgets show "No data":
1. Verify the metric names match your implementation
2. Check that tags are correctly applied
3. Ensure the Datadog agent is configured and running
4. Verify API keys have correct permissions

### Performance Issues
If dashboards load slowly:
1. Reduce the default time range
2. Limit the number of unique tag values
3. Use metric aggregation where possible
4. Consider splitting into multiple focused dashboards

## Support

For issues or questions:
1. Check Datadog agent logs for metric submission errors
2. Use Datadog's Metrics Explorer to verify metric availability
3. Consult the monitoring service implementation in `src/monitoring/`