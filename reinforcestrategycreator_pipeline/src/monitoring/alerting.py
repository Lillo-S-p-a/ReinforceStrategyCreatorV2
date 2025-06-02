"""
Manages alert rules and dispatches notifications through configured channels.
"""
from typing import Any, Dict, Optional, List
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import requests

from ..config.models import AlertManagerConfig, AlertRuleConfig, AlertChannelConfig, AlertChannelType
from .logger import get_logger

logger = get_logger("monitoring.alerting")

class AlertManager:
    """
    Handles incoming events, matches them against rules, and sends alerts
    through configured channels.
    """

    def __init__(self, config: AlertManagerConfig):
        """
        Initialize the AlertManager.

        Args:
            config: Configuration for the alert manager, including channels and rules.
        """
        self.config = config
        self.active_alerts: Dict[str, float] = {} # To track active alerts for deduplication: rule_name -> last_alert_timestamp
        logger.info("AlertManager initialized.")
        for channel in config.channels:
            logger.info(f"Alert channel loaded: {channel.name} (Type: {channel.type.value}), Enabled: {channel.enabled}")
            if channel.type == AlertChannelType.EMAIL and not channel.details.get("email_to"):
                logger.warning(f"Email channel '{channel.name}' is missing 'email_to' in details.")
            elif channel.type == AlertChannelType.SLACK and not channel.details.get("slack_webhook_url"):
                logger.warning(f"Slack channel '{channel.name}' is missing 'slack_webhook_url' in details.")
            elif channel.type == AlertChannelType.PAGERDUTY and not channel.details.get("pagerduty_service_key"):
                 logger.warning(f"PagerDuty channel '{channel.name}' is missing 'pagerduty_service_key' in details.")
        for rule in config.rules:
            logger.info(f"Alert rule loaded: {rule.name}, Event: {rule.event_type}, Severity: {rule.severity}, Enabled: {rule.enabled}")


    def handle_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        severity: str = "info",
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Process an event, check against rules, and dispatch alerts if necessary.

        Args:
            event_type: The type of event that occurred.
            event_data: Data associated with the event.
            severity: Severity of the event.
            tags: Optional tags for the event.
        """
        if not self.config.enabled:
            logger.debug(f"AlertManager is disabled. Skipping event: {event_type}")
            return

        logger.debug(f"Handling event: {event_type}, Severity: {severity}, Data: {event_data}")

        for rule in self.config.rules:
            if not rule.enabled:
                continue

            if rule.event_type == event_type and rule.severity == severity:
                # Basic condition check (can be expanded based on rule.conditions)
                conditions_met = self._check_rule_conditions(rule, event_data)
                
                if conditions_met:
                    logger.info(f"Rule '{rule.name}' matched for event '{event_type}'.")
                    
                    # Deduplication check
                    now = time.time()
                    if rule.name in self.active_alerts:
                        last_alert_time = self.active_alerts[rule.name]
                        if (now - last_alert_time) < rule.deduplication_window_seconds:
                            logger.info(f"Alert for rule '{rule.name}' deduplicated. Last alert was at {last_alert_time}.")
                            continue # Skip sending this alert
                    
                    self.active_alerts[rule.name] = now # Update last alert time

                    for channel_name in rule.channels:
                        channel_config = next((c for c in self.config.channels if c.name == channel_name and c.enabled), None)
                        if channel_config:
                            self._dispatch_alert(channel_config, rule, event_data, tags)
                        else:
                            logger.warning(f"Configured channel '{channel_name}' for rule '{rule.name}' not found or disabled.")
                    # break # Optional: process only the first matching rule or all
    
    def _check_rule_conditions(self, rule: AlertRuleConfig, event_data: Dict[str, Any]) -> bool:
        """Checks if the event_data meets the rule's conditions."""
        if not rule.conditions:
            return True # No specific conditions, so rule matches by event_type and severity

        for key_condition, value_target in rule.conditions.items():
            # Extract the base key by removing the operator suffix
            if key_condition.endswith(('_gt', '_gte', '_lt', '_lte', '_eq')):
                # Find the operator suffix
                for suffix in ['_gte', '_lte', '_gt', '_lt', '_eq']:
                    if key_condition.endswith(suffix):
                        base_key = key_condition[:-len(suffix)]
                        break
            else:
                base_key = key_condition
            
            actual_value = event_data.get(base_key)
            
            if actual_value is None: # Condition key not in event data
                return False

            try:
                actual_value = float(actual_value) # Assume numeric for now
            except (ValueError, TypeError):
                logger.warning(f"Could not convert event data value '{actual_value}' for condition key '{key_condition}' to float.")
                return False # Cannot compare

            if key_condition.endswith("_gt"):
                if not (actual_value > value_target):
                    return False
            elif key_condition.endswith("_gte"):
                if not (actual_value >= value_target):
                    return False
            elif key_condition.endswith("_lt"):
                if not (actual_value < value_target):
                    return False
            elif key_condition.endswith("_lte"):
                if not (actual_value <= value_target):
                    return False
            elif key_condition.endswith("_eq"):
                if not (actual_value == value_target):
                    return False
            # Add more condition types like "_contains", "_matches_regex" as needed
            else:
                logger.warning(f"Unsupported condition operator in '{key_condition}' for rule '{rule.name}'.")
                return False # Unknown condition type implies no match
        return True


    def _dispatch_alert(
        self,
        channel_config: AlertChannelConfig,
        rule: AlertRuleConfig,
        event_data: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Dispatch an alert through the specified channel.
        """
        alert_title = f"ALERT [{rule.severity.upper()}]: {rule.name} - {event_data.get('message', rule.description or rule.event_type)}"
        alert_body = f"Rule: {rule.name}\nEvent Type: {rule.event_type}\nSeverity: {rule.severity}\n"
        alert_body += f"Description: {event_data.get('message', rule.description or 'No specific message.')}\n"
        alert_body += f"Details: {event_data}\n"
        if tags:
            alert_body += f"Tags: {', '.join(tags)}\n"
        
        logger.info(f"Dispatching alert via channel '{channel_config.name}' (Type: {channel_config.type.value})")
        logger.debug(f"Alert Title: {alert_title}")
        logger.debug(f"Alert Body:\n{alert_body}")

        try:
            if channel_config.type == AlertChannelType.EMAIL:
                self._send_email_alert(channel_config, alert_title, alert_body, rule)
            elif channel_config.type == AlertChannelType.SLACK:
                self._send_slack_alert(channel_config, alert_title, alert_body, rule)
            elif channel_config.type == AlertChannelType.PAGERDUTY:
                self._send_pagerduty_alert(channel_config, alert_title, alert_body, rule, event_data)
            elif channel_config.type == AlertChannelType.DATADOG_EVENT:
                # This is typically handled by MonitoringService.log_event directly
                logger.info(f"Datadog event for rule '{rule.name}' should be handled by MonitoringService.log_event.")
            else:
                logger.warning(f"Alert channel type '{channel_config.type.value}' dispatch not implemented for rule '{rule.name}'.")
        except Exception as e:
            logger.error(f"Failed to dispatch alert via {channel_config.type.value}: {str(e)}")
    
    def _send_email_alert(self, channel_config: AlertChannelConfig, title: str, body: str, rule: AlertRuleConfig) -> None:
        """Send an email alert."""
        email_to = channel_config.details.get("email_to")
        email_from = channel_config.details.get("email_from", "noreply@monitoring.local")
        smtp_host = channel_config.details.get("smtp_host", "localhost")
        smtp_port = channel_config.details.get("smtp_port", 587)
        smtp_username = channel_config.details.get("smtp_username")
        smtp_password = channel_config.details.get("smtp_password")
        use_tls = channel_config.details.get("use_tls", True)
        
        if not email_to:
            logger.error(f"Email channel '{channel_config.name}' missing 'email_to' address")
            return
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_from
        msg['To'] = email_to if isinstance(email_to, str) else ', '.join(email_to)
        msg['Subject'] = title
        
        # Add body
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        try:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if use_tls:
                    server.starttls()
                if smtp_username and smtp_password:
                    server.login(smtp_username, smtp_password)
                
                recipients = [email_to] if isinstance(email_to, str) else email_to
                server.send_message(msg)
                
            logger.info(f"Email alert sent successfully to {email_to} for rule '{rule.name}'")
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            raise
    
    def _send_slack_alert(self, channel_config: AlertChannelConfig, title: str, body: str, rule: AlertRuleConfig) -> None:
        """Send a Slack alert via webhook."""
        webhook_url = channel_config.details.get("slack_webhook_url")
        channel = channel_config.details.get("slack_channel")  # Optional channel override
        username = channel_config.details.get("slack_username", "Monitoring Alert")
        icon_emoji = channel_config.details.get("slack_icon_emoji", ":warning:")
        
        if not webhook_url:
            logger.error(f"Slack channel '{channel_config.name}' missing 'slack_webhook_url'")
            return
        
        # Format message for Slack
        slack_message = {
            "username": username,
            "icon_emoji": icon_emoji,
            "attachments": [
                {
                    "color": self._get_severity_color(rule.severity),
                    "title": title,
                    "text": body,
                    "footer": "Monitoring System",
                    "ts": int(time.time())
                }
            ]
        }
        
        if channel:
            slack_message["channel"] = channel
        
        # Send to Slack
        try:
            response = requests.post(webhook_url, json=slack_message, timeout=10)
            response.raise_for_status()
            logger.info(f"Slack alert sent successfully for rule '{rule.name}'")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
            raise
    
    def _send_pagerduty_alert(self, channel_config: AlertChannelConfig, title: str, body: str, 
                             rule: AlertRuleConfig, event_data: Dict[str, Any]) -> None:
        """Send a PagerDuty alert."""
        service_key = channel_config.details.get("pagerduty_service_key")
        api_url = channel_config.details.get("pagerduty_api_url", "https://events.pagerduty.com/v2/enqueue")
        
        if not service_key:
            logger.error(f"PagerDuty channel '{channel_config.name}' missing 'pagerduty_service_key'")
            return
        
        # Create PagerDuty event
        pagerduty_event = {
            "routing_key": service_key,
            "event_action": "trigger",
            "payload": {
                "summary": title,
                "severity": self._map_severity_to_pagerduty(rule.severity),
                "source": "monitoring-system",
                "custom_details": {
                    "body": body,
                    "event_data": event_data,
                    "rule_name": rule.name
                }
            },
            "dedup_key": f"{rule.name}_{rule.event_type}"  # For deduplication
        }
        
        # Send to PagerDuty
        try:
            response = requests.post(api_url, json=pagerduty_event, timeout=10)
            response.raise_for_status()
            logger.info(f"PagerDuty alert sent successfully for rule '{rule.name}'")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send PagerDuty alert: {str(e)}")
            raise
    
    def _get_severity_color(self, severity: str) -> str:
        """Map severity to Slack color."""
        colors = {
            "info": "#36a64f",     # Green
            "warning": "#ff9800",  # Orange
            "error": "#f44336",    # Red
            "critical": "#9c27b0"  # Purple
        }
        return colors.get(severity.lower(), "#808080")  # Default gray
    
    def _map_severity_to_pagerduty(self, severity: str) -> str:
        """Map our severity to PagerDuty severity."""
        mapping = {
            "info": "info",
            "warning": "warning",
            "error": "error",
            "critical": "critical"
        }
        return mapping.get(severity.lower(), "warning")