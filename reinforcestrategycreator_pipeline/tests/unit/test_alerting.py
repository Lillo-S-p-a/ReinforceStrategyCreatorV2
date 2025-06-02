"""
Unit tests for alerting components.
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import smtplib
import requests

from src.config.models import (
    AlertManagerConfig, AlertRuleConfig, AlertChannelConfig, AlertChannelType
)
from src.monitoring.alerting import AlertManager


class TestAlertManager:
    """Test cases for AlertManager."""
    
    def test_init(self):
        """Test AlertManager initialization."""
        channels = [
            AlertChannelConfig(
                type=AlertChannelType.EMAIL,
                name="email-channel",
                enabled=True,
                details={"email_to": "test@example.com"}
            ),
            AlertChannelConfig(
                type=AlertChannelType.SLACK,
                name="slack-channel",
                enabled=False,
                details={"slack_webhook_url": "https://hooks.slack.com/test"}
            )
        ]
        rules = [
            AlertRuleConfig(
                name="test-rule",
                event_type="test_event",
                severity="warning",
                channels=["email-channel"]
            )
        ]
        
        config = AlertManagerConfig(
            enabled=True,
            channels=channels,
            rules=rules
        )
        
        manager = AlertManager(config)
        assert manager.config == config
        assert manager.active_alerts == {}
    
    def test_handle_event_manager_disabled(self):
        """Test that events are ignored when manager is disabled."""
        config = AlertManagerConfig(enabled=False)
        manager = AlertManager(config)
        
        # Should not raise any errors
        manager.handle_event("test_event", {"data": "test"})
    
    def test_handle_event_rule_disabled(self):
        """Test that disabled rules are skipped."""
        rules = [
            AlertRuleConfig(
                name="disabled-rule",
                event_type="test_event",
                severity="warning",
                enabled=False,
                channels=["test-channel"]
            )
        ]
        
        config = AlertManagerConfig(
            enabled=True,
            rules=rules
        )
        manager = AlertManager(config)
        
        with patch.object(manager, '_dispatch_alert') as mock_dispatch:
            manager.handle_event("test_event", {"data": "test"}, severity="warning")
            mock_dispatch.assert_not_called()
    
    def test_handle_event_matching_rule(self):
        """Test event handling with matching rule."""
        channels = [
            AlertChannelConfig(
                type=AlertChannelType.EMAIL,
                name="test-channel",
                enabled=True,
                details={"email_to": "test@example.com"}
            )
        ]
        rules = [
            AlertRuleConfig(
                name="test-rule",
                event_type="drift_detected",
                severity="warning",
                channels=["test-channel"]
            )
        ]
        
        config = AlertManagerConfig(
            enabled=True,
            channels=channels,
            rules=rules
        )
        manager = AlertManager(config)
        
        with patch.object(manager, '_dispatch_alert') as mock_dispatch:
            manager.handle_event("drift_detected", {"score": 0.5}, severity="warning")
            mock_dispatch.assert_called_once()
    
    def test_handle_event_deduplication(self):
        """Test alert deduplication within window."""
        channels = [
            AlertChannelConfig(
                type=AlertChannelType.EMAIL,
                name="test-channel",
                enabled=True,
                details={"email_to": "test@example.com"}
            )
        ]
        rules = [
            AlertRuleConfig(
                name="test-rule",
                event_type="test_event",
                severity="warning",
                channels=["test-channel"],
                deduplication_window_seconds=5
            )
        ]
        
        config = AlertManagerConfig(
            enabled=True,
            channels=channels,
            rules=rules
        )
        manager = AlertManager(config)
        
        with patch.object(manager, '_dispatch_alert') as mock_dispatch:
            # First event should trigger alert
            manager.handle_event("test_event", {"data": "test"}, severity="warning")
            assert mock_dispatch.call_count == 1
            
            # Second event within window should be deduplicated
            manager.handle_event("test_event", {"data": "test"}, severity="warning")
            assert mock_dispatch.call_count == 1  # Still 1
            
            # Wait for deduplication window to pass
            time.sleep(6)
            
            # Third event should trigger alert
            manager.handle_event("test_event", {"data": "test"}, severity="warning")
            assert mock_dispatch.call_count == 2
    
    def test_check_rule_conditions_no_conditions(self):
        """Test rule condition check with no conditions."""
        rule = AlertRuleConfig(
            name="test-rule",
            event_type="test_event",
            severity="warning",
            channels=[]
        )
        
        config = AlertManagerConfig(rules=[rule])
        manager = AlertManager(config)
        
        assert manager._check_rule_conditions(rule, {"any": "data"}) is True
    
    def test_check_rule_conditions_gt(self):
        """Test rule condition check with greater than."""
        rule = AlertRuleConfig(
            name="test-rule",
            event_type="test_event",
            severity="warning",
            channels=[],
            conditions={"score_gt": 0.5}
        )
        
        config = AlertManagerConfig(rules=[rule])
        manager = AlertManager(config)
        
        assert manager._check_rule_conditions(rule, {"score": 0.6}) is True
        assert manager._check_rule_conditions(rule, {"score": 0.4}) is False
        assert manager._check_rule_conditions(rule, {"score": 0.5}) is False
    
    def test_check_rule_conditions_gte(self):
        """Test rule condition check with greater than or equal."""
        rule = AlertRuleConfig(
            name="test-rule",
            event_type="test_event",
            severity="warning",
            channels=[],
            conditions={"score_gte": 0.5}
        )
        
        config = AlertManagerConfig(rules=[rule])
        manager = AlertManager(config)
        
        assert manager._check_rule_conditions(rule, {"score": 0.6}) is True
        assert manager._check_rule_conditions(rule, {"score": 0.5}) is True
        assert manager._check_rule_conditions(rule, {"score": 0.4}) is False
    
    def test_check_rule_conditions_lt(self):
        """Test rule condition check with less than."""
        rule = AlertRuleConfig(
            name="test-rule",
            event_type="test_event",
            severity="warning",
            channels=[],
            conditions={"score_lt": 0.5}
        )
        
        config = AlertManagerConfig(rules=[rule])
        manager = AlertManager(config)
        
        assert manager._check_rule_conditions(rule, {"score": 0.4}) is True
        assert manager._check_rule_conditions(rule, {"score": 0.6}) is False
        assert manager._check_rule_conditions(rule, {"score": 0.5}) is False
    
    def test_check_rule_conditions_missing_data(self):
        """Test rule condition check with missing event data."""
        rule = AlertRuleConfig(
            name="test-rule",
            event_type="test_event",
            severity="warning",
            channels=[],
            conditions={"score_gt": 0.5}
        )
        
        config = AlertManagerConfig(rules=[rule])
        manager = AlertManager(config)
        
        assert manager._check_rule_conditions(rule, {"other": "data"}) is False
    
    @patch('smtplib.SMTP')
    def test_send_email_alert(self, mock_smtp):
        """Test email alert sending."""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        channel_config = AlertChannelConfig(
            type=AlertChannelType.EMAIL,
            name="email-channel",
            enabled=True,
            details={
                "email_to": "test@example.com",
                "email_from": "alerts@monitoring.com",
                "smtp_host": "smtp.example.com",
                "smtp_port": 587,
                "smtp_username": "user",
                "smtp_password": "pass",
                "use_tls": True
            }
        )
        
        rule = AlertRuleConfig(
            name="test-rule",
            event_type="test_event",
            severity="warning",
            channels=["email-channel"]
        )
        
        config = AlertManagerConfig(channels=[channel_config], rules=[rule])
        manager = AlertManager(config)
        
        manager._send_email_alert(channel_config, "Test Alert", "Alert body", rule)
        
        mock_smtp.assert_called_once_with("smtp.example.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user", "pass")
        mock_server.send_message.assert_called_once()
    
    @patch('smtplib.SMTP')
    def test_send_email_alert_multiple_recipients(self, mock_smtp):
        """Test email alert with multiple recipients."""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        channel_config = AlertChannelConfig(
            type=AlertChannelType.EMAIL,
            name="email-channel",
            enabled=True,
            details={
                "email_to": ["test1@example.com", "test2@example.com"],
                "smtp_host": "localhost"
            }
        )
        
        rule = AlertRuleConfig(
            name="test-rule",
            event_type="test_event",
            severity="warning",
            channels=["email-channel"]
        )
        
        config = AlertManagerConfig(channels=[channel_config], rules=[rule])
        manager = AlertManager(config)
        
        manager._send_email_alert(channel_config, "Test Alert", "Alert body", rule)
        
        # Check that message was sent
        mock_server.send_message.assert_called_once()
        sent_msg = mock_server.send_message.call_args[0][0]
        assert "test1@example.com, test2@example.com" in sent_msg['To']
    
    @patch('requests.post')
    def test_send_slack_alert(self, mock_post):
        """Test Slack alert sending."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        channel_config = AlertChannelConfig(
            type=AlertChannelType.SLACK,
            name="slack-channel",
            enabled=True,
            details={
                "slack_webhook_url": "https://hooks.slack.com/test",
                "slack_channel": "#alerts",
                "slack_username": "Alert Bot",
                "slack_icon_emoji": ":robot_face:"
            }
        )
        
        rule = AlertRuleConfig(
            name="test-rule",
            event_type="test_event",
            severity="error",
            channels=["slack-channel"]
        )
        
        config = AlertManagerConfig(channels=[channel_config], rules=[rule])
        manager = AlertManager(config)
        
        manager._send_slack_alert(channel_config, "Test Alert", "Alert body", rule)
        
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://hooks.slack.com/test"
        
        json_data = call_args[1]['json']
        assert json_data['username'] == "Alert Bot"
        assert json_data['icon_emoji'] == ":robot_face:"
        assert json_data['channel'] == "#alerts"
        assert len(json_data['attachments']) == 1
        assert json_data['attachments'][0]['color'] == "#f44336"  # Red for error
    
    @patch('requests.post')
    def test_send_pagerduty_alert(self, mock_post):
        """Test PagerDuty alert sending."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        channel_config = AlertChannelConfig(
            type=AlertChannelType.PAGERDUTY,
            name="pagerduty-channel",
            enabled=True,
            details={
                "pagerduty_service_key": "test-service-key"
            }
        )
        
        rule = AlertRuleConfig(
            name="test-rule",
            event_type="critical_error",
            severity="critical",
            channels=["pagerduty-channel"]
        )
        
        config = AlertManagerConfig(channels=[channel_config], rules=[rule])
        manager = AlertManager(config)
        
        event_data = {"error": "System failure", "component": "database"}
        manager._send_pagerduty_alert(channel_config, "Critical Alert", "Alert body", rule, event_data)
        
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://events.pagerduty.com/v2/enqueue"
        
        json_data = call_args[1]['json']
        assert json_data['routing_key'] == "test-service-key"
        assert json_data['event_action'] == "trigger"
        assert json_data['payload']['severity'] == "critical"
        assert json_data['payload']['summary'] == "Critical Alert"
        assert json_data['dedup_key'] == "test-rule_critical_error"
    
    def test_get_severity_color(self):
        """Test severity to color mapping."""
        config = AlertManagerConfig()
        manager = AlertManager(config)
        
        assert manager._get_severity_color("info") == "#36a64f"
        assert manager._get_severity_color("warning") == "#ff9800"
        assert manager._get_severity_color("error") == "#f44336"
        assert manager._get_severity_color("critical") == "#9c27b0"
        assert manager._get_severity_color("unknown") == "#808080"
    
    def test_map_severity_to_pagerduty(self):
        """Test severity mapping for PagerDuty."""
        config = AlertManagerConfig()
        manager = AlertManager(config)
        
        assert manager._map_severity_to_pagerduty("info") == "info"
        assert manager._map_severity_to_pagerduty("warning") == "warning"
        assert manager._map_severity_to_pagerduty("error") == "error"
        assert manager._map_severity_to_pagerduty("critical") == "critical"
        assert manager._map_severity_to_pagerduty("unknown") == "warning"
    
    def test_dispatch_alert_error_handling(self):
        """Test error handling in alert dispatch."""
        channel_config = AlertChannelConfig(
            type=AlertChannelType.EMAIL,
            name="email-channel",
            enabled=True,
            details={}  # Missing required email_to
        )
        
        rule = AlertRuleConfig(
            name="test-rule",
            event_type="test_event",
            severity="warning",
            channels=["email-channel"]
        )
        
        config = AlertManagerConfig(channels=[channel_config], rules=[rule])
        manager = AlertManager(config)
        
        # Should not raise exception, just log error
        manager._dispatch_alert(channel_config, rule, {"test": "data"})
    
    def test_channel_not_found(self):
        """Test handling of missing channel in rule."""
        rules = [
            AlertRuleConfig(
                name="test-rule",
                event_type="test_event",
                severity="warning",
                channels=["non-existent-channel"]
            )
        ]
        
        config = AlertManagerConfig(
            enabled=True,
            channels=[],  # No channels defined
            rules=rules
        )
        manager = AlertManager(config)
        
        with patch.object(manager, '_dispatch_alert') as mock_dispatch:
            manager.handle_event("test_event", {"data": "test"}, severity="warning")
            mock_dispatch.assert_not_called()