"""
Tests for lineup_parser module.
"""

import pytest
import pandas as pd
from src.data.lineup_parser import (
    parse_lineup_from_description,
    assign_lineups_to_events,
    get_lineup_coverage_stats,
)


class TestParseLineupFromDescription:
    """Tests for parse_lineup_from_description function."""

    def test_valid_lineup(self):
        """Test parsing a valid lineup description."""
        desc = "Jayhawks lineup change (Lagerald Vick, Devonte' Graham, Marcus Garrett, Malik Newman, Udoka Azubuike)"
        result = parse_lineup_from_description(desc)

        assert result is not None
        assert len(result) == 5
        assert "Lagerald Vick" in result
        assert "Devonte' Graham" in result
        assert "Marcus Garrett" in result
        assert "Malik Newman" in result
        assert "Udoka Azubuike" in result

    def test_handles_apostrophes(self):
        """Test that player names with apostrophes are handled correctly."""
        desc = "Bears lineup change (D'Angelo Smith, O'Brien Jones, Mike Johnson, Tim O'Neil, Ray Jackson)"
        result = parse_lineup_from_description(desc)

        assert result is not None
        assert len(result) == 5
        assert "D'Angelo Smith" in result
        assert "O'Brien Jones" in result

    def test_invalid_player_count(self):
        """Test that lineups with != 5 players return None."""
        # Only 3 players
        desc = "Team lineup change (Player One, Player Two, Player Three)"
        assert parse_lineup_from_description(desc) is None

        # 6 players
        desc = "Team lineup change (P1, P2, P3, P4, P5, P6)"
        assert parse_lineup_from_description(desc) is None

    def test_missing_parentheses(self):
        """Test that descriptions without parentheses return None."""
        desc = "Jayhawks lineup change with no parentheses"
        assert parse_lineup_from_description(desc) is None

    def test_not_lineup_change(self):
        """Test that non-lineup-change events return None."""
        desc = "Player makes two point shot"
        assert parse_lineup_from_description(desc) is None

    def test_none_input(self):
        """Test that None input returns None."""
        assert parse_lineup_from_description(None) is None

    def test_empty_string(self):
        """Test that empty string returns None."""
        assert parse_lineup_from_description("") is None


class TestAssignLineupsToEvents:
    """Tests for assign_lineups_to_events function."""

    def test_lineup_propagation(self):
        """Test that lineups propagate to subsequent events."""
        # Create sample game events
        events = pd.DataFrame([
            {
                'game_id': 'game1',
                'elapsed_time_sec': 0,
                'home_id': 'team_home',
                'away_id': 'team_away',
                'team_id': 'team_home',
                'event_type': 'lineupchange',
                'event_description': 'Home lineup change (P1, P2, P3, P4, P5)',
                'possession_team_id': 'team_home',
            },
            {
                'game_id': 'game1',
                'elapsed_time_sec': 5,
                'home_id': 'team_home',
                'away_id': 'team_away',
                'team_id': 'team_away',
                'event_type': 'lineupchange',
                'event_description': 'Away lineup change (A1, A2, A3, A4, A5)',
                'possession_team_id': 'team_home',
            },
            {
                'game_id': 'game1',
                'elapsed_time_sec': 10,
                'home_id': 'team_home',
                'away_id': 'team_away',
                'team_id': 'team_home',
                'event_type': 'twopointmade',
                'event_description': 'P1 makes shot',
                'possession_team_id': 'team_home',
            },
        ])

        result = assign_lineups_to_events(events)

        # Check that lineups are assigned
        assert result.iloc[2]['home_lineup'] == ['P1', 'P2', 'P3', 'P4', 'P5']
        assert result.iloc[2]['away_lineup'] == ['A1', 'A2', 'A3', 'A4', 'A5']

        # Check offensive/defensive lineups
        assert result.iloc[2]['offensive_lineup'] == ['P1', 'P2', 'P3', 'P4', 'P5']
        assert result.iloc[2]['defensive_lineup'] == ['A1', 'A2', 'A3', 'A4', 'A5']

    def test_lineup_change_updates(self):
        """Test that lineup changes update the tracked lineup."""
        events = pd.DataFrame([
            {
                'game_id': 'game1',
                'elapsed_time_sec': 0,
                'home_id': 'team_home',
                'away_id': 'team_away',
                'team_id': 'team_home',
                'event_type': 'lineupchange',
                'event_description': 'Home lineup change (P1, P2, P3, P4, P5)',
                'possession_team_id': 'team_home',
            },
            {
                'game_id': 'game1',
                'elapsed_time_sec': 100,
                'home_id': 'team_home',
                'away_id': 'team_away',
                'team_id': 'team_home',
                'event_type': 'twopointmade',
                'event_description': 'P1 makes shot',
                'possession_team_id': 'team_home',
            },
            {
                'game_id': 'game1',
                'elapsed_time_sec': 200,
                'home_id': 'team_home',
                'away_id': 'team_away',
                'team_id': 'team_home',
                'event_type': 'lineupchange',
                'event_description': 'Home lineup change (P6, P7, P8, P9, P10)',
                'possession_team_id': 'team_home',
            },
            {
                'game_id': 'game1',
                'elapsed_time_sec': 250,
                'home_id': 'team_home',
                'away_id': 'team_away',
                'team_id': 'team_home',
                'event_type': 'twopointmade',
                'event_description': 'P6 makes shot',
                'possession_team_id': 'team_home',
            },
        ])

        result = assign_lineups_to_events(events)

        # First shot should have first lineup
        assert result.iloc[1]['home_lineup'] == ['P1', 'P2', 'P3', 'P4', 'P5']

        # Second shot should have updated lineup
        assert result.iloc[3]['home_lineup'] == ['P6', 'P7', 'P8', 'P9', 'P10']

    def test_events_before_lineup_change(self):
        """Test that events before any lineup change have None."""
        events = pd.DataFrame([
            {
                'game_id': 'game1',
                'elapsed_time_sec': 0,
                'home_id': 'team_home',
                'away_id': 'team_away',
                'team_id': 'team_home',
                'event_type': 'opentip',
                'event_description': 'Jump ball',
                'possession_team_id': 'team_home',
            },
            {
                'game_id': 'game1',
                'elapsed_time_sec': 5,
                'home_id': 'team_home',
                'away_id': 'team_away',
                'team_id': 'team_home',
                'event_type': 'lineupchange',
                'event_description': 'Home lineup change (P1, P2, P3, P4, P5)',
                'possession_team_id': 'team_home',
            },
        ])

        result = assign_lineups_to_events(events)

        # Event before lineup change should have None
        assert result.iloc[0]['home_lineup'] is None
        assert result.iloc[0]['away_lineup'] is None


class TestGetLineupCoverageStats:
    """Tests for get_lineup_coverage_stats function."""

    def test_coverage_calculation(self):
        """Test calculation of coverage statistics."""
        events = pd.DataFrame([
            {
                'game_id': 'game1',
                'event_type': 'lineupchange',
                'event_description': 'Team lineup change (P1, P2, P3, P4, P5)',
            },
            {
                'game_id': 'game1',
                'event_type': 'twopointmade',
                'event_description': 'P1 makes shot',
            },
            {
                'game_id': 'game2',
                'event_type': 'twopointmade',
                'event_description': 'P2 makes shot',
            },
        ])

        stats = get_lineup_coverage_stats(events)

        assert stats['total_games'] == 2
        assert stats['games_with_lineups'] == 1
        assert stats['coverage_pct'] == 50.0
        assert stats['total_lineup_events'] == 1
        assert stats['successful_parses'] == 1
        assert stats['parse_success_rate'] == 100.0
