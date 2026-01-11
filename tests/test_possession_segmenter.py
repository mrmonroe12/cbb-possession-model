"""
Tests for possession_segmenter module.
"""

import pytest
import pandas as pd
from src.data.possession_segmenter import (
    is_possession_ending_event,
    determine_outcome_type,
    segment_game_into_possessions,
    validate_possessions,
)


class TestIsPossessionEndingEvent:
    """Tests for is_possession_ending_event function."""

    def test_made_shots_end_possession(self):
        """Test that made shots end possessions."""
        event = pd.Series({'event_type': 'twopointmade'})
        assert is_possession_ending_event(event)

        event = pd.Series({'event_type': 'threepointmade'})
        assert is_possession_ending_event(event)

    def test_defensive_rebound_ends_possession(self):
        """Test that defensive rebounds end possessions."""
        event = pd.Series({'event_type': 'rebound', 'rebound_type': 'defensive'})
        assert is_possession_ending_event(event)

    def test_offensive_rebound_continues_possession(self):
        """Test that offensive rebounds don't end possessions."""
        event = pd.Series({'event_type': 'rebound', 'rebound_type': 'offensive'})
        assert not is_possession_ending_event(event)

    def test_turnover_ends_possession(self):
        """Test that turnovers end possessions."""
        event = pd.Series({'event_type': 'turnover'})
        assert is_possession_ending_event(event)

    def test_offensive_foul_ends_possession(self):
        """Test that offensive fouls end possessions."""
        event = pd.Series({'event_type': 'offensivefoul'})
        assert is_possession_ending_event(event)

    def test_end_period_ends_possession(self):
        """Test that end of period ends possessions."""
        event = pd.Series({'event_type': 'endperiod'})
        assert is_possession_ending_event(event)

    def test_missed_shot_continues_possession(self):
        """Test that missed shots don't end possessions (wait for rebound)."""
        event = pd.Series({'event_type': 'twopointmiss'})
        assert not is_possession_ending_event(event)


class TestDetermineOutcomeType:
    """Tests for determine_outcome_type function."""

    def test_made_2pt_outcome(self):
        """Test made 2pt outcome."""
        events = [
            pd.Series({'event_type': 'twopointmade'}),
        ]
        assert determine_outcome_type(events) == 'made_2pt'

    def test_made_3pt_outcome(self):
        """Test made 3pt outcome."""
        events = [
            pd.Series({'event_type': 'threepointmade'}),
        ]
        assert determine_outcome_type(events) == 'made_3pt'

    def test_turnover_outcome(self):
        """Test turnover outcome."""
        events = [
            pd.Series({'event_type': 'turnover'}),
        ]
        assert determine_outcome_type(events) == 'turnover'

    def test_missed_shot_outcome(self):
        """Test missed shot outcome."""
        events = [
            pd.Series({'event_type': 'twopointmiss'}),
            pd.Series({'event_type': 'rebound', 'rebound_type': 'defensive'}),
        ]
        assert determine_outcome_type(events) == 'missed'

    def test_offensive_foul_outcome(self):
        """Test offensive foul outcome."""
        events = [
            pd.Series({'event_type': 'offensivefoul'}),
        ]
        assert determine_outcome_type(events) == 'offensive_foul'


class TestSegmentGameIntoPossessions:
    """Tests for segment_game_into_possessions function."""

    def test_simple_possession_sequence(self):
        """Test a simple sequence of possessions."""
        events = pd.DataFrame([
            {
                'game_id': 'game1',
                'season': 2018,
                'elapsed_time_sec': 0,
                'period': 1,
                'possession_team_id': 'team_a',
                'home_id': 'team_a',
                'away_id': 'team_b',
                'event_type': 'twopointmade',
                'points_scored': 2.0,
                'home_lineup': ['P1', 'P2', 'P3', 'P4', 'P5'],
                'away_lineup': ['A1', 'A2', 'A3', 'A4', 'A5'],
            },
            {
                'game_id': 'game1',
                'season': 2018,
                'elapsed_time_sec': 20,
                'period': 1,
                'possession_team_id': 'team_b',
                'home_id': 'team_a',
                'away_id': 'team_b',
                'event_type': 'threepointmade',
                'points_scored': 3.0,
                'home_lineup': ['P1', 'P2', 'P3', 'P4', 'P5'],
                'away_lineup': ['A1', 'A2', 'A3', 'A4', 'A5'],
            },
        ])

        possessions = segment_game_into_possessions(events)

        assert len(possessions) == 2
        assert possessions[0].points_scored == 2
        assert possessions[0].outcome_type == 'made_2pt'
        assert possessions[1].points_scored == 3
        assert possessions[1].outcome_type == 'made_3pt'

    def test_possession_with_offensive_rebound(self):
        """Test possession with offensive rebound continues."""
        events = pd.DataFrame([
            {
                'game_id': 'game1',
                'season': 2018,
                'elapsed_time_sec': 0,
                'period': 1,
                'possession_team_id': 'team_a',
                'home_id': 'team_a',
                'away_id': 'team_b',
                'event_type': 'twopointmiss',
                'points_scored': 0,
                'home_lineup': ['P1', 'P2', 'P3', 'P4', 'P5'],
                'away_lineup': None,
            },
            {
                'game_id': 'game1',
                'season': 2018,
                'elapsed_time_sec': 2,
                'period': 1,
                'possession_team_id': 'team_a',
                'home_id': 'team_a',
                'away_id': 'team_b',
                'event_type': 'rebound',
                'rebound_type': 'offensive',
                'points_scored': 0,
                'home_lineup': ['P1', 'P2', 'P3', 'P4', 'P5'],
                'away_lineup': None,
            },
            {
                'game_id': 'game1',
                'season': 2018,
                'elapsed_time_sec': 5,
                'period': 1,
                'possession_team_id': 'team_a',
                'home_id': 'team_a',
                'away_id': 'team_b',
                'event_type': 'twopointmade',
                'points_scored': 2.0,
                'home_lineup': ['P1', 'P2', 'P3', 'P4', 'P5'],
                'away_lineup': None,
            },
        ])

        possessions = segment_game_into_possessions(events)

        # Should be one possession that includes all events
        assert len(possessions) == 1
        assert possessions[0].num_events == 3
        assert possessions[0].has_offensive_rebound
        assert possessions[0].points_scored == 2
        assert possessions[0].num_shot_attempts == 2

    def test_turnover_possession(self):
        """Test possession ending in turnover."""
        events = pd.DataFrame([
            {
                'game_id': 'game1',
                'season': 2018,
                'elapsed_time_sec': 0,
                'period': 1,
                'possession_team_id': 'team_a',
                'home_id': 'team_a',
                'away_id': 'team_b',
                'event_type': 'turnover',
                'points_scored': 0,
                'home_lineup': None,
                'away_lineup': None,
            },
        ])

        possessions = segment_game_into_possessions(events)

        assert len(possessions) == 1
        assert possessions[0].points_scored == 0
        assert possessions[0].outcome_type == 'turnover'

    def test_possession_duration(self):
        """Test that possession duration is calculated correctly."""
        events = pd.DataFrame([
            {
                'game_id': 'game1',
                'season': 2018,
                'elapsed_time_sec': 10,
                'period': 1,
                'possession_team_id': 'team_a',
                'home_id': 'team_a',
                'away_id': 'team_b',
                'event_type': 'twopointmiss',
                'points_scored': 0,
                'home_lineup': None,
                'away_lineup': None,
            },
            {
                'game_id': 'game1',
                'season': 2018,
                'elapsed_time_sec': 35,
                'period': 1,
                'possession_team_id': 'team_a',
                'home_id': 'team_a',
                'away_id': 'team_b',
                'event_type': 'rebound',
                'rebound_type': 'defensive',
                'points_scored': 0,
                'home_lineup': None,
                'away_lineup': None,
            },
        ])

        possessions = segment_game_into_possessions(events)

        assert len(possessions) == 1
        assert possessions[0].duration_sec == 25  # 35 - 10

    def test_possession_numbering(self):
        """Test that possessions are numbered sequentially."""
        events = pd.DataFrame([
            {
                'game_id': 'game1',
                'season': 2018,
                'elapsed_time_sec': 0,
                'period': 1,
                'possession_team_id': 'team_a',
                'home_id': 'team_a',
                'away_id': 'team_b',
                'event_type': 'twopointmade',
                'points_scored': 2.0,
                'home_lineup': None,
                'away_lineup': None,
            },
            {
                'game_id': 'game1',
                'season': 2018,
                'elapsed_time_sec': 20,
                'period': 1,
                'possession_team_id': 'team_b',
                'home_id': 'team_a',
                'away_id': 'team_b',
                'event_type': 'twopointmade',
                'points_scored': 2.0,
                'home_lineup': None,
                'away_lineup': None,
            },
            {
                'game_id': 'game1',
                'season': 2018,
                'elapsed_time_sec': 40,
                'period': 1,
                'possession_team_id': 'team_a',
                'home_id': 'team_a',
                'away_id': 'team_b',
                'event_type': 'twopointmade',
                'points_scored': 2.0,
                'home_lineup': None,
                'away_lineup': None,
            },
        ])

        possessions = segment_game_into_possessions(events)

        assert len(possessions) == 3
        assert possessions[0].possession_num == 0
        assert possessions[1].possession_num == 1
        assert possessions[2].possession_num == 2


class TestValidatePossessions:
    """Tests for validate_possessions function."""

    def test_basic_validation(self):
        """Test basic validation statistics."""
        possessions = pd.DataFrame([
            {
                'possession_id': 'game1_0',
                'game_id': 'game1',
                'points_scored': 2,
                'outcome_type': 'made_2pt',
                'duration_sec': 25.0,
                'home_lineup': ['P1', 'P2', 'P3', 'P4', 'P5'],
                'away_lineup': ['A1', 'A2', 'A3', 'A4', 'A5'],
            },
            {
                'possession_id': 'game1_1',
                'game_id': 'game1',
                'points_scored': 0,
                'outcome_type': 'missed',
                'duration_sec': 18.0,
                'home_lineup': ['P1', 'P2', 'P3', 'P4', 'P5'],
                'away_lineup': ['A1', 'A2', 'A3', 'A4', 'A5'],
            },
        ])

        stats = validate_possessions(possessions)

        assert stats['total_possessions'] == 2
        assert stats['total_games'] == 1
        assert stats['avg_possessions_per_game'] == 2
        assert stats['avg_points'] == 1.0  # (2 + 0) / 2
        assert stats['lineup_coverage_pct'] == 100.0
