"""
Parse lineup information from lineupchange event descriptions.

The BigQuery NCAA Basketball dataset contains lineupchange events with the full
5-player lineup embedded in the event_description field.

Example:
    "Jayhawks lineup change (Lagerald Vick, Devonte' Graham, Marcus Garrett,
     Malik Newman, Udoka Azubuike)"
"""

import re
from typing import Optional, List, Dict
import pandas as pd
from dataclasses import dataclass


@dataclass
class LineupChange:
    """Represents a parsed lineup change event."""
    game_id: str
    team_id: str
    team_name: str
    elapsed_time_sec: int
    period: int
    players: List[str]


def parse_lineup_from_description(event_description: str) -> Optional[List[str]]:
    """
    Extract 5 player names from lineup change description.

    Args:
        event_description: Event description string from BigQuery

    Returns:
        List of 5 player names, or None if parsing failed

    Examples:
        >>> desc = "Jayhawks lineup change (Lagerald Vick, Devonte' Graham, Marcus Garrett, Malik Newman, Udoka Azubuike)"
        >>> parse_lineup_from_description(desc)
        ['Lagerald Vick', "Devonte' Graham", 'Marcus Garrett', 'Malik Newman', 'Udoka Azubuike']
    """
    if not event_description or 'lineup change' not in event_description.lower():
        return None

    # Extract content between parentheses
    match = re.search(r'\(([^)]+)\)', event_description)
    if not match:
        return None

    # Split by comma and clean up
    players = [p.strip() for p in match.group(1).split(',')]

    # Validate we got exactly 5 players
    if len(players) != 5:
        return None

    # Filter out empty strings
    players = [p for p in players if p]

    return players if len(players) == 5 else None


def extract_lineup_changes(
    events_df: pd.DataFrame
) -> List[LineupChange]:
    """
    Extract all lineup changes from a DataFrame of events.

    Args:
        events_df: DataFrame with columns: game_id, team_id, team_market,
                   event_type, event_description, elapsed_time_sec, period

    Returns:
        List of LineupChange objects
    """
    lineup_changes = []

    # Filter to lineup change events
    lineup_events = events_df[events_df['event_type'] == 'lineupchange'].copy()

    for _, event in lineup_events.iterrows():
        players = parse_lineup_from_description(event['event_description'])

        if players:
            lineup_changes.append(
                LineupChange(
                    game_id=event['game_id'],
                    team_id=event['team_id'],
                    team_name=event.get('team_market', ''),
                    elapsed_time_sec=event['elapsed_time_sec'],
                    period=event['period'],
                    players=players
                )
            )

    return lineup_changes


def assign_lineups_to_events(game_events: pd.DataFrame) -> pd.DataFrame:
    """
    Propagate lineup information to all events in a game.

    When we see a lineupchange event for a team, we record those 5 players.
    All subsequent events for that team use that lineup until the next
    lineupchange.

    Args:
        game_events: DataFrame of events for a single game, must contain:
                     - elapsed_time_sec: for ordering
                     - home_id, away_id: team IDs
                     - team_id: team that this event belongs to
                     - event_type: to detect lineupchange events
                     - event_description: to parse lineups

    Returns:
        Same DataFrame with added columns:
            - home_lineup: List[str] of 5 player names for home team
            - away_lineup: List[str] of 5 player names for away team
            - offensive_lineup: List[str] of 5 players on offense (for this event)
            - defensive_lineup: List[str] of 5 players on defense (for this event)

    Note:
        Events before the first lineupchange will have None for lineups.
    """
    # Sort by elapsed time to process in order
    game_events = game_events.sort_values('elapsed_time_sec').copy()

    # Track current lineup for each team
    current_lineups: Dict[str, Optional[List[str]]] = {}

    # Get team IDs
    home_id = game_events['home_id'].iloc[0]
    away_id = game_events['away_id'].iloc[0]

    # Initialize tracking lists
    home_lineups = []
    away_lineups = []

    for _, event in game_events.iterrows():
        # Check if this is a lineup change
        if event['event_type'] == 'lineupchange' and pd.notna(event['team_id']):
            lineup = parse_lineup_from_description(event['event_description'])
            if lineup:
                current_lineups[event['team_id']] = lineup

        # Record current lineups for this event
        home_lineups.append(current_lineups.get(home_id))
        away_lineups.append(current_lineups.get(away_id))

    game_events['home_lineup'] = home_lineups
    game_events['away_lineup'] = away_lineups

    # Add offensive/defensive lineups based on possession
    def get_offensive_defensive_lineups(row):
        """Determine which lineup is on offense/defense."""
        if pd.isna(row['possession_team_id']):
            return None, None

        if row['possession_team_id'] == home_id:
            return row['home_lineup'], row['away_lineup']
        else:
            return row['away_lineup'], row['home_lineup']

    lineups = game_events.apply(get_offensive_defensive_lineups, axis=1)
    game_events['offensive_lineup'] = [l[0] for l in lineups]
    game_events['defensive_lineup'] = [l[1] for l in lineups]

    return game_events


def get_lineup_coverage_stats(events_df: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate statistics about lineup change coverage in the dataset.

    Args:
        events_df: DataFrame of play-by-play events

    Returns:
        Dictionary with coverage statistics
    """
    total_games = events_df['game_id'].nunique()
    games_with_lineups = events_df[
        events_df['event_type'] == 'lineupchange'
    ]['game_id'].nunique()

    lineup_events = events_df[events_df['event_type'] == 'lineupchange']

    # Parse success rate
    parsed_lineups = lineup_events['event_description'].apply(
        parse_lineup_from_description
    )
    successful_parses = parsed_lineups.notna().sum()
    total_lineup_events = len(lineup_events)

    return {
        'total_games': total_games,
        'games_with_lineups': games_with_lineups,
        'coverage_pct': 100 * games_with_lineups / total_games if total_games > 0 else 0,
        'total_lineup_events': total_lineup_events,
        'successful_parses': successful_parses,
        'parse_success_rate': 100 * successful_parses / total_lineup_events if total_lineup_events > 0 else 0,
    }
