"""
Segment play-by-play events into possessions.

A possession is a continuous sequence of events where one team has the ball.
A possession ends when:
- Made field goal (2pt or 3pt)
- Defensive rebound after a missed shot
- Turnover
- End of period
- Offensive foul
"""

from typing import List, Dict, Optional
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class Possession:
    """Represents a single possession with outcome information."""
    possession_id: str
    game_id: str
    possession_num: int
    season: int
    period: int

    start_time_sec: int
    end_time_sec: int
    duration_sec: float

    offensive_team_id: str
    defensive_team_id: str

    home_lineup: Optional[List[str]]
    away_lineup: Optional[List[str]]

    # Outcomes (labels)
    points_scored: int
    outcome_type: str

    # Metadata
    num_events: int
    has_offensive_rebound: bool
    num_shot_attempts: int

    # Raw events for detailed analysis (optional)
    events: Optional[List[Dict]] = None


def is_possession_ending_event(event: pd.Series) -> bool:
    """
    Check if an event ends a possession.

    Args:
        event: A single event row from the play-by-play data

    Returns:
        True if this event ends the possession
    """
    event_type = event['event_type']

    # Made shots end possession (except and-one situations)
    if event_type in ('twopointmade', 'threepointmade'):
        return True

    # Defensive rebounds end possession
    if event_type == 'rebound' and event.get('rebound_type') == 'defensive':
        return True

    # Turnovers end possession
    if event_type == 'turnover':
        return True

    # Offensive fouls end possession
    if event_type == 'offensivefoul':
        return True

    # End of period
    if event_type == 'endperiod':
        return True

    return False


def determine_outcome_type(events: List[pd.Series]) -> str:
    """
    Determine the outcome type from a list of events.

    Args:
        events: List of event rows in this possession

    Returns:
        Outcome type string (made_2pt, made_3pt, missed, turnover, etc.)
    """
    # Check events in order to find the terminal event
    for event in events:
        event_type = event['event_type']

        if event_type == 'twopointmade':
            return 'made_2pt'
        elif event_type == 'threepointmade':
            return 'made_3pt'
        elif event_type == 'turnover':
            return 'turnover'
        elif event_type == 'offensivefoul':
            return 'offensive_foul'
        elif event_type == 'endperiod':
            return 'end_period'

    # Check for missed shots
    for event in events:
        if event['event_type'] in ('twopointmiss', 'threepointmiss'):
            return 'missed'

    return 'unknown'


def summarize_possession(
    events: List[pd.Series],
    possession_num: int,
    game_id: str,
    season: int
) -> Possession:
    """
    Compute possession-level summary from constituent events.

    Args:
        events: List of event rows that make up this possession
        possession_num: Sequential number of this possession in the game
        game_id: Game identifier
        season: Season year

    Returns:
        Possession object with computed outcomes
    """
    if not events:
        raise ValueError("Cannot summarize empty possession")

    first_event = events[0]
    last_event = events[-1]

    # Compute total points scored
    import math
    points = sum(
        (event.get('points_scored', 0) or 0) if not (isinstance(event.get('points_scored'), float) and math.isnan(event.get('points_scored', 0))) else 0
        for event in events
    )

    # Determine outcome type
    outcome_type = determine_outcome_type(events)

    # Get team IDs
    offensive_team_id = first_event.get('possession_team_id')
    home_id = first_event['home_id']
    away_id = first_event['away_id']

    # Determine defensive team
    if offensive_team_id == home_id:
        defensive_team_id = away_id
    else:
        defensive_team_id = home_id

    # Check for offensive rebounds
    has_offensive_rebound = any(
        event['event_type'] == 'rebound' and
        event.get('rebound_type') == 'offensive'
        for event in events
    )

    # Count shot attempts
    num_shot_attempts = sum(
        1 for event in events
        if event['event_type'] in (
            'twopointmade', 'twopointmiss',
            'threepointmade', 'threepointmiss'
        )
    )

    # Get lineups (from first event where possession is established)
    home_lineup = first_event.get('home_lineup')
    away_lineup = first_event.get('away_lineup')

    # Create possession ID
    possession_id = f"{game_id}_{possession_num}"

    return Possession(
        possession_id=possession_id,
        game_id=game_id,
        possession_num=possession_num,
        season=season,
        period=first_event['period'],
        start_time_sec=first_event['elapsed_time_sec'],
        end_time_sec=last_event['elapsed_time_sec'],
        duration_sec=last_event['elapsed_time_sec'] - first_event['elapsed_time_sec'],
        offensive_team_id=offensive_team_id,
        defensive_team_id=defensive_team_id,
        home_lineup=home_lineup,
        away_lineup=away_lineup,
        points_scored=int(points) if not math.isnan(points) else 0,
        outcome_type=outcome_type,
        num_events=len(events),
        has_offensive_rebound=has_offensive_rebound,
        num_shot_attempts=num_shot_attempts,
        events=None,  # Don't store raw events by default to save memory
    )


def segment_game_into_possessions(
    game_events: pd.DataFrame,
    include_raw_events: bool = False
) -> List[Possession]:
    """
    Segment a single game's events into possessions.

    Args:
        game_events: DataFrame of events for one game, sorted by elapsed_time_sec
        include_raw_events: If True, include raw events in Possession objects

    Returns:
        List of Possession objects
    """
    if game_events.empty:
        return []

    # Ensure sorted by time
    game_events = game_events.sort_values('elapsed_time_sec').reset_index(drop=True)

    # Get game metadata
    game_id = game_events['game_id'].iloc[0]
    season = game_events['season'].iloc[0]

    possessions = []
    current_possession_events = []
    current_team = None
    possession_num = 0

    for idx, event in game_events.iterrows():
        # Convert row to dict for easier handling
        event_dict = event.to_dict()

        # Check for possession change
        new_team = event.get('possession_team_id')
        if pd.notna(new_team) and new_team != current_team:
            # Save previous possession if exists
            if current_possession_events:
                possession = summarize_possession(
                    current_possession_events,
                    possession_num,
                    game_id,
                    season
                )
                if include_raw_events:
                    possession.events = current_possession_events
                possessions.append(possession)
                possession_num += 1

            # Start new possession
            current_possession_events = []
            current_team = new_team

        # Add event to current possession
        current_possession_events.append(event_dict)

        # Check if this event ends the possession
        if is_possession_ending_event(event):
            if current_possession_events:
                possession = summarize_possession(
                    current_possession_events,
                    possession_num,
                    game_id,
                    season
                )
                if include_raw_events:
                    possession.events = current_possession_events
                possessions.append(possession)
                possession_num += 1

            # Reset for next possession
            current_possession_events = []
            current_team = None

    # Handle any remaining events
    if current_possession_events:
        possession = summarize_possession(
            current_possession_events,
            possession_num,
            game_id,
            season
        )
        if include_raw_events:
            possession.events = current_possession_events
        possessions.append(possession)

    return possessions


def segment_all_games(
    events_df: pd.DataFrame,
    include_raw_events: bool = False,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Segment all games in a DataFrame into possessions.

    Args:
        events_df: DataFrame of play-by-play events for multiple games
        include_raw_events: If True, include raw events in output
        show_progress: If True, show progress bar

    Returns:
        DataFrame where each row is a possession
    """
    all_possessions = []

    game_ids = events_df['game_id'].unique()

    # Setup progress bar if needed
    if show_progress:
        try:
            from tqdm import tqdm
            game_ids = tqdm(game_ids, desc="Segmenting games")
        except ImportError:
            pass

    for game_id in game_ids:
        game_events = events_df[events_df['game_id'] == game_id]
        possessions = segment_game_into_possessions(
            game_events,
            include_raw_events=include_raw_events
        )
        all_possessions.extend(possessions)

    # Convert to DataFrame
    possession_dicts = [asdict(p) for p in all_possessions]
    possessions_df = pd.DataFrame(possession_dicts)

    return possessions_df


def validate_possessions(
    possessions_df: pd.DataFrame,
    games_df: Optional[pd.DataFrame] = None
) -> Dict[str, any]:
    """
    Validate possession data and compute summary statistics.

    Args:
        possessions_df: DataFrame of possessions
        games_df: Optional DataFrame of game-level data for comparison

    Returns:
        Dictionary with validation results and statistics
    """
    stats = {
        'total_possessions': len(possessions_df),
        'total_games': possessions_df['game_id'].nunique(),
        'avg_possessions_per_game': len(possessions_df) / possessions_df['game_id'].nunique(),
        'avg_duration_sec': possessions_df['duration_sec'].mean(),
        'avg_points': possessions_df['points_scored'].mean(),
    }

    # Outcome type distribution
    stats['outcome_distribution'] = possessions_df['outcome_type'].value_counts().to_dict()

    # Points distribution
    stats['points_distribution'] = possessions_df['points_scored'].value_counts().to_dict()

    # Check lineup coverage
    has_lineups = (
        possessions_df['home_lineup'].notna() &
        possessions_df['away_lineup'].notna()
    ).sum()
    stats['possessions_with_lineups'] = has_lineups
    stats['lineup_coverage_pct'] = 100 * has_lineups / len(possessions_df)

    # If games data provided, validate point totals
    if games_df is not None:
        # Aggregate points by game
        game_points = possessions_df.groupby('game_id')['points_scored'].sum()

        # Compare with actual game scores
        # This would require game_df to have final scores
        stats['validation'] = 'Games data provided but validation not yet implemented'

    return stats
