"""
Process raw play-by-play data into possessions.

This script:
1. Loads raw play-by-play data from BigQuery export
2. Parses lineups from lineupchange events
3. Segments events into possessions
4. Saves processed possessions to parquet files

Usage:
    python scripts/process_possessions.py --input data/raw/ --output data/processed/
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.data.lineup_parser import (
    assign_lineups_to_events,
    get_lineup_coverage_stats,
)
from src.data.possession_segmenter import (
    segment_all_games,
    validate_possessions,
)


def process_pbp_to_possessions(
    input_file: Path,
    output_dir: Path,
    save_by_season: bool = True
) -> None:
    """
    Process play-by-play data into possessions.

    Args:
        input_file: Path to raw play-by-play parquet/csv file
        output_dir: Directory to save processed possessions
        save_by_season: If True, save separate files per season
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_file}...")
    if input_file.suffix == '.parquet':
        pbp_df = pd.read_parquet(input_file)
    elif input_file.suffix == '.csv':
        pbp_df = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")

    print(f"Loaded {len(pbp_df):,} events from {pbp_df['game_id'].nunique():,} games")

    # Step 1: Analyze lineup coverage
    print("\n" + "="*60)
    print("LINEUP COVERAGE ANALYSIS")
    print("="*60)
    lineup_stats = get_lineup_coverage_stats(pbp_df)
    for key, value in lineup_stats.items():
        print(f"{key}: {value}")

    # Step 2: Assign lineups to all events
    print("\n" + "="*60)
    print("ASSIGNING LINEUPS TO EVENTS")
    print("="*60)

    games_with_lineups = []
    game_ids = pbp_df['game_id'].unique()

    for game_id in tqdm(game_ids, desc="Processing games"):
        game_events = pbp_df[pbp_df['game_id'] == game_id].copy()
        game_events = assign_lineups_to_events(game_events)
        games_with_lineups.append(game_events)

    pbp_with_lineups = pd.concat(games_with_lineups, ignore_index=True)
    print(f"✓ Assigned lineups to {len(pbp_with_lineups):,} events")

    # Step 3: Segment into possessions
    print("\n" + "="*60)
    print("SEGMENTING INTO POSSESSIONS")
    print("="*60)

    possessions_df = segment_all_games(
        pbp_with_lineups,
        include_raw_events=False,
        show_progress=True
    )

    print(f"✓ Created {len(possessions_df):,} possessions")

    # Step 4: Validate possessions
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)

    validation_stats = validate_possessions(possessions_df)
    for key, value in validation_stats.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

    # Step 5: Save processed data
    print("\n" + "="*60)
    print("SAVING PROCESSED DATA")
    print("="*60)

    if save_by_season:
        # Save separate file per season
        for season in sorted(possessions_df['season'].unique()):
            season_possessions = possessions_df[possessions_df['season'] == season]
            output_file = output_dir / 'possessions' / f'possessions_{season}.parquet'
            output_file.parent.mkdir(parents=True, exist_ok=True)

            season_possessions.to_parquet(output_file, index=False)
            print(f"✓ Saved {len(season_possessions):,} possessions to {output_file}")
    else:
        # Save all to one file
        output_file = output_dir / 'possessions' / 'possessions_all.parquet'
        output_file.parent.mkdir(parents=True, exist_ok=True)

        possessions_df.to_parquet(output_file, index=False)
        print(f"✓ Saved {len(possessions_df):,} possessions to {output_file}")

    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Process play-by-play data into possessions"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input file (parquet or csv) with play-by-play data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--save-by-season',
        action='store_true',
        default=True,
        help='Save separate files per season'
    )

    args = parser.parse_args()

    process_pbp_to_possessions(
        input_file=Path(args.input),
        output_dir=Path(args.output),
        save_by_season=args.save_by_season
    )


if __name__ == '__main__':
    main()
