"""
Export play-by-play data from BigQuery NCAA Basketball dataset.

This script extracts all play-by-play events from the public BigQuery dataset
and saves them to local parquet files for further processing.

Usage:
    python scripts/export_bigquery.py --output data/raw/ [--seasons 2013,2014,2015]
"""

import argparse
from pathlib import Path
from typing import Optional, List

from google.cloud import bigquery
import pandas as pd
from tqdm import tqdm


def get_export_query(seasons: Optional[List[int]] = None) -> str:
    """
    Build SQL query to export play-by-play data.

    Args:
        seasons: List of seasons to export. If None, exports all available seasons.

    Returns:
        SQL query string
    """
    query = """
    SELECT
        game_id,
        season,
        scheduled_date,
        period,
        game_clock,
        elapsed_time_sec,
        home_id,
        home_market,
        home_name,
        away_id,
        away_market,
        away_name,
        team_id,
        team_market,
        player_id,
        player_full_name,
        jersey_num,
        event_id,
        event_type,
        event_description,
        timestamp,
        event_coord_x,
        event_coord_y,
        possession_team_id,
        points_scored,
        three_point_shot,
        shot_made,
        rebound_type,
        turnover_type
    FROM `bigquery-public-data.ncaa_basketball.mbb_pbp_sr`
    """

    if seasons:
        season_list = ','.join(str(s) for s in seasons)
        query += f"\nWHERE season IN ({season_list})"

    query += "\nORDER BY game_id, elapsed_time_sec"

    return query


def export_pbp_data(
    output_dir: Path,
    seasons: Optional[List[int]] = None,
    format: str = 'parquet'
) -> None:
    """
    Export play-by-play data from BigQuery to local files.

    Args:
        output_dir: Directory to save exported data
        seasons: List of seasons to export (e.g., [2013, 2014])
        format: Output format ('parquet' or 'csv')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing BigQuery client...")
    client = bigquery.Client()

    print("Building export query...")
    query = get_export_query(seasons)
    print(f"\nQuery:\n{query}\n")

    print("Executing query (this may take a few minutes)...")
    query_job = client.query(query)

    # Get results with progress bar
    print("Fetching results...")
    df = query_job.to_dataframe(progress_bar_type='tqdm')

    print(f"\nExported {len(df):,} rows")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Save to file
    if seasons:
        season_str = '_'.join(str(s) for s in sorted(seasons))
        filename = f"pbp_{season_str}.{format}"
    else:
        filename = f"pbp_all_seasons.{format}"

    output_path = output_dir / filename

    print(f"\nSaving to {output_path}...")
    if format == 'parquet':
        df.to_parquet(output_path, index=False, compression='snappy')
    elif format == 'csv':
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"✓ Saved {len(df):,} rows to {output_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total events: {len(df):,}")
    print(f"Unique games: {df['game_id'].nunique():,}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Date range: {df['scheduled_date'].min()} to {df['scheduled_date'].max()}")
    print(f"\nEvent type breakdown:")
    print(df['event_type'].value_counts().head(10))
    print(f"\nLineup change events: {(df['event_type'] == 'lineupchange').sum():,}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Export NCAA basketball play-by-play data from BigQuery"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw',
        help='Output directory for exported data'
    )
    parser.add_argument(
        '--seasons',
        type=str,
        default=None,
        help='Comma-separated list of seasons to export (e.g., "2013,2014,2015"). If not specified, exports all.'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['parquet', 'csv'],
        default='parquet',
        help='Output file format'
    )

    args = parser.parse_args()

    # Parse seasons if provided
    seasons = None
    if args.seasons:
        seasons = [int(s.strip()) for s in args.seasons.split(',')]

    export_pbp_data(
        output_dir=Path(args.output),
        seasons=seasons,
        format=args.format
    )


if __name__ == '__main__':
    main()
