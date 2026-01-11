# College Basketball Possession Prediction Model

A deep learning model for predicting college basketball possession outcomes using transformer architecture and player embeddings.

## Overview

Most existing basketball prediction models operate at the game level and treat lineups as bags-of-stats. This project takes a different approach by:

- **Modeling at the possession level** - Predict the outcome of each individual possession
- **Learning player embeddings** - Capture individual player tendencies and styles
- **Modeling lineup interactions** - How 5 players work together as a unit
- **Capturing matchup dynamics** - How offensive and defensive lineups interact
- **Using sequential context** - Prior possessions in the game inform predictions

## Data Source

Uses the public BigQuery NCAA Basketball dataset (`bigquery-public-data.ncaa_basketball`):
- **Seasons**: 2013-2018 (5 seasons)
- **Games**: ~30,000
- **Play-by-play events**: ~4 million
- **Coverage**: D1, D2, D3 men's basketball

## Model Architecture

### V0 - Single Possession Baseline
- Player embeddings → Lineup encoder → Outcome prediction
- No sequential model
- Goal: Validate data pipeline and establish baseline

### V1 - Transformer Sequence Model
- Add causal transformer to attend to prior possessions
- Test if game history improves predictions

### V2 - Enhanced Features
- Coach and program embeddings
- Profile encoder for cold-start (new players)
- Auxiliary prediction tasks

### V3 - Game Simulator
- Use trained model to simulate full games
- Evaluate on game outcome prediction

## Project Structure

```
cbb-possession-model/
├── data/
│   ├── raw/                    # BigQuery exports
│   ├── processed/              # Processed possessions, players, games
│   └── embeddings/             # Trained embeddings
├── src/
│   ├── data/                   # Data processing and loading
│   ├── model/                  # Model components
│   ├── training/               # Training logic
│   └── evaluation/             # Metrics and simulation
├── notebooks/                  # Exploratory analysis
├── scripts/                    # Standalone scripts
└── tests/                      # Unit tests
```

## Getting Started

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### Workflow

1. **Export data from BigQuery**
   ```bash
   python scripts/export_bigquery.py --output data/raw/
   ```

2. **Process into possessions**
   ```bash
   python scripts/process_possessions.py --input data/raw/ --output data/processed/
   ```

3. **Train V0 model**
   ```bash
   python scripts/train_v0.py --data data/processed/possessions/
   ```

4. **Train V1 model (with transformer)**
   ```bash
   python scripts/train_v1.py --data data/processed/possessions/
   ```

## Prediction Targets

For each possession, the model predicts:
- **Points scored**: 0, 1, 2, or 3
- **Outcome type**: made_2pt, made_3pt, missed, turnover, foul_drawn, etc.
- **Possession duration**: seconds elapsed

## Model Size

Designed to be trainable on a laptop:
- ~3-5M parameters total
- ~10-20 minutes per epoch on CPU for one season
- Uses efficient transformer architecture

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | In Progress | Data pipeline - export, parse lineups, segment possessions |
| Phase 1 | Planned | V0 model - single possession baseline |
| Phase 2 | Planned | V1 model - add transformer for sequences |
| Phase 3 | Planned | Enhancements - profile encoder, coach embeddings, simulator |

## License

MIT

## Acknowledgments

Data provided by Google BigQuery public datasets (Sportradar NCAA Basketball data)
