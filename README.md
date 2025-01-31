# Analysis of Centrality in Competition Networks

This repository provides data, code, models, and analysis pertaining to the [Analysis and Predictability of Centrality in Competition Networks](https://math.ryerson.ca/~abonato/papers/comp_013125.pdf) paper.

## Data

### 1. Survivor

Source: [survivoR2py](https://github.com/stiles/survivoR2py)
- [vote_history.csv](https://github.com/mariamwalaa/CON-CN/blob/main/output/Survivor/vote_history.csv)
- [castaways.csv](https://github.com/mariamwalaa/CON-CN/blob/main/output/Survivor/castaways.csv)

### 2. Chess.com 

Source: [Titled Tuesday website](https://www.chess.com/tournament/live/titled-tuesdays)
- [Titled Tuesday pairings](https://github.com/mariamwalaa/CON-CN/tree/main/output/Chess.com/Titled%20Tuesday%20Pairings)
- [Titled Tuesday results](https://github.com/mariamwalaa/CON-CN/tree/main/output/Chess.com/Titled%20Tuesday%20Results)

### 3. Dota 2

Source: [Kaggle](https://www.kaggle.com/datasets/bwandowando/dota-2-pro-league-matches-2023)
- [raw](https://github.com/mariamwalaa/CON-CN/tree/main/output/DOTA2/raw)
  - main_metadata: provides match ID, league ID, winning team, scores, team IDs
  - teams: provides team ID, name, and tag
- [processed](https://github.com/mariamwalaa/CON-CN/tree/main/output/DOTA2/processed)
  - dota2_final_20XX: merges columns from main_metadata and teams raw files

