import pandas as pd
import sys, pathlib
from src.config import RAW_GAMES, RAW_PLAYS

def main():
    # Just validate files exist & are loadable
    games = pd.read_csv(RAW_GAMES, parse_dates=['date'])
    plays = pd.read_csv(RAW_PLAYS)
    print(f"Loaded games: {games.shape}, plays: {plays.shape}")

if __name__ == "__main__":
    main()
