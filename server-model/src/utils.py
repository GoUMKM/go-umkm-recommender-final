from typing import Dict, Any
import json
import pandas as pd

def save_metadata(metadata: Dict[str, Any], filepath: str):
    """Save training metadata to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(metadata, f)

def load_metadata(filepath: str) -> Dict[str, Any]:
    """Load training metadata from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_dataframe(df: pd.DataFrame, filepath: str):
    """Save dataframe to CSV"""
    df.to_csv(filepath, index=False)

def load_dataframe(filepath: str) -> pd.DataFrame:
    """Load dataframe from CSV"""
    return pd.read_csv(filepath)