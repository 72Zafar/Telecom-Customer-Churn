import yaml
import pandas as pd
from pathlib import Path

schema_path = Path('config/schema.yaml')
artifacts_dir = Path('artifacts')

with open(schema_path, 'r', encoding='utf-8') as f:
    schema = yaml.safe_load(f)

schema_columns = [list(item.keys())[0] for item in schema.get('columns', [])]
print(f"Expected columns (schema): {len(schema_columns)}\n")

# find latest artifact folder
latest = None
if artifacts_dir.exists():
    # pick the most recent timestamped folder
    folders = sorted([p for p in artifacts_dir.iterdir() if p.is_dir()])
    if folders:
        latest = folders[-1]

if not latest:
    print('No artifacts folder found')
    raise SystemExit(1)

train_path = latest / 'data_ingestion' / 'ingested' / 'train.csv'
test_path = latest / 'data_ingestion' / 'ingested' / 'test.csv'

for p in (train_path, test_path):
    if not p.exists():
        print(f'Missing file: {p}')
        continue
    df = pd.read_csv(p)
    print(f'\nChecking file: {p}')
    print(f'Columns in file: {len(df.columns)}')
    missing = [c for c in schema_columns if c not in df.columns]
    extra = [c for c in df.columns if c not in schema_columns]
    print(f'Missing columns ({len(missing)}): {missing}')
    print(f'Extra columns ({len(extra)}): {extra[:20]}')
    # show a few examples where similar names may differ
    sim = []
    for sc in schema_columns:
        for fc in df.columns:
            if sc.replace(' ', '').lower() == fc.replace(' ', '').lower() and sc != fc:
                sim.append((sc, fc))
    if sim:
        print('\nPotential near-matches:')
        for a,b in sim:
            print(f'  {a}  <->  {b}')

print('\nDone')
