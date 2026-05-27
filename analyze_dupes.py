import csv
import collections
import sys

csv.field_size_limit(2147483647)

with open('data/processed/metasms_hss_master.csv', 'r', encoding='utf-8') as f:
    reader = list(csv.DictReader(f))
    
print(f'Total rows read: {len(reader)}')

prefixes = collections.defaultdict(list)
for row in reader:
    text = row['text']
    prefix = text[:60].lower().strip()
    prefixes[prefix].append(row)
    
dupes = {k: v for k, v in prefixes.items() if len(v) > 1}
print(f'Found {len(dupes)} prefixes that have multiple entries.')

count = 0
for k, v in list(dupes.items())[:10]:
    print(f'\n--- Group prefix: {k} ---')
    for i, row in enumerate(v[:5]):
        print(f'[{i}] {row["source"]}: {row["text"][:150]}...')
