"""
Product Deduplication Script
Removes duplicate product records by matching URL + description prefix.
"""

import pandas as pd
import numpy as np
import json

print("Starting deduplication process...")
print("=" * 70)

# Load data
print("\nLoading data...")
df = pd.read_parquet('veridion_product_deduplication_challenge.snappy.parquet')
print(f"Loaded {len(df):,} records with {len(df.columns)} columns")

# Extract description from available fields
def get_description(row):
    # Try multiple fields in order of preference
    for field in ['product_summary', 'description', 'product_title', 'product_name']:
        if field in row.index:
            value = row[field]
            if value and isinstance(value, str) and len(value.strip()) >= 3:
                return value.strip()
    return ""

print("\nExtracting descriptions...")
descriptions = []
for idx, row in df.iterrows():
    descriptions.append(get_description(row))
    if (idx + 1) % 1000 == 0:
        print(f"  Processed {idx + 1}/{len(df)} rows...")

df['description'] = descriptions
print(f"Found {sum(1 for d in descriptions if d)} valid descriptions")

# Normalize text for comparison
def normalize(text):
    if not text:
        return ""
    return ' '.join(text.lower().strip().split())

print("\nNormalizing descriptions...")
df['normalized_desc'] = df['description'].apply(normalize)

# Create deduplication key: URL + first 20 chars of description
print("\nCreating deduplication keys...")
df['desc_prefix'] = df['normalized_desc'].str[:20].fillna('NO_DESC')
df['dedup_key'] = df['page_url'] + '|||' + df['desc_prefix']

print(f"Created {df['dedup_key'].nunique():,} unique groups")
print(f"Found {len(df) - df['dedup_key'].nunique():,} potential duplicates")

# Check if value is empty
def is_empty(value):
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == '':
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    if isinstance(value, np.ndarray) and value.size == 0:
        return True
    try:
        if pd.isna(value):
            return True
    except:
        pass
    return False

# Merge duplicates - keep most complete record
def merge_group(group):
    if len(group) == 1:
        return group.iloc[0]
    
    # Find record with most non-empty fields
    temp_cols = ['description', 'normalized_desc', 'desc_prefix', 'dedup_key']
    best_idx = None
    best_count = -1
    
    for idx in group.index:
        count = sum(1 for col in group.columns 
                   if col not in temp_cols and not is_empty(group.loc[idx, col]))
        if count > best_count:
            best_count = count
            best_idx = idx
    
    # Start with best record
    merged = group.loc[best_idx].copy()
    
    # Fill missing fields from other records
    for idx in group.index:
        if idx == best_idx:
            continue
        for col in group.columns:
            if col in temp_cols:
                continue
            if is_empty(merged[col]) and not is_empty(group.loc[idx, col]):
                merged[col] = group.loc[idx, col]
    
    return merged

# Process groups
print("\n" + "=" * 70)
print("DEDUPLICATING...")
print("=" * 70)

deduplicated = []
processed = 0

for key, group in df.groupby('dedup_key'):
    deduplicated.append(merge_group(group))
    processed += 1
    if processed % 500 == 0:
        print(f"  Processed {processed:,} groups...")

print(f"  Processed {processed:,} groups... Done!")

# Create final dataset
print("\nCreating final dataset...")
result = pd.DataFrame(deduplicated)
result = result.drop(columns=['description', 'normalized_desc', 'desc_prefix', 'dedup_key'])
result = result.reset_index(drop=True)

# Results
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Original records:      {len(df):,}")
print(f"Deduplicated records:  {len(result):,}")
print(f"Duplicates removed:    {len(df) - len(result):,}")
dedup_rate = ((len(df) - len(result)) / len(df)) * 100
print(f"Deduplication rate:    {dedup_rate:.2f}%")

print(f"\nUnique URLs: {result['page_url'].nunique():,}")
if 'brand' in result.columns:
    print(f"Unique brands: {result['brand'].nunique():,}")

# Sample
print("\n" + "=" * 70)
print("SAMPLE (First 5 records)")
print("=" * 70)
cols = ['product_name', 'brand', 'page_url'] if 'brand' in result.columns else ['product_name', 'page_url']
print(result[cols].head())

# Save results
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Convert complex columns to strings for export
export_df = result.copy()
for col in export_df.columns:
    if len(export_df) > 0:
        sample = export_df[col].iloc[0]
        if isinstance(sample, (list, np.ndarray)):
            export_df[col] = export_df[col].apply(
                lambda x: str(x.tolist()) if isinstance(x, np.ndarray) else str(x)
            )

# Save files
try:
    export_df.to_parquet('deduplicated_products.parquet', index=False)
    print("✓ Saved: deduplicated_products.parquet")
except Exception as e:
    print(f"⚠ Parquet error: {e}")

export_df.to_csv('deduplicated_products.csv', index=False)
print("✓ Saved: deduplicated_products.csv")

result.to_pickle('deduplicated_products.pkl')
print("✓ Saved: deduplicated_products.pkl")

# Save summary
summary = {
    'original_records': int(len(df)),
    'deduplicated_records': int(len(result)),
    'duplicates_removed': int(len(df) - len(result)),
    'deduplication_rate_percent': round(dedup_rate, 2),
    'unique_urls': int(result['page_url'].nunique()),
    'method': 'URL + First 20 characters of description'
}

with open('deduplication_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("✓ Saved: deduplication_summary.json")

print("\n" + "=" * 70)
print("✓ COMPLETE!")
print("=" * 70)