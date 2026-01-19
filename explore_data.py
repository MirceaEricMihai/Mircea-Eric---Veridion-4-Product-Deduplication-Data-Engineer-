import pandas as pd

print("Loading data...")
# Load the Parquet file
df = pd.read_parquet('veridion_product_deduplication_challenge.snappy.parquet')

# Display basic information
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "=" * 60)
print("COLUMN NAMES AND TYPES")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("MISSING VALUES COUNT")
print("=" * 60)
missing_counts = df.isnull().sum()
missing_pct = (missing_counts / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing %': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

# Save sample to CSV for easier viewing
print("\n" + "=" * 60)
print("SAVING SAMPLE TO CSV")
print("=" * 60)
df.head(20).to_csv('sample_products.csv', index=False)
print("Saved first 20 rows to 'sample_products.csv'")