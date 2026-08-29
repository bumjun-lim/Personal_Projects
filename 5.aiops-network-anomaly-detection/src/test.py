import pandas as pd
import data_loader
# 1. 파일 불러오기
df = pd.read_parquet('../data/processed_data/sample_2017.parquet')
print(df.info())
print(df.head(3))

df = data_loader.load_2025_benign()
print(df.info())
print(df.head(3))
print(df['label_full'].value_counts())

df = data_loader.load_2025_attack()
print(df.info())
print(df.head(3))
print(df['label_full'].value_counts())