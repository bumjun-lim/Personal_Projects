import pandas as pd

df = pd.read_parquet(
    "../data/processed_data/master_scored_dataset.parquet"
)

# 2. CSV 파일로 내보내기 (한글 깨짐 방지를 위해 utf-8-sig 사용)
df.to_csv(
    "../data/processed_data/master_scored_dataset.csv",
    index=False,
    encoding="utf-8-sig",
)

print("🎉 CSV 변환 완료!")