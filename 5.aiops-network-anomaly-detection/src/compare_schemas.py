import pandas as pd
import glob

def compare_schemas():
    path_2017 = glob.glob("../data/processed_data/cic-ids-2017/*.parquet")[0]
    path_2025 = glob.glob("../data/processed_data/cic-iiot-2025/*.parquet")[0]
    
    df_2017 = pd.read_parquet(path_2017)
    df_2025 = pd.read_parquet(path_2025)
    
    # 컬럼명 공백 제거 및 대소문자 통일 정제
    cols_2017 = set([c.strip().lower() for c in df_2017.columns])
    cols_2025 = set([c.strip().lower() for c in df_2025.columns])
    
    print(f">> 2017년 데이터 컬럼 수: {len(cols_2017)}개")
    print(f">> 2025년 IIoT 데이터 컬럼 수: {len(cols_2025)}개")
    
    common_cols = cols_2017.intersection(cols_2025)
    only_2017 = cols_2017 - cols_2025
    only_2025 = cols_2025 - cols_2017
    
    print(f">> 공통 컬럼 수: {len(common_cols)}개")
    print(f">> 2017년에만 있는 컬럼: {list(only_2017)}")
    print(f">> 2025년에만 있는 컬럼: {list(only_2025)}")

if __name__ == "__main__":
    compare_schemas()