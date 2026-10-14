import pandas as pd
import os
from glob import glob
import pyarrow as pa
import pyarrow.parquet as pq

def stream_csv_to_parquet():
    raw_data_dir = "../data/raw_data"
    output_dir = "../data/processed_data"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # raw_data 하위의 모든 CSV 파일 검색
    csv_files = glob(os.path.join(raw_data_dir, "**/*.csv"), recursive=True)
    print(f">> 총 발견된 원본 CSV 파일: {len(csv_files)}개")
    
    if len(csv_files) == 0:
        print(f"[경고] '{raw_data_dir}' 경로 내에서 CSV 파일을 찾지 못했습니다.")
        return
    
    for file_path in csv_files:
        # raw_data 기준 상대 경로 계산 (예: cic-iiot-2025/attack_data/sample.csv)
        rel_path = os.path.relpath(file_path, raw_data_dir)
        
        # 경로를 부모/자식 단위로 쪼개기
        parts = pathlib_parts = rel_path.split(os.sep)
        
        # parts[0]은 연도 폴더 (예: cic-ids-2017 또는 cic-iiot-2025)
        # 마지막 항목은 파일 이름 (예: sample.csv)
        year_folder = parts[0]
        file_name = parts[-1]
        
        # 중간에 낀 attack_data, benign_data 같은 하위 폴더들은 무시하고
        # 연도 폴더 바로 아래에 processed_data가 생성되도록 경로 조립
        target_output_dir = os.path.join(output_dir, year_folder)
        os.makedirs(target_output_dir, exist_ok=True)
        
        # 최종 파케 파일명 (필요시 파일 이름에 접두사를 붙여 구분할 수도 있지만, 원본 이름 그대로 저장)
        parquet_name = file_name.replace(".csv", ".parquet")
        output_path = os.path.join(target_output_dir, parquet_name)
        
        # 이미 변환된 파일이 있다면 중복 작업 방지
        if os.path.exists(output_path):
            print(f"[스킵] 이미 존재함: {year_folder}/{parquet_name}")
            continue
            
        print(f"[변환 시작] {rel_path} -> {year_folder}/{parquet_name}")
        
        writer = None
        try:
            # 10만 행씩 쪼개서(Chunk) 메모리 폭발(OOM) 방지
            for chunk in pd.read_csv(file_path, chunksize=100000, low_memory=False):
                chunk.columns = chunk.columns.str.strip()
                
                table = pa.Table.from_pandas(chunk)
                
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
                
                writer.write_table(table)
                
            print(f"[변환 완료] {output_path}")
            
        except Exception as e:
            print(f"[에러 발생] {rel_path} 처리 중 오류: {e}")
            
        finally:
            if writer is not None:
                writer.close()

if __name__ == "__main__":
    stream_csv_to_parquet()