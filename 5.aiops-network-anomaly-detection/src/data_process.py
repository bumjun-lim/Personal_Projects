import pandas as pd
import os
from glob import glob
import pyarrow as pa
import pyarrow.parquet as pq

def stream_csv_to_parquet():
    # 경로 설정 (프로젝트 루트 기준)
    raw_data_dir = "../data/raw_data"
    output_dir = "../data/processed_data"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # raw_data 하위의 모든 CSV 파일 검색 (하위 폴더 포함)
    csv_files = glob(os.path.join(raw_data_dir, "**/*.csv"), recursive=True)
    print(f">> 총 발견된 원본 CSV 파일: {len(csv_files)}개")
    
    if len(csv_files) == 0:
        print(f"[경고] '{raw_data_dir}' 경로 내에서 CSV 파일을 찾지 못했습니다.")
        return
    
    for file_path in csv_files:
        # raw_data 기준 상대 경로 계산 (예: cic-iiot-2025/attack_data/sample.csv)
        rel_path = os.path.relpath(file_path, raw_data_dir)
        
        # 하위 폴더 경로 구분자를 모두 언더바(_)로 치환하여 단일 플랫 이름으로 만듦
        # 예: cic-iiot-2025/attack_data/sample.csv -> cic-iiot-2025_attack_data_sample.parquet
        flat_name = rel_path.replace(os.sep, "_").replace(".csv", ".parquet")
        
        # 최상위 output_dir 바로 아래에 연도별 폴더 구조가 포함된 이름으로 일괄 저장
        output_path = os.path.join(output_dir, flat_name)
        
        # 이미 변환된 파일이 있다면 중복 작업 방지
        if os.path.exists(output_path):
            print(f"[스킵] 이미 존재함: {flat_name}")
            continue
            
        print(f"[변환 시작] {rel_path} -> {flat_name}")
        
        writer = None
        try:
            # 10만 행씩 쪼개서(Chunk) 메모리 폭발(OOM) 방지
            for chunk in pd.read_csv(file_path, chunksize=100000, low_memory=False):
                # 컬럼명 앞뒤 공백 정제
                chunk.columns = chunk.columns.str.strip()
                
                # Pandas DataFrame을 PyArrow 테이블로 변환 후 실시간 디스크에 쓰기
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