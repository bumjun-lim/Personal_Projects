import glob
import os
import pandas as pd

def load_2017_data(file_path='../data/processed_data/sample_2017.parquet'):
    """2017년 표준 벤치마크 샘플 데이터 로드."""
    return pd.read_parquet(file_path)

def load_2025_benign(file_path='../data/processed_data/iiot_2025_benign.parquet'):
    """2025년 IIoT 오토인코더 학습용 '정상(Benign)' 데이터 로드."""
    return pd.read_parquet(file_path)

def load_2025_attack(file_path='../data/processed_data/iiot_2025_attack.parquet'):
    """2025년 IIoT 모델 평가 및 테스트용 '공격(Attack)' 데이터 로드."""
    return pd.read_parquet(file_path)


# 2. 파케이트 변환
if __name__ == '__main__':
    RAW_2017_PATH = '../data/raw_data/cic-ids-2017/'
    RAW_2025_BENIGN = '../data/raw_data/cic-iiot-2025/benign_data/'
    RAW_2025_ATTACK = '../data/raw_data/cic-iiot-2025/attack_data/'
    PROCESSED_PATH = '../data/processed_data/'

    os.makedirs(PROCESSED_PATH, exist_ok=True)

    print("=== [Day 1] 2017년 표준 데이터 샘플링 및 저장 ===")
    files_2017 = glob.glob(os.path.join(RAW_2017_PATH, '*.csv'))
    if files_2017:
        sample_2017 = pd.read_csv(files_2017[0], nrows=10000)
        df_2017_subset = sample_2017.sample(frac=0.1, random_state=42)
        out_2017 = os.path.join(PROCESSED_PATH, 'sample_2017.parquet')
        df_2017_subset.to_parquet(out_2017, index=False)
        print(f"완료: 2017년 샘플 저장 -> {out_2017} (행: {df_2017_subset.shape[0]}개)")
    else:
        print("경고: 2017년 데이터 경로를 확인해주세요.")

    print("\n=== [Day 1] 2025년 IIoT 정상(Benign) 데이터 압축 저장 ===")
    benign_files = glob.glob(os.path.join(RAW_2025_BENIGN, '*.csv'))
    benign_chunks = []
    for file_path in benign_files:
        print(f"정상 파일 읽는 중: {os.path.basename(file_path)}")
        for chunk in pd.read_csv(file_path, chunksize=50000, low_memory=False):
            benign_chunks.append(chunk)

    if benign_chunks:
        df_benign = pd.concat(benign_chunks, ignore_index=True)
        out_benign = os.path.join(PROCESSED_PATH, 'iiot_2025_benign.parquet')
        df_benign.to_parquet(out_benign, index=False)
        print(f"완료: 2025년 정상 데이터 저장 -> {out_benign} (총 행 수: {df_benign.shape[0]}행)")
    else:
        print("경고: 2025년 benign_data 경로를 확인해주세요.")

    print("\n=== [Day 1] 2025년 IIoT 공격(Attack) 데이터 압축 저장 ===")
    attack_files = glob.glob(os.path.join(RAW_2025_ATTACK, '*.csv'))
    attack_chunks = []
    for file_path in attack_files:
        print(f"공격 파일 읽는 중: {os.path.basename(file_path)}")
        for chunk in pd.read_csv(file_path, chunksize=50000, low_memory=False):
            attack_chunks.append(chunk)

    if attack_chunks:
        df_attack = pd.concat(attack_chunks, ignore_index=True)
        out_attack = os.path.join(PROCESSED_PATH, 'iiot_2025_attack.parquet')
        df_attack.to_parquet(out_attack, index=False)
        print(f"완료: 2025년 공격 데이터 저장 -> {out_attack} (총 행 수: {df_attack.shape[0]}행)")
    else:
        print("경고: 2025년 attack_data 경로를 확인해주세요.")
        
    print("\n[모든 Day 1 데이터 전처리 및 모듈 세팅 완료]")