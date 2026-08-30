import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
def load_2017_data(file_path='../data/processed_data/sample_2017.parquet', 
                   include_attack=True, 
                   split=None, 
                   test_size=0.2, 
                   random_state=42):
    """
    2017년 표준 벤치마크 샘플 데이터 로드 및 분할.
    - include_attack=False: BENIGN(정상) 데이터만 추출 (오토인코더/정상 학습용)
    - include_attack=True: 공격 데이터를 포함한 전체 데이터 반환 (모델 성능 평가용)
    - split=None: 전체 데이터 반환
    - split='train': 지도 학습용 학습 셋만 반환
    - split='test': 지도 학습용 테스트(평가) 셋만 반환
    """
    df = pd.read_parquet(file_path)
    
    # 동적으로 레이블 컬럼명 탐색 (공백 및 대소문자 무시)
    label_col = [c for c in df.columns if c.strip().lower() == 'label'][0]
    
    if not include_attack:
        # 정상 데이터만 필터링
        df = df[df[label_col].astype(str).str.strip().str.upper().str.contains('BENIGN')]
        
    # 만약 split 인자가 주어지면 train/test로 쪼개서 반환 (지도 학습용)
    if split is not None:
        # 정답 레이블 생성 (0: 정상, 1: 공격)
        y = df[label_col].astype(str).str.strip().str.upper().apply(lambda x: 0 if 'BENIGN' in x else 1)
        
        # 층화 추출(stratify)을 적용하여 정상/공격 비율을 유지하며 분할
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=y
        )
        
        if split.lower() == 'train':
            return train_df
        elif split.lower() == 'test':
            return test_df
            
    return df

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
        # 첫 번째 파일 전체를 읽어들임 (앞부분만 치우치게 가져오는 문제 방지)
        sample_2017 = pd.read_csv(files_2017[0], low_memory=False)
        
        # 전체 데이터를 완전히 섞은(Shuffle) 후 상위 일부를 샘플링
        df_2017_shuffled = sample_2017.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df_2017_subset = df_2017_shuffled.head(10000) # 필요에 따라 크기 조절
        
        out_2017 = os.path.join(PROCESSED_PATH, 'sample_2017.parquet')
        df_2017_subset.to_parquet(out_2017, index=False)
        print(f"완료: 2017년 샘플 저장 -> {out_2017} (총 행 수: {df_2017_subset.shape[0]}개)")
        
        # 저장된 파일 내부의 레이블 분포 확인 (정상과 공격이 잘 섞였는지 검증)
        label_col = [c for c in df_2017_subset.columns if c.strip().lower() == 'label'][0]
        print("--- 2017 샘플 내부 레이블 분포 ---")
        print(df_2017_subset[label_col].value_counts())
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