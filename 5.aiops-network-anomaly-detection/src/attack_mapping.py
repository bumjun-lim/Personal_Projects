import glob
import os
import numpy as np
import pandas as pd

# 1. 상위 공격 그룹 맵핑 정의 (2017 & 2025 통합용)
attack_group_mapping = {
    # DoS / DDoS 계열
    "DOS": "DoS / DDoS",
    "DDOS": "DoS / DDoS",
    "DOS GOLDENEYE": "DoS / DDoS",
    "DOS HULK": "DoS / DDoS",
    "DOS SLOWHTTPTEST": "DoS / DDoS",
    "DOS SLOWLORIS": "DoS / DDoS",
    # Recon / Scan 계열
    "PORTSCAN": "Recon / Scan",
    "RECON": "Recon / Scan",
    # Brute Force 계열
    "FTP-PATATOR": "Brute Force",
    "SSH-PATATOR": "Brute Force",
    "WEB ATTACK - BRUTE FORCE": "Brute Force",
    "BRUTEFORCE": "Brute Force",
    # Web Attack 계열
    "WEB ATTACK - SQL INJECTION": "Web Attack",
    "WEB ATTACK - XSS": "Web Attack",
    "WEB": "Web Attack",
    # Malware / Botnet 계열
    "BOT": "Malware / Botnet",
    "MALWARE": "Malware / Botnet",
    # 기타 특수 공격
    "MITM": "MITM",
    "INFILTRATION": "Infiltration",
    "HEARTBLEED": "Heartbleed",
    "BENIGN": "Normal",
    "NORMAL": "Normal",
}


def assign_attack_group(label_val):
  cleaned = str(label_val).strip().upper()

  # 1. 정확한 매칭 먼저 시도
  if cleaned in attack_group_mapping:
    return attack_group_mapping[cleaned]

  # 2. 특수 기호(∮ 등)나 띄어쓰기로 인해 깨진 경우를 위한 키워드 포함 검사
  if "WEB ATTACK" in cleaned or "SQL INJECTION" in cleaned or "XSS" in cleaned:
    return "Web Attack"
  if "BRUTE FORCE" in cleaned or "PATATOR" in cleaned:
    return "Brute Force"
  if "DOS" in cleaned:
    return "DoS / DDoS"
  if "PORTSCAN" in cleaned or "RECON" in cleaned:
    return "Recon / Scan"
  if "BOT" in cleaned or "MALWARE" in cleaned:
    return "Malware / Botnet"

  return "Other_Attack"

# 2. [체크 루프] 실제 데이터프레임의 고유 레이블과 맵핑 대조
def audit_labels(df_17, df_25):
  # 2017 레이블 컬럼 탐색
  col_17 = [c for c in df_17.columns if c.lower() == "label"][0]
  labels_17 = df_17[col_17].dropna().astype(str).str.strip().unique()

  # 2025 레이블 컬럼 탐색
  col_25 = [c for c in df_25.columns if c.lower() == "label2"][0]
  labels_25 = df_25[col_25].dropna().astype(str).str.strip().unique()

  print("================ [라벨 맵핑 검증 리포트] ================")

  for year, labels in [("2017", labels_17), ("2025", labels_25)]:
    print(f"\n>> [{year} 데이터셋 레이블 검사]")
    unmapped = []
    for lbl in labels:
      mapped = assign_attack_group(lbl)
      if mapped == "Other_Attack":
        unmapped.append(lbl)
      else:
        print(f"  [OK] '{lbl}' -> {mapped}")

    if unmapped:
      print(
          f"  [WARNING] ⚠️ 맵핑되지 않아 'Other_Attack'으로 빠지는 레이블 발견:"
          f" {unmapped}"
      )
    else:
      print(f"  [SUCCESS] ✨ 모든 레이블이 정상적으로 상위 그룹에 맵핑되었습니다!")
  print("==========================================================")

def main():
  path_2017 = "../data/processed_data/cic-ids-2017/*.parquet"
  path_2025 = "../data/processed_data/cic-iiot-2025/*.parquet"

  files_2017 = glob.glob(path_2017)
  files_2025 = glob.glob(path_2025)
    
  print(f">> 2017년 파일 개수: {len(files_2017)}개 발견")
  print(f">> 2025년 파일 개수: {len(files_2025)}개 발견")
    
  if not files_2017 or not files_2025:
    print(">> 경로를 다시 확인해주세요.")
  
    
  df_17 = pd.concat([pd.read_parquet(f) for f in files_2017], ignore_index=True)
  df_25 = pd.concat([pd.read_parquet(f) for f in files_2025], ignore_index=True)
  
  print(np.shape(df_17))
  print(np.shape(df_25))  
  df_17.columns = df_17.columns.str.strip()
  df_25.columns = df_25.columns.str.strip()
  audit_labels(df_17, df_25)
  print(np.shape(df_17))
  print(np.shape(df_25))

if __name__ == "__main__":
  main()