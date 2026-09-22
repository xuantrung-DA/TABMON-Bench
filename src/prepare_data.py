import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from folktables import ACSDataSource, ACSIncome

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

# Cấu hình thư mục
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROC_DIR = os.path.join(DATA_DIR, 'processed')

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast các kiểu dữ liệu để tiết kiệm tối đa RAM và SSD (đáp ứng giới hạn 50GB SSD)."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.float32)
        else:
            # Chuyển object thành category để nén file Parquet tốt hơn
            df[col] = df[col].astype('category')
    return df

def generate_splits_and_save(df: pd.DataFrame, target_col: str, dataset_name: str):
    """B3 & B4: Tạo Train (40%) / Calibration (20%) / Test Pool (40%) và lưu dạng Parquet."""
    print(f"\n--- Xử lý {dataset_name} ---")
    
    # B5: Kiểm tra missing và imbalance
    missing_count = df.isnull().sum().sum()
    class_dist = df[target_col].value_counts(normalize=True).to_dict()
    cat_features = df.select_dtypes(include=['category']).columns.tolist()
    
    print(f"[*] Total rows: {len(df)}")
    print(f"[*] Missing values: {missing_count}")
    print(f"[*] Class imbalance: {class_dist}")
    print(f"[*] Categorical features ({len(cat_features)}): {cat_features}")

    # B2: Xử lý missing values cơ bản (Fill Unknown cho category, median cho numeric)
    # Split: Train(40%), Calib(20%), Test(40%)
    df_train_calib, df_test = train_test_split(df, test_size=0.4, stratify=df[target_col], random_state=42)
    df_train, df_calib = train_test_split(df_train_calib, test_size=0.3333, stratify=df_train_calib[target_col], random_state=42)

    # B4: Lưu Parquet
    ds_dir = os.path.join(PROC_DIR, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)
    
    df_train.to_parquet(os.path.join(ds_dir, 'train.parquet'), engine='pyarrow', compression='snappy')
    df_calib.to_parquet(os.path.join(ds_dir, 'calibration.parquet'), engine='pyarrow', compression='snappy')
    df_test.to_parquet(os.path.join(ds_dir, 'test_pool.parquet'), engine='pyarrow', compression='snappy')
    print(f"[v] Đã lưu dạng Parquet tại: {ds_dir}")

# --- Các hàm Tải Dữ Liệu (B1) ---

def load_adult():
    adult = fetch_ucirepo(id=2)
    X = adult.data.features.copy()
    y = adult.data.targets.copy()
    # Gộp nhãn >50K và >50K.
    y['income'] = y['income'].astype(str).apply(lambda x: 1 if '>50K' in x else 0)
    df = pd.concat([X, y], axis=1)
    df = optimize_dtypes(df)
    generate_splits_and_save(df, 'income', 'adult')

def load_bank_marketing():
    bank = fetch_ucirepo(id=222)
    X = bank.data.features.copy()
    y = bank.data.targets.copy()
    y['y'] = y['y'].apply(lambda x: 1 if x == 'yes' else 0)
    df = pd.concat([X, y], axis=1)
    df = optimize_dtypes(df)
    generate_splits_and_save(df, 'y', 'bank_marketing')

def load_covertype():
    # Covertype rất lớn (~581k dòng), Parquet là bắt buộc
    covertype = fetch_ucirepo(id=31)
    X = covertype.data.features.copy()
    y = covertype.data.targets.copy()
    
    # Biến bài toán thành Binary (vd: Class 2 vs Rest để tạo Imbalance rõ ràng)
    y['Cover_Type'] = y['Cover_Type'].apply(lambda x: 1 if x == 2 else 0)
    df = pd.concat([X, y], axis=1)
    df = optimize_dtypes(df)
    generate_splits_and_save(df, 'Cover_Type', 'covertype')

def load_folktables_acs():
    print("\nDownloading Folktables ACSIncome (CA, 2018)...")
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person', root_dir=RAW_DIR)
    acs_data = data_source.get_data(states=["CA"], download=True)
    
    features, labels, _ = ACSIncome.df_to_numpy(acs_data)
    df = pd.DataFrame(features, columns=ACSIncome.features)
    df['PINCP'] = labels # 1 nếu thu nhập > 50k
    
    df = optimize_dtypes(df)
    generate_splits_and_save(df, 'PINCP', 'acs_income')

if __name__ == "__main__":
    print("=== KÍCH HOẠT PHASE 1: CHUẨN BỊ DỮ LIỆU ===")
    load_adult()
    load_bank_marketing()
    load_folktables_acs()
    load_covertype()
    print("\n=== PHASE 1 HOÀN TẤT ===")
