# Khoi tao moi truong ao
1. python -m venv .paper
2. .paper\Scripts\activate
3. pip install -r requirements.txt

# Tai dataset
1. python src/prepare_data.py

# Train Logistic, RF, XGBoost, MLP, thực hiện Isotonic Calibration và lưu Models + Logits
1. python experiments/run_phase2.py

# Unit test (dua 1 lỗi Covariate Shift giả lập vào tập Adult để xem Oracle Evaluator có chấm điểm đúng không)
1. python experiments/run_phase4_unit_test.py

# Core Benchmark
1. python experiments/run_core_benchmark.py
2. SMOKE_TEST = True de test 1 dataset/1 seed
3. mo run_core_benchmark.py chuyen SMOKE_TEST = False de chay toan bo ma tran thi nghiem

# Kiem tra pipeline truoc khi train du lieu that
1. python -m unittest tests.test_smoke_pipeline -v

