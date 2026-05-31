# Framework MO-CL-D3QN Traffic Signal Control

Repository ini berisi program eksperimen **Adaptive Traffic Signal Control (ATSC)** pada satu persimpangan menggunakan **Multi-Objective Curriculum Learning Dueling Double Deep Q-Network (MO-CL-D3QN)** dan beberapa metode pembanding.

Program mencakup seluruh alur kerja:

1. pengecekan skenario SUMO;
2. training model;
3. evaluasi lima metrik penelitian;
4. visualisasi hasil;
5. implementasi model ke SUMO GUI;
6. perbandingan model proposed, ablation models, dan baseline controllers.

---

## 1. Metode yang Dibandingkan

| Method key | Model |
|---|---|
| `mo_cl_d3qn` | MO-CL-D3QN (Proposed) |
| `mo_d3qn_no_curriculum` | MO-D3QN tanpa Curriculum Learning |
| `cl_ddqn_single_objective` | CL-DDQN dengan single-objective reward |
| `single_objective_ddqn` | Single-Objective DDQN |
| `actuated` | Actuated Controller |
| `fixed_time` | Fixed-Time Controller |

---

## 2. Metrik Evaluasi

Program menghitung lima metrik utama:

| Metrik | Satuan | Arah optimasi |
|---|---|---|
| Average Travel Time | detik | lebih rendah lebih baik |
| Average Queue Length | kendaraan | lebih rendah lebih baik |
| Average Delay | detik/kendaraan | lebih rendah lebih baik |
| Throughput | kendaraan/jam | lebih tinggi lebih baik |
| Fuel Consumption | liter/jam | lebih rendah lebih baik |

---

## 3. Struktur Repository

```text
single_intersection_mo_cl_d3qn_github/
├── scenario/
│   ├── environment.net.xml
│   ├── environment.sumocfg
│   └── routes_medium.rou.xml
├── single_intersection_experiment.py
├── plot_results.py
├── deploy_all_models_gui.py
├── run_full_pipeline.bat
├── run_quick_test.bat
├── run_scenario_gui.bat
├── run_deploy_all_models_gui.bat
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 4. Prasyarat

### 4.1 Install Python

Gunakan Python 3.10 atau versi yang lebih baru.

Cek versi Python:

```bash
python --version
```

### 4.2 Install SUMO

Install **Eclipse SUMO** dan pastikan perintah berikut dapat dijalankan:

```bash
sumo --version
sumo-gui --version
```

### 4.3 Set `SUMO_HOME` pada Windows

Contoh:

```bat
set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
set PATH=%PATH%;%SUMO_HOME%\bin
```

Jika SUMO dipasang pada folder lain, sesuaikan path tersebut.

### 4.4 Install Library Python

Dari folder repository, jalankan:

```bash
pip install -r requirements.txt
```

---

## 5. Cek Skenario SUMO Secara Manual

Buka simulasi menggunakan SUMO GUI:

```bash
sumo-gui -c scenario/environment.sumocfg
```

Atau jalankan:

```bat
run_scenario_gui.bat
```

Pastikan:

- network berhasil dimuat;
- kendaraan muncul;
- traffic light berubah fase;
- tidak ada error route;
- simulasi dapat berjalan sampai selesai.

---

## 6. Uji Cepat Sebelum Training Penuh

Jalankan:

```bash
python single_intersection_experiment.py --mode full_pipeline --episodes 5 --eval-episodes 2 --max-steps 500 --sumocfg scenario/environment.sumocfg
```

Atau gunakan:

```bat
run_quick_test.bat
```

Tujuan uji cepat adalah memastikan seluruh pipeline berjalan sebelum training penuh.

---

## 7. Training dan Evaluasi Semua Model

### 7.1 Jalankan Pipeline Lengkap

```bash
python single_intersection_experiment.py --mode full_pipeline --episodes 120 --eval-episodes 5 --sumocfg scenario/environment.sumocfg
```

Atau gunakan:

```bat
run_full_pipeline.bat
```

Pipeline ini:

1. melakukan training empat model berbasis RL;
2. menyimpan model terbaik dan model episode terakhir;
3. mengevaluasi empat model RL;
4. mengevaluasi Actuated Controller;
5. mengevaluasi Fixed-Time Controller;
6. menghasilkan tabel ringkasan lima metrik.

### 7.2 Training Semua Model RL Saja

```bash
python single_intersection_experiment.py --mode train_all --episodes 120 --sumocfg scenario/environment.sumocfg
```

### 7.3 Evaluasi Semua Model Saja

Gunakan setelah model selesai dilatih:

```bash
python single_intersection_experiment.py --mode evaluate_all --eval-episodes 5 --sumocfg scenario/environment.sumocfg
```

### 7.4 Evaluasi Baseline Saja

```bash
python single_intersection_experiment.py --mode baselines --eval-episodes 5 --sumocfg scenario/environment.sumocfg
```

---

## 8. Training Model Tertentu

### 8.1 MO-CL-D3QN Proposed

```bash
python single_intersection_experiment.py --mode train --method mo_cl_d3qn --episodes 120 --sumocfg scenario/environment.sumocfg
```

### 8.2 MO-D3QN tanpa Curriculum Learning

```bash
python single_intersection_experiment.py --mode train --method mo_d3qn_no_curriculum --episodes 120 --sumocfg scenario/environment.sumocfg
```

### 8.3 CL-DDQN Single Objective

```bash
python single_intersection_experiment.py --mode train --method cl_ddqn_single_objective --episodes 120 --sumocfg scenario/environment.sumocfg
```

### 8.4 Single-Objective DDQN

```bash
python single_intersection_experiment.py --mode train --method single_objective_ddqn --episodes 120 --sumocfg scenario/environment.sumocfg
```

---

## 9. Output Hasil Training dan Evaluasi

Output training tersimpan di:

```text
outputs_single_intersection/
```

Model terbaik:

```text
outputs_single_intersection/mo_cl_d3qn/best_model.pt
outputs_single_intersection/mo_d3qn_no_curriculum/best_model.pt
outputs_single_intersection/cl_ddqn_single_objective/best_model.pt
outputs_single_intersection/single_objective_ddqn/best_model.pt
```

Model episode terakhir:

```text
outputs_single_intersection/<method>/last_model.pt
```

Ringkasan lima metrik:

```text
outputs_single_intersection/evaluation_summary.csv
```

Data mentah seluruh evaluasi:

```text
outputs_single_intersection/evaluation_raw.csv
```

Ringkasan tanpa pembulatan:

```text
outputs_single_intersection/evaluation_summary_unrounded.csv
```

---

## 10. Membuat Visualisasi Hasil

Jalankan:

```bash
python plot_results.py --csv outputs_single_intersection/evaluation_summary.csv
```

Grafik tersimpan di:

```text
outputs_single_intersection/plots/
```

Grafik yang dihasilkan mencakup:

1. Average Travel Time;
2. Average Queue Length;
3. Average Delay;
4. Throughput;
5. Fuel Consumption;
6. Normalized Performance Index.

---

## 11. Implementasi Model ke SUMO GUI

### 11.1 Cek Import Program Deployment

Jalankan:

```bash
python deploy_all_models_gui.py --check-import
```

Jika berhasil, akan muncul pesan bahwa import program kompatibel.

### 11.2 Deploy Model Proposed Saja

```bash
python deploy_all_models_gui.py --method mo_cl_d3qn --gui --wait-before-start --keep-open --sumocfg scenario/environment.sumocfg
```

Alur program:

1. SUMO GUI terbuka pada waktu 0;
2. terminal menunggu input;
3. tekan `ENTER`;
4. model mulai mengendalikan traffic light;
5. metrik ditampilkan secara periodik;
6. GUI tetap terbuka setelah simulasi selesai sampai `ENTER` ditekan kembali.

### 11.3 Deploy Semua Model RL

```bash
python deploy_all_models_gui.py --method rl_all --gui --wait-before-start --keep-open --sumocfg scenario/environment.sumocfg
```

### 11.4 Deploy Baseline Controllers

```bash
python deploy_all_models_gui.py --method baseline_all --gui --wait-before-start --keep-open --sumocfg scenario/environment.sumocfg
```

### 11.5 Deploy Semua Model dan Baseline

```bash
python deploy_all_models_gui.py --method all --gui --wait-before-start --keep-open --sumocfg scenario/environment.sumocfg
```

Atau jalankan:

```bat
run_deploy_all_models_gui.bat
```

---

## 12. Output Deployment

Hasil implementasi GUI tersimpan di:

```text
outputs_deployment/
```

Ringkasan semua metode:

```text
outputs_deployment/deployment_comparison_summary.csv
```

Raw metrics:

```text
outputs_deployment/deployment_all_models_metrics.csv
```

Log aksi setiap metode:

```text
outputs_deployment/deployment_actions_mo_cl_d3qn.csv
outputs_deployment/deployment_actions_mo_d3qn_no_curriculum.csv
outputs_deployment/deployment_actions_cl_ddqn_single_objective.csv
outputs_deployment/deployment_actions_single_objective_ddqn.csv
outputs_deployment/deployment_actions_actuated.csv
outputs_deployment/deployment_actions_fixed_time.csv
```

---

## 13. Hyperparameter Default

| Hyperparameter | Nilai |
|---|---:|
| Episodes | 120 |
| Evaluation episodes | 5 |
| Gamma | 0.99 |
| Learning rate | 1e-4 |
| Batch size | 64 |
| Replay buffer | 50,000 |
| Minimum replay size | 1,000 |
| Target update frequency | 500 training steps |
| Hidden dimension | 256 |
| Epsilon start | 1.00 |
| Epsilon end | 0.05 |
| Epsilon decay steps | 25,000 |
| Decision interval | 10 detik |
| Yellow duration | 3 detik |
| Maximum simulation time | 3,600 detik |

---

## 14. Catatan Reproducibility

Nilai hasil eksperimen dapat berbeda jika terdapat perubahan pada:

- random seed;
- route kendaraan;
- network SUMO;
- versi SUMO;
- versi library Python;
- jumlah episode;
- durasi simulasi;
- hyperparameter;
- spesifikasi hardware.

Untuk pelaporan penelitian, gunakan hasil aktual dari:

```text
outputs_single_intersection/evaluation_summary.csv
```

---

## 15. Upload ke GitHub

Buat repository baru di GitHub, lalu dari folder project jalankan:

```bash
git init
git add .
git commit -m "Initial commit: single-intersection MO-CL-D3QN experiment"
git branch -M main
git remote add origin <URL_REPOSITORY_GITHUB>
git push -u origin main
```

File output training dan checkpoint model tidak diunggah secara default karena sudah dimasukkan ke `.gitignore`.

---

## License

MIT License.
"# MO-CL-D3QN-experiment" 
