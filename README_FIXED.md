# IT4653 — Học sâu và ứng dụng

## Đề tài 9. Mô hình sinh: từ Autoencoder tới VAE và GAN

So sánh hai họ mô hình sinh trên cùng dữ liệu MNIST và cùng ngân sách tính toán,
làm rõ đánh đổi giữa độ nét ảnh và tính ổn định khi huấn luyện.

| Thành viên | Phần phụ trách | File |
|---|---|---|
| Nguyễn Hà Minh | Autoencoder | `1_autoencoder.py` |
| Đỗ Thành Đạt | VAE, Conditional VAE, nội suy | `2_…`, `3_…`, `4_…`, `poc_interpolation.py` |
| Nguyễn Trường Chinh | DCGAN, Conditional GAN, FID | `5_gan.py` |

**Mục lục**

1. [Yêu cầu đề bài và chỗ đáp ứng](#1-yêu-cầu-đề-bài-và-chỗ-đáp-ứng)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Cài đặt](#3-cài-đặt)
4. [Xác nhận GPU](#4-xác-nhận-gpu)
5. [Chạy](#5-chạy)
6. [Đọc kết quả in ra màn hình](#6-đọc-kết-quả-in-ra-màn-hình)
7. [Kết quả nằm ở đâu](#7-kết-quả-nằm-ở-đâu)
8. [Thư mục `shared/` và tính so sánh được của FID](#8-thư-mục-shared-và-tính-so-sánh-được-của-fid)
9. [Tham số dòng lệnh](#9-tham-số-dòng-lệnh)
10. [Lỗi hay gặp](#10-lỗi-hay-gặp)
11. [Tài liệu tham khảo](#12-tài-liệu-tham-khảo)

---

## 1. Yêu cầu đề bài và chỗ đáp ứng

### Yêu cầu bắt buộc

| # | Yêu cầu | File | Kết quả |
|---|---|---|---|
| 1 | Nắm rõ hyperparameter, thử nghiệm ảnh hưởng của từng tham số | cả ba | `experiment_log.csv`, quét `--latent-dims`, quét `--beta`, `--seed` của GAN |
| 2 | Cài AE thường và VAE, viết rõ loss reconstruction + KL, cài reparameterization trick và nêu vì sao không lan truyền ngược qua phép lấy mẫu | `1_…`, `2_…` | docstring trong hai file, `experiment_log.csv` |
| 3 | Khảo sát chiều ẩn 2, 8, 32, 128 tới chất lượng tái tạo; với chiều ẩn = 2 vẽ bản đồ 2D | `1_…`, `2_…` | `recon_bce_latent_<L>.png`, `loss_comparison_by_latent_dim.png`, `latent_map_2d_bce.png` |
| 4 | Nội suy tuyến tính giữa hai điểm trong không gian ẩn của AE và VAE, đặt cạnh nhau | `poc_interpolation.py` | `interp_grid_ae_vs_vae.png`, `interp_step_deltas.png` |
| 5 | Cài DCGAN, mô tả G/D, ghi đường loss của cả hai, bình luận bất ổn định, ghi nhận mode collapse | `5_gan.py dcgan` | `dcgan_curves.png`, `dcgan_progress.png`, `instability.png`, `mode_hist.png` |
| 6 | So sánh AE / VAE / GAN bằng chỉ số định lượng (FID) và lưới ảnh sinh ra ở cùng số epoch | `5_gan.py compare` | `bang_so_sanh.csv`, `so_sanh_luoi_anh.png`, `so_sanh_cot.png` |

### Yêu cầu khác

| # | Yêu cầu | File | Kết quả |
|---|---|---|---|
| 7 | Conditional VAE hoặc Conditional GAN sinh ảnh theo nhãn cho trước | `3_…`, `4_…`, `5_gan.py cgan` | `samples_by_class_*.png`, `label_swap_*.png`, `demo_B_*.png`, `cgan_by_label.png` |
| 8 | Phát hiện bất thường bằng sai số tái tạo (train trên 9 chữ số, test trên chữ số còn lại) | `1_…`, `2_…`, `3_…` | `anomaly_digit_<d>.png`, `anomaly_log.csv` |

**Dữ liệu.** MNIST, tự tải về `data/` ở lần chạy đầu (~10 MB). Không cần chuẩn bị gì.

---

## 2. Cấu trúc thư mục

```
1_autoencoder.py                          AE — tái tạo, khảo sát latent, phát hiện bất thường
2_variational_autoencoder.py              VAE / beta-VAE — thêm sinh từ prior, latent manifold
3_conditional_variational_autoencoder.py  CVAE — sinh theo nhãn
4_conditional_generation_demo.py          demo nhãn điều khiển đầu ra, chỉ nạp checkpoint
poc_interpolation.py                      nội suy không gian ẩn, AE đặt cạnh VAE
5_gan.py                                  DCGAN, cGAN, FID, bảng so sánh ba mô hình

scripts/command.sh    toàn bộ lệnh tái hiện thí nghiệm, theo thứ tự
shared/               file nhị phân dùng chung của nhóm — ĐƯỢC commit, xem mục 8
runs/                 kết quả mỗi lần chạy — KHÔNG commit
data/                 MNIST tải về — KHÔNG commit
pyproject.toml        khai báo thư viện và nguồn wheel PyTorch
```

Bốn file đầu là một script độc lập mỗi file. `5_gan.py` gom cả phần GAN vào một
file, chọn việc cần làm bằng tham số đầu tiên (`prepare`, `dcgan`, `cgan`,
`compare`, `all`).

---

## 3. Cài đặt

Repo dùng [uv](https://docs.astral.sh/uv/) để quản lý môi trường.

### 3.1. Windows

Mở **PowerShell** và cài `uv`:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Đóng hẳn PowerShell rồi mở lại để PATH được cập nhật. Sau đó `cd` về thư mục dự án:

```powershell
uv --version
```

### 3.2. Cấu hình PyTorch CUDA cho Windows / Linux

Nếu máy có GPU NVIDIA và muốn dùng CUDA 12.4, kiểm tra cuối `pyproject.toml` phải có cấu hình nguồn PyTorch cho **cả Windows (`win32`) và Linux**:

```toml
[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cuda", marker = "sys_platform == 'win32' or sys_platform == 'linux'" }
]
torchvision = [
    { index = "pytorch-cuda", marker = "sys_platform == 'win32' or sys_platform == 'linux'" }
]
```

> **Quan trọng trên Windows:** nếu marker chỉ có `sys_platform == 'linux'`, `uv run` có thể bỏ qua CUDA index và đồng bộ lại về wheel CPU. Khi đó dù vừa cài `torch+cu124` bằng `uv pip install`, lần chạy `uv run ...` tiếp theo vẫn có thể gỡ nó ra.

Để kết quả khớp với README này, phần dependencies nên cố định cặp phiên bản:

```toml
"torch==2.6.0",
"torchvision==0.21.0",
```

Sau khi sửa `pyproject.toml`, nếu trước đó môi trường đã cài nhầm bản CPU, làm sạch một lần:

```powershell
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
Remove-Item uv.lock -Force -ErrorAction SilentlyContinue
uv lock
uv sync
```

Nếu dự án chưa từng cài sai thì chỉ cần:

```powershell
uv sync
```

`uv sync` tạo `.venv` ngay trong dự án. Từ đây chạy chương trình bằng `uv run python ...`; không cần `activate` môi trường thủ công.

### 3.3. macOS / Linux — cài uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Linux + NVIDIA có thể giữ cấu hình CUDA ở trên. macOS không dùng CUDA NVIDIA, xem mục CPU/macOS phía dưới.

### 3.4. Kiểm tra driver NVIDIA

Không cần cài Visual Studio hay CUDA Toolkit chỉ để chạy PyTorch wheel — wheel CUDA của PyTorch đã kèm CUDA runtime cần thiết.

Kiểm tra driver:

```text
nvidia-smi
```

Với CUDA 12.4 nên dùng NVIDIA driver đủ mới (README này lấy mốc Driver Version ≥ 551). Nếu driver quá cũ, cập nhật driver hoặc chuyển index PyTorch sang một bản CUDA tương thích hơn.

Con số `CUDA Version` trong `nvidia-smi` là mức CUDA tối đa mà driver hỗ trợ, không phải CUDA Toolkit mà project đang dùng. Phiên bản CUDA thực tế của PyTorch được kiểm tra bằng `torch.version.cuda` ở mục 4.

### 3.5. Máy CPU hoặc macOS

Nếu không dùng GPU NVIDIA, xoá hai khối sau khỏi `pyproject.toml`:

```toml
[[tool.uv.index]]
...

[tool.uv.sources]
...
```

Sau đó chạy lại:

```powershell
uv lock --refresh
uv sync
```

PyTorch sẽ dùng wheel mặc định phù hợp với nền tảng. Code vẫn chạy được nhưng huấn luyện trên CPU sẽ chậm hơn đáng kể.

### 3.6. Không dùng uv

Tạo virtual environment theo cách thông thường:

```bash
python -m venv .venv
```

Kích hoạt:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows CMD
.venv\Scripts\activate
```

Cài các thư viện chung:

```bash
pip install numpy scipy scikit-learn matplotlib pandas tqdm tensorboard
```

Nếu dùng Windows/Linux + NVIDIA CUDA 12.4:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
```

Sau đó chạy script trực tiếp, ví dụ:

```bash
python 5_gan.py all
```

---

## 4. Xác nhận GPU

Sau khi `uv sync`, chạy lệnh kiểm tra này **trước khi train**:

```powershell
uv run python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Với RTX 3060 và cấu hình CUDA 12.4, kết quả mong đợi:

```text
Torch: 2.6.0+cu124
CUDA: 12.4
Available: True
GPU: NVIDIA GeForce RTX 3060
```

### Nếu ra `2.x.x+cpu`, `CUDA: None` hoặc `Available: False`

Ví dụ lỗi:

```text
Torch: 2.13.0+cpu
CUDA: None
Available: False
GPU: NO CUDA
```

Điều này nghĩa là môi trường project hiện đang dùng **PyTorch CPU-only**.

Không nên chỉ chạy:

```powershell
uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```

rồi coi như đã xong. Lệnh đó có thể cài CUDA tạm thời vào `.venv`, nhưng `uv run` sẽ kiểm tra project và có thể đồng bộ lại theo `pyproject.toml` / `uv.lock`. Nếu source marker trong project sai, bản CUDA vừa cài sẽ bị gỡ và thay lại bằng bản CPU.

Cách sửa đúng:

1. Kiểm tra `[tool.uv.sources]` trong `pyproject.toml` có marker Windows:

```toml
marker = "sys_platform == 'win32' or sys_platform == 'linux'"
```

2. Đảm bảo version mong muốn là:

```toml
"torch==2.6.0",
"torchvision==0.21.0",
```

3. Làm sạch môi trường và lock cũ:

```powershell
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
Remove-Item uv.lock -Force -ErrorAction SilentlyContinue
uv lock
uv sync
```

4. Chạy lại lệnh kiểm tra GPU ở đầu mục này.

Nếu `torch.cuda.is_available()` vẫn là `False` dù version đã có `+cu124`, chạy:

```text
nvidia-smi
```

Nếu `nvidia-smi` không nhận GPU hoặc báo lỗi driver, xử lý driver NVIDIA trước.

Bỏ qua bước xác nhận GPU thì code vẫn có thể chạy bằng CPU mà không báo lỗi rõ ràng, nhưng thời gian train sẽ tăng mạnh.

---

## 5. Chạy

Toàn bộ lệnh nằm sẵn trong `scripts/command.sh`. Trên Windows đổi `--num-workers 4`
thành `2` — tiến trình con trên Windows dùng `spawn`, khởi động chậm hơn Linux, để
cao dễ bị treo lâu ở đầu mỗi epoch mà CPU vẫn 100%.

> **`--epochs` của cả ba mô hình phải bằng nhau** (mặc định đều là 10). Yêu cầu 6
> là so sánh *ở cùng số epoch*; chặng `compare` sẽ in cảnh báo nếu phát hiện lệch.

### 5.1. Autoencoder

```
uv run python 1_autoencoder.py --out-dir runs --losses bce --num-workers 2 --anomaly-loss bce
```

Quét lần lượt 4 chiều ẩn `2, 8, 32, 128`, mỗi chiều 10 epoch, rồi chạy tiếp phần
phát hiện bất thường (train trên 9 chữ số, test trên chữ số còn lại, lặp 10 lần).

Kết quả vào `runs/ae_mnist_seed42/`. Thêm `--skip-anomaly` để bỏ phần bất thường.

### 5.2. Variational Autoencoder

```
uv run python 2_variational_autoencoder.py --out-dir runs --losses bce --num-workers 2 --anomaly-loss bce --beta 1.0
```

Kết quả vào `runs/vae_mnist_beta1.0_seed42/`.

File đáng chú ý: `samples_bce_latent_32.png` — **ảnh sinh mới từ prior `N(0, I)`**,
thứ mà AE không làm được. Đặt cạnh `recon_bce_latent_32.png` (ảnh tái tạo) là thấy
ngay khác biệt giữa "tái tạo" và "sinh mới".

### 5.3. Conditional VAE

Bản đầy đủ, quét latent và phần bất thường:

```
uv run python 3_conditional_variational_autoencoder.py --out-dir runs --losses bce --num-workers 2 --anomaly-loss bce --anomaly-score-mode min-over-labels
```

Bản chính dùng cho báo cáo, latent 32, 10 epoch:

```
uv run python 3_conditional_variational_autoencoder.py --epochs 10 --latent-dims 32 --losses bce --skip-anomaly --samples-per-class 10 --num-workers 0
```

Quét `beta` — latent rộng giữ nhiều thông tin lớp trong `z` nên đổi nhãn kém trung
thành; tăng `beta` ép `z` bỏ bớt thông tin đó:

```
uv run python 3_conditional_variational_autoencoder.py --epochs 10 --latent-dims 32 --losses bce --skip-anomaly --samples-per-class 10 --num-workers 0 --beta 2
uv run python 3_conditional_variational_autoencoder.py --epochs 10 --latent-dims 32 --losses bce --skip-anomaly --samples-per-class 10 --num-workers 0 --beta 4
```

Đo được ở epoch 10 (test recon / test KL): `beta 1` → 65.9 / 23.4 tái tạo nét nhất
nhưng một hàng đổi nhãn không chịu đổi chữ số; `beta 2` → 75.2 / 13.8 điều khiển
nhãn sạch, vẫn giữ đa dạng nét chữ — **cân bằng tốt nhất**; `beta 4` → 92.1 / 6.8
điều khiển sạch nhưng `z` bị bóp quá, các hàng trông giống nhau.

Bản latent 2 cho các hình cần không gian ẩn 2 chiều:

```
uv run python 3_conditional_variational_autoencoder.py --epochs 5 --latent-dims 2 --losses bce --skip-anomaly --samples-per-class 10 --num-workers 0
```

Kết quả vào `runs/cvae_mnist_beta<beta>_seed42/`, mỗi `beta` một thư mục riêng.

### 5.4. Demo sinh ảnh theo nhãn và nội suy

Không huấn luyện gì, chỉ nạp checkpoint từ 5.3:

```
uv run python 4_conditional_generation_demo.py --latent-dim 32 --digits 2 0 2 6 --styles 3 --n-real 4
```

Chỉ định thẳng checkpoint của một `beta` cụ thể (nếu không, script tự lấy thư mục
sắp xếp cuối cùng):

```
uv run python 4_conditional_generation_demo.py --latent-dim 32 --checkpoint runs/cvae_mnist_beta2.0_seed42/cvae_bce_latent_32.pt --digits 2 0 2 6 --styles 3 --n-real 4
```

Nội suy AE so với VAE, cần checkpoint latent 32 từ 5.1 và 5.2:

```
uv run python poc_interpolation.py --cae-file 1_autoencoder.py --cvae-file 2_variational_autoencoder.py --latent-dim 32 --pairs 0-1 3-8 4-9 7-2 --steps 11
```

### 5.5. GAN

Chạy một phát cả bốn chặng:

```
uv run python 5_gan.py all --out-dir runs --epochs 10
```

Hoặc từng chặng, để dừng lại xem kết quả:

```
uv run python 5_gan.py prepare
uv run python 5_gan.py dcgan   --out-dir runs --epochs 10
uv run python 5_gan.py cgan    --out-dir runs --epochs 10
uv run python 5_gan.py compare --out-dir runs
```

| Chặng | Làm gì | Yêu cầu |
|---|---|---|
| `prepare` | dựng bộ đo FID dùng chung, ghi ra `shared/` | — |
| `dcgan` | huấn luyện DCGAN, đường loss, phân tích bất ổn định, mode collapse | 5 |
| `cgan` | huấn luyện cGAN, sinh ảnh theo nhãn, đo độ khớp nhãn | 7 |
| `compare` | bảng FID + lưới ảnh AE / VAE / GAN | 6 |
| `all` | chạy cả bốn, tự bỏ qua `prepare` nếu `shared/` đã có sẵn | |

`prepare` chỉ cần chạy khi `shared/` chưa có `feature_cnn.pt` và `fid_ref.npz`.
Repo đã kèm sẵn hai file này — **đừng chạy lại `prepare`**, vì nó sinh ra bộ đo FID
mới và số sẽ không còn khớp với báo cáo. Xem mục 8.

Kết quả vào `runs/gan_mnist_seed42/`.

### Thời gian ước tính trên RTX 3060 12GB

| Phần | Đầy đủ | Rút gọn (`--skip-anomaly`) |
|---|---|---|
| 5.1 AE | 8–12 phút | 4–5 phút |
| 5.2 VAE | 8–12 phút | 4–5 phút |
| 5.3 CVAE bản chính + beta + latent 2 | ~7 phút | ~7 phút |
| 5.3 CVAE bản quét đầy đủ | 15–20 phút | — |
| 5.4 Demo + nội suy | vài giây | vài giây |
| 5.5 GAN | ~6 phút | ~6 phút |
| **Tổng** | **45–55 phút** | **~22 phút** |

Lần chạy đầu tiên tải MNIST, cộng thêm khoảng 30 giây. Trên CPU nhân khoảng 10 lần.

---

## 6. Đọc kết quả in ra màn hình

### Lúc GAN khởi động

```
Device: cuda (NVIDIA GeForce RTX 3060) | run dir: runs/gan_mnist_seed42
Shared FID ruler loaded | FID(ref, ref) = -0.0003 | n = 5000
batches/epoch: 468
```

- `Device: cuda (...)` — đang chạy trên GPU. Thấy `cpu` là quay lại mục 4.
- `FID(ref, ref)` phải gần 0. Đây là chốt kiểm tra bộ đo FID: chấm chính tập ảnh
  thật bằng thống kê đã lưu; lệch nhiều nghĩa là các file dùng chung không khớp
  nhau và mọi FID sau đó đều sai.

### Mỗi epoch của GAN

```
[dcgan] ep   7/10 | loss_D=0.912 loss_G=1.834 | D(x)=0.681 D(G(z))=0.352 | FID=18.44 entropy=2.271 | 84s
```

| Cột | Ý nghĩa | Muốn thấy gì |
|---|---|---|
| `loss_D`, `loss_G` | mất mát của Discriminator và Generator | **không** cần giảm đều — GAN là trò chơi hai người, loss thấp không có nghĩa ảnh đẹp |
| `D(x)` | xác suất D chấm cho ảnh thật | tiến về 0.5 |
| `D(G(z))` | xác suất D chấm cho ảnh giả | tiến về 0.5 |
| `FID` | khoảng cách phân bố với ảnh thật | càng thấp càng tốt, lên xuống thất thường là bình thường |
| `entropy` | độ đa dạng của 10 chữ số sinh ra | tiến về `ln 10 = 2.303`; dưới 1.8 là mode collapse |

`D(x)` leo lên gần 1 còn `D(G(z))` tụt về 0 nghĩa là **D thắng áp đảo** — G không
còn gradient để học. Đây chính là kiểu hỏng cần ghi nhận cho yêu cầu 5.

### Sau khi DCGAN xong

Khối `PHÂN TÍCH BẤT ỔN ĐỊNH (10 epoch cuối)` in sẵn các con số dùng cho báo cáo:
chênh lệch `D(x) − D(G(z))`, độ dao động của hai đường loss, chuẩn gradient của G,
entropy so với `ln 10`, và FID cuối so với FID tốt nhất.

### Sau khi cGAN xong

```
Độ khớp nhãn: 96.4% ảnh sinh ra đúng chữ số được yêu cầu (đoán mò chỉ ~10%)
```

Bằng chứng định lượng cho yêu cầu 7.

---

## 7. Kết quả nằm ở đâu

```
runs/ae_mnist_seed42/              AE
runs/vae_mnist_beta1.0_seed42/     VAE
runs/cvae_mnist_beta1.0_seed42/    CVAE (mỗi beta một thư mục)
runs/gan_mnist_seed42/             DCGAN và cGAN
```

### AE — `runs/ae_mnist_seed42/`

| File | Nội dung |
|---|---|
| `recon_bce_latent_<L>.png` | ảnh tái tạo ở chiều ẩn `L` |
| `loss_comparison_by_latent_dim.png` | chất lượng tái tạo theo chiều ẩn |
| `latent_map_2d_bce.png` | bản đồ không gian ẩn 2 chiều |
| `interpolation_ae_bce.png` | nội suy giữa hai điểm |
| `anomaly_digit_<d>.png` | phát hiện bất thường, giữ lại chữ số `d` |
| `experiment_log.csv`, `anomaly_log.csv`, `loss_comparison_summary.csv` | số liệu |
| `ae_bce_latent_<L>.pt` | checkpoint |

### VAE — `runs/vae_mnist_beta1.0_seed42/`

Như trên, thêm:

| File | Nội dung |
|---|---|
| `samples_bce_latent_<L>.png` | **ảnh sinh mới từ prior `N(0, I)`** — chỉ VAE có |
| `latent_manifold_bce.png` | lưới giải mã trải trên không gian ẩn |
| `interpolation_vae_bce.png` | nội suy, đặt cạnh bản của AE |

### CVAE — `runs/cvae_mnist_beta<beta>_seed42/`

| File | Nội dung |
|---|---|
| `samples_by_class_bce_latent_<L>.png` | mỗi hàng một lớp được chỉ định |
| `label_swap_bce_latent_<L>.png` | giữ nguyên `z`, đổi nhãn |
| `latent_manifold_bce_class<c>.png` | manifold riêng cho từng lớp |
| `demo_A/B/C_*.png` | ba demo của `4_conditional_generation_demo.py` |

### Nội suy — cùng thư mục với checkpoint

| File | Nội dung |
|---|---|
| `interp_grid_ae_vs_vae.png` | AE và VAE đặt cạnh nhau |
| `interp_<a>_to_<b>.png` | từng cặp chữ số |
| `interp_step_deltas.png`, `interp_metrics.csv` | độ mượt từng bước |

### GAN — `runs/gan_mnist_seed42/`

| File | Nội dung | Yêu cầu |
|---|---|---|
| `dcgan_curves.png` | Hình 1 — loss D/G, `D(x)` vs `D(G(z))`, FID theo epoch | 5 |
| `dcgan_progress.png` | Hình 2 — cùng một `z`, ảnh rõ dần qua các epoch | 5 |
| `instability.png` | Hình 3 — biên độ nhảy loss, entropy | 5 |
| `mode_hist.png` | Hình 4 — phân bố 10 chữ số, thật vs sinh ra | 5 |
| `cgan_by_label.png` | Hình 5 — mỗi hàng một chữ số được chỉ định | 7 |
| `so_sanh_luoi_anh.png` | Hình 6 — lưới ảnh sinh mới AE / VAE / GAN | 6 |
| `khao_sat_latent.png` | Hình 6b — ảnh tái tạo theo chiều latent | 3 |
| `so_sanh_cot.png` | Hình 7 — biểu đồ cột FID và entropy | 6 |
| `bang_so_sanh.csv` | Bảng 1 — FID, entropy, số mode, pixel std | 6 |
| `lich_su_dcgan.csv`, `lich_su_cgan.csv` | số liệu từng epoch | 5 |
| `dcgan.pt`, `cgan.pt` | checkpoint G và D | — |
| `dcgan_ep<NNN>.png`, `cgan_ep<NNN>.png` | ảnh mẫu từng epoch | — |

---

## 8. Thư mục `shared/` và tính so sánh được của FID

FID so sánh **phân bố đặc trưng** của ảnh thật và ảnh sinh ra. Một con số FID chỉ
có nghĩa khi gắn với hai lựa chọn cố định:

1. **bộ trích đặc trưng** — mạng nào biến ảnh thành vector
2. **tập ảnh thật tham chiếu** — so với những ảnh thật nào

Nếu ba thành viên mỗi người tự huấn luyện một bộ trích đặc trưng và tự bốc 5000
ảnh thật khác nhau, ba con số đo bằng ba cái thước khác nhau, xếp chung một bảng
là vô nghĩa — **và không có lỗi nào được báo**, chương trình vẫn chạy trơn tru.

Nên chặng `prepare` cố định cả hai một lần rồi ghi ra:

| File | Nội dung | Kích thước |
|---|---|---|
| `shared/feature_cnn.pt` | bộ trích đặc trưng 128 chiều đã huấn luyện | ~0.3 MB |
| `shared/fid_ref.npz` | `mu`, `sigma` của 5000 ảnh thật tham chiếu | ~0.1 MB |
| `shared/anh_that_5000.npz` | chính 5000 ảnh thật đó | ~1.1 MB |

Ba file này **được commit** (`.gitignore` có ngoại lệ `!shared/feature_cnn.pt` vì
luật `*.pt` chặn mặc định). Ai muốn tái hiện đúng số trong báo cáo thì **bỏ qua
chặng `prepare`** và dùng file có sẵn. Chạy lại `prepare` sinh ra bộ đo mới, FID
vẫn ra nhưng không so được với báo cáo.

FID gốc dùng InceptionV3 huấn luyện trên ảnh màu ImageNet; trên ảnh xám 28×28 đặc
trưng của nó gần như vô nghĩa, nên ở đây thay bằng một CNN nhỏ huấn luyện trên
chính MNIST. Con số thu được là "MNIST-FID", chỉ so được với nhau trong đề tài
này, không so được với FID công bố ở các bài báo khác.

### Ảnh AE / VAE gửi sang

Chặng `compare` đọc mọi file `.npz` trong `--team-dir` (mặc định là `shared/`),
theo hai kiểu tên:

```
anh_sinh_<ten>.npz
<ae|vae>_<loai>_latent<L>.npz      ví dụ: vae_bce_reconstruction_latent32.npz
```

Khoá bắt buộc là `imgs` với shape `(N, 1, 28, 28)`; thêm `epochs` thì tốt, không có
thì script lấy theo `--epochs` và in dòng nhắc.

Script tự xử lý hai cái bẫy hay gặp:

- **Thang giá trị.** Decoder của AE/VAE kết thúc bằng sigmoid nên ảnh nằm trong
  `[0,1]`; Generator của GAN kết thúc bằng tanh nên ảnh nằm trong `[-1,1]`. Đưa
  nhầm thang không làm chương trình chết, chỉ làm FID sai lặng lẽ — nên script tự
  nhận thang và quy về `[-1,1]`, khai sai thì báo lỗi ngay.
- **Tái tạo và sinh mới.** Tên file chứa `recon` được xếp riêng sang Hình 6b. Ảnh
  tái tạo là đưa ảnh thật vào encoder rồi giải mã ra — FID của nó thấp gần như
  đương nhiên vì mô hình đã được xem trước đáp án. Chỉ ảnh sinh từ `z ~ N(0, I)`
  mới đặt cạnh GAN được.

---

## 9. Tham số dòng lệnh

### `1_`, `2_`, `3_` (AE / VAE / CVAE)

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--out-dir` | `runs` | thư mục cha chứa kết quả |
| `--data-dir` | `data` | nơi tải MNIST |
| `--dataset` | `MNIST` | hoặc `FashionMNIST` |
| `--epochs` | 10 | số epoch mỗi cấu hình |
| `--latent-dims` | `2 8 32 128` | các chiều ẩn cần quét |
| `--losses` | `bce mse` | hàm loss tái tạo cần so sánh |
| `--beta` | 1.0 | hệ số KL (chỉ `2_` và `3_`) |
| `--batch-size` | 128 | |
| `--lr` | 1e-3 | |
| `--seed` | 42 | |
| `--num-workers` | 2 | để 0–2 trên Windows |
| `--skip-anomaly` | tắt | bỏ phần phát hiện bất thường |
| `--anomaly-epochs` | 5 | số epoch mỗi mô hình bất thường |

### `5_gan.py`

`uv run python 5_gan.py --help` in ra đầy đủ.

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--out-dir` | `runs` | thư mục cha chứa kết quả |
| `--data-dir` | `data` | nơi tải MNIST |
| `--shared-dir` | `shared` | bộ đo FID và file `.npz` của nhóm |
| `--team-dir` | như `--shared-dir` | nơi để riêng `.npz` của AE/VAE |
| `--epochs` | 10 | phải bằng số epoch của AE và VAE |
| `--seed` | 42 | đổi seed để xem GAN bất ổn định thế nào |
| `--batch-size` | 128 | |
| `--latent-dim` | 100 | số chiều của `z` |
| `--lr` | 2e-4 | learning rate của cả G và D |
| `--fid-n` | 5000 | số ảnh tham chiếu, chỉ dùng ở chặng `prepare` |
| `--feat-epochs` | 2 | số epoch huấn luyện bộ trích đặc trưng |
| `--digits` | `0 … 9` | các chữ số vẽ trong lưới cGAN, mỗi số một hàng |

Mặc định khác của DCGAN lấy theo bài báo Radford, Metz & Chintala (2016): Adam
`beta1 = 0.5`, khởi tạo trọng số `N(0, 0.02)`, LeakyReLU(0.2) trong D, thay pooling
bằng conv stride 2, BatchNorm ở cả hai mạng.

---

## 10. Lỗi hay gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `uv run` tự đổi `torch 2.6.0+cu124` thành `2.x.x+cpu` | Kiểm tra `[tool.uv.sources]`: Windows phải có marker `sys_platform == 'win32'`; sau đó xoá `.venv`, `uv.lock`, chạy `uv lock` và `uv sync` |
| `torch.cuda.is_available()` ra `False` và `torch.version.cuda` là `None` | Đang dùng PyTorch CPU-only — xem mục 4 |
| Torch có `+cu124` nhưng CUDA vẫn `False` | Chạy `nvidia-smi`; kiểm tra/cập nhật driver NVIDIA |
| `torch.cuda.get_device_name(0)` báo `Torch not compiled with CUDA enabled` | PyTorch hiện tại là CPU-only; không gọi `get_device_name(0)` trước khi kiểm tra `torch.cuda.is_available()` |
| `uv : The term 'uv' is not recognized` | Đóng mở lại PowerShell, hoặc gọi `$HOME\.local\bin\uv.exe` |
| `running scripts is disabled on this system` | Dùng lệnh cài `uv` với `-ExecutionPolicy ByPass` như mục 3 |
| `uv lock` / `uv sync` báo không tìm được phiên bản phù hợp | Kiểm tra version `torch`, `torchvision` và CUDA index trong `pyproject.toml`; với README này dùng `torch==2.6.0`, `torchvision==0.21.0`, `cu124` |
| Treo lâu ở đầu mỗi epoch, CPU 100% | `--num-workers` quá cao trên Windows, hạ xuống 2 hoặc 0 |
| `FileNotFoundError: shared/feature_cnn.pt` | Thiếu `shared/` khi copy sang. Lấy từ repo, hoặc chạy `5_gan.py prepare` |
| `runs/…/dcgan.pt not found` | Chạy `compare` trước `dcgan`/`cgan`, hoặc `--out-dir`/`--seed` khác lúc train |
| `No checkpoint cvae_..._latent_32.pt` | Chạy `4_…demo.py` mà chưa chạy mục 5.3 trước |
| `SAI THANG: khai '[0,1]' nhưng min < 0` | File `.npz` không đúng thang đã khai. Kiểm tra `imgs.min()`, `imgs.max()` |
| `Found 0 team file(s)` | `.npz` của AE/VAE chưa có trong `--team-dir`, bảng chỉ còn DCGAN và cGAN |
| MNIST tải về lỗi | Đặt `mnist_train.csv` vào `data/`, script tự nhận |
| `CUDA out of memory` | Với RTX 3060 12 GB thường không xảy ra ở cấu hình mặc định. Nếu vẫn gặp, thử `--batch-size 64` |
| Ảnh GAN sinh ra toàn nhiễu sau vài epoch | GAN sập — đúng hiện tượng cần ghi nhận cho yêu cầu 5. Xem `instability.png` rồi đổi `--seed` hoặc giảm `--lr` |

---


## 11. Tài liệu tham khảo

- Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Higgins et al., *beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*, ICLR 2017.
- Sohn, Lee & Yan, *Learning Structured Output Representation using Deep Conditional Generative Models* (CVAE), NeurIPS 2015.
- Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- Radford, Metz & Chintala, *Unsupervised Representation Learning with Deep Convolutional GANs*, ICLR 2016. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
- Mirza & Osindero, *Conditional Generative Adversarial Nets*, 2014. [arXiv:1411.1784](https://arxiv.org/abs/1411.1784)
- Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* (FID), NeurIPS 2017. [arXiv:1706.08500](https://arxiv.org/abs/1706.08500)
