# IT4653 — Học sâu và ứng dụng

## Đề tài 9. Mô hình sinh: từ Autoencoder tới VAE và GAN

So sánh hai họ mô hình sinh trên cùng dữ liệu MNIST và cùng ngân sách tính toán,
làm rõ đánh đổi giữa độ nét ảnh và tính ổn định khi huấn luyện.

| Thành viên | Phần phụ trách | File |
|---|---|---|
| Nguyễn Hà Minh | Autoencoder | `1_autoencoder.py` |
| Đỗ Thành Đạt | VAE, Conditional VAE | `2_…`, `3_…`, `4_…`, `poc_interpolation.py` |
| Nguyễn Trường Chinh | DCGAN, Conditional GAN, FID | `5_gan.py` |

---

## 1. Cài đặt

Repo dùng [uv](https://docs.astral.sh/uv/). Cài uv rồi đồng bộ môi trường:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

cd it4653-main
uv sync
```

`uv sync` đọc `pyproject.toml` + `uv.lock` và dựng đúng phiên bản thư viện. Sau đó
mọi lệnh chạy qua `uv run python <file>`.

**GPU.** `pyproject.toml` đang trỏ wheel PyTorch về index CUDA 12.4 cho Linux.
Nếu máy dùng driver khác, sửa `cu124` trong mục `[[tool.uv.index]]` cho khớp
(`nvidia-smi` để xem). Trên macOS hoặc máy chỉ có CPU thì xoá cả khối
`[[tool.uv.index]]` và `[tool.uv.sources]`, wheel mặc định của PyPI là đúng.

**Không dùng uv?** Cũng chạy được bằng pip:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision numpy scipy scikit-learn matplotlib pandas tqdm tensorboard
python 5_gan.py all                # bỏ tiền tố "uv run"
```

**Dữ liệu.** MNIST tự tải về `data/` ở lần chạy đầu (~10 MB), không cần chuẩn bị gì.

---

## 2. Chạy nhanh

Toàn bộ lệnh của cả nhóm nằm trong `scripts/command.sh`. Riêng phần GAN, chạy
lần lượt bốn lệnh dưới đây từ thư mục gốc của repo:

Cả phần GAN nằm trong **một file duy nhất** `5_gan.py`, chọn việc cần làm bằng
tham số đầu tiên:

```bash
uv run python 5_gan.py all --out-dir runs --epochs 10
```

Hoặc chạy từng chặng để dừng lại xem kết quả:

```bash
uv run python 5_gan.py prepare                        # chỉ chạy nếu shared/ còn trống
uv run python 5_gan.py dcgan   --out-dir runs --epochs 10
uv run python 5_gan.py cgan    --out-dir runs --epochs 10
uv run python 5_gan.py compare --out-dir runs
```

| Chặng | Làm gì | Yêu cầu |
|---|---|---|
| `prepare` | dựng bộ đo FID dùng chung, ghi ra `shared/` | — |
| `dcgan` | huấn luyện DCGAN, đường loss, phân tích bất ổn định, mode collapse | 1, 4 |
| `cgan` | huấn luyện cGAN, sinh ảnh theo nhãn, đo độ khớp nhãn | 3 |
| `compare` | bảng FID + lưới ảnh AE / VAE / GAN | 2 |
| `all` | chạy cả bốn, tự bỏ qua `prepare` nếu `shared/` đã có sẵn | |

Thời gian tham khảo trên GPU T4: `prepare` khoảng 1 phút, `dcgan` và `cgan` mỗi
chặng 6–8 phút với 10 epoch, `compare` dưới 1 phút. Trên CPU chậm hơn khoảng 10 lần.

Kết quả ghi vào `runs/gan_mnist_seed42/`.

---

## 3. Cấu trúc thư mục

```
1_autoencoder.py                        AE — tái tạo, khảo sát latent, phát hiện bất thường
2_variational_autoencoder.py            VAE / beta-VAE
3_conditional_variational_autoencoder.py  CVAE
4_conditional_generation_demo.py        demo sinh ảnh theo nhãn từ checkpoint CVAE
poc_interpolation.py                    nội suy không gian ẩn AE vs VAE

5_gan.py                                toàn bộ phần GAN: DCGAN, cGAN, FID, so sánh
                                        (prepare / dcgan / cgan / compare / all)

shared/        file nhị phân dùng chung, ĐƯỢC commit (xem mục 4)
runs/          kết quả mỗi lần chạy, KHÔNG commit
data/          MNIST tải về, KHÔNG commit
scripts/command.sh   toàn bộ lệnh tái hiện thí nghiệm, theo thứ tự
```

---

## 4. Vì sao có thư mục `shared/`

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
| `shared/fid_ref.npz` | `mu`, `sigma` và chính 5000 ảnh thật tham chiếu | ~1.2 MB |

Hai file này **được commit** (`.gitignore` có ngoại lệ `!shared/feature_cnn.pt`).
Ai muốn tái hiện đúng số trong báo cáo thì **bỏ qua chặng `prepare`** và dùng file
có sẵn. Chạy lại `prepare` sẽ sinh ra bộ đo mới, FID vẫn ra nhưng không so được
với báo cáo.

FID gốc dùng InceptionV3 huấn luyện trên ảnh màu ImageNet; trên ảnh xám 28×28 đặc
trưng của nó gần như vô nghĩa, nên ở đây thay bằng một CNN nhỏ huấn luyện trên
chính MNIST. Con số thu được là "MNIST-FID", chỉ so được với nhau trong đề tài
này, không so được với FID công bố ở các bài báo khác.

### Ảnh AE / VAE gửi sang

Chặng `compare` đọc mọi file `.npz` trong `--team-dir` (mặc định là
`shared/`), theo hai kiểu tên:

```
anh_sinh_<ten>.npz
<ae|vae>_<loai>_latent<L>.npz      ví dụ: vae_bce_reconstruction_latent32.npz
```

Khoá bắt buộc là `imgs` với shape `(N, 1, 28, 28)`; thêm được `epochs` thì tốt.
Script tự xử lý hai cái bẫy hay gặp:

- **Thang giá trị.** Decoder của AE/VAE kết thúc bằng sigmoid nên ảnh nằm trong
  `[0,1]`; Generator của GAN kết thúc bằng tanh nên ảnh nằm trong `[-1,1]`. Đưa
  nhầm thang không làm chương trình chết, chỉ làm FID sai lặng lẽ — nên script tự
  nhận thang và quy về `[-1,1]`.
- **Tái tạo và sinh mới.** Tên file chứa `recon` được xếp riêng. Ảnh tái tạo là
  đưa ảnh thật vào encoder rồi giải mã ra, FID của nó thấp gần như đương nhiên vì
  mô hình đã được xem trước đáp án; chỉ ảnh sinh từ `z ~ N(0, I)` mới đặt cạnh GAN
  được. Ảnh tái tạo dùng cho Hình 6b (khảo sát chiều latent).

---

## 5. Kết quả sinh ra

Trong `runs/gan_mnist_seed42/`:

| File | Nội dung | Yêu cầu |
|---|---|---|
| `dcgan_curves.png` | Hình 1 — loss D/G, `D(x)` vs `D(G(z))`, FID theo epoch | 1 |
| `dcgan_progress.png` | Hình 2 — cùng một `z` cố định, ảnh rõ dần qua các epoch | 1 |
| `instability.png` | Hình 3 — biên độ nhảy loss giữa hai epoch, entropy lớp | 4 |
| `mode_hist.png` | Hình 4 — phân bố 10 chữ số, thật vs sinh ra | 1 |
| `cgan_by_label.png` | Hình 5 — mỗi hàng một chữ số được chỉ định | 3 |
| `so_sanh_luoi_anh.png` | Hình 6 — lưới ảnh sinh mới AE / VAE / GAN cùng số epoch | 2 |
| `khao_sat_latent.png` | Hình 6b — ảnh tái tạo theo chiều latent | — |
| `so_sanh_cot.png` | Hình 7 — biểu đồ cột FID và entropy | 2 |
| `bang_so_sanh.csv` | Bảng 1 — FID, entropy, số mode, pixel std | 2 |
| `lich_su_dcgan.csv`, `lich_su_cgan.csv` | số liệu từng epoch | 1, 4 |
| `dcgan.pt`, `cgan.pt` | checkpoint G và D | — |

Ngoài các hình, chặng `dcgan` in ra phần **phân tích bất ổn định** (yêu cầu 4):
chênh lệch `D(x) − D(G(z))` (lớn hơn 0.8 là D thắng áp đảo), độ dao động của hai
đường loss, chuẩn gradient của G, entropy lớp so với `ln 10 = 2.303`, và so sánh
FID cuối với FID tốt nhất. Chặng `cgan` in ra **độ khớp nhãn** — tỉ lệ
ảnh sinh ra được bộ phân loại đọc đúng chữ số đã yêu cầu (đoán mò chỉ ~10%).

---

## 6. Tham số dòng lệnh

`uv run python 5_gan.py --help` in ra đầy đủ. Các cờ hay dùng:

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--out-dir` | `runs` | thư mục cha chứa kết quả |
| `--data-dir` | `data` | nơi tải MNIST về |
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

Mặc định khác của DCGAN, lấy theo bài báo Radford, Metz & Chintala (2016):
Adam `beta1 = 0.5`, khởi tạo trọng số `N(0, 0.02)`, LeakyReLU(0.2) trong D.

> `--epochs` của GAN phải bằng `--epochs` của AE và VAE. Yêu cầu 2 nói rõ là so
> sánh **ở cùng số epoch**; chặng `compare` sẽ in cảnh báo nếu phát hiện lệch.

---

## 7. Lỗi hay gặp

| Hiện tượng | Nguyên nhân và cách xử lý |
|---|---|
| `FileNotFoundError: shared/feature_cnn.pt` | Chưa có bộ đo dùng chung. Lấy `shared/` từ repo, hoặc chạy `5_gan.py prepare`. |
| `runs/…/dcgan.pt not found` | Chạy `compare` trước `dcgan`/`cgan`, hoặc dùng `--out-dir`/`--seed` khác lúc train. |
| `SAI THANG: khai '[0,1]' nhưng min < 0` | File `.npz` không đúng thang đã khai. Kiểm tra `imgs.min()`, `imgs.max()`. |
| `Found 0 team file(s)` | `.npz` của AE/VAE chưa có trong `--team-dir`, bảng chỉ còn DCGAN và cGAN. |
| MNIST tải về lỗi | Đặt file `mnist_train.csv` vào `data/`, script tự nhận. |
| Ảnh sinh ra toàn nhiễu sau vài epoch | GAN sập — đúng hiện tượng cần ghi nhận cho yêu cầu 4. Xem `instability.png` rồi đổi `--seed` hoặc giảm `--lr` để chạy lại. |

---

## 8. Yêu cầu của đề bài

| Mức độ | GAN có thể huấn luyện thất bại; hãy làm AE/VAE trước để chắc chắn có kết quả |
|---|---|
| Tài nguyên tính toán | Cần GPU. MNIST: ~15 phút/mô hình. CelebA 64×64: ~1 giờ cho DCGAN trên T4. |

**Dữ liệu.** MNIST hoặc Fashion-MNIST cho phần cơ sở; CelebA cắt 64×64 (lấy 20–30k
ảnh) cho phần nâng cao.

### Yêu cầu bắt buộc

- Nắm rõ được các hyperparameter trong quá trình xây dựng mô hình, cần thử nghiệm sự ảnh hưởng, tác dụng của mỗi tham số đóng góp trong quá trình xây dựng mô hình
- Cài Autoencoder thường và Variational Autoencoder: viết rõ hàm mất mát gồm reconstruction + KL, giải thích và cài đặt reparameterization trick (nêu rõ vì sao không lan truyền ngược qua phép lấy mẫu được).
- Khảo sát số chiều không gian ẩn (2, 8, 32, 128) tới chất lượng tái tạo; với chiều ẩn = 2, vẽ bản đồ không gian 2D.
- Nội suy tuyến tính giữa hai điểm trong không gian ẩn của AE và của VAE, đặt cạnh nhau để cho thấy VAE cho không gian ẩn liên tục hơn.
- Cài DCGAN: mô tả generator/discriminator, ghi lại đường loss của cả hai và bình luận về tính bất ổn định; ghi nhận mode collapse nếu xảy ra.
- So sánh AE / VAE / GAN bằng cả chỉ số định lượng (FID hoặc Inception Score) và lưới ảnh sinh ra ở cùng số epoch.

### Yêu cầu khác

- Cài Conditional VAE hoặc Conditional GAN để sinh ảnh theo nhãn cho trước.
- Ứng dụng phát hiện bất thường bằng sai số tái tạo của autoencoder (huấn luyện trên 9 chữ số, kiểm thử trên chữ số còn lại).

### Sản phẩm

Mã 3 mô hình + lưới ảnh sinh theo epoch + bảng FID + hình nội suy không gian ẩn + phân tích sự bất ổn định của GAN.

---

## Tài liệu tham khảo

- Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- Radford, Metz & Chintala, *Unsupervised Representation Learning with Deep Convolutional GANs*, ICLR 2016. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
- Mirza & Osindero, *Conditional Generative Adversarial Nets*, 2014. [arXiv:1411.1784](https://arxiv.org/abs/1411.1784)
- Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* (FID), NeurIPS 2017. [arXiv:1706.08500](https://arxiv.org/abs/1706.08500)
- Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
