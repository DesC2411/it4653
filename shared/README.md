# `shared/` — file nhị phân dùng chung của nhóm

Thư mục này giữ mọi thứ phải **giống hệt nhau** giữa ba thành viên. Thiếu chúng thì
`5_gan.py compare` vẫn chạy nhưng ba con số FID không so được với nhau.

| File | Ai tạo | Bắt buộc |
|---|---|---|
| `feature_cnn.pt` | `5_gan.py prepare` | có — bộ trích đặc trưng 128 chiều |
| `fid_ref.npz` | `5_gan.py prepare` | có — `mu`, `sigma` và 5000 ảnh thật tham chiếu |
| `anh_that_5000.npz` | `5_gan.py prepare` (lần đầu) | không — chỉ để chia lại tập ảnh thật |
| `*_latent*.npz`, `anh_sinh_*.npz` | thành viên làm AE và VAE | có, nếu muốn bảng so sánh đủ ba mô hình |

Định dạng file ảnh gửi sang: khoá `imgs`, shape `(N, 1, 28, 28)`, thêm `epochs` nếu có.
Thang `[0,1]` hay `[-1,1]` đều được, script tự nhận. Tên file chứa `recon` sẽ được
hiểu là **ảnh tái tạo** và tách sang hình riêng, không xếp cạnh GAN.

Chi tiết xem mục 4 của `README.md` ở thư mục gốc.
