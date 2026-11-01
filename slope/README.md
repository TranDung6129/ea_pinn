# slope/: mái dốc đàn dẻo, biến dạng phẳng

Thư mục độc lập với `src/` và `config.py` của bài 2D, không import chéo.

## Hai môi trường

Oracle chạy trong `fenicsx-env` (dolfinx 0.9, gmsh, không có torch). PINN chạy
trong `pinn_venv` và chỉ đọc file `.npz` mà oracle ghi ra. Chạy mọi lệnh từ thư
mục gốc của repo.

## Thứ tự chạy lần đầu

```bash
conda activate fenicsx-env          # nhớ deactivate pinn_venv trước

python -m slope.constitutive                     # ~1 s, cần numpy
python -m slope.oracle --check-constitutive      # vài giây
python -m slope.oracle --check-steps             # ~1 phút
python -m slope.oracle --check-fos               # vài phút (~8 lần giải)
python -m slope.oracle --plot                    # nghiệm tham chiếu cho PINN
```

Ngưỡng đạt của từng phép tự kiểm được chốt sẵn trong code:

| phép kiểm | kiểm cái gì | ngưỡng |
|---|---|---|
| `constitutive` | ứng suất sau chiếu nằm trên hoặc trong mặt chảy, điểm đàn hồi giữ nguyên | 1e-8·k |
| `--check-constitutive` | bản UFL trong dolfinx khớp bản numpy | 1e-6·k |
| `--check-steps` | Hencky không có lịch sử: 4 bước tải và 10 bước tải ra cùng u | 1e-6 tương đối |
| `--check-fos` | hệ số an toàn so với Griffiths & Lane (1999) ví dụ 1 | lệch dưới 10% |

Không đạt phép nào thì dừng, gửi output, chưa đi tiếp sang PINN.

## Mô hình

Hencky (biến dạng toàn phần) + Drucker-Prager kết hợp khớp Mohr-Coulomb biến
dạng phẳng. Công thức và lý do nằm ở đầu `constitutive.py`. Giới hạn cần ghi
trong luận văn: là xấp xỉ của dẻo tăng dần (chính xác khi tải tỉ lệ), không mô
tả dỡ tải, quy luật chảy kết hợp đánh giá quá độ giãn nở.

## Lưu ý về E

Trọng lực bật từ trạng thái không ứng suất nên u gồm cả độ lún đàn hồi do
trọng lượng bản thân. Không ảnh hưởng cổng 1 (PINN và FEM giải cùng bài), nhưng
phải định nghĩa lại E theo số gia chuyển vị trước cổng 2. Xem đầu `oracle.py`.

## PINN (cổng 1), chạy trong pinn_venv

```bash
python -m slope.energy                  # tự kiểm: torch khớp numpy, dW/deps = sig
python -m slope.pinn --ref results_slope/gate1_srf13_fine.npz --srf 1.3
```

Huấn luyện bằng cực tiểu hoá năng lượng (xem đầu `energy.py`), không dùng dữ liệu
FEM. Nghiệm FEM chỉ để chấm điểm ở cuối, in thẳng ĐẠT / KHÔNG ĐẠT theo ba ngưỡng
cổng 1 trong `compare.py`.
