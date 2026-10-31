# FMD-PINN: trạng thái ngày 26/09/2026

Tài liệu bàn giao giữa các cuộc hội thoại. Mọi con số dưới đây là số đã chạy thật, 5 seed (42 đến 46) trừ khi ghi khác.

## 1. Mục tiêu và định vị đóng góp

Bài toán: tìm tập tham số CÓ THỂ dẫn tới phá hủy, tức tập mức không E(p) = 0, bằng ít lời gọi oracle. Đây là bài toán tìm nghiệm, không phải cực đại hóa E.

Đích cuối: cảnh báo sớm sạt lở từ quan trắc đa cảm biến. Ngoài hiện trường p được ước lượng từ cảm biến và luôn có sai số, nên cần một phán quyết an toàn tức thì kèm độ nhạy ∂E/∂p để truyền sai số cảm biến.

Đóng góp so với baseline (BO tìm biên, cũng như họ AK-MCS trong độ tin cậy kết cấu, Echard, Gayton, Lemaire, Structural Safety 2011): không nằm ở "ít lời gọi" mà ở chỗ mô hình thay thế có vật lý: dùng được realtime, giải thích được bằng vật lý, cho gradient đúng, và hướng tới bài toán đa cảm biến. Nhắm hội nghị trước, Q1 sau.

Lưu ý thuật ngữ: oracle 2D là bộ giải sai phân hữu hạn theo phương pháp đường (method of lines, lưới 32×32, RK45), KHÔNG phải FEM. Chỉ bài 3D mới dùng FEM.

## 2. Kết quả bài 2D đã chốt (MỐC CHÍNH)

PDE: ∂u/∂t = D∇²u + αu − βu³ trên [0,1]², biên Neumann, u0 = 0.1 sin(πx) sin(πy), E = max u − 1.5, p = (α, β, D).

Bộ mốc đóng băng: tag `cap`. GT lưới 25³, lưới đo 24³, độ phân giải tham chiếu 0.038 (không phải sàn cứng: p95 có thể xuống dưới một chút).

| phương pháp | δ_H p95 | chamfer | FSR biên | báo động giả | CR |
|---|---|---|---|---|---|
| BO tìm biên (chạy lại toàn miền 26/09) | 0.0345 ± 0.0008 | 0.0238 | 0.5% | 1.9% | 99.8% |
| B3 (levelset + bisection) + hiệu chỉnh hằng số | 0.0388 ± 0.0012 | 0.0244 | 2.7% | 4.6% | 99.3% |
| B3 không hiệu chỉnh | 0.0593 ± 0.0200 | 0.0276 | 8.4% | 8.3% | 97.6% |
| B2 (levelset) không hiệu chỉnh | 0.0532 ± 0.0089 | 0.0259 | 1.8% | 11.8% | 99.5% |
| B1 (ngẫu nhiên đều sau 10 điểm LHS) | 0.0641 ± 0.0251 | 0.0286 | 13.8% | 3.9% | 96.2% |

Hiệu chỉnh hằng số: E_final = E_pinn + b, b = trung bình (E_true − E_pinn) trên các lời gọi |E_true| < 0.3 của CHÍNH 200 lời gọi của phương pháp. Không dùng ground truth. Không đổi ∇E.

Đọc: về hình học gần như bằng BO; về FSR kém BO khoảng 5 lần (2.7% so với 0.5%). Độ bất ổn giữa các seed chủ yếu là độ lệch mức (std B3 0.0200 → 0.0012 sau hiệu chỉnh). BO chạm độ phân giải sau khoảng 30 lời gọi.

Không báo cáo bản logD2 (0.0362): dạng hàm chữ U theo D được chọn sau khi nhìn ground truth, là rò rỉ đáp án. GP trên phần dư dùng cấu hình của BO bị suy biến (độ dài tương quan chạm 1e-5, không có số hạng nhiễu), chưa sửa.

## 3. Gradient và realtime

B3 `cap`, 5 seed × 50 điểm trên biên thật (từ `export_figdata.py`), autodiff so với sai phân trung tâm trên oracle, tọa độ chuẩn hóa:
- cos trung vị 0.9968 ± 0.0012; sai số |∇E| 11.4 ± 4.6%; ∂E/∂α 10.4 ± 4.4%; ∂E/∂β 14.1 ± 5.2%; đúng dấu 100%; σ_E (sai số cảm biến 5%) lệch 10.3 ± 3.9%
- PHÁT HIỆN MỚI: ở β < 1.5 (vùng biên dốc nhất, |∇E| trung vị 10.2 so với 1.75) hướng vẫn đúng (cos 0.996) nhưng biên độ bị đánh giá THẤP khoảng 32% (tỉ số 0.68, p10 0.45). Nghĩa là σ_E bị đánh giá thấp đúng chỗ nhạy nhất: điểm yếu an toàn phải ghi.
- ∂E/∂D không đo được: nằm dưới nhiễu sai phân của oracle.

Độ trễ (B2 seed 42, cùng kiến trúc): E + ∇E cho 1 cấu hình 1.73 ms; 0.21 ms/cấu hình khi 1024 cấu hình; đỉnh VRAM 956 MiB; trọng số 0.22 MiB.

ĐÍNH CHÍNH chi phí oracle: 20.9 ms chỉ là tại p = (8, 1, 0.01), cấu hình nhanh. Trên 40 cấu hình ngẫu nhiên trong miền: trung vị 89 ms, dải 21 đến 255 ms (D lớn làm hệ cứng). Nên tỉ số tăng tốc là khoảng 12 lần ở cấu hình thuận lợi nhất và khoảng 50 lần ở trung vị; các con số 13 lần / 73 lần trước đây là thiên thấp. Comment "0.119–0.155 s/lần gọi" trong config.py là số từ trước khi chuyển sang ma trận thưa.

## 4. Phát hiện vật lý: Định lý 1 (đã chứng minh trong bài báo)

Với biên Neumann, tích phân phương trình trên miền làm số hạng khuếch tán triệt tiêu: d⟨u⟩/dt = α⟨u⟩ − β⟨u³⟩, không có D. Chứng minh (nguyên lý so sánh, chặn trên bằng ODE, chặn dưới bằng giá trị trung bình, tính đơn điệu chặt theo α và β): với mọi (β, D) trong miền, E = 0 có đúng một nghiệm α*(β, D), liên tục, tăng chặt theo β, và ln 15 < α* ≤ 3.611 + 2.25β; chặn không phụ thuộc D.

Kiểm trên lưới GT 25³: 625/625 cột có đúng một lần đổi dấu, 0 vi phạm chặn, α* không giảm theo β ở 25/25 lát D. Tính đơn điệu chặt theo α chỉ vỡ ở 75 cột, tất cả tại E ≈ 2.24, do oracle dừng tích phân khi max u = 3.75.

α* tại β = 0.5 cho D = 0.0059 / 0.0387 / 0.2529: 3.05 / 3.53 / 3.78 (số đọc bằng mắt trên hình trước đây là 3.5 / 3.7 / 4.2, đã thay). Công thức cân bằng α = 2π²D + 2.25β cho 1.24 / 1.89 / 6.12: SAI ở β thấp, không được trích. Độ nghiêng dα*/dD: +2.36 ở β < 1.5, +0.010 ở β ≥ 1.5. Bài 2D thực chất là bài hai tham số, giải thích vì sao BO mạnh.

## 5. Lần chạy `proj` (26/09): chiếu Newton về E_pinn = 0. Kết quả: hỗn hợp

Mục đích: tăng mật độ lời gọi sát biên. Cách làm: sau khi chọn nút lưới quét 16³, chiếu về E_pinn = 0 bằng 3 bước Newton dọc gradient autodiff (`_project_to_levelset`; `PROJECT_ITERS = 0` là tắt, quay về hành vi `cap`).

Mật độ: |E_true| < 0.1 tăng 27% → 61% (B3), 30% → 67% (B2); dự đoán trước khi chạy là 52 đến 85% theo seed trên cả lần chạy. Cái giá: khoảng cách láng giềng gần nhất giảm 31% (0.105 → 0.072), lời gọi ở β < 1.5 giảm 14% → 12%.

| | δ_H p95 | FSR biên | báo động giả |
|---|---|---|---|
| B3 `proj` + hằng số | 0.0432 ± 0.0096 | 2.5% | 4.4% |
| B3 `proj` + GP phần dư | 0.0416 ± 0.0077 | 1.7% | 7.0% |
| B2 `proj` + hằng số | 0.0631 ± 0.0195 | 6.1% | 8.2% |

ĐÍNH CHÍNH cách đọc theo seed: so với B3 `cap` CHƯA hiệu chỉnh cùng seed, chiếu cải thiện 4/5 seed (0.0544 → 0.0370, 0.0401 → 0.0388, 0.0931 → 0.0409, 0.0688 → 0.0582; trung bình 0.0593 → 0.0513) và làm tệ hẳn 1 seed (45: 0.0401 → 0.0817). Nhưng SAU hiệu chỉnh hằng số thì trung bình tệ hơn (0.0388 → 0.0432): không chiếu thì lỗi chủ yếu là lệch mức, hằng số gỡ được; có chiếu thì seed 45 và 46 còn lỗi hình dạng, hằng số không gỡ được. Hình seed 45: đường B3 gần như thẳng, không bắt được đoạn biên dựng đứng ở β thấp.

Bài học cho 3D: cân bằng khai thác (sát biên) và thăm dò (độ phủ) phải được thiết kế. Mốc 2D vẫn là `cap` + hiệu chỉnh hằng số.

### Bất đồng giữa các seed có thay con mắt ở 7D được không

Trên 5 checkpoint B3 `proj`, 760 điểm sai trong dải biên, 27% rơi vào 10% vùng bất đồng cao nhất (ngẫu nhiên 10%; ngưỡng chốt trước 40%). Theo seed: 16 / 14 / 38 / 32 / 36%. Tín hiệu thật nhưng yếu: chỉ bắt được chỗ các mô hình sai khác nhau. Ở 7D chỉ làm cờ phụ; định vị chỗ sai dựa vào tập kiểm tra oracle độc lập, tóm tắt bằng luật đọc được.

## 6. Code và file

Mới trong phiên này: `measure_grad.py`, `measure_dslope.py`, `eval_corrected.py`, `export_figdata.py`, và bản sửa `_level_offset` + `_project_to_levelset` trong `src/fmd_pinn.py` cùng `LEVEL_CORRECT`, `LEVEL_BAND`, `PROJECT_ITERS`, `PROJECT_MAX_STEP` trong `config.py` (đặt `PROJECT_ITERS = 0` cho mốc `cap`). `results/figdata.npz` chứa mọi dữ liệu vẽ hình.

Lưu ý: chạy `run_03_bayesian_opt.py` với `ORACLE_REGION` (holdout) sẽ ghi đè `bo_boundary_seed*.json` toàn miền. `plot_compare.py` ghi đè cùng một tên ảnh cho mọi seed.

## 7. Bài báo (26/09)

Bản IEEE conference 9 trang, tiếng Anh: `main.tex`, `IEEEtran.cls`, `fig/` (5 hình PDF + `schematic.tex` TikZ), `figs.py`, `analysis.py`, sơ đồ `fmd_pinn_method.drawio`. Gồm: Mệnh đề 1 (tồn tại và phân tách, đa tạp C¹ khi 0 là giá trị chính quy), Mệnh đề 2 (biểu diễn đồ thị khi đơn điệu), Định lý 1 cho bài 2D (đã chứng minh), Mệnh đề 3 cho mái dốc 3D dưới giả thiết H1 (FS liên tục), H2 (có cấu hình ổn định và sập; mọi cấu hình ổn định tĩnh sẽ sập ở k_h lớn nhất), H3 (FS giảm chặt theo k_h). E cho 3D định nghĩa qua hệ số an toàn giảm cường độ: E = 1 − FS, ∇E = ∇_pΨ / ∂_λΨ. Đã có một agent độc lập soát chứng minh và đối chiếu số liệu; các chỗ nó chỉ ra đã sửa.

## 8. Bài 3D: kế hoạch theo cổng

Mái dốc đàn dẻo 3D, p = (θ, H, c', φ', γ, r_u, k_h) ∈ ℝ⁷ (hình học, vật liệu, thủy văn, địa chấn), tiêu chí Matsuoka-Nakai (trơn), FEM 2 đến 8 giờ mỗi lần: 200 lời gọi là 400 đến 1600 giờ CPU, thêm 100 đến 400 giờ cho tập kiểm tra 50 cấu hình.

Giữ lại: vòng lặp chủ động, bisection, hiệu chỉnh mức, BO, đo FSR/báo động giả, cách kiểm gradient.
Phải đổi: lưới quét (16⁷ không khả thi; cách chọn ứng viên thay thế CHƯA có lời giải, phải có số hạng thăm dò); ground truth và δ_H (thay bằng tập kiểm tra oracle độc lập); kiểm gradient bằng sai phân (14 lời gọi mỗi điểm).
Viết mới: oracle FEM 3D, mạng cho trường chuyển vị 3 thành phần, phần dư cân bằng + đàn dẻo, collocation trên miền mái dốc, E qua hệ số an toàn.

Cổng: (1) PINN một bộ tham số cố định so với một lời giải FEM; (2) 2 đến 3 tham số, kiểm E, gradient, và H1, H3; (3) vòng lặp 7 chiều so với baseline kiểu AK-MCS. Rủi ro lớn nhất: cổng 1.

## 9. Quy tắc làm việc

- Trước mỗi lần chạy dài: đưa một phép kiểm tra vài giây trên dữ liệu đã có kèm ngưỡng quyết định chốt trước. Không kiểm tra được thì nói rõ.
- Sửa code: chỉ gửi đoạn sửa kèm vị trí trong từng file.
- Không đặt tên cho một hiện tượng trước khi đo nó (bài học "knee").
- Sửa phép đo trước, thiết kế thí nghiệm sau, mô hình là cuối cùng.
- Báo cáo trung bình theo ngưỡng chốt trước, nhưng luôn xem thêm từng seed và hình.
