"""
slope/config_slope.py — cấu hình bài mái dốc đàn dẻo, biến dạng phẳng.

Tách hẳn khỏi config.py của bài 2D: không import, không sửa gì ở đó.

Đơn vị: m, kN, kPa (kN/m^2), kN/m^3.

Bộ tham số cố định cho cổng 1 lấy theo ví dụ 1 của Griffiths & Lane (1999,
Géotechnique 49(3)): mái dốc đồng nhất 2:1, phi' = 20 độ, c'/(gamma*H) = 0.05.
Lý do chọn: bài báo đó có hệ số an toàn tham chiếu, nên oracle có một phép tự
kiểm độc lập với PINN (xem check_fos trong oracle.py). Hình học miền dưới đây
KHÔNG trùng hoàn toàn lưới của Griffiths & Lane, nên chỉ so trong một dung sai.
"""

# ── Hình học (mặt dốc đi xuống về bên phải) ─────────────────────────────────
H_SLOPE   = 10.0     # chiều cao mái dốc
SLOPE_RUN = 20.0     # hình chiếu ngang của mặt dốc (2:1 -> 26.57 độ)
H_FOUND   = 2.0      # chiều dày lớp nền dưới chân dốc
L_CREST   = 10.0     # bề rộng phần đỉnh phía sau mép đỉnh
L_TOE     = 10.0     # bề rộng phần nền phía trước chân dốc
MESH_H    = 0.5      # kích thước phần tử mục tiêu (gmsh)

# ── Vật liệu (bộ cố định cho cổng 1) ────────────────────────────────────────
GAMMA   = 20.0       # trọng lượng riêng
YOUNG   = 1.0e5      # mô đun đàn hồi. Không ảnh hưởng hệ số an toàn,
                     # nhưng quyết định độ lớn chuyển vị, tức quyết định E.
POISSON = 0.4        # K0 = nu/(1-nu) = 0.667 ≈ 1 - sin(phi) (Jaky). Với 0.3, K0 = 0.43
                     # thấp hơn trạng thái chủ động, nền phẳng tự chảy dẻo dưới ~3.8 m
                     # khi bật trọng lực, vùng dẻo không còn phản ánh cơ cấu trượt.
COHESION = 10.0      # c' = 0.05 * GAMMA * H_SLOPE
PHI_DEG  = 20.0      # góc ma sát. Quy luật chảy KẾT HỢP (góc giãn nở = phi),
                     # để ánh xạ ứng suất theo biến dạng là duy nhất.

# ── Tiêu chí cảnh báo: E = |u(mép đỉnh)| / U_ALARM - 1 ──────────────────────
# Giá trị tạm. Cổng 1 chỉ so trường chuyển vị và u_đỉnh giữa PINN và FEM, nên
# chưa cần chốt U_ALARM. Xem ghi chú "chuyển vị bật trọng lực" trong oracle.py
# trước khi chốt ngưỡng cho cổng 2.
U_ALARM = 0.05

# ── Bộ giải ─────────────────────────────────────────────────────────────────
# Hencky không có lịch sử: các bước tải dưới đây chỉ là tiếp nối để Newton hội
# tụ từ một điểm xuất phát tốt, KHÔNG phải lịch sử tải. Kết quả cuối không phụ
# thuộc số bước (có kiểm trong oracle.py: --check-steps).
LOAD_STEPS   = (0.25, 0.5, 0.75, 1.0)
MIN_DLOAD    = 1e-3      # bước tải nhỏ nhất trước khi kết luận không có cân bằng
NEWTON_RTOL  = 1e-8
NEWTON_ATOL  = 1e-10
NEWTON_MAXIT = 30
QUAD_DEGREE  = 2         # bậc cầu phương cho số hạng dẻo

# ── Tìm hệ số an toàn bằng giảm cường độ (chỉ để tự kiểm oracle) ────────────
SRF_LO, SRF_HI = 0.8, 2.5
SRF_TOL        = 0.01
FOS_REFERENCE  = 1.4     # ví dụ 1 của Griffiths & Lane; đối chiếu lại bảng
                         # trong bài trước khi trích con số này
FOS_TOLERANCE  = 0.10    # sai khác tương đối chấp nhận (hình học khác, DP khác MC)
