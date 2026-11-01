# Chuyển project FMD-PINN sang tài khoản Claude khác

Anthropic chưa hỗ trợ chuyển dữ liệu giữa hai tài khoản, nên project phải dựng lại bằng tay. Thư mục này chứa mọi thứ cần để dựng lại. Làm theo thứ tự dưới đây.

## Thư mục này có gì

| File | Dùng để làm gì |
|---|---|
| `PROJECT_INSTRUCTIONS.md` | Dán vào ô Custom instructions của project mới. Đã cập nhật đến 01/11/2026 (bản cũ trong project vẫn là của 26/09 và có vài số đã được đính chính). |
| `TRANG_THAI_2026-09-26.md` | Bản sao nguyên văn tài liệu bàn giao `claude/TRANG_THAI_2026-09-26.md` trong project cũ. |
| `MEMORY_PROJECT.md` | Bộ nhớ của project cũ, phần kỹ thuật và cách làm việc, đã dịch sang tiếng Việt. |
| `MAI_DOC_2026-09-28.md` | CHƯA CÓ. Bạn tải về ở bước 2 rồi thêm vào. |

Repo này công khai, nên những ghi nhớ mang tính cá nhân (về thầy hướng dẫn, kinh phí, kế hoạch nếu không đạt Q2) KHÔNG nằm ở đây. Chúng nằm trong file `MEMORY_RIENG_TU.md` được gửi riêng trong cuộc hội thoại; hãy lưu file đó ở máy, đừng commit.

## Bước 1: đẩy phần còn thiếu lên repo (làm ở máy của bạn)

Commit mới nhất trên `main` là `57df1ea` (26/09 13:24), đã có `measure_grad.py`, `measure_dslope.py`, `eval_corrected.py`, `export_figdata.py` và bản sửa trong `src/fmd_pinn.py`, `config.py`. Nhưng repo CHƯA có:

- thư mục code mái dốc (oracle FEniCSx, module cấu tử numpy/torch, PINN năng lượng, quét FEM 125 điểm, tái dựng trường đàn hồi, các script cổng 1 và cổng 2);
- bản thảo bài báo: `main.tex`, `IEEEtran.cls`, `fig/`, `figs.py`, `analysis.py`, `fmd_pinn_method.drawio`;
- `results/figdata.npz` và các file kết quả `cap`/`proj` nếu `.gitignore` đang chặn chúng (kiểm bằng `git status --ignored`).

Đẩy những thứ này lên trước. Nếu không muốn bản thảo bài báo công khai trước khi nộp, để nó trong một repo riêng tư hoặc chuyển `ea_pinn` sang riêng tư.

## Bước 2: xuất từ tài khoản cũ

1. **Bản xuất dữ liệu.** Settings, mục Privacy, Export data. Link tải gửi qua email, hết hạn sau 24 giờ. Giữ file zip làm kho lưu trữ; nó không nhập được vào tài khoản mới.
2. **Doc mái dốc.** Mở doc "Bài toán mái dốc đàn dẻo: phát biểu và những gì đã làm" (claude.ai/artifact/6HyDhL5NLVWYs89wmwVLQn), tải về dạng Markdown, lưu thành `handover/MAI_DOC_2026-09-28.md` và commit. Tải thêm bản PDF để giữ hai hình (bản Markdown thay hình bằng chữ). Đây là tài liệu đầy đủ nhất về bài mái dốc: phát biểu, luật cấu tử, định nghĩa E₁ và E₂, bảng các lần chạy.
3. **Trang "FMD-PINN: Tối ưu tính toán GPU"** (claude.ai/artifact/1aVm4FxDfgMiY4fA2TwD51). Mở, lưu nội dung nếu còn cần.
4. **File trong project:** `FMD_PINN_Research_Report_v2.pdf` (báo cáo tháng 4, thời A5, đã lỗi thời so với hướng hiện tại). Đã được gửi kèm trong cuộc hội thoại; tải về nếu muốn giữ.
5. **Các hội thoại quan trọng** để tra lại trong bản xuất (tìm theo tiêu đề trong `conversations.json`): "Landslide detection multi-sensor research" (27/09, toàn bộ bài mái dốc, cổng 1 và 2, tổng quan tài liệu PINN cho độ tin cậy), "Greeting" (26/09, dựng lại phương pháp và mốc 2D), "Hoàn thiện đề cương nghiên cứu và xuất PDF" (24/09), "Code lấy thông tin cho các chương 5, 6, 7", "Quy trình warm up: kết quả và đánh giá", "Bài báo liên quan tới phương pháp nghiên cứu" (tổng quan có phản biện).

Các link artifact ở trên sẽ ngừng hoạt động khi tài khoản cũ bị xóa, nên tải hết trước bước 6.

## Bước 3: dựng project ở tài khoản mới

1. Tạo project tên FMD-PINN.
2. Dán nội dung `PROJECT_INSTRUCTIONS.md` vào Custom instructions.
3. Nối GitHub vào project (đồng bộ repo `TranDung6129/ea_pinn`, nhánh `main`). Project cũ đồng bộ `/src/`, `/config.py` và các `run_0*.py`; thêm `/handover/`, các script đo (`measure_grad.py`, `eval_corrected.py`, `export_figdata.py`) và thư mục code mái dốc.
4. Nếu không nối GitHub được, tải trực tiếp ba file trong `handover/` (và `MAI_DOC_2026-09-28.md`) lên làm tài liệu của project.

## Bước 4: đưa trí nhớ vào project mới

Có ba cách, nên làm cách A và B.

**Cách A, tài liệu project (chắc chắn nhất).** Bước 3 đã làm: `MEMORY_PROJECT.md` nằm trong tài liệu của project, và instructions bảo Claude đọc nó đầu mỗi hội thoại. Cách này không phụ thuộc vào tính năng nhớ.

**Cách B, cho Claude tự ghi vào bộ nhớ của project.** Mở một hội thoại mới TRONG project, dán nội dung `MEMORY_PROJECT.md` kèm câu sau:

> Đây là bộ nhớ của project FMD-PINN từ tài khoản cũ của tôi. Hãy lưu nó vào bộ nhớ của project này, giữ nguyên ý từng dòng, không thêm suy diễn, và cho tôi biết dòng nào không lưu được.

Sau đó vào trang của project, xem phần memory để kiểm. Một số dòng có thể bị bỏ qua theo quy tắc riêng tư của hệ thống nhớ; điều đó không sao vì cách A vẫn giữ đủ.

**Cách C, nhập bộ nhớ cấp tài khoản.** Settings, Capabilities, Import memory. Tính năng này đang thử nghiệm, nạp vào bộ nhớ chung của tài khoản chứ không vào riêng project, và có thể mất đến 24 giờ. Chỉ dùng cho các sở thích chung (trả lời tiếng Việt, văn xuôi, không dùng dấu gạch ngang dài, viết toán bằng LaTeX), không dùng cho nội dung FMD-PINN.

Lưu ý thiết lập bộ nhớ của project: mỗi project có một lựa chọn giữ bộ nhớ riêng hay dùng chung với bộ nhớ tài khoản, nằm trên trang của project. Nếu muốn FMD-PINN tách hẳn khỏi các việc khác (VinUni, môn học), chọn riêng.

## Dùng với Claude Code trong VS Code

File `CLAUDE.md` ở gốc repo được Claude Code tự đọc khi bạn mở thư mục `ea_pinn`. Nó nạp `PROJECT_INSTRUCTIONS.md` và `MEMORY_PROJECT.md`, nên Claude Code có đủ bộ nhớ ngay từ câu đầu mà không cần project trên claude.ai. Phần riêng tư: chép nội dung `MEMORY_RIENG_TU.md` vào file `CLAUDE.local.md` ở gốc repo. File này cũng được tự đọc, và đã nằm trong `.gitignore` nên không bị đẩy lên GitHub.

## Bước 5: kiểm tra project mới

Mở hội thoại mới trong project và hỏi ba câu. Nếu Claude trả lời đúng mà không cần bạn giải thích lại, việc chuyển đã xong.

1. "Mốc 2D hiện tại là gì và tái tạo nó thế nào?" Đáp án: `cap` + hiệu chỉnh hằng số, δ_H p95 0.0388 ± 0.0012, FSR 2.7%; đặt `PROJECT_ITERS = 0`.
2. "Có được dùng công thức α = 2π²D + 2.25β không?" Đáp án: không, sai vì biên Neumann làm mode trung bình không phụ thuộc D.
3. "Bài mái dốc đang ở đâu?" Đáp án: cổng 1 đạt, cổng 2 không đạt (đúng dấu 97.3% nhưng không có dải trượt), cổng 3 chưa làm; còn mở việc chọn E₁ hay E₂.

## Bước 6: đóng tài khoản cũ (nếu muốn)

Chỉ làm sau khi bước 5 đã qua và bản xuất đã tải về. Hủy gói trả phí trước, chờ hết kỳ, rồi xóa tài khoản trong Settings, Account. Muốn dùng lại số điện thoại thì nhờ bộ phận hỗ trợ gỡ liên kết khỏi tài khoản cũ.
