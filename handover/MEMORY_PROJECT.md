# FMD-PINN: bộ nhớ project (xuất ngày 01/11/2026)

File này gom những gì Claude đã ghi nhớ trong project FMD-PINN ở tài khoản cũ, chỉ phần kỹ thuật và cách làm việc. Mỗi dòng là điều Tran đã nói hoặc đã chốt. Phần lịch sử (thời A5, min-max) được giữ lại để hiểu vì sao đã đổi hướng, không phải trạng thái hiện tại.

## Bài học phương pháp

- Không chấp nhận đánh đổi FSR lấy δ_H: cải thiện hình học biên mà FSR tăng không phải đóng góp đáng tin; hai thứ phải cùng tốt lên.
- Mỗi lần chỉ đổi một thứ. Chồng nhiều thay đổi cùng lúc làm không cô lập được nguyên nhân. Thử một seed trước khi chạy đủ 5 seed.
- Nút thắt thật là oracle, không phải huấn luyện PINN; tối ưu GPU cho PINN không giải quyết nút thắt đó.
- δ_H tính trên các điểm gọi oracle đo phân bố lời gọi, không đo độ chính xác của biên dự đoán; cần thước đo riêng cho chất lượng mô hình thay thế.
- Thuật ngữ: E(u; p) là phiếm hàm (functional), không phải hàm chỉ thị; 1_C(p) ∈ {0, 1} mới là hàm chỉ thị.
- Oracle 2D là `solve_ivp` của scipy theo phương pháp đường, không phải FEniCS hay FEM.
- Không chạy qua đêm trên code chưa kiểm.
- Không chỉnh loss, trọng số hay anchor để số đẹp hơn; đó là tạo tham số đẹp chứ không phải nghiên cứu. Tìm và sửa lỗi thiết kế.
- Bỏ các lập luận dựa trên thời gian chạy FEM (kiểu T_train + N_δ·T_FEM ≤ N_BO·T_FEM); đếm số lời gọi oracle thay vào.
- Dự án cần một mốc neo: điểm đủ để viết, công bố và trình diễn bằng chứng khái niệm. Cải thiện tiếp đi qua bài 3D, không xoay vòng trên bài 2D.
- Sửa phép đo là chấp nhận được, sửa thiết kế thí nghiệm là mức tối đa; sửa mô hình liên tục không có đích là kiểu thất bại phải tránh.
- Mục đích bài 2D là chứng minh thuật toán giải được bài toán và ra kết quả, không phải tối ưu để thắng baseline hàng chục năm.
- Mục tiêu là tìm các bộ tham số có thể dẫn tới phá hủy, không phải giải toàn bộ PDE.
- Thước đo theo lát và thước đo hình học mất ý nghĩa trong không gian 7 chiều: một lát trong hàng nghìn lát không kiểm được và không khái quát được.
- Trước khi đề nghị một lần chạy dài, đưa phép kiểm tra vài giây trên dữ liệu có sẵn kèm ngưỡng quyết định chốt trước. Tran không phải "máy cày" chạy lại để kiểm ý tưởng có thể kiểm trên giấy.
- Với bài 3D, mọi giai đoạn cần một cổng kiểm tra rẻ trước khi bỏ ra hàng giờ đến hàng ngày tính toán; không chấp nhận kiểu "cắm vào là chạy".
- Không đặt tên cho hiện tượng trước khi đo: gọi vùng β thấp là "knee" đã dẫn tới chuỗi suy luận sai (tưởng gradient suy biến, thực tế gradient ở đó lớn nhất) và một thành phần làm ngược ý định.

## Cách làm việc

- Phát triển lặp, thử trước: một seed, kiểm, rồi mới 5 seed. Gỡ lỗi bằng cô lập có hệ thống, script chẩn đoán, đổi một biến một lần.
- Muốn lời giải đầy đủ, có lập luận, đưa ra ngay thay vì hỏi lại từng chút.
- Phát biểu lý thuyết phải có cơ sở toán học, không mô tả suông. Muốn con số dứt khoát kèm nguồn rõ ràng.
- Sửa code: chỉ gửi đoạn thay đổi kèm chỉ dẫn sửa ở đâu trong từng file, không gửi lại cả file.
- Viết thẳng, tập trung nội dung, không trang trí. Báo cáo tiếng Việt kiểu học thuật, toán viết kiểu LaTeX, hình vẽ bằng script matplotlib.
- Trong hội thoại cũng viết toán bằng LaTeX, không viết công thức dạng chữ thường hay khối code; nêu đầy đủ phát biểu toán (PDE, phương trình, ý nghĩa tham số) trước khi bàn về lần chạy.
- Hình khoa học: tối thiểu chữ trên hình, phần giải thích in ra ngoài chứ không ghi chú trên hình.

## Môi trường và công cụ

- Máy huấn luyện chính: Windows, RTX 4070 Ti Super. Máy phụ: Ubuntu ThinkPad P15 (Quadro T2000), dùng vẽ hình và chạy dự phòng; Ubuntu được ưu tiên để chạy vì tránh lỗi DLL trên Windows (`libiomp5md.dll` xung đột với OpenMP của torch).
- Stack: PyTorch 2.6.0+cu124, Python 3.10.11, scipy (`solve_ivp`), Adam + L-BFGS.
- Môi trường Windows: `pinn_venv_new`; PATH phải đặt tay mỗi phiên.
- Repo chính `TranDung6129/ea_pinn`, đẩy thẳng lên `main`, không dùng tag: mốc `cap` tái tạo bằng `PROJECT_ITERS = 0` (3 là `proj`). Nhánh `clean-from-dbcad31` chứa kết quả A5 cũ.
- Mái dốc: oracle chạy trong conda env `fenicsx-env` (dolfinx 0.9), PINN trong `pinn_venv`, tách riêng. Code mái dốc ở thư mục mới trong cùng repo, không đụng code 2D.

## Định vị và đóng góp

- Dùng thực tế là realtime: ngoài hiện trường p ước lượng từ cảm biến luôn có sai số, câu hỏi cần trả lời tức thì là cấu hình hiện tại CÓ THỂ rơi vào vùng phá hủy không. Một lượt autodiff qua mô hình thay thế cho E(p) (khoảng cách tới biên) và ∇E (hướng sai số cảm biến đẩy qua biên). Vì vậy cần mô hình khả vi, không phải bảng tra hay tái dựng hình học đa tạp.
- Danh sách đóng góp chốt 26/09: (1) phán quyết nhanh bộ tham số hiện tại có thể dẫn tới phá hủy hay không, kèm gradient nhanh, chạy trên thiết bị biên; (2) giải thích trực tiếp bằng vật lý (vị trí mặt trượt); (3) áp dụng cho bài đàn dẻo 7 tham số. Tran thấy đóng góp lớn nhất ở góc AI-IoT và thiết bị biên hơn là bản thân mô hình.
- Đích xuất bản: hội nghị trước cho kết quả 2D; tối thiểu Q2, Q1 nếu thuận lợi. Không nhằm thắng baseline.
- Phạm vi luận văn: đa tạp phá hủy trong không gian tham số; cảm biến, thời gian và cảnh báo là việc thực tế về sau.
- Họ baseline cho 3D: Kriging chủ động kiểu AK-MCS.

## Quyết định đã chốt cho bài mái dốc (26 đến 28/09)

- E định nghĩa theo tiêu chí báo động chuyển vị (khớp với cái cảm biến đo), không theo sập; bản thảo bài báo lại dùng E = 1 − FS. Hai định nghĩa này chưa được thống nhất.
- Làm biến dạng phẳng trước khi lên 3D đầy đủ. Oracle FEniCSx với thuật toán chiếu ngược tự viết.
- Dẻo Hencky (biến dạng toàn phần) cho cả FEM và PINN, ghi rõ giả thiết và giới hạn.
- PINN theo phương pháp năng lượng (cực tiểu thế năng toàn phần).
- Cổng 2 mở c, φ và hệ số động đất giả tĩnh k_h làm đầu vào của PINN (hình học, γ, E, ν cố định; mực nước ngầm để sau), sau một phép kiểm độ nhạy.
- Kế hoạch 3D theo cổng: PINN một bộ tham số cố định so với FEM, rồi 2 đến 3 tham số, rồi vòng lặp đủ 7 chiều.
- Kết quả: cổng 1 đạt; cổng 2 không đạt (đúng dấu E ở 97.3% điểm nhưng không tái tạo được dải biến dạng dẻo tập trung); cổng 3 chưa làm. Chi tiết trong doc "Bài toán mái dốc đàn dẻo: phát biểu và những gì đã làm".

## Lịch sử (thời A5, trước khi đổi hướng tháng 9)

- Bản A5 (min-max đối kháng) đạt δ_H ≈ 0.1816 ± 0.0071 so với BO+FEM 1.2534 theo thước đo cũ, FSR 10.2% (dải chết τ = 0.05), CR 98.85%. Thước đo cũ đã được thay (p95 và chamfer) nên các số này không so được với mốc hiện tại.
- Tháng 9/2026 đổi câu hỏi nghiên cứu: đối tượng là tập mức không E(p) = 0, bài toán tìm nghiệm. Vòng ngoài cực đại E bị coi là sai hướng (nó đuổi theo cấu hình sập sâu nhất, nằm trong vùng phá hủy chứ không trên biên) và được thay bằng tìm nghiệm kiểu bisection. Kết quả A5 thành baseline nội bộ.
- Thứ tự Tran đặt ra: có vòng lặp nhanh trước, khớp đúng vùng β thấp và đóng bài 2D như phần kiểm chứng, rồi sang bài 3D là mục tiêu chính.
- Thời gian chạy từng phụ thuộc thuật toán chứ không phần cứng (cùng thời gian trên i7-10H, X99 nhiều nhân và i9-11F).
