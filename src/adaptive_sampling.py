"""
src/adaptive_sampling.py  —  ĐÃ LOẠI BỎ

AdaptiveSampler cũ (pool 81920 điểm, KD-tree rebuild, np.argsort trên toàn
pool mỗi lần gọi, np.random.choice có trọng số lấy 8192/81920 mỗi bước
huấn luyện) là nguồn nghẽn đơn luồng chính trong vòng lặp. Ngoài ra
update() chỉ chạy đúng một lần trong cả 200 oracle call, nên nó không có
tác dụng như thiết kế, và biến thể ablation "adaptive only" thực chất
không bật adaptive.

File này chỉ còn là lớp tương thích để src/baselines.py import _sample_bc.
Code mới dùng src/sampling.py.
"""

from src.sampling import boundary_points as _sample_bc   # noqa: F401
from src.sampling import boundary_points                 # noqa: F401

__all__ = ["_sample_bc", "boundary_points"]