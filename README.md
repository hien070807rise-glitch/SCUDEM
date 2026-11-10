# SCUDEM XI 2026 — Problem B: Group Dynamics in Foraging

> **Đề tài:** Động lực học nhóm trong hành vi kiếm ăn  
> **Phương pháp:** Mô hình hóa Phương trình Vi phân Ngẫu nhiên Langevin Ba Giai đoạn (Overdamped Langevin SDE)  
> **Thí sinh thực hiện:** Lê Kế Hiển, Phạm Thanh Thư, Đặng Huyền Trâm  
> **Cuộc thi:** Student Challenge Using Differential Equations Modeling (SIMIODE / COMAP)

---

## 📌 Tổng quan Nghiên cứu

Dự án giải quyết trọn vẹn **Bài toán B (SCUDEM XI 2026)** về cơ chế chia sẻ thông tin bổ sung và phối hợp nhóm trong hành vi kiếm ăn của loài khỉ nhện *Ateles geoffroyi* tại Khu bảo tồn Punta Laguna. Mô hình được xây dựng trên nền tảng hệ phương trình vi phân ngẫu nhiên Langevin quá cản lồng nhau qua ba giai đoạn, tích hợp:
1. **Khoảng cách không gian:** Lực lò xo xã hội $d_0$ và hàm suy giảm âm thanh Gauss $\rho(d)$ với bán kính nghe rõ $R_c \approx 50\text{ m}$.
2. **Chất lượng và độ chi tiết thông điệp:** 4 bậc chi tiết $b \in \{1..4\}$ kết hợp hàm chi phí sinh lý $c_0 b$.
3. **Độ chính xác kênh truyền:** Quá trình nhiễu tọa độ Ornstein--Uhlenbeck $\eta_j(t)$ và chốt chặn phòng vệ của cổng cảm nhận tại chỗ $(1-q_i)^\gamma$.
4. **Lừa dối chiến lược:** Tín hiệu giả lệch hướng $(t_j, \Delta)$, phân tích động lực học niềm tin nhị phân (*bang-bang*), và chân dung pha hệ nhân bản Replicator Dynamics.

Toàn bộ kết luận định lượng được nghiệm thu trên tập dữ liệu từ **$R = 5.000$ bản sao Monte Carlo** sử dụng biến ngẫu nhiên chung (CRN) với bước tích phân $\dd t = 0{,}25\text{ phút}$.

---

## 📂 Cấu trúc Thư mục

```text
SCUDEM/
├── SCUDEM_Bao_cao_Hoan_chinh.tex     # Mã nguồn báo cáo hoàn chỉnh (XeLaTeX)
├── SCUDEM_Bao_cao_Hoan_chinh.pdf     # Bản PDF báo cáo hoàn chỉnh (25 trang)
├── README.md                          # Tài liệu hướng dẫn dự án
├── .gitignore                         # Danh mục loại trừ file rác biên dịch
├── python/                            # Toàn bộ mã nguồn mô phỏng và kiểm toán
│   ├── super_sim.py                   # Động cơ mô phỏng Langevin SDE 3 giai đoạn
│   ├── run_super_simulation.py        # Kịch bản chạy siêu mô phỏng R=5.000 và TOST
│   ├── super_simulation_results.json  # Dữ liệu nghiệm thu số liệu thống kê
│   ├── gd4/                           # Module phân tích lừa dối và quét cự ly
│   └── gd5/                           # Module phân tích độ nhạy toàn cục Sobol
├── fig/                               # Đồ thị và hình ảnh minh họa cho báo cáo
└── fig_en/                            # Phiên bản hình ảnh định dạng quốc tế
```

---

## 🚀 Hướng dẫn Chạy & Biên dịch

### 1. Biên dịch Báo cáo PDF (XeLaTeX)
Báo cáo sử dụng trình biên dịch **XeLaTeX** với gói ngôn ngữ tiếng Việt `polyglossia` và font `TeX Gyre Pagella`:
```bash
xelatex -interaction=nonstopmode SCUDEM_Bao_cao_Hoan_chinh.tex
xelatex -interaction=nonstopmode SCUDEM_Bao_cao_Hoan_chinh.tex
```

### 2. Chạy Siêu mô phỏng Python ($R = 5.000$ bản sao)
Yêu cầu môi trường Python 3.9+ với `numpy`, `scipy`, `matplotlib`:
```bash
cd python
python run_super_simulation.py
```
Kết quả thống kê, khoảng tin cậy 95% và kiểm định tương đương TOST sẽ tự động xuất ra file `super_simulation_results.json`.
