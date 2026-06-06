---
name: vllm-serving-education-and-optimization
description: Quy trình biên soạn tài liệu kỹ thuật chuyên sâu về vLLM Serving và tối ưu hóa tài nguyên phần cứng GPU/VRAM cho các AI Serving Engineer.
version: 1.0.0
author: Antigravity AI Coding Assistant
tools_required:
  - nodejs >= 18.0.0
  - npm >= 9.0.0
  - python >= 3.8.0
  - git >= 2.30.0
---

# Kỹ năng Biên soạn Tài liệu & Tối ưu hóa vLLM Serving

Kỹ năng này định nghĩa các quy tắc thiết kế hệ thống, lập lộ trình học tập, phong cách viết tài liệu kỹ thuật chuyên sâu và các quy trình tự động hóa xác minh khi biên soạn chuỗi bài giảng về kiến trúc phục vụ mô hình ngôn ngữ lớn (LLM Serving) và mô hình đa phương thức (VLM Serving).

---

## 1. ĐỊNH HƯỚNG TƯ DUY & PHONG CÁCH VIẾT (Persona & Writing Persona)

Khi viết tài liệu kỹ thuật hoặc hướng dẫn cho các kỹ sư serving, phong cách viết và tư duy lập luận đóng vai trò quyết định đến khả năng tiếp thu.

### Tông giọng & Ngôn ngữ chủ đạo:
- **Tông giọng:** Chuyên nghiệp, khách quan, mang tính hướng dẫn và thực chiến cao (tập trung vào môi trường production và thực tế vận hành).
- **Ngôn ngữ:** Tiếng Việt kỹ thuật chuẩn xác. Giữ nguyên các thuật ngữ chuyên ngành tiếng Anh phổ biến (ví dụ: *KV Cache*, *Prefill*, *Decode*, *Throughput*, *Latency*, *Page Table*, *Radix Tree*) để người đọc dễ dàng đối chiếu với tài liệu quốc tế và mã nguồn.

### Cách tiếp cận bài toán (Problem-First Approach):
Khi giải thích bất kỳ tính năng hoặc tham số tối ưu nào, cấu trúc trình bày phải tuân thủ nghiêm ngặt trình tự sau:
1. **Giải thích xung đột hệ thống và hạn chế phần cứng vật lý trước:**
   - Ví dụ: Tại sao pha Prefill (Compute-bound GEMM) lại tranh chấp tài nguyên và làm nghẽn (stall) pha Decode (Memory-bound GEMV).
   - Chỉ ra vấn đề thực tế: Inter-Token Latency (ITL) tăng vọt, người dùng gặp hiện tượng khựng chữ khi có prompt dài nạp vào.
2. **Trình bày cơ chế giải quyết ở tầng thuật toán và cấu trúc dữ liệu:**
   - Đưa ra giải pháp (ví dụ: bẻ nhỏ prompt prefill thành các chunk kích thước cố định chạy đan xen).
   - Vẽ sơ đồ ASCII để trực quan hóa luồng dữ liệu hoặc mô hình lập lịch.
3. **Minh họa bằng công thức toán học và mã nguồn thực tế:**
   - Sử dụng các biểu thức toán học LaTeX để mô tả mối quan hệ tài nguyên (ví dụ: cách tính số block KV Cache vật lý).
   - Trích dẫn trực tiếp tên file, tên hàm hoặc đoạn code cốt lõi từ codebase của công cụ serving (ví dụ: `{REPOSITORY_NAME}`) để người học có thể tự tra cứu sâu hơn.

---

## 2. CÁC RÀNG BUỘC BẢO MẬT & QUYỀN RIÊNG TƯ (Security & Privacy)

Bảo vệ thông tin và tài sản trí tuệ là yêu cầu bắt buộc trong mọi dự án serving.

### Ẩn danh hóa thông tin nhạy cảm:
- Tuyệt đối không được hardcode các thông tin nhạy cảm của hệ thống hoặc cá nhân vào tài liệu.
- Mọi giá trị cấu hình cụ thể phải được thay thế bằng các biến môi trường hoặc biến động dạng `{BIẾN}`. Ví dụ:
  - Đường dẫn cục bộ: `{PROJECT_DIR}/docs` thay vì `/Users/username/project/docs`.
  - Thông tin xác thực Git: `{TARGET_GIT_USERNAME}` và `{TARGET_GIT_EMAIL}`.
  - Thông tin định danh kho chứa: `{GITHUB_OWNER}` và `{REPOSITORY_NAME}`.

### Bảo mật môi trường Serving:
- Cảnh báo người dùng về nguy cơ bảo mật khi sử dụng các cờ như `--trust-remote-code` trong production. Khuyến khích cơ chế kiểm duyệt mã nguồn cục bộ (audit) trước khi chạy.
- Cấu hình tệp `robots.txt` và `sitemap.xml` phù hợp đối với các trang tài liệu nội bộ để ngăn chặn các công cụ tìm kiếm tự động (web crawlers) lập chỉ mục (index) dữ liệu nhạy cảm của doanh nghiệp.
- Nếu có yêu cầu bảo mật đặc biệt đối với mã nguồn, tệp `README.md` tại một số thư mục nhạy cảm có thể được duy trì ở dung lượng đúng $0$ bytes để không để lộ cấu trúc thư mục ra bên ngoài.

---

## 3. QUY TRÌNH THỰC THI & TỰ ĐỘNG HÓA (Execution Workflow)

Quy trình chuẩn mực để cập nhật tài liệu và tích hợp hệ thống bao gồm các bước sau:

```
+------------------+     +------------------+     +------------------+
| 1. Config Git    | --> | 2. Edit & Write  | --> | 3. Build & Test  |
| (Local Identity) |     | (Markdown Docs)  |     | (npm run build)  |
+------------------+     +------------------+     +------------------+
                                                               │
                                                               ▼
+------------------+     +------------------+     +------------------+
| 6. CD Trigger    | <-- | 5. Git Push      | <-- | 4. Commit Local  |
| (GitHub Pages)   |     | (origin main)    |     | (git commit)     |
+------------------+     +------------------+     +------------------+
```

### Bước 1: Cấu hình thông tin Git cục bộ (Local Identity)
Trước khi thực hiện bất kỳ commit nào, phải thiết lập thông tin định danh git trong thư mục làm việc để tránh ghi nhận thông tin sai lệch:
```bash
git config --local user.name "{TARGET_GIT_USERNAME}"
git config --local user.email "{TARGET_GIT_EMAIL}"
```

### Bước 2: Biên soạn tài liệu kỹ thuật
Viết các bài giảng dưới dạng Markdown chất lượng cao, lưu vào đúng thư mục chuyên đề quy định (ví dụ: `docs/10_production_reference/`). Sử dụng các liên kết tương đối hợp lệ để liên kết giữa các bài giảng.

### Bước 3: Kiểm tra biên dịch tĩnh (Static Build)
Chạy lệnh kiểm tra biên dịch Docusaurus cục bộ để phát hiện sớm các lỗi cú pháp Markdown hoặc các liên kết bị hỏng (broken links):
```bash
npm run build
```

### Bước 4: Commit thay đổi cục bộ
Đưa các file đã thay đổi vào vùng chờ (stage) và tạo commit kèm theo thông điệp (commit message) chuẩn hóa:
```bash
git add docs/roadmap.md docs/10_production_reference/lesson_10_vram_allocation_offline_batching.md
git commit -m "docs: expand Lesson 10 with deep-dive technical sections"
```

### Bước 5: Đẩy mã nguồn lên Remote Repository
Đẩy nhánh làm việc lên GitHub để lưu trữ:
```bash
git push origin main
```

### Bước 6: Kích hoạt quy trình CI/CD qua GitHub API
Khi cần tự động hóa việc kích hoạt quy trình triển khai (GitHub Pages deployment) mà không cần tạo commit rỗng, hãy gửi request trực tiếp đến GitHub Actions API:
```bash
curl -X POST \
  -H "Authorization: token {GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/{GITHUB_OWNER}/{REPOSITORY_NAME}/actions/workflows/{WORKFLOW_ID}/dispatches \
  -d '{"ref":"main"}'
```

---

## 4. XỬ LÝ LỖI THƯỜNG GẶP (Troubleshooting Guide)

Trong quá trình xây dựng trang tài liệu và cấu hình vLLM, một số lỗi phổ biến sau đây có thể xảy ra:

### Lỗi 1: Docusaurus build thất bại do broken links (Cảnh báo biến thành Lỗi)
- **Triệu chứng:** Lệnh `npm run build` kết thúc với mã lỗi không bằng 0 và báo lỗi liên kết bị hỏng (broken links).
- **Nguyên nhân:** Docusaurus tự động loại bỏ các tiền tố số (ví dụ: `01_`, `02_`) của thư mục khi tạo URL slug trên web. Do đó, các liên kết tuyệt đối dạng `/docs/01_system_foundations/file` sẽ bị hỏng.
- **Khắc phục:** Thay thế toàn bộ bằng các liên kết tương đối dạng `./system_foundations/file` (loại bỏ tiền tố số của thư mục trong đường dẫn liên kết) hoặc cập nhật cấu hình `onBrokenLinks: 'warn'` tạm thời trong file cấu hình để debug.

### Lỗi 2: CUDA OOM khi khởi tạo vLLM Engine
- **Triệu chứng:** Mô hình báo lỗi Out-Of-Memory ngay khi vừa khởi động engine, trước khi xử lý bất kỳ request nào.
- **Nguyên nhân:** Giá trị đặt cho cờ `--max-num-seqs` hoặc `--max-num-batched-tokens` quá lớn so với dung lượng VRAM thực tế của GPU, làm pha Profiling Run của vLLM chiếm dụng quá nhiều bộ nhớ activation đỉnh.
- **Khắc phục:** Giảm `--max-num-seqs` xuống mức vừa phải (ví dụ từ $512$ xuống $256$ hoặc $128$) hoặc giảm `--max-model-len` để giải phóng không gian bộ nhớ.

### Lỗi 3: Lỗi xác thực quyền đẩy mã nguồn (Git Push Permission Denied)
- **Triệu chứng:** Git báo lỗi không có quyền ghi khi chạy lệnh `git push`.
- **Nguyên nhân:** SSH Key hoặc Personal Access Token (PAT) cấu hình cục bộ không khớp với tài khoản sở hữu repository `{GITHUB_OWNER}`.
- **Khắc phục:** Kiểm tra lại URL của remote bằng lệnh `git remote -v`. Cấu hình lại PAT hợp lệ trong thông tin xác thực git hoặc cập nhật URL git chứa token: `git remote set-url origin https://{GITHUB_TOKEN}@github.com/{GITHUB_OWNER}/{REPOSITORY_NAME}.git`.

---

## 5. TIÊU CHUẨN XÁC MINH HOÀN THÀNH (Verification Checklist)

Để đảm bảo tài liệu được biên soạn và xuất bản đạt tiêu chuẩn chất lượng cao nhất, trước khi bàn giao cho người dùng, Agent phải chạy các lệnh shell tự động sau để xác minh:

| Hạng mục xác minh | Lệnh Shell kiểm tra | Tiêu chuẩn chất lượng yêu cầu |
| :--- | :--- | :--- |
| **Kiểm tra biên dịch Docusaurus** | `npm run build` | Biên dịch thành công 100%, kết thúc với mã thoát (exit code) bằng $0$. |
| **Kiểm tra ký tự cấm (Em-Dash)** | `python3 -c "assert '\u2014' not in open('{FILE_PATH}').read()"` | Không được xuất hiện ký tự gạch ngang dài em-dash (mã unicode U+2014) trong tất cả các file tài liệu. |
| **Kiểm tra trạng thái HTTP cổng deploy** | `curl -s -o /dev/null -w "%{http_code}" https://{GITHUB_OWNER}.github.io/{REPOSITORY_NAME}/` | Trả về mã trạng thái HTTP là `200` sau khi quy trình CD hoàn tất. |
| **Kiểm tra quyền truy cập của Bot** | `curl -s https://{GITHUB_OWNER}.github.io/{REPOSITORY_NAME}/robots.txt` | Phải chứa các chỉ thị cấm phù hợp (ví dụ: `Disallow: /`) nếu trang web là tài liệu bảo mật nội bộ. |
| **Kiểm tra thẻ meta chặn index** | `curl -s https://{GITHUB_OWNER}.github.io/{REPOSITORY_NAME}/ | grep -i "robots"` | Phải chứa `<meta name="robots" content="noindex, nofollow">` nếu được yêu cầu bảo mật riêng tư. |
