# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Thành Lộc — 2A202600817
**Ngày nộp**: 2026-06-25
**Submission option**: A (lightweight)

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (Dự đoán dựa trên T4 notebook)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, ~300 samples
- **max_seq_length**: 512 (p95)
- **GPU**: Tesla T4 (Google Colab), 16 GB VRAM
- **Training cost**: $0 (Colab Free)

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 1,843,200       | 3.77 min   | 7.22 GB   | 1.558     | 4.748      |
| 16   | 3,686,400       | 3.85 min   | 6.61 GB   | 1.516     | 4.554      |
| 64   | 14,745,600      | 3.74 min   | 8.00 GB   | 1.477     | 4.379      |
| Base | -               | -          | -         | N/A       | N/A        |

## 3. Loss Curve Analysis
[Đính kèm loss_curve.png]
- Quan sát: Không có dấu hiệu overfitting rõ rệt, train loss và eval loss đều có xu hướng giảm ổn định. Tuy nhiên, ở các epoch cuối, eval loss có thể bắt đầu chững lại.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
**Base**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, nó có nghĩa là máy tính học tập từ dữ liệu. Mô hình học được từ dữ liệu này có thể thực hiện các tác...
**Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo) và sử dụng thuật toán và các mô hình học máy để học từ dữ liệu và từ đó đưa ra các dự đoán hoặc...
**Nhận xét**: Cả hai đều giải thích đúng, tuy nhiên mô hình fine-tuned có cách hành văn mang tính kỹ thuật và tự nhiên hơn.

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
**Base**: 
```python
def fibonacci(n):
    if n <= 0:
        return "N phải là một số dương"
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
```
**Fine-tuned (r=16)**: 
```python
def fibonacci(n):
    if n < 0:
        raise ValueError("Input phải là một số nguyên dương.")
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range
```
**Nhận xét**: Cả hai đều sinh ra code cơ bản nhưng chưa hoàn thiện (do giới hạn token sinh ra). Fine-tuned model đưa ra cách xử lý lỗi `ValueError` chuyên nghiệp hơn.

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
**Base**: 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục, màu sắc, font chữ, hình ảnh và các yếu tố khác để đảm bảo người dùng có thể dễ dàng sử dụng sản phẩm. 2. Tru...
**Fine-tuned (r=16)**: 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản: UI/UX thiết kế phải dễ dàng sử dụng và dễ hiểu. 4. Tương thích: UI/UX thiết kế phải dễ dàng tương thích với...
**Nhận xét**: Mô hình fine-tuned liệt kê trực tiếp và ngắn gọn vào đúng 5 nguyên tắc, trong khi base model giải thích dài dòng ở điểm số 1. Cải thiện rõ rệt về định dạng trả lời (format).

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
**Base**: LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp cải thiện hiệu năng của mô hình bằng cách thay đổi các phép biến đổi trong mô hình bằng các phép biến đổi nhỏ hơ...
**Fine-tuned (r=16)**: LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển để cải thiện hiệu quả và độ ổn định của các mạng neural network trong các mô hình có số lượng các layer lớn. LoRA là một kỹ thuật regularization được sử dụng để cải thiệ...
**Nhận xét**: Cả hai mô hình đều có dấu hiệu hallucination nhẹ về mặt học thuật (base model mô tả chưa chuẩn xác hoàn toàn, còn fine-tuned model thì bịa ra từ "Layer-wise Adaptive Regularization Optimization"). Kiến thức base model chưa được cải thiện nhiều sau SFT do đây không phải là mục đích của fine-tuning (theo lý thuyết).

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
**Base**: Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất của mô hình bằng cách cung cấp cho nó một câu hỏi hoặc câu lệnh để dựa vào, thay vì cung cấp dữ liệu đầu...
**Fine-tuned (r=16)**: Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp hệ thống AI giải quyết các vấn đề và thực hiện các tác vụ. Prompt được sử dụng để cung cấp cho hệ th...
**Nhận xét**: Cả hai đều chưa hoàn thành hết câu trả lời do max new tokens, tuy nhiên nội dung bước đầu đều có ý nghĩa hợp lý.

## 5. Conclusion về Rank Trade-off
- **Rank nào cho ROI tốt nhất trên dataset này? Tại sao?**
Rank = 16 cho ROI tốt nhất. Rank 8 có perplexity cao nhất (4.748), trong khi rank 64 đạt perplexity tốt nhất (4.379) nhưng đòi hỏi nhiều VRAM (8.00 GB) và params hơn gấp 4 lần so với r=16. R=16 cân bằng tốt giữa thời gian huấn luyện (~3.85 phút), yêu cầu bộ nhớ (6.61 GB) và độ chính xác (perplexity 4.554).
- **Khi nào tăng rank không còn cải thiện perplexity (diminishing returns)?**
Khi rank tăng từ 16 lên 64 (gấp 4 lần số lượng tham số huấn luyện), perplexity chỉ giảm nhẹ từ 4.554 xuống 4.379 (giảm khoảng 3.8%). Sự cải thiện này không tương xứng với tài nguyên bị tiêu tốn thêm, cho thấy hiệu ứng diminishing returns.
- **Recommendation: nếu deploy production, bạn chọn rank nào? Tại sao?**
Tôi sẽ chọn rank = 16. Lý do vì mức cải thiện chất lượng từ r=16 lên r=64 là không đủ lớn để bù đắp cho sự chênh lệch chi phí VRAM và thời gian suy luận (khi chạy multi-tenant hoặc lưu trữ adapter). r=16 đảm bảo mô hình học được đủ tốt phong cách dữ liệu mà vẫn cực kỳ hiệu quả tài nguyên.

## 6. What I Learned
- Hiểu được rõ ảnh hưởng thực tế của giá trị rank đối với tài nguyên GPU (VRAM) và thời gian huấn luyện.
- Thấy được sự thật là fine-tuning chỉ cải thiện "style/format" chứ không dễ dàng dạy thêm kiến thức mới (ví dụ như câu LoRA/QLoRA mô hình tự chế ra "Layer-wise Adaptive Regularization Optimization").
- Nắm được cách đánh giá mô hình không chỉ dựa vào số perplexity mà còn qua Qualitative test (sinh thử văn bản và đọc để so sánh độ tự nhiên, chính xác).
