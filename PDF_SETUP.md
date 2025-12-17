# 📄 PDF Viewer Setup Guide

## Tính năng mới: Xem PDF trong Streamlit App

Mode 6 cho phép xem trực tiếp các file PDF tài liệu VinFast trong app.

## 🎯 Cách sử dụng

### Bước 1: Cấu hình đường dẫn PDF

Mở file `app.py` và chỉnh sửa đường dẫn:

```python
# PDF Configuration
PDF_DIRECTORY = "/Users/apple/VSF/pageindex/document"  # Thay đổi đường dẫn này
PDF_FILES = {
    "doc1.pdf": "VinFast VF8 User Manual",
    "doc2.pdf": "VinFast VF9 User Manual", 
    "doc3.pdf": "VinFast Safety Guide",
    "doc4.pdf": "VinFast Maintenance Guide"
}
```

### Bước 2: Đặt file PDF

Đảm bảo các file PDF có tên đúng trong thư mục:
- `doc1.pdf` → VinFast VF8 User Manual
- `doc2.pdf` → VinFast VF9 User Manual
- `doc3.pdf` → VinFast Safety Guide
- `doc4.pdf` → VinFast Maintenance Guide

### Bước 3: Sử dụng trong app

1. Chạy app: `python run_local.py`
2. Chọn **Mode 6: 📄 PDF Viewer** ở sidebar
3. Xem PDF trong các tab khác nhau

## 📋 Tùy chỉnh

### Thêm PDF mới

Trong `app.py`, thêm vào dictionary `PDF_FILES`:

```python
PDF_FILES = {
    "doc1.pdf": "VinFast VF8 User Manual",
    "doc2.pdf": "VinFast VF9 User Manual", 
    "doc3.pdf": "VinFast Safety Guide",
    "doc4.pdf": "VinFast Maintenance Guide",
    "doc5.pdf": "Tên file PDF mới của bạn"  # Thêm dòng này
}
```

### Đổi đường dẫn thư mục

```python
PDF_DIRECTORY = "/path/to/your/pdf/folder"
```

### Đổi tên hiển thị

Chỉnh sửa value trong dictionary `PDF_FILES`:

```python
"doc1.pdf": "Tên mới cho doc1"
```

## 🎨 Tính năng

- ✅ **Tabs**: Mỗi PDF hiển thị trong tab riêng
- ✅ **Embedded viewer**: PDF hiển thị trực tiếp trong app
- ✅ **Responsive**: Tự động điều chỉnh kích thước
- ✅ **Error handling**: Thông báo rõ ràng nếu thiếu file
- ✅ **Auto-detect**: Chỉ hiển thị các file PDF tồn tại

## 🔧 Troubleshooting

### Không thấy PDF?

1. Kiểm tra đường dẫn `PDF_DIRECTORY` đúng chưa
2. Kiểm tra tên file trong `PDF_FILES` khớp với file thực tế
3. Kiểm tra quyền đọc file PDF

### PDF không hiển thị?

- Một số browser không hỗ trợ PDF embedded
- Thử browser khác (Chrome/Firefox khuyến nghị)
- Kiểm tra file PDF không bị lỗi

### Hiển thị chậm?

- File PDF lớn sẽ load chậm hơn
- Cân nhắc giảm kích thước file PDF
- Hoặc chia nhỏ thành nhiều file

## 📝 Ví dụ cấu hình

### Cấu hình cho Windows:

```python
PDF_DIRECTORY = "C:/Users/YourName/Documents/PDFs"
```

### Cấu hình cho Linux/Mac:

```python
PDF_DIRECTORY = "/home/username/documents/pdfs"
# hoặc
PDF_DIRECTORY = "/Users/username/Documents/PDFs"
```

### Cấu hình tương đối:

```python
PDF_DIRECTORY = "./pdfs"  # Thư mục pdfs trong cùng folder với app.py
```

## 🚀 Demo

```bash
# 1. Đặt PDF vào thư mục
mkdir -p pdfs
cp your_document.pdf pdfs/doc1.pdf

# 2. Chỉnh sửa app.py
PDF_DIRECTORY = "./pdfs"

# 3. Chạy app
python run_local.py

# 4. Chọn Mode 6 trong app
```

## 🎯 Kết quả

Bạn sẽ thấy:
- Tabs với tên của từng PDF
- PDF viewer embedded
- Có thể scroll, zoom trong PDF
- Chuyển đổi giữa các PDF dễ dàng

---

**Lưu ý**: Đảm bảo file PDF không quá lớn (< 10MB) để tốc độ load tốt.
