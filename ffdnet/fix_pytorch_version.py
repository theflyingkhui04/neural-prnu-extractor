import os
import re

def fix_train_file():
    """Fix PyTorch version compatibility and indentation issues in train.py"""
    # Sử dụng đường dẫn động để hoạt động cả trên local và Kaggle
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_file = os.path.join(base_dir, "ffdnet", "train.py")
    
    print(f"Đang sửa file: {train_file}")
    
    try:
        with open(train_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 1. Sửa vấn đề autocast
        if "autocast(device_type='cuda')" in content:
            content = content.replace("autocast(device_type='cuda')", "torch.cuda.amp.autocast()")
        
        # 2. Sửa vấn đề import
        if "from torch.cuda.amp import autocast" in content:
            content = content.replace("from torch.cuda.amp import autocast", "# from torch.cuda.amp import autocast")
            
            # Thêm import torch nếu chưa có
            if "import torch" not in content:
                content = "import torch\n" + content
        
        # 3. Sửa vấn đề thụt đầu dòng trong khối try-except
        content = re.sub(r'try:\s*\nexcept', 'try:\n    pass\nexcept', content)
        
        # 4. Sửa vấn đề indentation cho dòng "for idx in range(2):"
        content = re.sub(r'(\s*)for idx in range\(2\):', r'  for idx in range(2):', content)
        
        # 5. Đảm bảo đúng thụt đầu dòng trong toàn bộ file
        # Tách thành từng dòng để xử lý
        lines = content.split('\n')
        fixed_lines = []
        
        # Fix các block code phổ biến
        in_function = False
        indentation_level = 0
        
        for line in lines:
            # Bỏ qua dòng trống
            if not line.strip():
                fixed_lines.append(line)
                continue
                
            # Xử lý đặc biệt cho khối with trong train()
            if "with torch.cuda.amp.autocast():" in line or "with autocast():" in line:
                # Đảm bảo khối with có đúng thụt đầu dòng
                fixed_line = ' ' * (indentation_level * 2) + line.strip()
                fixed_lines.append(fixed_line)
                indentation_level += 1
                continue
                
            # Kiểm tra các dòng kết thúc block
            if line.strip() in ['else:', 'elif:', 'except:', 'finally:', 'except ImportError:']:
                if indentation_level > 0:
                    indentation_level -= 1
                    
            # Giữ nguyên thụt đầu dòng của dòng hiện tại
            # Đây là chiến lược an toàn nhất để tránh phá vỡ code
            fixed_lines.append(line)
            
            # Tăng indentation cho dòng tiếp theo nếu cần
            if line.strip().endswith(':') and not line.strip().startswith(('#', '"""', "'''")):
                indentation_level += 1
                
        # Ghép lại thành nội dung hoàn chỉnh
        content = '\n'.join(fixed_lines)
        
        # Lưu file đã sửa
        with open(train_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        print("✅ Đã sửa xong vấn đề compatibility và indentation trong train.py")
        
    except Exception as e:
        print(f"❌ Lỗi khi sửa file train.py: {e}")

if __name__ == "__main__":
    fix_train_file()