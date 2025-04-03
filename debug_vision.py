import os
from pathlib import Path

# Kiểm tra thư mục đầu vào
input_path = '/kaggle/input/ai-real/datasets'
print(f"Checking directory: {input_path}")

# Liệt kê các file
files = list(Path(input_path).glob('**/*.jpg'))
if len(files) == 0:
    files = list(Path(input_path).glob('**/*.*'))

print(f"Found {len(files)} files")
for f in files[:5]:
    print(f"  {f}")
    
# Kiểm tra cấu trúc file names
if len(files) > 0:
    print("\nFilename examples:")
    for f in files[:3]:
        print(f"  {f.name}")