# convert_png_to_jpg.py
import random
import os
from PIL import Image
from pathlib import Path

# 경로 설정
val_dir = "/media/jemo/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/yolo_dataset/images/val"
calib_dir = "./calibration_data"

# 캘리브레이션 디렉터리 생성
os.makedirs(calib_dir, exist_ok=True)

# PNG 파일 목록 가져오기
png_files = list(Path(val_dir).glob("*.png"))
print(f"전체 PNG 파일: {len(png_files)}개")

# 랜덤하게 300개 선택
selected_files = random.sample(png_files, 300)
print(f"선택된 파일: {len(selected_files)}개")

# PNG → JPG 변환
for i, png_file in enumerate(selected_files):
    try:
        # PNG 이미지 열기
        img = Image.open(png_file)
        
        # RGBA → RGB 변환 (투명도 제거)
        if img.mode in ('RGBA', 'LA', 'P'):
            # 흰색 배경으로 변환
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        
        # JPG로 저장
        jpg_path = Path(calib_dir) / f"{png_file.stem}.jpg"
        img.save(jpg_path, 'JPEG', quality=95)
        
        if (i + 1) % 50 == 0:
            print(f"진행상황: {i + 1}/300 완료")
            
    except Exception as e:
        print(f"변환 실패: {png_file} -> {e}")

print("✅ 변환 완료!")