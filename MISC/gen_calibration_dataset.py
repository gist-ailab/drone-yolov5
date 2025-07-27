#!/usr/bin/env python3
"""
JPEG 이미지들을 Hailo 캘리브레이션용 .npy 파일로 변환
"""

import os
import numpy as np
import cv2
from pathlib import Path
import glob
from tqdm import tqdm

def preprocess_calibration_images(
    image_dir, 
    output_path, 
    target_size=(640, 640),
    max_images=300,
    normalize=True
):
    """
    JPEG 이미지들을 Hailo 캘리브레이션용 numpy 배열로 변환
    
    Args:
        image_dir: 이미지가 있는 디렉토리 경로
        output_path: 출력할 .npy 파일 경로
        target_size: 리사이즈할 크기 (height, width)
        max_images: 사용할 최대 이미지 개수
        normalize: 0-1로 정규화 여부
    """
    
    print(f"🔄 캘리브레이션 데이터 전처리 시작")
    print(f"📁 입력 디렉토리: {image_dir}")
    print(f"💾 출력 파일: {output_path}")
    print(f"📏 타겟 크기: {target_size}")
    print(f"📊 최대 이미지 수: {max_images}")
    
    # 이미지 파일 찾기 (다양한 포맷 지원)
    image_extensions = [
        'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp', 
        'ppm', 'pgm', 'pbm', 'sr', 'ras', 'dib'
    ]
    image_files = []
    
    print(f"🔍 지원하는 이미지 포맷: {', '.join(image_extensions)}")
    
    for ext in image_extensions:
        # 소문자 확장자
        pattern = os.path.join(image_dir, f'*.{ext}')
        found_files = glob.glob(pattern)
        image_files.extend(found_files)
        
        # 대문자 확장자
        pattern = os.path.join(image_dir, f'*.{ext.upper()}')
        found_files = glob.glob(pattern)
        image_files.extend(found_files)
        
        if found_files:
            print(f"  📁 {ext.upper()}: {len(found_files)}개 파일")
    
    # 중복 제거
    image_files = list(set(image_files))
    
    print(f"📷 발견된 이미지 수: {len(image_files)}")
    
    if len(image_files) == 0:
        print("❌ 이미지 파일을 찾을 수 없습니다!")
        return False
    
    # 이미지 개수 제한
    if len(image_files) > max_images:
        image_files = image_files[:max_images]
        print(f"📊 {max_images}개 이미지만 사용합니다.")
    
    # 전처리된 이미지들을 저장할 배열
    height, width = target_size
    processed_images = np.zeros((len(image_files), height, width, 3), dtype=np.float32)
    
    print("🔄 이미지 전처리 중...")
    
    valid_count = 0
    failed_files = []
    
    for i, img_path in enumerate(tqdm(image_files, desc="이미지 처리")):
        try:
            # 이미지 읽기 (다양한 방법 시도)
            img = None
            
            # 방법 1: OpenCV로 읽기
            img = cv2.imread(img_path)
            
            # 방법 2: PIL로 읽기 (OpenCV 실패시)
            if img is None:
                try:
                    from PIL import Image
                    pil_img = Image.open(img_path)
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    img = np.array(pil_img)
                    # PIL은 RGB, OpenCV는 BGR이므로 변환 필요
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                except ImportError:
                    print(f"⚠️ PIL을 사용할 수 없습니다. OpenCV만 사용합니다.")
                except Exception as e:
                    print(f"⚠️ PIL로 이미지 읽기 실패 {img_path}: {e}")
            
            if img is None:
                failed_files.append(img_path)
                print(f"⚠️ 이미지 읽기 실패: {os.path.basename(img_path)}")
                continue
            
            # 이미지 정보 확인
            original_height, original_width = img.shape[:2]
            if len(img.shape) != 3 or img.shape[2] != 3:
                print(f"⚠️ 지원하지 않는 이미지 형태 {img.shape}: {os.path.basename(img_path)}")
                failed_files.append(img_path)
                continue
            
            # BGR to RGB 변환
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 리사이즈
            img_resized = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
            
            # 데이터 타입 변환
            img_float = img_resized.astype(np.float32)
            
            # 정규화 (0-255 -> 0-1)
            if normalize:
                img_float = img_float / 255.0
            
            processed_images[valid_count] = img_float
            valid_count += 1
            
            # 처음 몇 개 이미지 정보 출력
            if valid_count <= 3:
                print(f"  📷 {os.path.basename(img_path)}: {original_width}x{original_height} -> {width}x{height}")
            
        except Exception as e:
            failed_files.append(img_path)
            print(f"⚠️ 이미지 처리 오류 {os.path.basename(img_path)}: {e}")
            continue
    
    # 유효한 이미지만 선택
    if valid_count > 0:
        processed_images = processed_images[:valid_count]
        
        # .npy 파일로 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, processed_images)
        
        print(f"\n✅ 캘리브레이션 데이터 저장 완료!")
        print(f"📊 총 이미지: {len(image_files)}개")
        print(f"📊 처리 성공: {valid_count}개")
        print(f"📊 처리 실패: {len(failed_files)}개")
        if failed_files:
            print(f"📊 실패한 파일들:")
            for fail_file in failed_files[:5]:  # 처음 5개만 표시
                print(f"     {os.path.basename(fail_file)}")
            if len(failed_files) > 5:
                print(f"     ... 및 {len(failed_files) - 5}개 더")
        
        print(f"📏 최종 배열 크기: {processed_images.shape}")
        print(f"💾 파일 경로: {output_path}")
        print(f"💾 파일 크기: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        print(f"📈 픽셀 값 범위: {processed_images.min():.3f} ~ {processed_images.max():.3f}")
        
        # 샘플 통계 출력
        mean_vals = processed_images.mean(axis=(0,1,2))
        std_vals = processed_images.std(axis=(0,1,2))
        print(f"📊 RGB 평균값: R={mean_vals[0]:.3f}, G={mean_vals[1]:.3f}, B={mean_vals[2]:.3f}")
        print(f"📊 RGB 표준편차: R={std_vals[0]:.3f}, G={std_vals[1]:.3f}, B={std_vals[2]:.3f}")
        
        return True
    else:
        print("❌ 처리된 이미지가 없습니다!")
        print("📋 확인사항:")
        print("  1. 이미지 디렉토리 경로가 올바른지 확인")
        print("  2. 지원하는 이미지 포맷인지 확인 (jpg, png, bmp 등)")
        print("  3. 이미지 파일이 손상되지 않았는지 확인")
        return False

def create_yolo_calibration_data():
    """YOLOv5용 캘리브레이션 데이터 생성"""
    
    # 경로 설정
    image_dir = "/media/jemo/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/yolo_3class_dataset_0723/images/calibration_data"
    output_dir = "/ailab_mat2/dataset/drone/250312_sejong/yolo_weights/250724_yolov5/converted_models"
    output_file = os.path.join(output_dir, "calibration_data.npy")
    
    # 디렉토리 확인
    if not os.path.exists(image_dir):
        print(f"❌ 이미지 디렉토리를 찾을 수 없습니다: {image_dir}")
        return False
    
    # 디렉토리 내용 확인
    files = os.listdir(image_dir)
    print(f"📁 디렉토리 내용 (처음 10개):")
    for file in files[:10]:
        print(f"  {file}")
    if len(files) > 10:
        print(f"  ... 및 {len(files) - 10}개 더")
    
    # 전처리 실행
    success = preprocess_calibration_images(
        image_dir=image_dir,
        output_path=output_file,
        target_size=(640, 640),  # YOLOv5 입력 크기
        max_images=64,           # Hailo 권장 캘리브레이션 크기
        normalize=True           # 0-1 정규화
    )
    
    if success:
        print(f"\n🎉 캘리브레이션 데이터 준비 완료!")
        print(f"📁 사용할 파일: {output_file}")
        print(f"\n다음 명령어에서 사용하세요:")
        print(f"hailo optimize \\")
        print(f"    /ailab_mat2/dataset/drone/250312_sejong/yolo_weights/250724_yolov5/converted_models/yolov5m_3class.har \\")
        print(f"    --calib-set-path {output_file} \\")
        print(f"    --hw-arch hailo8 \\")
        print(f"    --output-har-path /ailab_mat2/dataset/drone/250312_sejong/yolo_weights/250724_yolov5/converted_models/yolov5m_3class_optimized.har")
        
        return output_file
    
    return None

def main():
    print("=" * 70)
    print("Hailo 캘리브레이션 데이터 전처리기")
    print("=" * 70)
    
    create_yolo_calibration_data()

if __name__ == "__main__":
    main()