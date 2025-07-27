import os
import random
from pathlib import Path
import glob
from PIL import Image

def create_calibration_dataset_png_to_jpg(source_dir, target_dir, num_samples=300, seed=42, jpg_quality=95):
    """
    지정된 디렉토리에서 랜덤하게 PNG 이미지를 선택하여 JPG로 변환한 캘리브레이션 데이터셋을 생성합니다.
    
    Args:
        source_dir (str): 원본 PNG 이미지가 있는 디렉토리 경로
        target_dir (str): 캘리브레이션 JPG 이미지를 저장할 디렉토리 경로
        num_samples (int): 선택할 이미지 수 (기본값: 300)
        seed (int): 랜덤 시드 (재현 가능한 결과를 위해)
        jpg_quality (int): JPG 품질 (1-100, 기본값: 95)
    """
    
    # 랜덤 시드 설정 (재현 가능한 결과를 위해)
    random.seed(seed)
    
    # PNG 이미지 파일 찾기
    png_extensions = ['*.png', '*.PNG']
    
    # 원본 디렉토리에서 모든 PNG 파일 찾기
    all_png_images = []
    for ext in png_extensions:
        pattern = os.path.join(source_dir, ext)
        all_png_images.extend(glob.glob(pattern))
    
    print(f"원본 디렉토리에서 {len(all_png_images)}개의 PNG 이미지를 찾았습니다.")
    
    # 이미지가 충분한지 확인
    if len(all_png_images) < num_samples:
        print(f"경고: 요청한 샘플 수({num_samples})가 사용 가능한 PNG 이미지 수({len(all_png_images)})보다 많습니다.")
        num_samples = len(all_png_images)
        print(f"샘플 수를 {num_samples}로 조정합니다.")
    
    # 랜덤하게 이미지 선택
    selected_images = random.sample(all_png_images, num_samples)
    print(f"{num_samples}개의 PNG 이미지를 랜덤하게 선택했습니다.")
    
    # 타겟 디렉토리 생성
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    print(f"타겟 디렉토리 생성: {target_dir}")
    
    # 선택된 PNG 이미지들을 JPG로 변환하여 저장
    converted_count = 0
    failed_count = 0
    
    for img_path in selected_images:
        try:
            # 파일명에서 확장자 제거하고 .jpg로 변경
            img_filename = os.path.basename(img_path)
            img_name_without_ext = os.path.splitext(img_filename)[0]
            jpg_filename = f"{img_name_without_ext}.jpg"
            target_path = os.path.join(target_dir, jpg_filename)
            
            # PNG 이미지 열기
            with Image.open(img_path) as img:
                # RGBA 또는 투명도가 있는 이미지를 RGB로 변환
                if img.mode in ('RGBA', 'LA', 'P'):
                    # 흰색 배경 생성
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    
                    if img.mode == 'P':
                        # 팔레트 모드를 RGBA로 변환
                        img = img.convert('RGBA')
                    
                    if img.mode in ('RGBA', 'LA'):
                        # 투명도 채널을 마스크로 사용하여 합성
                        background.paste(img, mask=img.split()[-1])
                    else:
                        background.paste(img)
                    
                    img = background
                elif img.mode != 'RGB':
                    # 다른 모드는 RGB로 변환
                    img = img.convert('RGB')
                
                # JPG로 저장
                img.save(target_path, 'JPEG', quality=jpg_quality, optimize=True)
                
            converted_count += 1
            
            # 진행상황 출력 (50개마다)
            if converted_count % 50 == 0:
                print(f"진행상황: {converted_count}/{num_samples} 완료")
                
        except Exception as e:
            print(f"이미지 변환 실패: {img_path} -> {e}")
            failed_count += 1
    
    print(f"\n=== 작업 완료 ===")
    print(f"성공적으로 변환된 이미지: {converted_count}개")
    print(f"변환 실패한 이미지: {failed_count}개")
    print(f"타겟 디렉토리: {target_dir}")
    print(f"JPG 품질: {jpg_quality}")
    
    return converted_count, failed_count

def verify_converted_images(target_dir):
    """변환된 JPG 이미지들을 검증합니다."""
    jpg_files = glob.glob(os.path.join(target_dir, "*.jpg"))
    print(f"\n=== 변환 결과 검증 ===")
    print(f"생성된 JPG 파일 수: {len(jpg_files)}개")
    
    # 몇 개 샘플 이미지 정보 출력
    for i, jpg_file in enumerate(jpg_files[:5]):
        try:
            with Image.open(jpg_file) as img:
                file_size = os.path.getsize(jpg_file) / 1024  # KB
                print(f"  {i+1}. {os.path.basename(jpg_file)}: {img.size} {img.mode} ({file_size:.1f}KB)")
        except Exception as e:
            print(f"  {i+1}. {os.path.basename(jpg_file)}: 검증 실패 - {e}")
    
    if len(jpg_files) > 5:
        print(f"  ... 외 {len(jpg_files) - 5}개 파일")

def main():
    # 설정값들
    source_directory = "/media/jemo/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/yolo_3class_dataset_0723/images/val"
    target_directory = "/media/jemo/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/yolo_3class_dataset_0723/images/calibration_data"
    sample_count = 300
    jpg_quality = 95  # JPG 품질 (95는 고품질)
    
    print("=== PNG → JPG 캘리브레이션 데이터셋 생성기 ===")
    print(f"원본 디렉토리: {source_directory}")
    print(f"타겟 디렉토리: {target_directory}")
    print(f"선택할 이미지 수: {sample_count}")
    print(f"JPG 품질: {jpg_quality}")
    print("=" * 60)
    
    # 원본 디렉토리 존재 확인
    if not os.path.exists(source_directory):
        print(f"❌ 오류: 원본 디렉토리가 존재하지 않습니다: {source_directory}")
        return
    
    # Pillow 라이브러리 확인
    try:
        from PIL import Image
        print("✅ PIL(Pillow) 라이브러리 확인 완료")
    except ImportError:
        print("❌ 오류: PIL(Pillow) 라이브러리가 설치되어 있지 않습니다.")
        print("다음 명령어로 설치하세요: pip install Pillow")
        return
    
    # 캘리브레이션 데이터셋 생성
    try:
        converted, failed = create_calibration_dataset_png_to_jpg(
            source_dir=source_directory,
            target_dir=target_directory,
            num_samples=sample_count,
            seed=42,  # 재현 가능한 결과를 위한 시드
            jpg_quality=jpg_quality
        )
        
        if converted > 0:
            print(f"\n✅ PNG → JPG 캘리브레이션 데이터셋이 성공적으로 생성되었습니다!")
            print(f"📂 위치: {target_directory}")
            print(f"📊 변환된 이미지 수: {converted}개")
            
            # 변환 결과 검증
            verify_converted_images(target_directory)
            
            print(f"\n🚀 다음 명령어로 Hailo 최적화를 진행하세요:")
            print(f"hailo optimize best.har --calib-set-path {target_directory} --hw-arch hailo8")
            
        else:
            print(f"\n❌ 캘리브레이션 데이터셋 생성에 실패했습니다.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()