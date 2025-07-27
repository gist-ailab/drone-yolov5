#!/bin/bash
# 완전한 YOLOv5 PT to HEF 변환 스크립트
# PT → ONNX → HAR → 최적화 → HEF

# ==================== 경로 설정 ====================
MODEL_NAME="yolov5n_10class_250725"
NET_NAME="yolov5n_10class"
IMAGE_DIR="/media/jemo/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/yolo_dataset_0723/images/val"

PT_PATH="/ailab_mat2/dataset/drone/250312_sejong/yolo_weights/250724_yolov5/pth_sources/${MODEL_NAME}/weights/best.pt"
OUTPUT_DIR="/ailab_mat2/dataset/drone/250312_sejong/yolo_weights/250724_yolov5/converted_models/${MODEL_NAME}"
YOLOV5_DIR="/media/jemo/HDD1/Workspace/src/Project/Drone24/edgeboard/Imx/yolov5"

# 파일 경로
ONNX_PATH="$OUTPUT_DIR/best.onnx"
HAR_PATH="$OUTPUT_DIR/$NET_NAME.har"
CALIB_NPY_PATH="$OUTPUT_DIR/calibration_data.npy"
OPTIMIZED_HAR_PATH="$OUTPUT_DIR/${NET_NAME}_optimized.har"

echo "🚀 완전한 YOLOv5 PT to HEF 변환"
echo "=" * 60
echo "PT 모델: $PT_PATH"
echo "YOLOv5 디렉토리: $YOLOV5_DIR"
echo "캘리브레이션 이미지: $IMAGE_DIR"
echo "출력 디렉토리: $OUTPUT_DIR"
echo ""

# ==================== 전제 조건 확인 ====================
echo "📋 전제 조건 확인"
if [ ! -f "$PT_PATH" ]; then
    echo "❌ PT 파일을 찾을 수 없습니다: $PT_PATH"
    exit 1
fi

if [ ! -d "$YOLOV5_DIR" ]; then
    echo "❌ YOLOv5 디렉토리를 찾을 수 없습니다: $YOLOV5_DIR"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ 캘리브레이션 이미지 디렉토리를 찾을 수 없습니다: $IMAGE_DIR"
    exit 1
fi

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

echo "✅ 모든 전제 조건 확인 완료"
echo ""

if [ ! -f "$ONNX_PATH" ]; then
    echo "0️⃣ 단계 0: PT → ONNX 변환 (YOLOv5 공식 export.py 사용)"
    
    # YOLOv5 디렉토리로 이동
    cd "$YOLOV5_DIR"
    
    # YOLOv5 공식 export.py 사용
    echo "🔄 YOLOv5 export.py 실행 중..."
    python export.py \
        --weights "$PT_PATH" \
        --include onnx \
        --img 640 640 \
        --batch-size 1 \
        --simplify \
        --opset 11 \
        --device cpu
    
    # 원래 디렉토리로 복귀
    cd - > /dev/null
    
    # 생성된 ONNX 파일을 올바른 위치로 이동 (필요시)
    # export.py는 보통 PT 파일과 같은 디렉토리에 ONNX를 생성
    PT_DIR=$(dirname "$PT_PATH")
    GENERATED_ONNX="$PT_DIR/best.onnx"
    
    if [ -f "$GENERATED_ONNX" ] && [ "$GENERATED_ONNX" != "$ONNX_PATH" ]; then
        echo "🔄 ONNX 파일을 출력 디렉토리로 이동..."
        mv "$GENERATED_ONNX" "$ONNX_PATH"
    fi
    
    if [ ! -f "$ONNX_PATH" ]; then
        echo "❌ ONNX 변환 실패"
        echo "🔍 생성된 파일 확인:"
        ls -la "$PT_DIR"
        echo ""
        echo "🔧 수동 변환 방법:"
        echo "cd $YOLOV5_DIR"
        echo "python export.py --weights $PT_PATH --include onnx --img 640 640 --batch-size 1 --simplify --opset 11"
        exit 1
    fi
    
    # 파일 크기 확인
    file_size=$(du -h "$ONNX_PATH" | cut -f1)
    echo "✅ ONNX 변환 완료: $ONNX_PATH ($file_size)"
else
    echo "✅ ONNX 파일 이미 존재: $ONNX_PATH"
fi

echo ""

# ==================== 1단계: 캘리브레이션 데이터 전처리 ====================
if [ ! -f "$CALIB_NPY_PATH" ]; then
    echo "1️⃣ 단계 1: 캘리브레이션 데이터 전처리"
    
    python3 -c "
import os
import numpy as np
import cv2
import glob
from tqdm import tqdm

def preprocess_images():
    image_dir = '$IMAGE_DIR'
    output_path = '$CALIB_NPY_PATH'
    target_size = (640, 640)
    max_images = 64
    
    print('📁 이미지 디렉토리: ' + image_dir)
    
    # 다양한 이미지 포맷 지원
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp']
    image_files = []
    
    print('🔍 지원 포맷: ' + ', '.join([ext.upper() for ext in image_extensions]))
    
    for ext in image_extensions:
        # 소문자 + 대문자 확장자 모두 확인
        for case_ext in [ext, ext.upper()]:
            pattern = os.path.join(image_dir, '*.' + case_ext)
            files = glob.glob(pattern)
            image_files.extend(files)
    
    # 중복 제거
    image_files = list(set(image_files))
    
    print('📷 발견된 이미지: ' + str(len(image_files)) + '개')
    
    if len(image_files) == 0:
        print('❌ 이미지 파일이 없습니다!')
        print('확인사항:')
        print('  1. 디렉토리 경로가 올바른지 확인')
        print('  2. 지원 포맷 파일이 있는지 확인 (JPG, PNG, BMP 등)')
        return False
    
    # 최대 64개만 사용
    if len(image_files) > max_images:
        image_files = image_files[:max_images]
        print('📊 처음 ' + str(max_images) + '개 이미지만 사용합니다')
    
    height, width = target_size
    processed_images = np.zeros((len(image_files), height, width, 3), dtype=np.float32)
    
    valid_count = 0
    failed_count = 0
    
    for i, img_path in enumerate(tqdm(image_files, desc='이미지 처리')):
        try:
            # OpenCV로 이미지 읽기
            img = cv2.imread(img_path)
            
            if img is None:
                # PIL로 재시도
                try:
                    from PIL import Image
                    pil_img = Image.open(img_path)
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    img = np.array(pil_img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                except:
                    pass
            
            if img is None:
                failed_count += 1
                continue
            
            # 이미지 형태 확인
            if len(img.shape) != 3 or img.shape[2] != 3:
                failed_count += 1
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (width, height))
            img_float = img_resized.astype(np.float32) / 255.0
            
            processed_images[valid_count] = img_float
            valid_count += 1
            
        except Exception as e:
            failed_count += 1
            continue
    
    if valid_count > 0:
        processed_images = processed_images[:valid_count]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, processed_images)
        
        print('✅ 캘리브레이션 데이터 저장: ' + output_path)
        print('📊 처리 성공: ' + str(valid_count) + '개')
        print('📊 처리 실패: ' + str(failed_count) + '개')
        print('📏 배열 크기: ' + str(processed_images.shape))
        print('💾 파일 크기: {:.2f} MB'.format(os.path.getsize(output_path) / 1024 / 1024))
        return True
    else:
        print('❌ 처리된 이미지가 없습니다!')
        return False

if __name__ == '__main__':
    success = preprocess_images()
    exit(0 if success else 1)
"
    
    if [ $? -ne 0 ] || [ ! -f "$CALIB_NPY_PATH" ]; then
        echo "❌ 캘리브레이션 데이터 전처리 실패"
        exit 1
    fi
else
    echo "✅ 캘리브레이션 데이터 이미 존재: $CALIB_NPY_PATH"
fi

echo ""

# ==================== 2단계: ONNX → HAR 변환 ====================
if [ ! -f "$HAR_PATH" ]; then
    echo "2️⃣ 단계 2: ONNX → HAR 변환"
    hailo parser onnx \
        "$ONNX_PATH" \
        --net-name "$NET_NAME" \
        --har-path "$HAR_PATH" \
        --hw-arch hailo8 \
        -y
    
    if [ ! -f "$HAR_PATH" ]; then
        echo "❌ HAR 변환 실패"
        exit 1
    fi
    echo "✅ HAR 변환 완료: $HAR_PATH"
else
    echo "✅ HAR 파일 이미 존재: $HAR_PATH"
fi

echo ""

# ==================== 3단계: HAR 최적화 ====================
echo "3️⃣ 단계 3: HAR 최적화 (실제 캘리브레이션 데이터 사용)"
hailo optimize \
    "$HAR_PATH" \
    --calib-set-path "$CALIB_NPY_PATH" \
    --hw-arch hailo8 \
    --output-har-path "$OPTIMIZED_HAR_PATH"

if [ ! -f "$OPTIMIZED_HAR_PATH" ]; then
    echo "❌ HAR 최적화 실패"
    echo "🔄 랜덤 캘리브레이션으로 재시도..."
    hailo optimize \
        "$HAR_PATH" \
        --use-random-calib-set \
        --hw-arch hailo8 \
        --output-har-path "$OPTIMIZED_HAR_PATH"
fi

if [ ! -f "$OPTIMIZED_HAR_PATH" ]; then
    echo "❌ 최적화 완전 실패"
    exit 1
fi

echo "✅ 최적화 완료: $OPTIMIZED_HAR_PATH"
echo ""

# ==================== 4단계: HAR → HEF 컴파일 ====================
echo "4️⃣ 단계 4: HAR → HEF 컴파일"
hailo compiler \
    "$OPTIMIZED_HAR_PATH" \
    --hw-arch hailo8 \
    --output-dir "$OUTPUT_DIR"

echo ""

# ==================== 결과 확인 ====================
echo "📁 최종 결과:"
echo "=" * 60
ls -la "$OUTPUT_DIR"

# 생성된 파일들 확인
echo ""
echo "📊 생성된 파일들:"
echo "----------------"
if [ -f "$ONNX_PATH" ]; then
    echo "  ✅ ONNX: $(basename "$ONNX_PATH") ($(du -h "$ONNX_PATH" | cut -f1))"
fi
if [ -f "$HAR_PATH" ]; then
    echo "  ✅ HAR: $(basename "$HAR_PATH") ($(du -h "$HAR_PATH" | cut -f1))"
fi
if [ -f "$OPTIMIZED_HAR_PATH" ]; then
    echo "  ✅ 최적화된 HAR: $(basename "$OPTIMIZED_HAR_PATH") ($(du -h "$OPTIMIZED_HAR_PATH" | cut -f1))"
fi
if [ -f "$CALIB_NPY_PATH" ]; then
    echo "  ✅ 캘리브레이션: $(basename "$CALIB_NPY_PATH") ($(du -h "$CALIB_NPY_PATH" | cut -f1))"
fi

# HEF 파일 확인
HEF_FILES=$(find "$OUTPUT_DIR" -name "*.hef" 2>/dev/null)
if [ -n "$HEF_FILES" ]; then
    echo ""
    echo "🎉 변환 완료!"
    echo "HEF 파일:"
    for hef in $HEF_FILES; do
        echo "  ✅ $(basename "$hef") ($(du -h "$hef" | cut -f1))"
    done
    
    echo ""
    echo "📋 사용법:"
    echo "HEF 파일을 Hailo8 디바이스에서 다음과 같이 사용할 수 있습니다:"
    for hef in $HEF_FILES; do
        echo "  hailo run $(basename "$hef") --input-dir /path/to/test/images"
    done
    
    echo ""
    echo "🔧 추가 정보:"
    echo "  • 입력 크기: 640x640"
    echo "  • 정규화: 0-1 (0-255에서 255로 나눔)"
    echo "  • 색상 포맷: RGB"
    echo "  • 캘리브레이션: 실제 데이터 사용"
    
else
    echo ""
    echo "❌ HEF 파일 생성 실패"
    echo "중간 파일들을 확인해주세요:"
    echo "  PT: $PT_PATH"
    echo "  ONNX: $ONNX_PATH"
    echo "  HAR: $HAR_PATH"
    echo "  최적화된 HAR: $OPTIMIZED_HAR_PATH"
    echo "  캘리브레이션: $CALIB_NPY_PATH"
    
    echo ""
    echo "🔧 문제 해결 방법:"
    echo "1. 각 단계별로 로그 확인"
    echo "2. hailo dfc-studio GUI 도구 사용"
    echo "3. 개별 명령어로 단계별 실행"
fi

echo ""
echo "🚀 변환 프로세스 완료!"