#!/usr/bin/env python3
"""
YOLOv5 PT 모델을 Hailo8 HEF 형식으로 변환하는 스크립트
단계: PT -> ONNX -> HEF
"""

import torch
import os
import sys
import subprocess
import yaml
from pathlib import Path

# 모델 경로 설정
MODEL_PATH = "/ailab_mat2/dataset/drone/250312_sejong/yolo_weights/250724_yolov5/pth_sources/yolov5m_3class_250725/weights/best.pt"
OUTPUT_DIR = "/ailab_mat2/dataset/drone/250312_sejong/yolo_weights/250724_yolov5/converted_models"

class YOLOv5ToHEFConverter:
    def __init__(self, model_path, output_dir):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일명에서 확장자 제거
        self.model_name = Path(model_path).stem
        self.onnx_path = self.output_dir / f"{self.model_name}.onnx"
        self.hef_path = self.output_dir / f"{self.model_name}.hef"
        
    def step1_pt_to_onnx(self):
        """1단계: PT 모델을 ONNX로 변환"""
        print("=" * 50)
        print("Step 1: PT 모델을 ONNX로 변환")
        print("=" * 50)
        
        try:
            # YOLOv5 리포지토리 경로 확인 및 추가
            import sys
            yolov5_paths = [
                '/media/jemo/HDD1/Workspace/src/Project/Drone24/edgeboard/Imx/yolov5',
                './yolov5',
                '../yolov5',
                '/content/yolov5'  # Colab용
            ]
            
            yolov5_path = None
            for path in yolov5_paths:
                if os.path.exists(path):
                    yolov5_path = path
                    break
            
            if yolov5_path:
                sys.path.insert(0, yolov5_path)
                print(f"YOLOv5 경로 추가: {yolov5_path}")
            
            # YOLOv5 모델 로드 (개선된 방법)
            try:
                # 방법 1: YOLOv5 모듈을 사용한 로드
                import models
                model = torch.load(self.model_path, map_location='cpu')
                if isinstance(model, dict):
                    if 'model' in model:
                        model = model['model']
                    elif 'ema' in model:
                        model = model['ema']
            except:
                # 방법 2: state_dict만 사용
                print("YOLOv5 모듈을 찾을 수 없습니다. torch.jit.load를 시도합니다...")
                try:
                    model = torch.jit.load(self.model_path, map_location='cpu')
                except:
                    print("torch.jit.load 실패. 기본 torch.load를 시도합니다...")
                    checkpoint = torch.load(self.model_path, map_location='cpu')
                    
                    # 체크포인트에서 모델 추출
                    if isinstance(checkpoint, dict):
                        if 'model' in checkpoint:
                            model = checkpoint['model'].float()
                        elif 'ema' in checkpoint:
                            model = checkpoint['ema'].float()
                        else:
                            raise Exception("모델을 찾을 수 없습니다.")
                    else:
                        model = checkpoint
            
            # 모델을 evaluation 모드로 설정
            model.eval()
            model.float()  # FP32로 변환
            
            # 입력 크기 설정 (일반적으로 640x640)
            input_size = (1, 3, 640, 640)
            dummy_input = torch.randn(input_size)
            
            # 모델이 함수 호출 가능한지 확인
            try:
                with torch.no_grad():
                    test_output = model(dummy_input)
                print(f"모델 테스트 성공. 출력 형태: {[out.shape if hasattr(out, 'shape') else type(out) for out in test_output] if isinstance(test_output, (list, tuple)) else test_output.shape}")
            except Exception as e:
                print(f"모델 테스트 실패: {e}")
                # 모델 구조 출력
                print("모델 타입:", type(model))
                if hasattr(model, 'modules'):
                    print("모델 모듈 수:", len(list(model.modules())))
            
            # ONNX로 내보내기
            torch.onnx.export(
                model,
                dummy_input,
                str(self.onnx_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['images'],  # YOLOv5는 'images' 사용
                output_names=['output0', 'output1', 'output2'],  # YOLOv5 다중 출력
                dynamic_axes={
                    'images': {0: 'batch_size'},
                    'output0': {0: 'batch_size'},
                    'output1': {0: 'batch_size'},
                    'output2': {0: 'batch_size'}
                }
            )
            
            print(f"✅ ONNX 변환 완료: {self.onnx_path}")
            return True
            
        except Exception as e:
            print(f"❌ ONNX 변환 실패: {e}")
            return False
    
    def step2_create_yaml_config(self):
        """2단계: Hailo 변환을 위한 YAML 설정 파일 생성"""
        print("=" * 50)
        print("Step 2: Hailo 변환 설정 파일 생성")
        print("=" * 50)
        
        yaml_config = {
            'model_name': self.model_name,
            'onnx': str(self.onnx_path),
            'hw_arch': 'hailo8',
            'calibration': {
                'calib_type': 'uniform_noise',
                'calib_set_size': 64
            },
            'input_conversion': {
                'input': {
                    'input_1': {
                        'preprocessing': {
                            'input_format': 'RGB',
                            'normalization': {
                                'mean_values': [0.0, 0.0, 0.0],
                                'std_values': [255.0, 255.0, 255.0]
                            }
                        }
                    }
                }
            },
            'output_conversion': {
                'quantization': {
                    'output_1': {
                        'data_type': 'uint8'
                    }
                }
            }
        }
        
        yaml_path = self.output_dir / f"{self.model_name}_config.yaml"
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
        
        print(f"✅ 설정 파일 생성 완료: {yaml_path}")
        return yaml_path
    
    def step3_onnx_to_hef(self, yaml_config_path):
        """3단계: ONNX를 HEF로 변환 (Hailo DataFlow Compiler 사용)"""
        print("=" * 50)
        print("Step 3: ONNX를 HEF로 변환")
        print("=" * 50)
        
        # 방법 1: Hailo Model Zoo 사용 (권장)
        cmd_zoo = [
            'hailo_model_zoo', 'optimize', 'yolov5m',
            '--onnx', str(self.onnx_path),
            '--output-dir', str(self.output_dir),
            '--name', self.model_name
        ]
        
        # 방법 2: Hailo DFC 직접 사용
        cmd_dfc = [
            'hailo', 'compiler', 'compile',
            str(self.onnx_path),
            '--output-dir', str(self.output_dir)
        ]
        
        print(f"방법 1 (권장): {' '.join(cmd_zoo)}")
        print(f"방법 2 (대안): {' '.join(cmd_dfc)}")
        print("\n먼저 Hailo Model Zoo 방법을 시도합니다...")
        
        try:
            # 먼저 Hailo Model Zoo 시도
            result = subprocess.run(cmd_zoo, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ HEF 변환 완료 (Model Zoo): {self.output_dir}")
                return True
            else:
                print(f"❌ Model Zoo 변환 실패:")
                print(f"Error: {result.stderr}")
                print("\nHailo DFC 직접 사용을 시도합니다...")
                
                # Hailo DFC 직접 사용 시도
                result2 = subprocess.run(cmd_dfc, capture_output=True, text=True)
                
                if result2.returncode == 0:
                    print(f"✅ HEF 변환 완료 (DFC): {self.output_dir}")
                    return True
                else:
                    print(f"❌ DFC 변환도 실패:")
                    print(f"Error: {result2.stderr}")
                    return False
                
        except FileNotFoundError as e:
            print(f"❌ 명령어를 찾을 수 없습니다: {e}")
            print("Hailo 도구가 올바르게 설치되었는지 확인해주세요.")
            return False
    
    def convert(self):
        """전체 변환 프로세스 실행"""
        print(f"YOLOv5 모델 변환 시작: {self.model_path}")
        print(f"출력 디렉토리: {self.output_dir}")
        
        # Step 1: PT to ONNX
        if not self.step1_pt_to_onnx():
            return False
        
        # Step 2: Create config
        yaml_path = self.step2_create_yaml_config()
        
        # Step 3: ONNX to HEF
        if not self.step3_onnx_to_hef(yaml_path):
            print("\n📋 수동 변환 방법:")
            print("1. Hailo Model Zoo에서 유사한 YOLOv5 모델 찾기")
            print("2. hailo_model_zoo optimize 명령어 사용")
            print("3. 또는 Hailo DFC (DataFlow Compiler) GUI 도구 사용")
            return False
        
        print("\n🎉 변환 완료!")
        print(f"HEF 파일: {self.hef_path}")
        return True

def main():
    converter = YOLOv5ToHEFConverter(MODEL_PATH, OUTPUT_DIR)
    converter.convert()

if __name__ == "__main__":
    main()