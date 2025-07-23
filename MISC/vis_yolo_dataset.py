import cv2
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
import matplotlib.cm as cm # 컬러맵 임포트

def visualize_yolo_dataset(yaml_path):
    """
    YOLOv5 데이터셋을 시각화합니다.
    - 키보드 좌우 화살표 또는 'a', 'd' 키로 이전/다음 프레임 이동
    - Matplotlib 슬라이더로 원하는 프레임으로 바로 이동
    - 클래스 인덱스별 다른 색상 적용
    """
    try:
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"오류: YAML 파일을 찾을 수 없습니다: {yaml_path}")
        return
    except yaml.YAMLError as e:
        print(f"오류: YAML 파일을 파싱하는 중 오류 발생: {e}")
        return

    # YAML 파일에서 정보 추출
    base_path = data_config.get('path')
    train_images_relative = data_config.get('train')
    val_images_relative = data_config.get('val')
    class_names = data_config.get('names')
    nc = data_config.get('nc')

    if not all([base_path, train_images_relative, val_images_relative, class_names, nc is not None]):
        print("오류: YAML 파일에 'path', 'train', 'val', 'names', 'nc' 정보가 모두 포함되어야 합니다.")
        return

    # 이미지 및 라벨 경로 구성
    train_images_path = os.path.join(base_path, train_images_relative)
    val_images_path = os.path.join(base_path, val_images_relative)

    train_labels_path = os.path.join(base_path, train_images_relative.replace('images', 'labels'))
    val_labels_path = os.path.join(base_path, val_images_relative.replace('images', 'labels'))

    print(f"데이터셋 기본 경로: {base_path}")
    print(f"학습 이미지 경로: {train_images_path}")
    print(f"학습 라벨 경로: {train_labels_path}")
    print(f"검증 이미지 경로: {val_images_path}")
    print(f"검증 라벨 경로: {val_labels_path}")
    print(f"클래스 이름: {class_names}")
    print(f"클래스 개수 (nc): {nc}")

    # 클래스별 색상 생성
    # Matplotlib의 컬러맵을 사용하여 클래스 개수만큼 고유한 색상을 생성합니다.
    # 'tab10'은 최대 10가지 색상을 제공하며, 더 많은 클래스에는 'hsv'나 'jet' 같은 다른 컬러맵을 고려할 수 있습니다.
    colors = [cm.get_cmap('hsv', nc)(i) for i in range(nc)]
    # RGBA (0-1) 형식의 색상을 255 스케일의 RGB (0-1) 문자열로 변환하여 Matplotlib에 사용
    class_colors = {i: tuple(int(c * 255) for c in color[:3]) for i, color in enumerate(colors)}
    # Matplotlib의 Rectangle 및 Text에 사용할 HEX 또는 RGB 튜플 형식으로 변환
    class_colors_mpl = {i: color[:3] for i, color in enumerate(colors)} # 0-1 스케일의 RGB 튜플

    # 이미지 파일 목록 가져오기 (훈련 및 검증 세트 모두 포함)
    all_image_paths = []
    if os.path.exists(train_images_path):
        for img_name in sorted(os.listdir(train_images_path)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                all_image_paths.append(os.path.join(train_images_path, img_name))
    else:
        print(f"경고: 훈련 이미지 디렉토리를 찾을 수 없습니다: {train_images_path}")

    if os.path.exists(val_images_path):
        for img_name in sorted(os.listdir(val_images_path)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                all_image_paths.append(os.path.join(val_images_path, img_name))
    else:
        print(f"경고: 검증 이미지 디렉토리를 찾을 수 없습니다: {val_images_path}")

    if not all_image_paths:
        print("오류: 시각화할 이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    total_images = len(all_image_paths)
    current_idx = 0

    # Matplotlib 설정
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25) # 슬라이더를 위한 공간 확보

    # 슬라이더 추가
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
    slider = Slider(ax_slider, 'Image Index', 0, total_images - 1, valinit=0, valstep=1)

    def update_image(idx):
        nonlocal current_idx
        current_idx = int(idx)
        if current_idx < 0:
            current_idx = 0
        elif current_idx >= total_images:
            current_idx = total_images - 1

        img_path = all_image_paths[current_idx]
        
        # 라벨 파일 경로 유추
        label_filename = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        
        # 이미지 경로에 따라 라벨 경로 결정
        # 이전에 'images'를 'labels'로 바꾸는 로직을 사용했으므로, 그대로 유지
        if "train" in img_path:
            label_path = os.path.join(train_labels_path, label_filename)
        elif "val" in img_path:
            label_path = os.path.join(val_labels_path, label_filename)
        else:
            # 예상치 못한 경로일 경우, 라벨 경로를 찾지 못할 수 있음
            print(f"경고: '{img_path}'에 대한 라벨 경로를 유추할 수 없습니다. 라벨을 표시하지 않습니다.")
            label_path = None

        img = cv2.imread(img_path)
        if img is None:
            print(f"경고: 이미지를 로드할 수 없습니다: {img_path}")
            ax.clear()
            ax.text(0.5, 0.5, "Image Load Error", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f"Image {current_idx+1}/{total_images}")
            plt.draw()
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        ax.clear()
        ax.imshow(img)
        ax.axis('off')

        title_text = f"[{current_idx+1}/{total_images}] {os.path.basename(img_path)}\n"
        has_labels = False

        if label_path and os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = f.readlines()
            
            for label in labels:
                try:
                    parts = list(map(float, label.strip().split()))
                    class_id = int(parts[0])
                    x_center, y_center, bbox_width, bbox_height = parts[1:]

                    # YOLO 포맷 (정규화된 xywh)을 픽셀 좌표 (xywh)로 변환
                    x_tl = int((x_center - bbox_width / 2) * w)
                    y_tl = int((y_center - bbox_height / 2) * h)
                    bbox_w_px = int(bbox_width * w)
                    bbox_h_px = int(bbox_height * h)

                    # 클래스 이름 가져오기
                    class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"Unknown({class_id})"
                    
                    # 클래스별 색상 선택
                    color_to_use = class_colors_mpl.get(class_id, (1, 1, 1)) # 기본값은 흰색 (RGB 0-1)

                    # 바운딩 박스 그리기
                    rect = Rectangle((x_tl, y_tl), bbox_w_px, bbox_h_px,
                                     linewidth=2, edgecolor=color_to_use, facecolor='none')
                    ax.add_patch(rect)
                    
                    # 라벨 텍스트 추가 (클래스 인덱스와 이름 표시)
                    text_label = f"{class_id}: {class_name}"
                    ax.text(x_tl, y_tl - 10, text_label, color='white', fontsize=10, 
                            bbox=dict(facecolor=color_to_use, alpha=0.7, edgecolor='none', pad=1))
                    has_labels = True

                except (ValueError, IndexError):
                    print(f"경고: 라벨 파일 {label_path}에서 잘못된 라인 형식: {label.strip()}")
        
        if not has_labels:
            title_text += "(No Objects Detected or Label File Missing)"

        ax.set_title(title_text)
        plt.draw() # 변경사항 즉시 반영

    def on_key_press(event):
        nonlocal current_idx
        if event.key == 'right' or event.key == 'd':
            current_idx = (current_idx + 1) % total_images
            slider.set_val(current_idx)
        elif event.key == 'left' or event.key == 'a':
            current_idx = (current_idx - 1 + total_images) % total_images
            slider.set_val(current_idx)
        
        # update_image(current_idx) # set_val이 update_image를 호출하므로 불필요

    slider.on_changed(update_image)
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # 초기 이미지 로드
    update_image(current_idx)
    plt.show()

# --- 사용 예시 ---
if __name__ == "__main__":
    # 당신의 data.yaml 파일 경로를 여기에 정확하게 입력하세요.
    # 이전 단계에서 생성한 'yolo_dataset_0723'의 data.yaml 경로를 권장합니다.
    yaml_file_path = '/media/jemo/HDD1/Workspace/src/Project/Drone24/edgeboard/Imx/yolov5/data/sejong_new_3class_0723.yaml'

    # 주의: 위 yaml_file_path는 예시이므로, 실제 data.yaml 파일이 생성된 경로에 맞게 수정해주세요.
    # 만약 기존 10-클래스 데이터셋을 시각화하고 싶다면:
    # yaml_file_path = '/media/jemo/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/yolo_dataset/data.yaml'
    
    # 3-클래스 데이터셋을 시각화하고 싶고, 만약 data.yaml 파일 이름이
    # 'sejong_new_3class_0724.yaml'이고 경로가
    # '/media/jemo/HDD1/Workspace/src/Project/Drone24/edgeboard/Imx/yolov5/data/'라면:
    # yaml_file_path = '/media/jemo/HDD1/Workspace/src/Project/Drone24/edgeboard/Imx/yolov5/data/sejong_new_3class_0724.yaml'


    visualize_yolo_dataset(yaml_file_path)