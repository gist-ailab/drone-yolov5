import os
import cv2
import numpy as np

def make_raw_vid_from_images_fixed(image_folder, output_file, max_images=500):
    """
    이미지들을 RAW 비디오 파일로 변환 (수정된 버전)
    max_images: 처리할 최대 이미지 수 (기본값: 500)
    """
    # Get all image files in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort images by filename to maintain order
    images.sort()
    
    # Limit to max_images
    if len(images) > max_images:
        images = images[:max_images]
        print(f"Limited to first {max_images} images out of {len(os.listdir(image_folder))} total images")

    if not images:
        print("No images found in the folder.")
        return

    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    print(f"Video dimensions: {width}x{height}")
    print(f"Total frames: {len(images)}")
    print(f"Output file path: {os.path.abspath(output_file)}")

    # Try different codecs for .raw extension
    success = False
    codecs_to_try = [
        ('I420', cv2.VideoWriter_fourcc(*'I420')),
        ('IYUV', cv2.VideoWriter_fourcc(*'IYUV')),
        ('YUY2', cv2.VideoWriter_fourcc(*'YUY2')),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
    ]
    
    for codec_name, fourcc in codecs_to_try:
        print(f"Trying codec: {codec_name}")
        out = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))
        
        if out.isOpened():
            print(f"Success with codec: {codec_name}")
            success = True
            break
        else:
            print(f"Failed with codec: {codec_name}")
            out.release()
    
    if not success:
        print("All codecs failed. Using binary raw method instead.")
        make_binary_raw_from_images(image_folder, output_file, max_images)
        return

    # Write frames
    for i, image in enumerate(images):
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        if frame is not None:
            out.write(frame)
        if i % 100 == 0:
            print(f"Processed {i+1}/{len(images)} frames")

    out.release()
    
    # Check if file was actually created
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"RAW video saved as {output_file}")
        print(f"File size: {file_size / (1024*1024):.2f} MB")
    else:
        print("File was not created. Trying binary raw method.")
        make_binary_raw_from_images(image_folder, output_file, max_images)

def make_binary_raw_from_images(image_folder, output_file, max_images=500):
    """
    이미지들을 순수 바이너리 RAW 파일로 변환
    max_images: 처리할 최대 이미지 수 (기본값: 500)
    """
    # Get all image files in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()
    
    # Limit to max_images
    if len(images) > max_images:
        original_count = len(images)
        images = images[:max_images]
        print(f"Limited to first {max_images} images out of {original_count} total images")

    if not images:
        print("No images found in the folder.")
        return

    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    print(f"Creating binary RAW file...")
    print(f"Image dimensions: {width}x{height}x{layers}")
    print(f"Total frames: {len(images)}")

    # Write raw binary data
    with open(output_file, 'wb') as f:
        for i, image in enumerate(images):
            image_path = os.path.join(image_folder, image)
            frame = cv2.imread(image_path)
            if frame is not None:
                # Keep BGR format as is, or convert to RGB if needed
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                f.write(frame.tobytes())
            
            if i % 100 == 0:
                print(f"Processed {i+1}/{len(images)} frames")

    # Check file creation and size
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"Binary RAW file saved as {output_file}")
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        
        # Save metadata file
        metadata_file = output_file.replace('.raw', '_metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write(f"Width: {width}\n")
            f.write(f"Height: {height}\n")
            f.write(f"Channels: {layers}\n")
            f.write(f"Frames: {len(images)}\n")
            f.write(f"Format: BGR888 (OpenCV default)\n")
            f.write(f"Frame rate: 30 FPS\n")
            f.write(f"Bytes per frame: {width * height * layers}\n")
            f.write(f"Total file size: {file_size} bytes\n")
        
        print(f"Metadata saved as {metadata_file}")
    else:
        print("ERROR: File was not created!")

def check_file_status():
    """
    현재 디렉토리의 파일 상태 확인
    """
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    raw_files = [f for f in os.listdir('.') if f.endswith('.raw')]
    if raw_files:
        print(f"Found .raw files: {raw_files}")
        for f in raw_files:
            size = os.path.getsize(f)
            print(f"  {f}: {size / (1024*1024):.2f} MB")
    else:
        print("No .raw files found in current directory")

# 사용 예시
if __name__ == "__main__":
    image_folder = "/media/jemo/HDD1/Workspace/dset/Drone-Detection-Custom/250312_sejong/yolo_dataset/images/val"
    output_file = "val.raw"
    
    print("=== Before processing ===")
    check_file_status()
    
    print("\n=== Processing ===")
    # 500장만 처리하도록 설정 (기본값)
    make_raw_vid_from_images_fixed(image_folder, output_file, max_images=500)
    
    print("\n=== After processing ===")
    check_file_status()