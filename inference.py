import torch
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os
import sys

# 필요한 클래스 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_rlhf import PolicyModel

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 정책 모델 로드
    policy_model = PolicyModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    policy_model.load_state_dict(checkpoint['policy_model'])
    policy_model.eval()
    
    return policy_model

def predict_image(model, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 이미지 로드 및 변환
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 예측
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = output
        prediction = torch.argmax(probabilities).item()
        confidence = probabilities[0][prediction].item()
    
    # 결과 반환
    label = "개미" if prediction == 0 else "꿀벌"
    return label, confidence

def main():
    # 모델 로드
    model_path = 'rlhf_models.pth'
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다. ({model_path})")
        print("먼저 train_rlhf.py를 실행하여 모델을 학습시키세요.")
        return
        
    model = load_model(model_path)
    
    # GUI 생성
    root = tk.Tk()
    root.title("개미/꿀벌 분류기")
    root.geometry("400x500")
    
    def predict_file():
        file_path = filedialog.askopenfilename(
            filetypes=[("이미지 파일", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            label, confidence = predict_image(model, file_path)
            result_label.config(text=f"예측: {label}\n확률: {confidence:.2%}")
            
            # 이미지 표시
            img = Image.open(file_path)
            img.thumbnail((300, 300))
            photo = tk.PhotoImage(file=file_path)
            image_label.config(image=photo)
            image_label.image = photo
    
    # GUI 구성요소
    title_label = tk.Label(root, text="RLHF로 학습된 개미/꿀벌 분류기", font=("Arial", 14))
    title_label.pack(pady=10)
    
    image_label = tk.Label(root)
    image_label.pack(pady=10)
    
    button = tk.Button(root, text="이미지 선택", command=predict_file)
    button.pack(pady=5)
    
    result_label = tk.Label(root, text="", font=("Arial", 12))
    result_label.pack(pady=5)
    
    info_label = tk.Label(root, text="인식 가능: 개미(0), 꿀벌(1)", font=("Arial", 10))
    info_label.pack(pady=20)
    
    root.mainloop()

if __name__ == "__main__":
    main()