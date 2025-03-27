import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.distributions import Categorical
from PIL import Image
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import numpy as np
import json

# 데이터셋 클래스
class AntBeeDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = []
        self.transform = transform
        
        print(f"이미지 검색 시작: {image_dir}")
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    self.image_paths.append(full_path)
                    print(f"이미지 발견: {full_path}")
        
        print(f"총 {len(self.image_paths)}개의 이미지를 찾았습니다.")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"폴더에서 이미지를 찾을 수 없습니다: {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_path

# 보상 모델
class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        
    def forward(self, x):
        return self.model(x)

# 정책 모델
class PolicyModel(nn.Module):
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        
    def forward(self, x):
        return torch.softmax(self.model(x), dim=1)

# PPO 알고리즘
class PPO:
    def __init__(self, policy_model, reward_model, lr=0.0003, gamma=0.99, epsilon=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy_model
        self.reward_model = reward_model
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon

    def update(self, states, actions, rewards, old_probs):
        # 모든 텐서를 동일한 디바이스로 이동
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        old_probs = torch.stack(old_probs).to(self.device)

        # 현재 정책으로 액션 확률 계산
        current_probs = self.policy(states)
        current_probs = current_probs.gather(1, actions.unsqueeze(1))

        # ratio 계산
        ratio = (current_probs / old_probs)

        # PPO 손실 함수 계산
        surr1 = ratio * rewards.unsqueeze(1)
        surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * rewards.unsqueeze(1)
        loss = -torch.min(surr1, surr2).mean()

        # 정책 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

# GUI 피드백 함수
def get_human_feedback(image_path):
    feedback_result = [None]
    
    root = tk.Tk()
    root.title("이미지 분류")
    
    window_width = 800
    window_height = 600
    root.geometry(f"{window_width}x{window_height}")
    
    # 이미지 로드 및 리사이징
    image = Image.open(image_path)
    image.thumbnail((window_width-100, window_height-100))
    photo = ImageTk.PhotoImage(image)
    
    image_label = tk.Label(root, image=photo)
    image_label.pack(pady=10)
    
    filename_label = tk.Label(root, text=f"파일: {os.path.basename(image_path)}")
    filename_label.pack(pady=5)
    
    def on_ant():
        feedback_result[0] = '0'
        root.quit()
        
    def on_bee():
        feedback_result[0] = '1'
        root.quit()
        
    def on_skip():
        feedback_result[0] = 's'
        root.quit()
    
    button_frame = ttk.Frame(root)
    button_frame.pack(pady=10)
    
    style = ttk.Style()
    style.configure('Custom.TButton', padding=10)
    
    ant_btn = ttk.Button(button_frame, text="개미 (0)", command=on_ant, style='Custom.TButton')
    bee_btn = ttk.Button(button_frame, text="꿀벌 (1)", command=on_bee, style='Custom.TButton')
    skip_btn = ttk.Button(button_frame, text="건너뛰기 (s)", command=on_skip, style='Custom.TButton')
    
    ant_btn.pack(side=tk.LEFT, padx=5)
    bee_btn.pack(side=tk.LEFT, padx=5)
    skip_btn.pack(side=tk.LEFT, padx=5)
    
    root.bind('0', lambda e: on_ant())
    root.bind('1', lambda e: on_bee())
    root.bind('s', lambda e: on_skip())
    
    root.mainloop()
    root.destroy()
    
    return feedback_result[0]

# 메인 학습 함수
def train_rlhf(image_dir, num_epochs=10):
    # 데이터 변환 정의
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")
    
    # 모델 초기화
    policy_model = PolicyModel().to(device)
    reward_model = RewardModel().to(device)
    ppo = PPO(policy_model, reward_model)
    
    # 데이터 로더 설정
    dataset = AntBeeDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 학습 기록용
    training_data = {
        'episodes': [],
        'human_feedback': {}
    }
    
    print("RLHF 학습을 시작합니다...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        states = []
        actions = []
        rewards = []
        old_probs = []
        episode_rewards = []
        
        for batch_idx, (images, paths) in enumerate(dataloader):
            images = images.to(device)
            path = paths[0]
            
            # 정책 모델로 액션 선택
            with torch.no_grad():
                action_probs = policy_model(images)
            dist = Categorical(action_probs)
            action = dist.sample()
            
            # 휴먼 피드백 수집
            human_feedback = get_human_feedback(path)
            if human_feedback == 's':
                continue
            
            # 피드백 저장
            training_data['human_feedback'][path] = int(human_feedback)
            
            # 보상 계산
            reward = 1.0 if int(human_feedback) == action.item() else -1.0
            episode_rewards.append(reward)
            
            # 배치 데이터 수집
            states.append(images.squeeze(0))
            actions.append(action.to(device))
            rewards.append(reward)
            old_probs.append(action_probs[0, action].to(device))
            
            # 미니배치 크기에 도달하면 PPO 업데이트
            if len(states) >= 8:  # 미니배치 크기
                loss = ppo.update(states, actions, rewards, old_probs)
                print(f"Batch {batch_idx}, PPO Loss: {loss:.4f}")
                states, actions, rewards, old_probs = [], [], [], []
        
        # 에피소드 정보 저장
        episode_info = {
            'epoch': epoch + 1,
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'total_samples': len(episode_rewards)
        }
        training_data['episodes'].append(episode_info)
        
        print(f"Epoch {epoch+1} 완료 - 평균 보상: {episode_info['mean_reward']:.4f}")
    
    # 학습 데이터 저장
    print("\n학습 데이터 저장 중...")
    with open('rlhf_training_data.json', 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    # 모델 저장
    print("모델 저장 중...")
    torch.save({
        'policy_model': policy_model.state_dict(),
        'reward_model': reward_model.state_dict()
    }, 'rlhf_models.pth')
    
    print("학습이 완료되었습니다!")
    return policy_model, reward_model

if __name__ == "__main__":
    image_directory = r"C:\Users\bemor\Pictures\hymenoptera_data\hymenoptera_data\val\all"
    train_rlhf(image_directory)