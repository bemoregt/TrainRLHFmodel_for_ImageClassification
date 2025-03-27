# RLHF for Image Classification

이 프로젝트는 강화학습에서 인간 피드백(Reinforcement Learning from Human Feedback, RLHF)을 사용하여 이미지 분류 모델을 훈련시키는 구현을 제공합니다. 개미와 꿀벌을 분류하는 예제를 통해 RLHF 방법론을 보여줍니다.

![RLHF 이미지 분류](https://raw.githubusercontent.com/bemoregt/TrainRLHFmodel_for_ImageClassification/main/docs/rlhf_flow.png)

## 기능

- ResNet18을 사용한 이미지 분류 모델
- PPO(Proximal Policy Optimization) 알고리즘을 통한 RLHF 구현
- 그래픽 사용자 인터페이스(GUI)를 통한 인간 피드백 수집
- 학습된 모델을 사용한 이미지 추론 기능

## 필요 조건

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- Pillow
- tkinter

## 설치 방법

```bash
git clone https://github.com/bemoregt/TrainRLHFmodel_for_ImageClassification.git
cd TrainRLHFmodel_for_ImageClassification
pip install -r requirements.txt
```

## 사용 방법

### 모델 학습하기

```bash
python train_rlhf.py
```

- 실행하면 지정된 이미지 폴더에서 개미와 꿀벌 이미지를 로드합니다.
- 각 이미지에 대해 GUI가 나타나면 이미지가 개미(0) 또는 꿀벌(1)인지 선택하거나 건너뛸 수 있습니다.
- 모델은 사용자의 피드백을 기반으로 학습되며, 학습 데이터와 모델이 저장됩니다.

### 학습된 모델로 추론하기

```bash
python inference.py
```

- GUI가 나타나면 이미지를 선택할 수 있습니다.
- 모델은 이미지가 개미인지 꿀벌인지 예측하고 결과를 표시합니다.

## 코드 설명

### 주요 구성 요소

1. **데이터셋 클래스 (AntBeeDataset)**
   - 지정된 디렉토리에서 이미지를 로드하고 전처리합니다.

2. **모델 아키텍처**
   - **정책 모델 (PolicyModel)**: 이미지를 입력으로 받아 개미 또는 꿀벌일 확률을 출력하는 ResNet18 기반 모델입니다.
   - **보상 모델 (RewardModel)**: 이미지의 가치를 평가하는 ResNet18 기반 모델입니다.

3. **PPO 알고리즘**
   - 개미/꿀벌 분류를 위한 PPO(Proximal Policy Optimization) 구현입니다.
   - 클리핑된 서러게이트 목적 함수를 사용하여 정책을 최적화합니다.

4. **인간 피드백 인터페이스**
   - tkinter를 사용한 GUI로 사용자가 이미지에 라벨을 지정할 수 있습니다.
   - 키보드 단축키(0, 1, s)를 지원합니다.

5. **학습 루프**
   - 모델이 이미지를 분류합니다.
   - 사용자가 피드백을 제공합니다.
   - 사용자 피드백과 모델 예측 간의 일치 여부에 따라 보상이 계산됩니다.
   - PPO 알고리즘을 사용하여 정책 모델이 업데이트됩니다.

6. **추론 스크립트**
   - 학습된 모델을 로드하고 새 이미지에 대한 예측을 수행합니다.
   - 사용자 친화적인 GUI를 통해 결과를 시각화합니다.

## RLHF 작동 방식

1. **초기화**: 사전 훈련된 ResNet18 모델로 정책 모델과 보상 모델을 초기화합니다.
2. **샘플링**: 정책 모델이 이미지 분류를 시도합니다.
3. **인간 피드백**: 사용자가 올바른 분류(개미 또는 꿀벌)를 제공합니다.
4. **보상 계산**: 모델의 예측이 사용자 피드백과 일치하면 양의 보상, 그렇지 않으면 음의 보상이 주어집니다.
5. **정책 업데이트**: PPO 알고리즘을 사용하여 정책 모델이 사용자 피드백에 맞게 업데이트됩니다.
6. **반복**: 지정된 에포크 수만큼 과정을 반복합니다.

## 참고

RLHF는 대형 언어 모델(LLM)의 훈련에 주로 사용되는 기술이지만, 이 프로젝트는 이미지 분류 작업에 적용하여 개념을 시연합니다. 실제 응용에서는 더 복잡한 피드백 메커니즘과 더 큰 데이터셋이 필요할 수 있습니다.