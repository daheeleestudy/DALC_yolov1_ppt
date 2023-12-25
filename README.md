# 동덕여대 인공지능 동아리 DALC Yolo v1 paper review

# . Introduction

이미지 탐지 분류 시스템 object detection의 하나
You only look once라는 의미로 이미지를 한번만 보고 예측할 수 있다는 의미 

localization(bounding box 선정) & classification(이미지 분류)를 한번에 수행 - 1 stage detector

원본 이미지를 그리드화 시켜 그리드 중심으로 미리 정의 된 형태로 지정된 경계박스 갯수 기반 신뢰도 계산, 이미지에 객체가 포함되어 있는지, 배경만 단독으로 있는지에 대한 여부 포함, 높은 객체 신뢰도를 가진 위치를 선택해 객체 카테고리 파악한다. 단일CNN과 다르게 전체적으로 이미지를 학습

![image](https://github.com/daheeleestudy/DALC_yolov1_ppt/assets/139957707/a141191d-faf2-48ac-9f40-77ae54935e04)


대표적인 object detection 방식인 R-CNN은 많은 객체를 탐지하는데 한계가 있음.

R- CNN 

물체나 사람을 탐지해서 임의의 테두리 상자 지정->IOU(interesection of union) 계산- (영역의 교집합/영역의 합집합) - IOU값이 임의의값 이상이 되도록 조정, -> selective search (특정 기준에 따라 탐색 실시
테두리 상자를 세분화하고 중복 탐지를 제거하며 장면의 다른 객체를 기준으로 상자를 재평가하는 데 사용함, 그래서 CNN, SVM, regressor 각각 따로 학습시켜야 하고, 모두 CNN을 통과해야하기 때문에 시간이 많이 걸림

관련 참고 영상

[십분딥러닝_10_R-CNN(Regions with CNNs)_1](https://www.youtube.com/watch?v=W0U2mf9pf8o&t=435s)

YOLO 장점

- Object detection을 regression 문제로 변환해 단순화 하여 실시간으로 detection이 가능해졌다. (엄청나게 빠른 속도)
- 기존 detection 방식은 예측된 bounding box 내부만을 이용해서 클래스를 예측하는데 YOLO는 전체 이미지를 통해 bounding box의 class를 예측
- 학습한 이미지에 대한 예측 뿐 아니라 다른 도메인의 이미지에도 어느 정도 괜찮은 성능을 보임

# 2. Unified Detection

![image](https://github.com/daheeleestudy/DALC_yolov1_ppt/assets/139957707/5f3dc136-0331-4843-a186-4d95b16bd026)


7*7 그리드 2개의 bounding box를 예측 = 총 98개 바운딩 박스

각 바운딩 박스 confidence score

Confidence score 식

![image](https://github.com/daheeleestudy/DALC_yolov1_ppt/assets/139957707/4215cdc3-601e-40b4-9e13-757093edd5e4)


각 그리드셀- c개의 conditional class probability

**conditional class probability =  Pr(Classi | Object)**  물체가 bbbox에 있을때 그리드셀에 있는 object가 클래스에 속할 확률

![image](https://github.com/daheeleestudy/DALC_yolov1_ppt/assets/139957707/231a7943-c2b2-4b2a-8b7a-01cbfa4b2327)


output= 바운딩 박스의 중심좌표인 x,y

이미지의 폭인 w, 길이인 h, (normalize 된값, 0과 1 사이)

마지막으로 class specific confidence score를 계산

Class Specific Confidence Score

![image](https://github.com/daheeleestudy/DALC_yolov1_ppt/assets/139957707/3cf17074-7ba5-42c6-bad5-6d21c641370d)


## **2.1 Network Design**

Yolo에서 사용한 네트워크 디자인
googlenet- conv layer를 많이 쌓을수록 연산량 증가- 중간에 원바이원 reduction layer로 연산량 감소

24 Convolutional Layer + 2 fully connected Layer

Fast  Yolo (Yolo를 경량화하여 속도 높임) 

9 Convolutional Layer + 2 fully connected Layer

## 2.2 Training

![image](https://github.com/daheeleestudy/DALC_yolov1_ppt/assets/139957707/9da1379f-716c-43e1-af59-3af1debcc9d3)


특정 object에 대해 ground truth 중심이 위치하는 셀이 해당
학습할때는 bounding box 한개만 사용- IOU가 가장높은(ground truth와 가장많이 겹치는) 박스 한개
사용하는 loss function은 SSE 사용- 그리드셀의 object 존재하는 경우의 오차, predictor box로 선정된 오차만 학습

첫줄 - x,y (bbx 중심좌표) 오차

두번째 줄 - w,h 오차

세번째줄 - object가 있는 곳의 SSE(오차제곱합)

네번째줄 - object가 없는 곳의 SSE(오차제곱합)

막줄- class 별 오차제곱합

다 더하여 계산

## 2.3 Inference

성능 확인을 위한 최종적인 bounding box 예측 

첫번째 bounding box output 값 (x,y,w,h, confidence score)-5

두번째 bounding box output 값 (x,y,w,h, confidence score)-5

나머지 20 - 20 class conditional class probability

![image](https://github.com/daheeleestudy/DALC_yolov1_ppt/assets/139957707/cb054d9e-0502-4c1a-b274-204aceb93d2c)



첫번째 output에서 예측된 bounding box  confidence score와 conditional class probability를 곱하면 = class confidence score가 나옴 

두번째 bounding box도 위와 같이 구함 

![image](https://github.com/daheeleestudy/DALC_yolov1_ppt/assets/139957707/92340734-9adf-441e-ab22-ff30b43afbdb)



98개 class specific confidence score 얻을 수 있음 

![image](https://github.com/daheeleestudy/DALC_yolov1_ppt/assets/139957707/477a6d33-0b96-4fe5-81c2-0a757d732a52)


1440개 값 중 0.2 (threshold)보다 작은 값 - 모두 0으로 변환

그후 클래스별 내림차순 정렬 → Non - max supression 기법 통해서 최종 output 만들어냄

![image](https://github.com/daheeleestudy/DALC_yolov1_ppt/assets/139957707/be41ed66-d015-4672-b462-8c67906b8caf)


모든 클래스에 대해 NMS 적용하면 대부분의 값이 0으로 치환

output: 마지막 bbx 에 대해서 예측값이 크고 0보다 큰 클래스만 추출

Non - max supression 관련 자료

[NMS(Non Max Suppression)](https://visionhong.tistory.com/11)

- 가장 높은 확률값의 bbx를 메인박스로 둠
- 메인박스 주위 박스들 IOU 연산 하여 0.5보다 크면 제거 (물체 붙어 있을 때 문제가 발생함으로 0.5인 값만 지워야함)

만약 여러 class  사진 있으면 각 종류에 맞는 bbx 끼리 진행해야함

## 2.4 Limitations of YOLO

1. grid cell 한개당 한개의 클래스만 취급, 여러개의 물체들이 밀집되어 있는 객체에 대한 탐지가 어렵게됨.
2.  일정한 ratio의 bbx로만 예측을 하다보니 다른 ratio를 가진 object detection이 어렵게 됨.
3. 작은 bounding box나 큰 bounding box나 loss를 동일하게 처리하는 단점  ( 큰 상자의 작은 움직임<<작은 상자의 작은 움직임 영향이 큼)

# 3. **Comparison to Other Detection Systems**

DPM (Deformable parts models)

슬라이딩 윈도우 방식(sliding window) 사용 - 일정 간격의 bbx 건너 뛰고 bbx 그려줌

![image](https://github.com/daheeleestudy/DALC_yolov1_ppt/assets/139957707/bdda8470-f9f7-47b3-ac3b-eea1887bbb4f)


서로 분리된 파이프라인 - 특징 추출, 위치 파악, bbx 예측 수행 (yolo는 이 과정을 하나의 convolution 신경망으로 한번에 처리) 

DPM 관련 자료
https://89douner.tistory.com/82

# 3-2 
### OverFeat.

1. Source: Integrated Recognition, Localization, Detection; 지역화(Localization)를 하고, 탐지를 위해 Localizer를 적용하는 것
2. Sliding Window Algorithm사용: 일정한 범위의 Window, box를 original image 에 슬라이드. 이로써 추출한 aspect ratio(형상비율)-사이즈, 각도, 모양-을 CNN에 넣어 분류를 수행함→ Window를 bounding box 와 동일한 역할임
3. Overfeat. 은 Detection performance 보다 Localization(지역화) performance에 최적화된 알고리즘.
4. DPM 과 마찬가지로, Localizer는 Local 정보만 보기 때문에, 전반적인 context 를 설명하기는 어려움. 

### MultiGrasp.

1. 매우 간단한 알고리즘
2. 사이즈, 위치, 객체의 바운더리를 측정할 필요가 없음 (grasping 하기에 “적절한” 지역을 찾을 뿐)
3. YOLO 역시, 이미지 당 다중 class의 다중 객체들을 탐지하기 위하여 bounding box 와 class확률을 예측한다.

# 4. Experiments

YOLO vs. Other real time detection systems [PASCAL VOC. 2007]

YOLO와 Fast R-CNN, 두 모델 간 오류를 살펴야 함(Fast R-CNN 은 가장 높은 성능을 보이는  R-CNN 버전 중 하나) So, YOLO와 R-CNN 포함 다른 버전들 간 차이를 알 수 있음.

# 4-1 Comparison to Other Real-Time System
![image](https://github.com/daheeleestudy/DALC_yolov1_ppt/assets/139957707/3cee8eef-1dd4-4e1b-a8fd-bc8eca3c5d03)


1. Fast YOLO 가 PASCAL 에서 제일 빠른 객체 탐지를 수행함. 
- 52.7%의 mAP 를 기록하면서, 앞 DPM 보다, 높은 정확률을 보임.
1. YOLO는 63.4% 의 mAP를 보여줌과 둥시에, 실시간 객체 탐지를 수행함
2. YOLO 모델을 VGG-16을 적용하여 훈련시키기도 하나, YOLO보다 속도 측면에서 현저히 낮은 기록 보유 
3. Fast R-CNN : R-CNN 에서 ‘분류’ 단계의 처리 속도를 줄일 수 있다. 그러나, bounding box를 생성하기 위해 selective search (한 이미지당 2초) 에 의존함—>문제점이 될 수 있음. mAP측면에서는 높은 성능을 보이나, 속도 측면에서는 성능이 현저하게 떨어져 (fps=0.5) 실시간 객체 탐지에 사용하기 적합하지 않음
4. Faster R-CNN : 바운딩 박스를 생성하기 위해 selective search 알고리즘 대신 신경망 (neural network)를 사용함. 두 가지 버전 중, 제일 높은 정확률을 보이는 모델(VGG-16)은 속도 7을, 낮은 정확률이 떨어지는 모델 (ZF)은 속도 18을 기록. 
- VGG-16 version 은 ZF보다 약 10mAP 높은 정확도를 보이지만, YOLO 모델보다 6배 느림.
- ZF (Zeiler-Fergus) 은 YOLO보다 2.5배나 더 느리지만 (VGG-16보다 빠름) 정확도에서 적합하지 않음

# 4-2 VOC 2007 Error Analysis
