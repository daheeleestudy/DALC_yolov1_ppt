# . Introduction

이미지 탐지 분류 시스템 object detection의 하나
You only look once라는 의미로 이미지를 한번만 보고 예측할 수 있다는 의미 

localization(bounding box 선정) & classification(이미지 분류)를 한번에 수행 - 1 stage detector

원본 이미지를 그리드화 시켜 그리드 중심으로 미리 정의 된 형태로 지정된 경계박스 갯수 기반 신뢰도 계산, 이미지에 객체가 포함되어 있는지, 배경만 단독으로 있는지에 대한 여부 포함, 높은 객체 신뢰도를 가진 위치를 선택해 객체 카테고리 파악한다. 단일CNN과 다르게 전체적으로 이미지를 학습

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e2536b5c-84e6-4865-8d01-35716fbac3e8/Untitled.png)

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

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/27dae789-a845-4a20-93cc-317e4b4a2e67/Untitled.png)

7*7 그리드 2개의 bounding box를 예측 = 총 98개 바운딩 박스

각 바운딩 박스 confidence score

Confidence score 식

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/06253099-79a0-4be7-80dc-471f1db0e7e2/Untitled.png)

각 그리드셀- c개의 conditional class probability

**conditional class probability =  Pr(Classi | Object)**  물체가 bbbox에 있을때 그리드셀에 있는 object가 클래스에 속할 확률

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c88ba58b-6695-48d0-83cc-48c00c85ff0e/Untitled.png)

output= 바운딩 박스의 중심좌표인 x,y

이미지의 폭인 w, 길이인 h, (normalize 된값, 0과 1 사이)

마지막으로 class specific confidence score를 계산

Class Specific Confidence Score

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2b1c3234-6cfd-4db9-8754-45d9e4323294/Untitled.png)

## **2.1 Network Design**

Yolo에서 사용한 네트워크 디자인
googlenet- conv layer를 많이 쌓을수록 연산량 증가- 중간에 원바이원 reduction layer로 연산량 감소

24 Convolutional Layer + 2 fully connected Layer

Fast  Yolo (Yolo를 경량화하여 속도 높임) 

9 Convolutional Layer + 2 fully connected Layer

## 2.2 Training

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/63f739ba-6408-4e01-8b2b-abdc7c492ef7/Untitled.png)

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

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/323d9b6a-969c-4660-9a70-6773b3c8262c/Untitled.png)

첫번째 output에서 예측된 bounding box  confidence score와 conditional class probability를 곱하면 = class confidence score가 나옴 

두번째 bounding box도 위와 같이 구함 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5c073e85-c8ed-4584-801a-f76a1cac2862/Untitled.png)

98개 class specific confidence score 얻을 수 있음 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5e3707f4-8863-4475-ac33-05be7ad32442/Untitled.png)

1440개 값 중 0.2 (threshold)보다 작은 값 - 모두 0으로 변환

그후 클래스별 내림차순 정렬 → Non - max supression 기법 통해서 최종 output 만들어냄

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/940c9793-97d2-4028-a71f-90241b212614/Untitled.png)

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

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2e01fa79-e7a7-465e-87ed-30d56709b166/Untitled.png)

서로 분리된 파이프라인 - 특징 추출, 위치 파악, bbx 예측 수행 (yolo는 이 과정을 하나의 convolution 신경망으로 한번에 처리) 

DPM 관련 자료
