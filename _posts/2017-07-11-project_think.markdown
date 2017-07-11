---
layout: post
title: "[project] 프로젝트 진행 상황 및 생각"
subtitle:   "프로젝트 진행"
categories: project
tags: project progress
---

### 2017-07-10

vgg-s에서 vgg16으로 모델을 바꿈 <br>
layer수를 증가시켜 학습시킴. <br>

결과 : <br>
![vgg16_test1](/img/project_result/vgg16_test1.jpg)
![vgg16_test2](/img/project_result/vgg16_test2.jpg)
![vgg16_test3](/img/project_result/vgg16_test3.jpg)

아직 don-care에서 결과임. all-care로 학습시키면 성능이 더 향상되지 않을까 생각

<br>
<br>

### 2017-07-11

modified_model_version
기존의 vgg16 model은 fc가 3개(4096,4096,1000) <br>
우리의 모델은 fc가 2개여서 3개로 올리면 성능이 올라가지 않을까 생각 <br>
현재 우리의 모델 : 25088->2048->128 <br>

m1 model : 25088->6272->2048->128 <br>
m1 결과 : <br>
![vgg16_m1_test1](/img/project_result/vgg16_m1_test1.jpg)
![vgg16_m1_test2](/img/project_result/vgg16_m1_test2.jpg)
![vgg16_m1_test3](/img/project_result/vgg16_m1_test3.jpg)

m2 model : 25088->2048->512->128 <br>
m2 결과 : <br>
![vgg16_m2_test1](/img/project_result/vgg16_m2_test1.jpg)
![vgg16_m2_test2](/img/project_result/vgg16_m2_test2.jpg)
![vgg16_m2_test3](/img/project_result/vgg16_m2_test3.jpg)

오히려 성능이 떨어짐..

이유? 주관적인 생각 : fc가 많아서 scratch로 학습시키기에는 데이터의 수가 부족함, 그래서 성능이 오히려 떨어지지 않을까 생각함