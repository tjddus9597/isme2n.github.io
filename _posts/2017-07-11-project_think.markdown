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

<br>

### 2017-07-14

*새로운 loss 제안*
1. 
$$
||f_a-f_i||_2^2 - ||f_a-f_j||_2^2 + \delta(1+w) \\
norm(d_j - d_i) = w 
$$
* d_i : ground truth distance \\
ex) i = 3, j = 4 => w = 0.07, i = 3, j= 15 => w= 0.16

2. 
pairwise ranking loss? <br>
pairwise mean squared error (PMSE)? 세현이한테 pairwise에 대해서 물어보기 <br>
$$
w[||f_i -f_j||^2_2 - ||g_i - g_j||^2_2]
$$ <br>
loss가 0 보다 작을 때는 무시 => w = 0 <br>
loss가 0 보다 클 때는 => w = 1


