---
layout: post
title: "[project] 프로젝트 진행 상황 및 생각"
subtitle:   "프로젝트 진행"
categories: project
tags: project progress
---

### 2017-07-11

modified_model_version
기존의 vgg16 model은 fc가 3개(4096,4096,1000)
우리의 모델은 fc가 2개여서 3개로 올리면 성능이 올라가지 않을까 생각
현재 우리의 모델 : 25088->2048->128

m1 model : 25088->6272->2048->128
m1 결과 : <br />
![vgg16_m1_test1](/img/project_result/vgg16_m1_test1.jpg)
![vgg16_m1_test2](/img/project_result/vgg16_m1_test2.jpg)
![vgg16_m1_test3](/img/project_result/vgg16_m1_test3.jpg)

m2 model : 25088->2048->512->128
m2 결과 : <br />
![vgg16_m2_test1](/img/project_result/vgg16_m2_test1.jpg)
![vgg16_m2_test2](/img/project_result/vgg16_m2_test2.jpg)
![vgg16_m2_test3](/img/project_result/vgg16_m3_test3.jpg)