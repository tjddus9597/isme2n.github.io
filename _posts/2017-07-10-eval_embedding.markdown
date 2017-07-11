---
layout: post
title: "[project] 프로젝트 파일들 참고파일"
subtitle:   "프로젝트 lua, m 파일들 참고용"
categories: project
tags: project files
---

### eval_embedding.lua

```lua
embed_all = torch.FloatTensor(num_data_trn, dim_embed)
 ```
`embed_all` : num_data_trn(12366) * dim_embed(128)

```lua
idx_trn_stt = (idx_bat - 1) * size_batch + 1
idx_trn_end = math.min(idx_bat * size_batch, num_data_trn)
```

`idx_trn_stt` : 현재 mini_batch에서의 시작 image의 index <br />
`idx_trn_end` : 현재 mini_batch에서의 마지막 image의 index


```lua
 -- embedding
embed = convnet:forward(inputs:cuda())
embed = embed:float()
embed_all:sub(idx_trn_stt, idx_trn_end):copy(embed:sub(1, idx_trn_end - idx_trn_stt + 1))
```
`embed` : inputs를 입력으로 받아 학습 <br />
`embed_all` : (1~64), (65~128) ... to 12366 까지 입력받은
모든 embed 값들

<br />
<br />
### eval_retrieval.m

```Matlab
load(fullfile(path_dataset, 'dist_pose_val.mat'));
dist_pose = dist_pose_val(1:num_data_val, 1:num_data_val);

% pose distances between query and DB
qdist_pose = dist_pose(is_query, ~is_query);
[qdist_pose_sorted, NN_pose] = sort(qdist_pose, 2);
```
dist_pose : 포즈 distance를 나타낸 matrix

ex) dist_pose(1:5, 1:5)
```
        0   14.5595   15.3691   36.7705   24.1138
14.5595         0   14.0480   30.2159   21.2464
15.3691   14.0480         0   31.8619   22.6487
36.7705   30.2159   31.8619         0   32.4974
24.1138   21.2464   22.6487   32.4974         0
```

qdist_pose : query distance pose로 dist_pose(1:1919, 1920:9919) <br />
단, data idx는 거리별로 정리되어있지 않음. 그냥 이미지 번호

```Matlab
% pairwise distances between embedded vectors
dist_emb = bsxfun(@plus, sum(embed_val .* embed_val, 1)', (-2) * embed_val' * embed_val);
```

`embed_val .* embed_val` : embed_val element들 제곱 <br />
`sum(embed_val .* embed_val, 1)'` : 각 image들의 embedding의 각 차원의 value를 모두 합친 것 <br />
```
d1_1^2 + d1_2^2 ... d1_128^2
d2_1^2 + d2_2^2 ... d2_128^2
```


`(-2) * embed_val' * embed_val` : <br />

$$
\begin{bmatrix}
       ({d_{11}}^2+{d_{12}}^2+{d_{13}}^2...)*(-2) & ({d_{11}}*{d_{21}}+{d_{12}}*{d_{22}}+{d_{13}}*{d_{23}}...)*(-2) & ({d_{11}}*{d_{31}}+{d_{12}}*{d_{32}}+{d_{13}}*{d_{33}}...)*(-2) \\
       ({d_{11}}*{d_{21}}+{d_{12}}*{d_{22}}+{d_{13}}*{d_{23}}...)*(-2) & ({d_{21}}^2+{d_{22}}^2+{d_{23}}^2...)*(-2) & \cdots \\
       ({d_{11}}*{d_{31}}+{d_{12}}*{d_{32}}+{d_{13}}*{d_{33}}...)*(-2) & \vdots & \ddots
\end{bmatrix}
$$

``` Matlab
dist_emb = bsxfun(@plus, sum(embed_val .* embed_val, 1), dist_emb);
```

$$
\begin{bmatrix}
0 & (({d_{11}}^2+{d_{12}}^2 ... ) + ({d_{21}}^2+{d_{22}}^2 ... ) + ({d_{11}}*{d_{21}}+{d_{12}}*{d_{12}})*(-2)) & \cdots \\ 
(({d_{11}}^2+{d_{12}}^2 ... ) + ({d_{21}}^2+{d_{22}}^2 ... ) + ({d_{11}}*{d_{21}}+{d_{12}}*{d_{22}})*(-2)) & 0 & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}
$$

즉

$$
\begin{bmatrix}
0 & (({d_{11}}-{d_{21}})^2+({d_{12}}-{d_{22}})^2 ...) & (({d_{11}}-{d_{31}})^2+({d_{12}}-{d_{32}})^2 ...) \\
(({d_{11}}-{d_{21}})^2+({d_{12}}-{d_{22}})^2 ...) & 0 & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}
$$