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

$$

=

\begin{bmatrix}
0 & (({d_{11}}-{d_{21}})^2+({d_{12}}-{d_{22}})^2 ...) & (({d_{11}}-{d_{31}})^2+({d_{12}}-{d_{32}})^2 ...) \\
(({d_{11}}-{d_{21}})^2+({d_{12}}-{d_{22}})^2 ...) & 0 & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}
$$


### FTEmbed.lua
```lua
local idxs = torch.range(1, m):cuda()
local indc = torch.lt(idxs:repeatTensor(m, 1):t(), idxs:repeatTensor(m, 1))   -- 1(i < j)
local dist = (a:repeatTensor(m, 1) - p):pow(2):sum(2):reshape(m)     
```

idxs : 1, 2, 3 .. 63
indc : 
```
0 1 1 1 1 1
0 0 1 1 1 1
0 0 0 1 1 1
0 0 0 0 1 1
0 0 0 0 0 1 ...
```
dist : ({1,2 distance}, {1,3 distance}) ... 63개

```lua
-- ground-truth (pose) distances
gt_dist[gt_dist:lt(self.mnd)] = self.mnd
```
gt_dist에서 self.mnd보다 less than(작은) element들을 self.mnd로 바꿔줌

```lua
nneg = math.max(math.min(gt_dist:gt(self.mxd):sum(), m-1), 1)
```
gt_dist에서 self.mxd보다 큰 것들의 개수, (단 m-1보다 작야 하고, 1보단 커야함)

```lua
-- uniform weight coefficients (except "don't care" triplets)
self.wgt:resize(indc:size()):copy(indc)            -- GT-dist based weights & order constraints
self.wgt[{{m-nneg+1, m}, {m-nneg+1, m}}] = 0       -- excluding don't care triplets
self.wgt:div(self.wgt:sum())                       -- normalization
```
self.wgt:resize(indc:size()):copy(indc) <br>
self.wgt를 indc크기로 맞추고 indc와 같게 만듬

```
0 1 1 1 1 1
0 0 1 1 1 1
0 0 0 1 1 1
0 0 0 0 1 1
0 0 0 0 0 1 ...
```
self.wgt[{{m-nneg+1, m}, {m-nneg+1, m}}] = 0 <br>
nneg 개수에 따라서 아래 부분을 0으로 만들어줌 ex) nneg = 3
```
0 1 1 1 1 1
0 0 1 1 1 1
0 0 0 0 0 0
0 0 0 0 0 0 
0 0 0 0 0 0 ...
```
self.wgt:div(self.wgt:sum()) <br>
총 개수 만큼 나눔 (normalization)
```
0.001 *
  0       5.4645  5.4645  5.4645  5.4645  5.4645
  0       0       5.4645  5.4645  5.4645  5.4645
  0       0       0       0       0       0
  0       0       0       0       0       0
  0       0       0       0       0       0
  0       0       0       0       0       0
```

```lua
-- loss per individual triplet
local loss = (dist:repeatTensor(m, 1):t() - dist:repeatTensor(m, 1)):add(self.mrg)
loss:cmul(self.wgt):clamp(0, loss:max())
```
> dist : ({1,2 distance}, {1,3 distance}) ... 63개 <br>
loss =>
$$
\begin{bmatrix}
d(1,2)-d(1,2) & d(1,2)-d(1,3) \\
d(1,3)-d(1,2) & d(1,2)-d(1,3) \\
d(1,4)-d(1,2) & \ddots
\end{bmatrix}
$$