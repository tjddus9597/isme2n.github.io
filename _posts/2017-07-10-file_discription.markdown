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

<br />
### FTEmbed.lua - function FTEmbedCriterion:updateOutput(input, gt_dist)
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
self.wgt[(m-nneg+1, m), (m-nneg+1, m))] = 0       -- excluding don't care triplets
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

self.wgt[(m-nneg+1, m), (m-nneg+1, m))] = 0  <br>
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

loss => <br>
$$
\begin{bmatrix}
d(1,2)-d(1,2) & d(1,2)-d(1,3) & d(1,2)-d(1,4)\\
d(1,3)-d(1,2) & d(1,3)-d(1,3) & d(1,3)-d(1,4)\\
d(1,4)-d(1,2) & d(1,4)-d(1,3) & \ddots
\end{bmatrix} + self.mrg
$$

일반적으로 d(1,2)-d(1,3)<0 일 것임 <br>
ex) d(1,2) = 0.01, d(1,3) = 0.03 <br>
d(1,2)-d(1,3)<0 이면 잘 mapping 된 것, 따라서 고려할 필요 X <br>

loss:cmul(self.wgt):`clamp(0, loss:max())`
위에서 얻은 loss에 self.wgt(normalization한 weight)를 곱하고
`0보다 작은 것들을 0으로 만들어줌`

```lua
-- update the weight coefficients
self.wgt:cmul(loss:gt(0):cuda())
```
loss가 0보다 큰 것들만 남기고 작은 것들은 0을 곱해서 없애줌

```lua
-- loss of this mini-batch
self.output = loss:sum()
```
loss들을 다 더함 -> 최종 loss

### 결론
loss => d(1,2) - d(1,3) + 0.03(margin) = 0으로 만들고자함<br>
만약 d(1,2) + 0.03 < d(1,3) 이면 고려 대상 X
d(1,2) + 0.03 = d(1,3)
d(1,2) + 0.03 = d(1,4) ...
 

### FTEmbed.lua - function FTEmbedCriterion:updateGradInput(input)

self.gradInput:resize(input:size()) - 64 * 128
$$
L = \sum_{i=1,j=1}^{N}[||f_a-f_i||^2_2-||f_a-f_j||^2_2] 
\\
\frac{\delta L}{\delta f_a} = \sum_{i=1,j=1}^{N}[2(f_a - f_i) - 2(f_a - f_j)] 
= \sum_{i=1,j=1}^{N}[2f_j-2f_i]
$$

```lua
-- grad for anchor = (f_j - f_i) * coeffs
local fj_minus_fi = p:t():reshape(r, 1, m):repeatTensor(1, m, 1) - p:t():reshape(r, m, 1):repeatTensor(1, 1, m)
```
p => 63, 128 <br>
p:t() => 128, 63 <br>
p:t() => reshape(r,1,m) => 128, 1, 63
p:t():reshape(r,1,m):repeatTensor(1,m,1) => 128,63,63

```lua
fj_minus_fi = p:t():reshape(r, 1, m):repeatTensor(1, m, 1) - p:t():reshape(r, m, 1):repeatTensor(1, 1, m)
```

1st dimension <br>
(2,1) p의 2번째 항목(즉, 3번째 image)과 p의 1번째 항목의 1차원
원소의 차
$$
\begin{bmatrix}
(1,1) & (2,1) & (3,1) & \cdots \\
(1,2) & (2,2) & (3,2) & \cdots \\
(1,3) & (2,3) & (3,3) & \ddots
\end{bmatrix}
$$ X 128 <br>
=> 128, 63, 63

```lua
self.gradInput[1]:copy(torch.cmul(fj_minus_fi, self.wgt:repeatTensor(r, 1, 1)):sum(2):sum(3):reshape(r))
```
fj_minus_fi의 각 차원에 wgt를 곱함. <br>
sum(2), sum(3) 2차원과 3차원을 다합쳐줌 <br>
reshape(r) : 1차원 r개의 벡터로 matrix를 수정함


```lua
local fp_minus_fa = (p - a:repeatTensor(m, 1))
   self.gradInput:sub(2, m+1):copy(torch.cmul(fp_minus_fa, (self.wgt - self.wgt:t()):sum(2):repeatTensor(1, r)))
```
$$
L = \sum_{p=1}^{N}||f_a-f_p||^2_2 = (f_a-f_p)^T(f_a-f_p)
\\
\frac{\delta L}{\delta f_a} = \sum_{p=1}^{N} \frac{\delta (f_a^Tf_a-f_a^Tf_p-f_p^Tf_a+f_p^Tf_p)}{\delta f_a}
=2f_a-2f_p
$$