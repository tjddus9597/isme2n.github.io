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
embed_all : num_data_trn(12366) * dim_embed(128)

```lua
idx_trn_stt = (idx_bat - 1) * size_batch + 1
idx_trn_end = math.min(idx_bat * size_batch, num_data_trn)
```

idx_trn_stt : 현재 mini_batch에서의 시작 image의 index
idx_trn_end : 현재 mini_batch에서의 마지막 image의 index


```lua
 -- embedding
embed = convnet:forward(inputs:cuda())
embed = embed:float()
embed_all:sub(idx_trn_stt, idx_trn_end):copy(embed:sub(1, idx_trn_end - idx_trn_stt + 1))
```
embed : inputs를 입력으로 받아 학습 <br />
embed_all : (1~64), (65~128) ... to 12366 까지 입력받은
모든 embed 값들



### eval_retrieval.m

```Matlab
   % pairwise distances between embedded vectors
   dist_emb = bsxfun(@plus, sum(embed_val .* embed_val, 1)', (-2) * embed_val' * embed_val);
   dist_emb = bsxfun(@plus, sum(embed_val .* embed_val, 1), dist_emb);

   % distances between query and DB, on the embedding space
   qdist_emb = dist_emb(is_query, ~is_query);
   [~, NN_emb] = sort(qdist_emb, 2);
```

embed_val .* embed_val = embed_val element들 제곱
sum(embed_val .* embed_val, 1)' : 