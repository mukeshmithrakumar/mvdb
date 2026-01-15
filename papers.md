# Papers

**References**:
* [ ] [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320)
* [ ] [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)



## HNSW

I am reading the attached research paper, my goal is to finish reading this and implement hnsw using rust from scratch. I dont want you to write any code. I will let you know before I start implementation. Right now, below are some questions I have on the paper:
1. what are navigable small world graphs? is it a new thing when the paper was released or does this paper build upon an existing technique?
2. "The proposed solution is fully graph-based, without any need for additional search structures, which  are  typically  used  at  the  coarse  search  stage  of  the  most  proximity  graph  techniques." what are these additional search structures mean, I dont have context on this. Whats the coarse search stage, what are proximity graph techniques?
3. 


### Introduction

Hierarchical Navigable Small World (HNSW) incrementally builds a multi-layer structure consisting from hierarchical set of proximity graphs (layers) for nested subsets of the stored elements.

![hnsw](./images/hnsw.png)

