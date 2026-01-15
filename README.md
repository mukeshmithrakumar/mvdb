# My Vector Database

This project is to be done at two scales. We will start with a 1 Million Vector database and then move onto 1 Billion. This shifts moves the project from a pure **Algorithmic** problem to a **Systems Engineering** problem (optimizing disk I/O, page alignment, compression).


## 1 Million Vector

The industry standard for an **In-Memory** Vector Database is **HNSW (Hierarchical Navigable Small World)**.

**Target:**
* **Capacity:** 1 Million Vectors (1024D).
* **RAM Usage:** ~4GB for raw data + ~1-2GB for the graph structure = Fits easily on a laptop.
* **Speed:** < 2ms query time.

Here is a 7-day roadmap for building a high-performance HNSW index in Rust.

**The Baseline Evaluation**
Before writing the custom Rust implementation, we must establish a baseline using existing non-clustered solutions.
* Candidates: LanceDB, Qdrant (standalone mode), and Faiss (IVF-PQ/DiskANN).
* Objective: Push these tools vertically until they break to identify whether the limits are conceptual (ANN theory) or implementation-driven (GC pauses, overhead). This "failure analysis" will directly inform the custom Rust architecture.

### The Architecture: HNSW

Think of HNSW as a "Skip List" for graphs.
1. **Layer 0 (Bottom):** Contains every single vector in a dense graph.
2. **Upper Layers:** Sparse "express lanes." Layer 1 has 50% of the nodes, Layer 2 has 25%, etc.
3. **Search Logic:** You start at the top layer, zoom across the map with long jumps, drop down a layer, zoom in closer, drop down again, until you reach the bottom for fine-grained search.

### Required Reading

* **Primary Paper:** *Efficient and Robust Approximate Nearest Neighbor Search using Hierarchical Navigable Small World Graphs* (Malkov & Yashunin, 2016). **Focus on:** Algorithm 1 (Search) and Algorithm 4 (Insertion).
* **SIMD (Single Instruction, Multiple Data):** You cannot calculate 1 billion or even 1 million distances one-by-one. You need AVX2 or AVX-512 instructions to calculate 8 or 16 distances per CPU cycle. *Rust Topic:* `std::simd` (portable) or `std::arch` (x86 specific).

### 7-Day Sprint Plan (In-Memory HNSW)

#### Phase 1: The Basics (Days 1-2)

**Day 1: High-Performance Distance Metrics (SIMD)**
* **Goal:** Even in RAM, comparing 1024 floats is slow. You still need SIMD.
* **Task:** Implement Dot Product and L2 Distance using `std::simd` or `Simd<f32, 16>`.
* **Milestone:** A micro-benchmark proving your SIMD distance function is 8x faster than a loop.

**Day 2: The "NSW" (Navigable Small World) - No Hierarchy Yet**
* **Goal:** Build a single-layer graph (Layer 0).
* **Concept:** A graph where every node connects to its  closest neighbors.
* **Task:** Implement a struct `Node { vector: Vec<f32>, neighbors: Vec<usize> }`.
* **Algorithm:** Implement **Greedy Search**. Start at entry point -> look at neighbors -> move to the one closer to query -> repeat.
* **Milestone:** Insert 10k vectors into a flat graph and search it. (It will be slow, $O(\log N)$ish, but functional).

#### Phase 2: The HNSW Algorithm (Days 3-5)

**Day 3: Adding the Layers (The "H" in HNSW)**
* **Goal:** Implement the multi-layer structure to speed up search.
* **Logic:**
* When inserting a vector, flip a coin to decide its "max layer" (exponential decay: most nodes are layer 0, very few are layer 5).
* Store `layers: Vec<Vec<usize>>` in your Node struct (each layer has different neighbors).
* **Task:** Implement the "Zoom-in" logic: Search Layer 5 -> Find closest node -> Drop to Layer 4 starting at that node -> Search Layer 4 -> Drop to Layer 3...
* **Milestone:** Search time should drop drastically compared to Day 2.

**Day 4: The Heuristic (heuristic for selecting neighbors)**
* **Goal:** This is the "secret sauce" of HNSW's accuracy.
* **Concept:** When connecting a node to neighbors, don't just pick the absolute closest ones. Pick ones that are close *and* widely spaced (diverse). This prevents "clustering" and allows the graph to span long distances.
* **Task:** Implement the `SelectNeighborsHeuristic` from the paper (Algorithm 4).
* **Milestone:** Accuracy (Recall) improves significantly on your test set.

**Day 5: Concurrency (RwLock & Atomic)**
* **Goal:** Allow simultaneous searches and insertions.
* **Challenge:** Rust's borrow checker hates graphs (self-referential structures).
* **Tool:** `parking_lot::RwLock` or `dashmap` (concurrent hashmap).
* **Task:** Wrap your nodes in `RwLock` or use a concurrent arena allocator so multiple threads can read the graph while one thread inserts.
* **Milestone:** Run a benchmark with 1 writer thread and 8 reader threads without deadlocks.

#### Phase 3: Polish (Days 6-7)

**Day 6: Serialization (Save/Load)**
* **Goal:** Don't rebuild the index every restart.
* **Tool:** `bincode` or `serde`.
* **Task:** Serialize the vector data and the neighbor lists to a binary file on disk.
* **Milestone:** `db.save("index.bin")` and `db.load("index.bin")`.

**Day 7: The Server**
* **Goal:** Make it usable.
* **Task:** Wrap it in a web server (Axum or Actix-web).
* **Endpoints:**
	* `POST /insert { vector: [..] }`
	* `POST /search { vector: [..], k: 10 }`
* **Milestone:** A working API you can curl from your terminal.

### Next Step

**"Arena Allocator" pattern** is the standard way to implement graph algorithms like HNSW without fighting the borrow checker (using indices `u32` instead of pointers/references).


## 1B Vector

Building a 1024-dimensional, 1-billion vector database on a single instance is an **extreme systems engineering challenge**.

1 billion vectors  1024 dimensions  4 bytes (f32)  **4 Terabytes** of raw data.
This exceeds the RAM of almost any single instance.

To achieve we must build a **Disk-Based Index** (Out-of-Core) algorithm. The State-of-the-Art (SOTA) for this specific constraint is **DiskANN (Vamana graph)**.

### Core Architecture: The DiskANN Model

We are building a two-tier system to bypass the RAM limit:
1. **SSD Layer:** Stores the full-precision vectors and the full Graph Index (Adjacency lists).
2. **RAM Layer:** Stores a compressed representation (Quantized vectors) and a small set of navigation entry points.
3. **Search Logic:** You "beam search" the graph on the SSD, reading only the nodes you visit, using the cached compressed vectors in RAM to guide the search distance calculations.

**The Architecture: DiskANN**

To fit 4TB of vectors on a machine with ~64GB RAM, you will build a two-tier system:
1. RAM (Navigation): Holds a compressed "rough" graph and quantized vectors to guide the search.
2. SSD (Storage): Holds the full 4TB of raw vectors and the detailed graph adjacency lists.
3. Mechanism: You use the RAM index to find candidate neighbors, then issue asynchronous disk reads (io_uring) to fetch full vectors from SSD for the final re-ranking.

### Preparation: The Syllabus

Before writing code, you must understand these four concepts.
1. **Memory Mapping (`mmap`) & Async I/O:** How to treat a 4TB file on disk as if it were a slice in memory, or alternatively, how to queue disk reads without blocking the CPU.
    * *Rust Topic:* `memmap2` crate, `io_uring` (Linux).
2. **Product Quantization (PQ):** How to compress a 4KB vector into 64 bytes to fit the navigation index in RAM.
	* *Product Quantization for Nearest Neighbor Search* (Hervé Jégou) – [Read mainly for the concept of sub-quantizers].
	* **Focus on:** How to compress 1024D vectors into small byte codes (e.g., 64 bytes).
3. **The Vamana Graph:** *DiskANN: Fast Accurate Billion-Point Nearest Neighbor Search on a Single Node* (NeurIPS 2019) – [Read Sections 3 & 4 specifically]. [Paper](https://papers.nips.cc/paper_files/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html). [DiskANN Algorithm Explained](https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3Dd_Z9g6s7l0k) *This video breaks down the DiskANN architecture and Vamana graph specifically for SSD-resident billion-scale datasets.*
	* **Focus on:** The **Vamana** graph construction algorithm and **RobustPrune**.

**References**
* [ ] Hybrid inverted+disk: **SPANN** (inverted methodology + disk posting lists) [SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search](https://arxiv.org/abs/2111.08566)
* Production reference designs: **Faiss** design principles (survey-ish, very practical) [docs](https://faiss.ai/index.html), [The Faiss library](https://arxiv.org/abs/2401.08281)
* [ ] Benchmarking landscape: ANN-Benchmarks + BigANN/NeurIPS challenge pages [Billion-Scale Approximate Nearest Neighbor Search Challenge: NeurIPS'21 competition track](https://big-ann-benchmarks.com/neurips21.html), [Info](https://ann-benchmarks.com/index.html)
* [ ] "Recent/SOTA to skim": PageANN (2025) to see where disk ANNS is heading [Scalable Disk-Based Approximate Nearest Neighbor Search with Page-Aligned Graph](https://arxiv.org/abs/2509.25487v2)
* 
* [ ] [Supercharge your I/O in Rust with io_uring](https://www.youtube.com/watch?v=IHAPVK1nOrQ) *Relevant because for a disk-based vector database, efficient non-blocking I/O is the critical bottleneck; this video explains how `io_uring` in Rust solves that.*


### 7-Day Sprint Plan

#### Phase 1: Foundations & The Math (Days 1-2)

**Day 1: Vector Primitives & SIMD**
* **Goal:** Create the fastest possible distance calculation (Dot Product / Euclidean) for 1024D vectors.
* **Topics:** `std::simd` (portable SIMD), AVX2/AVX-512 intrinsics, Fused Multiply-Add (FMA).
* **Rust Crates:** `bytemuck` (casting bytes to floats), `rand` (generating dummy data).
* **Milestone:** A benchmark demonstrating you can calculate distance between two 1024D vectors in nanoseconds using explicit SIMD instructions.
* **Why:** At 1 billion scale, every CPU cycle in the distance function counts.

**Day 2: Quantization (Compression)**
* **Goal:** Compress your 4TB dataset into something that might fit in RAM (or cache).
* **Topics:** Scalar Quantization (converting `f32` to `u8` or `i8`) and Product Quantization (PQ).
* **Implementation:** Implement a simple linear scalar quantizer: .
* **Milestone:** A function that compresses a 1024D `f32` vector (4KB) into a 1024D `u8` vector (1KB) and estimates distance with <5% error.

#### Phase 2: The Graph Engine (Days 3-5)

**Day 3: In-Memory Greedy Search (The Prototype)**
* **Goal:** Build the search logic before adding disk complexity.
* **Topics:** Greedy Search algorithm. Adjacency lists ( `Vec<Vec<u32>>`).
* **Task:** Create a random graph of 10k vectors in RAM. Implement a `search(query_vector)` function that starts at a random node and greedily moves to the neighbor closer to the query.
* **Milestone:** Functioning greedy search finding nearest neighbors in a small in-memory dataset.

**Day 4: The Vamana Construction (RobustPrune)**
* **Goal:** Implement the "secret sauce" of DiskANN.
* **Algorithm:** The **RobustPrune** step. Unlike HNSW, Vamana prunes edges aggressively to create a high-degree, small-diameter graph that minimizes "hops" (and thus disk reads).
* **Task:** Implement the logic: "Connect point P to neighbor N *only if* N is not closer to any of P's existing neighbors."
* **Milestone:** A graph construction script that builds a Vamana graph for 100k vectors.

**Day 5: Disk Layout & Memory Mapping**
* **Goal:** Move data from Heap to SSD.
* **Topics:** `mmap` (Memory Mapped Files), Page Alignment (4KB).
* **Rust Crates:** `memmap2`.
* **Implementation:**
    * Store vectors in a flat binary file (`vectors.bin`).
    * Store the graph adjacency list in a separate file (`graph.bin`).
    * Use `mmap` to view these files as slices `&[u8]`.
* **Milestone:** Perform the "Greedy Search" from Day 3, but reading purely from the memory-mapped file on disk.

#### Phase 3: Optimization & Scale (Days 6-7)

**Day 6: Parallel Build & IO Optimization**
* **Goal:** Speed up indexing and retrieval.
* **Topics:** Rayon (Data Parallelism), `io_uring` (Async I/O for Linux).
* **Task:** Use `rayon::par_iter` to parallelize the distance calculations during the build phase. (Note: Building the index for 1B vectors takes *days*; for this week, verify architecture handles 1M+ efficiently).
* **Milestone:** maximize CPU usage during index build.

**Day 7: The API & The "Billion" Test**
* **Goal:** Interface and final architecture check.
* **Task:** Wrap the search in a simple gRPC service (`tonic`).
* **Reality Check:** You won't index 1B vectors in a day. Generate 1B dummy vectors on disk (sparse/zeros to save space if needed, or just calculate the file offsets). Prove your `mmap` logic can seek to the 900-millionth vector and read it without crashing.


## Industry 

**Strategic Context & Operational Philosophy**

While the technical roadmap focuses on how to build the index, this project is defined by why we are building it on a single node. The goal is to prove that "billion-scale" does not inherently require complex, sharded Kubernetes clusters.

**1. The "Underserved Middle"**

We are targeting the gap between hyperscalers (who have dedicated infrastructure teams) and managed services (who charge a premium for abstraction). This project validates a "Database Appliance" model: a system optimized for teams with strong systems engineers who possess tight cost constraints but access to modern hardware. The aim is to trade infinite horizontal scalability for predictability, saturation, and simplicity.

**2. Hardware & Vertical Scaling Strategy**

Unlike web-scale search, we are targeting the realistic upper bound for most recommender and retrieval workloads (100M–1B vectors).
* The Hardware Hypothesis: We assume access to a "beefy" single node (≥512 GB to 2 TB RAM, fast NVMe) rather than a fleet of small instances.
* No GPUs (Initially): To keep TCO low and deployment simple, we rely purely on CPU (AVX-512) and optimized Disk I/O.
* The Goal: High Availability (HA) should be achieved via a simple Active/Warm-Standby pair (2 nodes), not a complex distributed mesh.

**3. Primary Research Questions**

The implementation must explicitly answer the following to validate the single-node hypothesis:
* Single-Node Feasibility: What is the true "tipping point" for RAM vs. Recall? (e.g., Can 100M, 300M, or 1B vectors fit before latency degrades unacceptably?)
* Cost Efficiency: How does the TCO and $/QPS compare to managed services (Vertex, Pinecone) or distributed clusters?
* Operational Simplicity: Can rebuilds, restarts, and recovery be made deterministic and "boring"?
* The Cluster Threshold: What specific workload characteristics definitively force a move to multi-node designs?

**4. Non-Goals**

To maintain focus, we explicitly exclude:
* Infinite Horizontal Scalability: We are not building a system to index the entire internet.
* Multi-Tenancy: This is a single-tenant system for a specific workload.
* Kubernetes Orchestration: The system should run as a standard binary/service without requiring a container orchestration platform.


## References

**Code**
* [ ] [MTEB: Massive Text Embedding Benchmark](https://huggingface.co/blog/mteb)
* [ ] [Reveal Hidden Pitfalls and Navigate Next Generation of Vector Similarity Search from Task-Centric Views](https://huggingface.co/papers/2512.12980)
* [ ] [Iceberg: Task-Centric Benchmarks for Vector Similarity Search](https://huggingface.co/datasets/PIIR/Iceberg-dataset)
* [ ] [Benchmarks of approximate nearest neighbor libraries in Python](https://github.com/erikbern/ann-benchmarks)
* [ ] [Framework for evaluating ANNS algorithms on billion scale datasets](https://github.com/harsha-simhadri/big-ann-benchmarks)
* [ ] [Benchmarking nearest neighbors](https://github.com/erikbern/ann-benchmarks)
* [ ] [Benchmarks for Billion-Scale Similarity Search](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search)
* [ ] [Billion-Scale Approximate Nearest Neighbor Search Challenge: NeurIPS'21 competition track](https://big-ann-benchmarks.com/neurips21.html)
* [ ] [Neural Search in Action](https://matsui528.github.io/cvpr2023_tutorial_neural_search/)
* [ ] [Puffer - High-Performance Vector Database](https://github.com/harishsg993010/open-puffer)
* [ ] [TurboPuffer: Object Storage-First Vector Database Architecture](https://567-labs.github.io/systematically-improving-rag/talks/turbopuffer-engine/)
* [ ] [RAG on Everything with LEANN. Enjoy 97% storage savings while running a fast, accurate, and 100% private RAG application on your personal device.](https://github.com/yichuan-w/LEANN)


**Blogs**
* [ ] [Introduction to SIMD programming in pure Rust](https://kerkour.com/introduction-rust-simd)
* [ ] [How to Build a Database without a Server](https://www.infoq.com/presentations/arcticdb-db-no-server/)
* [ ] [CPU Optimized Embeddings with 🤗 Optimum Intel and fastRAG](https://huggingface.co/blog/intel-fast-embedding)
* [ ] [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization)
* [ ] [HNSW at Scale: Why Your RAG System Gets Worse as the Vector Database Grows](https://towardsdatascience.com/hnsw-at-scale-why-your-rag-system-gets-worse-as-the-vector-database-grows/)
* [ ] [Scaling HNSWs](https://antirez.com/news/156)


**Papers**
* [ ] [Kids! Use hnswlib for HNSW](https://terencezl.github.io/blog/2022/09/28/kids-use-hnswlib/)
* [ ] [How We Store and Search 30 Billion Faces](https://www.clearview.ai/post/how-we-store-and-search-30-billion-faces)
* [ ] [Online clustering: algorithms, evaluation, metrics, application and benchmarking](https://hoanganhngo610.github.io/river-clustering.kdd.2022/)
* [ ] [Optimizing Data Stream Representation: An Extensive Survey on Stream Clustering Algorithms](https://link.springer.com/article/10.1007/s12599-019-00576-5)
* [ ] [BIRCH: an efficient data clustering method for very large databases](https://dl.acm.org/doi/10.1145/233269.233324)
* [ ] [Incremental Cluster Validity Indices for Online Learning of Hard Partitions: Extensions and Comparative Study](https://ieeexplore.ieee.org/document/8970493)
* [ ] [Quake: Adaptive Indexing for Vector Search](https://www.usenix.org/conference/osdi25/presentation/mohoney)
* [ ] [Achieving Low-Latency Graph-Based Vector Search via Aligning Best-First Search Algorithm with SSD](https://www.usenix.org/conference/osdi25/presentation/guo)
* [ ] [Compass: Encrypted Semantic Search with High Accuracy](https://www.usenix.org/conference/osdi25/presentation/zhu-jinhao)
* [ ] [ScaleDB: A Scalable, Asynchronous In-Memory Database](https://www.usenix.org/conference/osdi23/presentation/zhang-qianxi)
* [ ] [SEPH: Scalable, Efficient, and Predictable Hashing on Persistent Memory](https://www.usenix.org/conference/osdi23/presentation/wang-chao)
* [ ] [SMART: A High-Performance Adaptive Radix Tree for Disaggregated Memory](https://www.usenix.org/conference/osdi23/presentation/luo)
* [ ] [zIO: Accelerating IO-Intensive Applications with Transparent Zero-Copy IO](https://www.usenix.org/conference/osdi22/presentation/stamler)
* [ ] [TriCache: A User-Transparent Block Cache Enabling High-Performance Out-of-Core Processing with In-Memory Programs](https://www.usenix.org/conference/osdi22/presentation/feng)
* [ ] [Modernizing File System through In-Storage Indexing](https://www.usenix.org/conference/osdi21/presentation/koo)
* [ ] [Nap: A Black-Box Approach to NUMA-Aware Persistent Memory Indexes](https://www.usenix.org/conference/osdi21/presentation/wang-qing)
* [ ] [Optimizing Storage Performance with Calibrated Interrupts](https://www.usenix.org/conference/osdi21/presentation/tai)
* [ ] [ZNS+: Advanced Zoned Namespace Interface for Supporting In-Storage Zone Compaction](https://www.usenix.org/conference/osdi21/presentation/han)
* [ ] [Retrofitting High Availability Mechanism to Tame Hybrid Transaction/Analytical Processing](https://www.usenix.org/conference/osdi21/presentation/shen)
* [ ] [Beyond malloc efficiency to fleet efficiency: a hugepage-aware memory allocator](https://www.usenix.org/conference/osdi21/presentation/hunter)
* [ ] [Marius: Learning Massive Graph Embeddings on a Single Machine](https://www.usenix.org/conference/osdi21/presentation/mohoney)
* [ ] [Replicating Persistent Memory Key-Value Stores with Efficient RDMA Abstraction](https://www.usenix.org/conference/osdi23/presentation/wang-qing)
* [ ] [Cobra: Making Transactional Key-Value Stores Verifiably Serializable](https://www.usenix.org/conference/osdi20/presentation/tan)
* [ ] [Storage Systems are Distributed Systems (So Verify Them That Way!)](https://www.usenix.org/conference/osdi20/presentation/hance)
* [ ] [Fast RDMA-based Ordered Key-Value Store using Remote Learned Cache](https://www.usenix.org/conference/osdi20/presentation/wei)
* [ ] [CrossFS: A Cross-layered Direct-Access File System](https://www.usenix.org/conference/osdi20/presentation/ren)
* [ ] [Caladan: Mitigating Interference at Microsecond Timescales](https://www.usenix.org/conference/osdi20/presentation/fried)
* [ ] [Performance-Optimal Read-Only Transactions](https://www.usenix.org/conference/osdi20/presentation/lu)
* [ ] [FlightTracker: Consistency across Read-Optimized Online Stores at Facebook](https://www.usenix.org/conference/osdi20/presentation/shi)
* [ ] [Assise: Performance and Availability via Client-local NVM in a Distributed File System](https://www.usenix.org/conference/osdi20/presentation/anderson)
* [ ] [Focus: Querying Large Video Datasets with Low Latency and Low Cost](https://www.usenix.org/conference/osdi18/presentation/hsieh)
* [ ] [Write-Optimized and High-Performance Hashing Index Scheme for Persistent Memory](https://www.usenix.org/conference/osdi18/presentation/zuo)
* [ ] [FlashShare: Punching Through Server Storage Stack from Kernel to Firmware for Ultra-Low Latency SSDs](https://www.usenix.org/conference/osdi18/presentation/zhang)
* [ ] [Splinter: Bare-Metal Extensions for Multi-Tenant Low-Latency Storage](https://www.usenix.org/conference/osdi18/presentation/kulkarni)
* [ ] [ZERO Results Problem on Vector DBs: Qdrant’s ACORN Algorithm Fixes the Broken Filter Problem](https://blog.stackademic.com/zero-results-problem-on-vector-dbs-qdrants-acorn-algorithm-fixes-the-broken-filter-problem-b2623b765267)
* [ ] [Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph](https://arxiv.org/abs/1707.00143)
* [ ] [SVFusion: A CPU-GPU Co-Processing Architecture for Large-Scale Real-Time Vector Search](https://arxiv.org/abs/2601.08528)
* [ ] [Towards Building efficient Routed systems for Retrieval](https://arxiv.org/abs/2601.06389)


**Videos**
* [ ] [Internet-Scale Semantic, Structural, and Text Search in Real Time by Ash Vardanian](https://youtu.be/yn87sxRsOj0?si=UkeRLD770MfRHnQI)




