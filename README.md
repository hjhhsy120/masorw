# masorw

Memory-Aware Second-Order Random Walk

We have int version for small, weighted and directed graphs, long long version for large, unweighted and undirected graphs.

## Usage

Compile example:

g++ node2vec.cpp -fopenmp -lpthread -std=c++11 -o node2vec

g++ autoreg.cpp -fopenmp -lpthread -std=c++11 -o autoreg

Excution example:

./node2vec -onlywalk -input blogcatalog.bin -walkname walk_blogcatalog -memory 50 -verbose 1 -threads 16 -p 0.25 -q 4

./autoreg -input blogcatalog.bin -startpoints blogcatalog_start.txt -memory 50 -verbose 1 -threads 16 -alpha 0.5

## Dataset format

### Graph for Int Version

The graph is weighted and directed.

Format of binary file:

number of nodes (long long), number of edges (long long),

offsets of each node (array of int, the 1st is 0, 2nd is #neighbors of node 1, 3rd is sum of #neighbors of node 1&2, ..., length equals number of nodes),

neighbors of each node (arrays of int32, length of each array equals node degree, neighbors' ids in ascending order),

weights of each edge (array of float, corresponding to the neighbors list)

### Graph for Long Long version

The graph must be unweighted and undirected with edges for both directions.

Format of binary file:

number of nodes (long long), number of edges (long long),

offsets of each node (array of long long, the 1st is 0, 2nd is #neighbors of node 1, 3rd is sum of #neighbors of node 1&2, ..., length equals number of nodes),

neighbors of each node (array of int32, length of each array equals node degree, neighbors' ids in ascending order)

### Start points of auto regression

Text file, the 1st line is the number of start points ns, the next ns lines are identifier numbers of each point.

## Arg List

(Notice: do not put those with no parameters at the end, like -recursive, -anothercost, etc.)

-input x: (required) name of input dataset

-verbose x: amount of output information, 0 very little, 1 moderate, default 2 detailed

-threads x: number of threads

-memory x: memory limitation in MB for the data structures of sampling methods, default 1024

-rejbound x: when the degree exceeds x, sample x neighbors to estimate the average time of rejection method, x = 0 indicates precise time

-alias: whether only to use alias sampling

-reject: whether only to use reject sampling

-online: whether only to use online (naive) sampling

-recursive: whether to use recursive greedy algorithm

-anothercost: whether to use degree-based greedy algorithm

-rejectcnt: whether to output the accept rate of each node

-cntname x: (required if using -rejectcnt) file name for accept rates

-autotypes x: file name for sampling methods of each node

-seed x: random seed (notice that multi-threading also leads to randomness)

-memorychange x y z ... 0: only to show the estimated time and memory changes when memory size changes. x is the initial size, y, z, ... are increments, end with 0.

### Node2vec specific:

-p x: parameter of node2vec, default 1

-q x: parameter of node2vec, default 1

-nwalks x: number of walks from each node, default 10

-walklen x: length of walks, default 80

-window x: window size for skipgram, default 10

-onlywalk: (currently required) whether only to sample the random walks without training embeddings

-walkname x: (required) preffix of names of temporary files

(the following parameters are for training embeddings, which is currently unavailable)

-dim x: dimension of embedding, default 128

-lr x: learning rate (which decays), default 0.025

-nsamples x: ratio of #negative samples to #positive samples, default 5

-output x: (required without -onlywalk) file name for embeddings

### Auto Regression specific

-startpoints x: (required) file name for start points

-alpha x: parameter of auto regression, default 0.2

-pidivn x: ratio of #walks from each node to #nodes, default 4

-dumpfac x: probability of taking next step, default 0.85

-maxlen x: max length of walks, default 20

-output x: file name for frequency of each node
