/**
Random, malloc, sigmoid, alias(draw) and graph related functions
Ramdom functions have 2 versions here
*/

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <queue>
#include <string.h>

// ADD memory info output
#include<unistd.h>


#if defined(__AVX2__) ||                                                       \
    defined(__FMA__) // icpc, gcc and clang register __FMA__, VS does not
#define VECTORIZE 1
#define AVX_LOOP _Pragma("omp simd")
#else
#define AVX_LOOP // empty
#endif

#ifndef UINT64_C // VS can not detect the ##ULL macro
#define UINT64_C(c) (c##ULL)
#endif

#define SIGMOID_BOUND 6.0  // computation range for fast sigmoid lookup table
// #define DEFAULT_ALIGN 128  // default align in bytes; abandoned
#define MAX_CODE_LENGTH 64 // maximum HSM code length. sufficient for nv < int64

using namespace std;

typedef long long LL;
typedef unsigned int uint;
typedef unsigned char byte;

// ADD size of memory in MB
int memory_size = 1024; // 1GB
// ADD space at the moment in byte
LL space_now = 0;

const int sigmoid_table_size = 1024; // This should fit in L1 cache
const float SIGMOID_RESOLUTION = sigmoid_table_size / (SIGMOID_BOUND * 2.0f);
float *sigmoid_table;

int n_threads = 1; // number of threads program will be using

int verbosity = 2; // verbosity level. 2 = report progress and tell jokes, 1 =
                   // report time and hsm size, 0 = errors, <0 = shut up
LL nv = 0, ne = 0; // number of nodes and edges
                    // We use CSR format for the graph matrix (unweighted).
                    // Adjacent nodes for vertex i are stored in
                    // edges[offsets[i]:offsets[i+1]]
LL *offsets;       // CSR index pointers for nodes.
int *edges;         // CSR offsets
// int *degrees;       // Node degrees

float alpha = 0.2;

/**
class for random number
*/
class myrandom {
public:
// http://xoroshiro.di.unimi.it/#shootout
// We use xoroshiro128+, the fastest generator available
uint64_t rng_seed0, rng_seed1;

myrandom(uint64_t seed) {
  for (int i = 0; i < 2; i++) {
    LL z = seed += UINT64_C(0x9E3779B97F4A7C15);
    z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);
    if (i == 0)
      rng_seed0 = z ^ (z >> 31);
    else
      rng_seed1 = z ^ (z >> 31);
  }
}

void reinit(uint64_t seed) {
  for (int i = 0; i < 2; i++) {
    LL z = seed += UINT64_C(0x9E3779B97F4A7C15);
    z = (z ^ z >> 30) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ z >> 27) * UINT64_C(0x94D049BB133111EB);
    if (i == 0)
      rng_seed0 = z ^ (z >> 31);
    else
      rng_seed1 = z ^ (z >> 31);
  }
}

static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

uint64_t lrand() {
  const uint64_t s0 = rng_seed0;
  uint64_t s1 = rng_seed1;
  const uint64_t result = s0 + s1;
  s1 ^= s0;
  rng_seed0 = rotl(s0, 55) ^ s1 ^ (s1 << 14);
  rng_seed1 = rotl(s1, 36);
  return result;
}

double drand() {
  const union un {
    uint64_t i;
    double d;
  } a = {UINT64_C(0x3FF) << 52 | lrand() >> 12};
  return a.d - 1.0;
}

int irand(int max) { return lrand() % max; }

int irand(int min, int max) { return lrand() % (max - min) + min; }
};


/**
// Malloc
// simply use malloc instead. aligned version is abandoned

inline void *
aligned_malloc(size_t size,
               size_t align) { // universal aligned allocator for win & linux
#ifndef _MSC_VER
  void *result;
  if (posix_memalign(&result, align, size))
    result = 0;
#else
  void *result = _aligned_malloc(size, align);
#endif
  return result;
}

inline void aligned_free(void *ptr) { // universal aligned free for win & linux
#ifdef _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}
*/

/**
sigmoid and shuffle functions
*/
void init_sigmoid_table() { // this shoould be called before fast_sigmoid once
  sigmoid_table = static_cast<float *>(
      malloc((sigmoid_table_size + 1) * sizeof(float)));
  for (int k = 0; k != sigmoid_table_size; k++) {
    float x = 2 * SIGMOID_BOUND * k / sigmoid_table_size - SIGMOID_BOUND;
    sigmoid_table[k] = 1 / (1 + exp(-x));
  }
}

float fast_sigmoid(float x) {
  if (x > SIGMOID_BOUND)
    return 1;
  if (x < -SIGMOID_BOUND)
    return 0;
  int k = (x + SIGMOID_BOUND) * SIGMOID_RESOLUTION;
  return sigmoid_table[k];
}

void shuffle(int *a, int n, myrandom &r) { // shuffles the array a of size n
  for (int i = n - 1; i >= 0; i--) {
    int j = r.irand(i + 1);
    int temp = a[j];
    a[j] = a[i];
    a[i] = temp;
  }
}


/**
Alias functions
*/
void init_walker(int n, int *j, float *probs) { // assumes probs are normalized
  vector<int> smaller, larger;
  for (int i = 0; i < n; i++) {
    if (probs[i] < 1)
      smaller.push_back(i);
    else
      larger.push_back(i);
  }
  while (smaller.size() != 0 && larger.size() != 0) {
    int small = smaller.back();
    smaller.pop_back();
    int large = larger.back();
    larger.pop_back();
    j[small] = large;
    probs[large] += probs[small] - 1;
    if (probs[large] < 1)
      smaller.push_back(large);
    else
      larger.push_back(large);
  }
}

int walker_draw(const int n, float *q, int *j, myrandom &r) {
  int kk = int(floor(r.drand() * n));
  return r.drand() < q[kk] ? kk : j[kk];
}


/**
Memory info function
*/
float showMemoryInfo(void)
{
    int pid = (int)getpid();
    struct {
        unsigned long size, resident, share, text, lib, data, dt;
    } result = {0,0,0,0,0,0,0};

    char FILE_NAME[255];
    sprintf(FILE_NAME, "/proc/%d/statm", pid);

    FILE *fp = fopen(FILE_NAME, "r");
    fscanf(fp, "%lu %lu %lu %lu %lu %lu %lu",
           &result.size, &result.resident, &result.share, &result.text, &result.lib, &result.data, &result.dt);
    fclose(fp);
    if (verbosity > 1) {
      printf("Process %d Memory Use:\n", pid);
      printf("size \tresident \tshare \t text \tlib \tdata \tdt\n");
      printf("==========================================================\n");
      printf("%lu \t%lu \t\t%lu \t%lu \t%lu \t%lu \t%lu\n",
              result.size, result.resident, result.share, result.text, result.lib, result.data, result.dt);
    }
    double usage = (double)sysconf(_SC_PAGESIZE) * result.resident / 1024 / 1024;
    cout << "Memory Use: " << usage << " MB" << endl;
    return usage;
}

double get_memory(void)
{
    int pid = (int)getpid();
    struct {
        unsigned long size, resident, share, text, lib, data, dt;
    } result = {0,0,0,0,0,0,0};

    char FILE_NAME[255];
    sprintf(FILE_NAME, "/proc/%d/statm", pid);

    FILE *fp = fopen(FILE_NAME, "r");
    fscanf(fp, "%lu %lu %lu %lu %lu %lu %lu",
           &result.size, &result.resident, &result.share, &result.text, &result.lib, &result.data, &result.dt);
    fclose(fp);
    return (double)sysconf(_SC_PAGESIZE) * result.resident / 1024 / 1024;
}



/**
Graph structure related
*/
inline int has_edge(int from, int to) {
  return binary_search(&edges[offsets[from]], &edges[offsets[from + 1]], to);
}

// binary search to find index of edge from src to dst
LL find_edge(int src, int dst) {
  LL l = offsets[src], r = offsets[src + 1], mid;
  while (l < r) {
    mid = (l + r) / 2;
    if (edges[mid] == dst)
      return mid;
    if (edges[mid] > dst)
      r = mid;
    else
      l = mid + 1;
  }
  return -1;
}

int degree(int id) {
  return offsets[id+1] - offsets[id];
}

// Success: 0; Failure: -1
int init_graph(string network_file) {
  ifstream inputFile(network_file, ios::in | ios::binary);
  if (inputFile.is_open()) {
    inputFile.seekg(0, ios::beg);
    inputFile.read(reinterpret_cast<char *>(&nv), sizeof(long long));
    inputFile.read(reinterpret_cast<char *>(&ne), sizeof(long long));
    offsets = static_cast<LL *>(
        malloc((nv + 1) * sizeof(LL)));
    edges =
        static_cast<int *>(malloc(ne * sizeof(int32_t)));
    inputFile.read(reinterpret_cast<char *>(offsets), nv * sizeof(LL));
    offsets[nv] = static_cast<LL>(ne);
    inputFile.read(reinterpret_cast<char *>(edges), sizeof(int32_t) * ne);
    if (verbosity > 0)
      cout << "nv: " << nv << ", ne: " << ne << endl;
    inputFile.close();
    return 0;
  } else {
    return -1;
  }
}
