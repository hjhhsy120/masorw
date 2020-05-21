#pragma once

#include "walklib.h"

using namespace std;

int **edge_js;
float **edge_qs;

// True weight of edges[dstadji], should be implemented in files like node2vec.cpp
// src -> edges[lastedgeidx] -> edges[dstadji]
extern inline float new_weight(int src, int lastedgeidx, int dstadji);

// NOTICE: only once!
inline void init_alias() {
  edge_js = static_cast<int **>(
      malloc(ne * sizeof(int *)));
  edge_qs = static_cast<float **>(
      malloc(ne * sizeof(float *)));
  memset(edge_js, 0, ne * sizeof(int *));
  memset(edge_qs, 0, ne * sizeof(float *));
}

// edge_js[src][lastedgeidx-offsets[src]] is the edge alias table for
// src -> edge[lastedgeidx] -> ?
// NOTICE: init first!
void one_edge_alias(int src, int lastedgeidx) {
  if (edge_js[lastedgeidx] != nullptr)
    return;
  int *one_edge_js;
  float *one_edge_qs;
  int dst = edges[lastedgeidx];
  int dst_degree = degrees[dst];
  one_edge_js = static_cast<int *>(
      malloc(dst_degree * sizeof(int)));
  one_edge_qs = static_cast<float *>(
      malloc(dst_degree * sizeof(float)));
  memset(one_edge_js, 0, dst_degree * sizeof(int));
  double sum = 0;
  for (int dstadji = offsets[dst]; dstadji < offsets[dst + 1]; dstadji++) {
    int dstadj = edges[dstadji];
    one_edge_qs[dstadji - offsets[dst]] = new_weight(src, lastedgeidx, dstadji);
    sum += one_edge_qs[dstadji - offsets[dst]];
  }
  for (int i = 0; i < dst_degree; i++)
    one_edge_qs[i] *= dst_degree / sum;
  init_walker(dst_degree, one_edge_js, one_edge_qs);
  if (edge_js[lastedgeidx] == nullptr) {
    edge_js[lastedgeidx] = one_edge_js;
    edge_qs[lastedgeidx] = one_edge_qs;
  }
  else {
    free(one_edge_js);
    free(one_edge_qs);
  }
}

// NOTICE: init first!
// edges[ed] -edges_r[ed]-> dst -> ?
void one_node_edge_alias(int dst) {
  for (int ed = offsets[dst]; ed < offsets[dst + 1]; ed++)
    one_edge_alias(edges[ed], edges_r[ed]);
}


// NOTICE: do not parallel!!!
void delete_one_edge_alias(int lastedgeidx) {
  if (edge_js[lastedgeidx] != nullptr) {
    free(edge_js[lastedgeidx]);
    edge_js[lastedgeidx] = nullptr;
  }
  if (edge_qs[lastedgeidx] != nullptr) {
    free(edge_qs[lastedgeidx]);
    edge_qs[lastedgeidx] = nullptr;
  }
}

// NOTICE: do not parallel!!!
void delete_one_node_edge_alias(int dst) {
  for (int ed = offsets[dst]; ed < offsets[dst + 1]; ed++)
    delete_one_edge_alias(edges_r[ed]);
}


// NOTICE: init first!
inline void all_edge_alias() {
  if (verbosity > 0) {
    LL sum = 0;
    for (int i = 0; i < nv; i++)
      sum += (LL)degrees[i] * degrees[i];
    cout << "Need " << sum * 8 / 1024 / 1024
         << " Mb for storing second-order degrees" << endl;
  }
#pragma omp parallel for schedule(dynamic)
  for (int src = 0; src < nv; src++)
    one_node_edge_alias(src); /*
    for (int lastedgeidx = offsets[src]; lastedgeidx < offsets[src + 1]; lastedgeidx++)
      one_edge_alias(src, lastedgeidx);*/
}

// NOTICE: exist first!
inline int sample_edge_alias(int src, int lastedgeidx, myrandom &r) {
  int dst = edges[lastedgeidx];
  return offsets[dst] + walker_draw(degrees[dst],
                                    edge_qs[lastedgeidx],
                                    edge_js[lastedgeidx],
                                    r);
}
