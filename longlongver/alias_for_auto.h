#pragma once

#include "walklib.h"

using namespace std;

int ***node_edge_js;
float ***node_edge_qs;

// True weight of edges[dstadji], should be implemented in files like node2vec.cpp
// src -> edges[lastedgeidx] -> edges[dstadji]
extern inline float new_weight(int src, LL lastedgeidx, LL dstadji);

// NOTICE: only once!
inline void init_alias_for_auto() {
  node_edge_js = static_cast<int ***>(
      malloc(nv * sizeof(int **)));
  node_edge_qs = static_cast<float ***>(
      malloc(nv * sizeof(float **)));
  memset(node_edge_js, 0, nv * sizeof(int ***));
  memset(node_edge_qs, 0, nv * sizeof(float ***));
}

/** alias table is stored in dst vertex **/
// edge_js[src][lastedgeidx-offsets[src]] is the edge alias table for
// src -> edge[lastedgeidx] -> ?
// NOTICE: init first!
void one_edge_alias(int dst, LL r_lastedgeidx) {
    LL lastedge_offset = r_lastedgeidx - offsets[dst];
    int dst_degree = degree(dst);

    int *one_edge_js;
    float *one_edge_qs;

    one_edge_js = static_cast<int *>(
        malloc(dst_degree * sizeof(int)));
    one_edge_qs = static_cast<float *>(
        malloc(dst_degree * sizeof(float)));
    memset(one_edge_js, 0, dst_degree * sizeof(int));
    double sum = 0;
    int src = edges[r_lastedgeidx];
    LL src_lastedgeidx = find_edge(src, dst);
    for (LL dstadji = offsets[dst]; dstadji < offsets[dst + 1]; dstadji++) {
      float w2 = new_weight(src, src_lastedgeidx, dstadji);
      one_edge_qs[dstadji - offsets[dst]] = w2;
      sum += w2;
    }
    for (int i = 0; i < dst_degree; i++)
      one_edge_qs[i] *= dst_degree / sum;
    init_walker(dst_degree, one_edge_js, one_edge_qs);
    //cout << "asign: "<< dst <<endl;
    node_edge_js[dst][lastedge_offset] = one_edge_js;
    node_edge_qs[dst][lastedge_offset] = one_edge_qs;
}

// NOTICE: init first
// The function means that any incoming edge of dst use alias for the next edge sampling.
// edges[ed] -edges_r[ed]-> dst -> ?
void one_node_edge_alias(int dst) {
  int dst_degree = degree(dst);
  node_edge_js[dst] = static_cast<int **>(malloc(dst_degree * sizeof(int *)));
  memset(node_edge_js[dst], 0, dst_degree * sizeof(int *));
  node_edge_qs[dst] = static_cast<float **>(malloc(dst_degree * sizeof(float *)));
  memset(node_edge_qs[dst], 0, dst_degree * sizeof(float *));
  for (LL ed = offsets[dst]; ed < offsets[dst + 1]; ed++)
    one_edge_alias(dst, ed);
}


// NOTICE: do not parallel!!!
void delete_one_edge_alias(int dst, LL r_lastedgeidx) {
  //TODO: current do nonthing!!!!
  LL lastedge_offset = r_lastedgeidx - offsets[dst];
  free(node_edge_js[dst][lastedge_offset]);
  free(node_edge_qs[dst][lastedge_offset]);
}

// NOTICE: do not parallel!!!
void delete_one_node_edge_alias(int dst) {
  //TODO: current do nonthing!!!!
  for (LL ed = offsets[dst]; ed < offsets[dst + 1]; ed++) {
    delete_one_edge_alias(dst, ed);
  }
  free(node_edge_js[dst]);
  free(node_edge_qs[dst]);
}


// NOTICE: init first!
inline void all_edge_alias() {
  if (verbosity > 0) {
    LL sum = 0;
    for (int i = 0; i < nv; i++) {
      LL d = degree(i);
      sum += d * d; //degrees[i] * degrees[i];
    }
    cout << "Need " << sum * 8 / 1024 / 1024
         << " Mb for storing second-order degrees" << endl;
  }
#pragma omp parallel for schedule(dynamic)
  for (int src = 0; src < nv; src++)
    one_node_edge_alias(src);
}

// NOTICE: exist first!
inline LL sample_edge_alias(int src, LL lastedgeidx, myrandom &r) {
  int dst = edges[lastedgeidx];
  LL r_lastedgeidx = find_edge(dst, src);
  LL lastedge_offset = r_lastedgeidx - offsets[dst];
  return offsets[dst] + walker_draw(degree(dst), //degrees[dst],
                                    node_edge_qs[dst][lastedge_offset],
                                    node_edge_js[dst][lastedge_offset],
                                    r);
}
