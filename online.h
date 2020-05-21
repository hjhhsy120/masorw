#pragma once

#include "walklib.h"

using namespace std;

double **prob;

extern inline float new_weight(int src, LL lastedgeidx, LL dstadji);

// NOTICE: only once!
inline void init_online() {
  int maxd = 0;
  for (int i = 0; i < nv; i++)
    maxd = (maxd < degree(i)) ? degree(i) : maxd;
  prob = static_cast<double **>(
      malloc(n_threads * sizeof(double *)));
  for (int i = 0; i < n_threads; i++) {
    double *pr = static_cast<double *>(
      malloc(maxd * sizeof(double)));
    prob[i] = pr;
  }
}

// NOTICE: tid is neede here!
inline LL sample_start_edge_online(int n1, myrandom &r, int tid) {
  LL n1_offset = offsets[n1];
  int n1_degree = degree(n1);
  prob[tid][0] = 1.0;
  for (int i = 1; i < n1_degree; i++)
    prob[tid][i] = prob[tid][i - 1] + 1.0;
  double pp;
  pp = r.drand() * prob[tid][n1_degree - 1];
  for (int i = 0; i < n1_degree; i++)
    if (pp <= prob[tid][i])
      return n1_offset + i;
  return offsets[n1 + 1] - 1;
}

// NOTICE: tid is neede here!
inline LL sample_edge_online(int n0, LL lastedgeidx, myrandom &r, int tid) {
  int n1 = edges[lastedgeidx];
  LL n1_offset = offsets[n1];
  int n1_degree = degree(n1);
  prob[tid][0] = new_weight(n0, lastedgeidx, n1_offset);
  for (int i = 1; i < n1_degree; i++)
    prob[tid][i] = prob[tid][i - 1] + new_weight(n0, lastedgeidx, n1_offset + i);
  double pp;
  pp = r.drand() * prob[tid][n1_degree - 1];
  for (int i = 0; i < n1_degree - 1; i++)
    if (pp <= prob[tid][i])
      return n1_offset + i;
  return offsets[n1 + 1] - 1;
}
