#pragma once

#include "walklib.h"

using namespace std;

// ADD alias table for node (neighbor) sampling with weights
int **node_js;
float **node_qs;

LL *cnt_total; // total_number
LL *cnt_acc;   // accept_number

bool is_node2vec = false;
float **node_fac;

// compute new weight w'_src, should be implemented in files like node2vec.cpp
extern inline float new_weight(int src, float lastedgeweight, int dst, LL dstadji);
extern inline float new_weight(int src, LL lastedgeidx, LL dstadji);

// NOTICE: only once!
inline void init_reject() {
  node_js = static_cast<int **>(
      malloc(nv * sizeof(int *)));
  node_qs = static_cast<float **>(
      malloc(nv * sizeof(float *)));
  node_fac = static_cast<float **>(
      malloc(nv * sizeof(float *)));
  memset(node_js, 0, nv * sizeof(int *));
  memset(node_qs, 0, nv * sizeof(float *));
  memset(node_fac, 0, nv * sizeof(float *));
}

// NOTICE: init first!
void one_node_alias_reject(int dst) {
  int *one_node_js;
  float *one_node_qs;
  int src_degree = degree(src);
  one_node_js = static_cast<int *>(
      malloc(src_degree * sizeof(int)));
  one_node_qs = static_cast<float *>(
      malloc(src_degree * sizeof(float)));
  memset(one_node_js, 0, src_degree * sizeof(int));

  double sum = 0;
  for (LL dsti = offsets[src]; dsti < offsets[src + 1]; dsti++) {
    sum += 1.0;
  }
  for (LL dsti = offsets[src]; dsti < offsets[src + 1]; dsti++)
    one_node_qs[dsti - offsets[src]] = 1.0 * src_degree / sum;
  init_walker(src_degree, one_node_js, one_node_qs);
  if (node_js[src] == nullptr) {
    node_js[src] = one_node_js;
    node_qs[src] = one_node_qs;
    return true;
  }
  else {
    free(one_node_js);
    free(one_node_qs);
    return false;
  }
}

// NOTICE: init first!
// return true if succeeds, false if the fac table has been built
// TODO: the node factor should be simialr to be alias table to be stored in incoming edges.
bool one_node_fac_reject(int src) {
  return true;
}

bool one_node_reject(int src) {
  if (one_node_alias_reject(src))
    return one_node_fac_reject(src);
  else
    return false;
}


// NOTICE: do not parallel!!!
void delete_one_node_alias_reject(int src) {
  if (node_js[src] != nullptr) {
    free(node_js[src]);
    node_js[src] = nullptr;
  }
  if (node_qs[src] != nullptr) {
    free(node_qs[src]);
    node_qs[src] = nullptr;
  }
}

// NOTICE: do not parallel!!!
void delete_one_node_fac_reject(int src) {
  return;
}

// NOTICE: do not parallel!!!
void delete_one_node_reject(int src) {
  delete_one_node_alias_reject(src);
  delete_one_node_fac_reject(src);
}



inline void all_node_alias_reject() {
  if (verbosity > 0)
    cout << "Need " << ne * 8 / 1024 / 1024
         << " Mb for storing first-order degrees" << endl;

#pragma omp parallel for schedule(dynamic)
  for (int dst = 0; dst < nv; dst++){
    one_node_alias_reject(dst);
  }
}

// NOTICE: init first!
inline void all_node_reject() {
  if (verbosity > 0)
    cout << "Need " << ne * 12 / 1024 / 1024
         << " Mb for storing first-order degrees" << endl;

#pragma omp parallel for schedule(dynamic)
  for (int src = 0; src < nv; src++)
    one_node_reject(src);
}


inline LL sample_start_edge(int n1, myrandom &r) {
  return offsets[n1] + walker_draw(degree(n1),
                                    node_qs[n1],
                                    node_js[n1],
                                    r);
}


inline bool acc(int n0, LL lastedgeidx, LL nextedgeidx, myrandom &r) {
  if (is_node2vec) {
    return r.drand() <= (float)new_weight(n0, lastedgeidx, nextedgeidx)
            / node_fac_const;
  } else{
    int dst = edges[lastedgeidx];
    return r.drand() <= (float)new_weight(n0, lastedgeidx, nextedgeidx)
            / ((1 - alpha) / degree(dst) + alpha / degree(n0));
  }
}



// NOTICE: exist first!
inline LL sample_edge_reject(int n0, LL lastedgeidx, myrandom &r) {
  int n1 = edges[lastedgeidx], n2;
  LL nextedgeidx;
  do {
    nextedgeidx = offsets[n1] + walker_draw(degree(n1),
                                    node_qs[n1],
                                    node_js[n1],
                                    r);
  } while (!acc(n0, lastedgeidx, nextedgeidx, r));
  return nextedgeidx;
}

/**
functions below are for counting accept_number, total_number
*/

// NOTICE: init before count, only once
inline void init_reject_cnt() {
  cnt_total = static_cast<LL *>(malloc(n_threads * nv * sizeof(LL)));
  memset(cnt_total, 0, n_threads * nv * sizeof(LL));
  cnt_acc = static_cast<LL *>(malloc(n_threads * nv * sizeof(LL)));
  memset(cnt_acc, 0, n_threads * nv * sizeof(LL));
  cout << "cnt array length : " << n_threads * nv << endl;
}

// with tid, it counts how many accepted and how many in total
inline LL sample_edge_reject(int n0, LL lastedgeidx, myrandom &r, int tid) {
  int n1 = edges[lastedgeidx], n2;
  LL nextedgeidx, num = 0;
  do {
    ++num;
    nextedgeidx = offsets[n1] + walker_draw(degree(n1),
                                    node_qs[n1],
                                    node_js[n1],
                                    r);
  } while (!acc(n0, lastedgeidx, nextedgeidx, r));
  cnt_total[tid * nv + n1] += num;
  cnt_acc[tid * nv + n1]++;
  return nextedgeidx;
}

// output to file
// the ith line is info of node i (with the ith smallest name in original dataset)
// 3 numbers split by space: accept_number total_number accept_rate
inline void output_reject_cnt(char *reject_cnt_name) {
  for (int i = 1; i < n_threads; i++) {
    for (int j = 0; j < nv; j++) {
      cnt_total[j] += cnt_total[i * nv + j];
      cnt_acc[j] += cnt_acc[i * nv + j];
    }
  }
  FILE *fp;
  fp = fopen(reject_cnt_name, "w");
  for (int i = 0; i < nv; i++) {
    fprintf(fp, "%llu %llu ", cnt_acc[i], cnt_total[i]);
    if (cnt_total > 0)
      fprintf(fp, "%lf\n", (double)(cnt_acc[i]) / cnt_total[i]);
    else
      fprintf(fp, "0\n");
  }
  fclose(fp);
}


// compute all reject time
inline void compute_rej_time(float * rej_time) {
#pragma omp parallel for schedule(dynamic)
  for (int dst = 0; dst < nv; dst++) {
    double sum = 0;
    // edges[ed] -edges_r[ed]-> dst -ed2-> ?
    for (LL ed = offsets[dst]; ed < offsets[dst + 1]; ed++) {
      double max_fac = 0;
      double totw = 0; // W
      double totw2 = 0; // W'
      for (LL ed2 = offsets[dst]; ed2 < offsets[dst + 1]; ed2++) {
        double w2 = new_weight(edges[ed], 1.0, dst, ed2);
        totw += 1.0;
        totw2 += w2;
        double fac = w2 / 1.0;
        if (max_fac < fac)
          max_fac = fac;
      }
      // k = W * max{w'_i/w_i} / W'
      sum += totw * max_fac / totw2;
    }
    sum /= degree(dst);
    rej_time[dst] = sum;
  }
}

// compute all reject time with bound
inline void compute_rej_time_with_bound(float * rej_time, int rej_bound) {
  all_node_alias_reject();
#pragma omp parallel
{
  myrandom myr(time(nullptr));
#pragma omp for schedule(dynamic)
  for (int dst = 0; dst < nv; dst++) {
    double sum = 0;
    // edges[ed] -edges_r[ed]-> dst -ed2-> ?
    for (LL ed = offsets[dst]; ed < offsets[dst + 1]; ed++) {
      double max_fac = 0;
      double totw = 0; // W
      double totw2 = 0; // W'
      if (degree(dst) <= rej_bound) {
        for (LL ed2 = offsets[dst]; ed2 < offsets[dst + 1]; ed2++) {
          double w2 = new_weight(edges[ed], 1.0, dst, ed2);
          totw += 1.0;
          totw2 += w2;
          double fac = w2;
          if (max_fac < fac)
            max_fac = fac;
        }
      } else {
        one_node_alias_reject(dst);
        for (int i = 0; i < rej_bound; i++) {
          LL ed2 = offsets[dst] + walker_draw(degree(dst),
                            node_qs[dst],
                            node_js[dst],
                            myr);
          double w2 = new_weight(edges[ed], 1.0, dst, ed2);
          totw += 1.0;
          totw2 += w2;
          double fac = w2;
          if (max_fac < fac)
            max_fac = fac;
        }
      }
      // k = W * max{w'_i/w_i} / W'
      sum += totw * max_fac / totw2;
    }
    sum /= degree(dst);
    rej_time[dst] = sum;
    delete_one_node_alias_reject(dst);
  }
}
}

