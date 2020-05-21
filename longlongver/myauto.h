#pragma once

#include "alias_for_auto.h"
#include "rej.h"
#include "online.h"

using namespace std;

#define USEALIAS 0
#define USEREJECT 1
#define USEONLINE 2
// USEAUTO is not for single node!
#define USEAUTO 3

#define ONLINE2REJECT 0
#define ONLINE2ALIAS 1
#define REJECT2ALIAS 2

#define AUTO_FLAG 0

// ADD node type: use alias, reject or online
char *node_type;
char *prev_node_type;

// ADD whether recursively search slopes
bool recursive = false;
// ADD whether to use degree * slope (true) or just slope (false)
bool another_cost = false;
// if degree > rej_bound then sample to get k
// if rej_bound=0 then do not sample
int rej_bound = 0;

float *rej_time;

class slope {
public:
  float val;
  int node;
  LL space_inc;
  char change_type;
  char used;
  void set_slope(float f, int i, LL u, char c) {
    val = f;
    node = i;
    space_inc = u;
    change_type = c;
    used = 0;
  }

  bool can_use_slope() {
    if (change_type == ONLINE2REJECT && node_type[node] == USEONLINE) return true;
    if (change_type == ONLINE2ALIAS && node_type[node] == USEONLINE) return true;
    if (change_type == REJECT2ALIAS && node_type[node] == USEREJECT) return true;
    return false;
  }

  // NOTICE: used is not handled in functions below!

  // DO NOT handle space or slope_pointer here
  // NOTICE: DO use it before real_use_slope
  void use_slope() {
    switch(change_type) {
    case ONLINE2REJECT:
      node_type[node] = USEREJECT;
      break;
    case ONLINE2ALIAS:
    default: // REJECT2ALIAS
      node_type[node] = USEALIAS;
    }
  }

  // DO NOT handle space or slope_pointer here
  // NOTICE: Do use it before restore_slope
  void cancel_slope() {
    switch(change_type) {
    case ONLINE2REJECT:
    case ONLINE2ALIAS:
      node_type[node] = USEONLINE;
      break;
    default: // REJECT2ALIAS
      node_type[node] = USEREJECT;
    }
  }

  // NOTICE: DO use it after use_slope()
  // ONLINE->REJECT->ALIAS dealt in REJECT2ALIAS
  // can parallelize
  void real_use_slope() {
    switch(change_type) {
    case ONLINE2REJECT:
      // ONLINE->REJECT
      if (node_type[node] == USEREJECT)
        one_node_reject(node);
      break;
    case ONLINE2ALIAS:
    default: // REJECT2ALIAS
      // ONLINE->ALIAS or ONLINE->REJECT->ALIAS
      if (prev_node_type[node] == USEONLINE) {
        one_node_alias_reject(node);
        one_node_edge_alias(node);
      } else { // REJECT->ALIAS
        delete_one_node_fac_reject(node);
        one_node_edge_alias(node);
      }
    }
  }

  // NOTICE: DO use it after cancel_slope()
  // ALIAS->REJECT->ONLINE in ONLINE2REJECT
  // can parallelize
  void real_cancel_slope() {
    switch(change_type) {
    case ONLINE2REJECT:
    case ONLINE2ALIAS:
      // REJECT->ONLINE
      if (prev_node_type[node] == USEREJECT)
        delete_one_node_reject(node);
      else { // ALIAS->ONLINE or ALIAS->REJECT->ONLINE
        delete_one_node_alias_reject(node);
        delete_one_node_edge_alias(node);
      }
      break;
    default: // REJECT2ALIAS
      // ALIAS->REJECT
      if (node_type[node] == USEREJECT) {
        delete_one_node_edge_alias(node);
        one_node_fac_reject(node);
      }
    }
  }
};

bool slope_cmp(const slope &s1, const slope &s2) {
  if(s1.val < s2.val)
    return true;
  return false;
}

slope *slopes;
int slope_cnt; //, slope_pointer;

/** Helper functions **/
void output_auto_types(char *auto_types_name) {
  FILE *fp = fopen(auto_types_name, "w");
  fprintf(fp, "type,degree\n");
  for (int i = 0; i < nv; i++) {
    switch(node_type[i]) {
    case USEALIAS:
      fprintf(fp, "alias");
      break;
    case USEREJECT:
      fprintf(fp, "reject");
      break;
    default: // USEONLINE
      fprintf(fp, "online");
    }
    fprintf(fp, ",%d\n", degree(i));
  }
  fclose(fp);
}

void compute_node_type_dist(char *auto_types_name) {
  int count[4] = {0,0,0,0};
  for (int i = 0; i < nv; i++) {
    count[node_type[i]]++;
  }
  cout <<"Node Type Dist: A:"<< count[0]
      <<", R: " << count[1]
      <<", D: " << count[2]
      <<", O: " << count[3] <<endl;

  if(auto_types_name != nullptr)
      output_auto_types(auto_types_name);
}

void compute_mem_accumlative_dist(char const *auto_types_name) {
  int slope_pointer = 0;
  FILE *fp = fopen(auto_types_name, "w");
  char *node_type_tmp = static_cast<char *>(malloc(nv * sizeof(char)));
  for (int node = 0; node < nv; node++) {
    node_type_tmp[node] = USEONLINE;
  }
  while (slope_pointer < slope_cnt) {
    LL space_inc = slopes[slope_pointer].space_inc;
    char change_type = slopes[slope_pointer].change_type;
    int node = slopes[slope_pointer].node;

    if ((change_type == ONLINE2REJECT && node_type_tmp[node] == USEONLINE) ||
        (change_type == ONLINE2ALIAS && node_type_tmp[node] == USEONLINE) ||
        (change_type == REJECT2ALIAS && node_type_tmp[node] == USEREJECT)) {
        space_now += space_inc;
        switch(change_type) {
           case ONLINE2REJECT:
              node_type_tmp[node] = USEREJECT;
              break;
           case ONLINE2ALIAS:
           default: // REJECT2ALIAS
              node_type_tmp[node] = USEALIAS;
        }
        fprintf(fp, "%d %lld %lld\n", slope_pointer, space_inc, space_now);
    }
    slope_pointer++;
  }
  fclose(fp);
  free(node_type_tmp);
}


// compute_reject_time has been moved to rej.h

void update_memory() {
  auto begin1 = chrono::steady_clock::now();

  // int prev_slope_pointer = slope_pointer;
  LL prev_space_now = space_now;
  space_now = 0;
  LL bound = (LL)memory_size << 20; // transfer to byte
  memcpy(prev_node_type, node_type, nv * sizeof(char));
  for (int i = 0; i < nv; i++)
    node_type[i] = USEONLINE;
  // increase memory
  int slope_pointer = 0;
  int skipped = 0;
  int full_slope_pointer = slope_cnt;
  while (slope_pointer < slope_cnt) {
    LL space_inc = slopes[slope_pointer].space_inc;
    if (space_now + space_inc > bound || !slopes[slope_pointer].can_use_slope()) {
      if (skipped == 0) {
        if (!slopes[slope_pointer].can_use_slope())
          skipped--;
        else {
          full_slope_pointer = slope_pointer;
          if (!recursive)
            break;
        }
      }
      skipped++;
      slope_pointer++;
      continue;
    }
    space_now += space_inc;
    slopes[slope_pointer].use_slope();
    slope_pointer++;
  }
  auto end1 = chrono::steady_clock::now();
  if (verbosity > 0)
    cout << "Finished greedy algorithm, using "
         << chrono::duration_cast<chrono::duration<float>>(end1 - begin1).count()
         << " s to run. "
         << "space_now: " << space_now <<", total memory: " << bound
         <<endl;

  auto begin2 = chrono::steady_clock::now();

/**
NOTICE: free slopes here!
*/
  free(slopes);
// initialize space
// TODO: this solution cannot support incremental memory update.
init_alias_for_auto();

compute_node_type_dist(nullptr);

cout << "Memory Info before update memory stucture after initliaze" <<endl;
showMemoryInfo();
#pragma omp parallel for schedule(dynamic)
  for (int node = 0; node < nv; node++) {
    if (node_type[node] == USEONLINE) {
      if (prev_node_type[node] == USEALIAS) {
        delete_one_node_alias_reject(node);
        delete_one_node_edge_alias(node);
      } else if (prev_node_type[node] == USEREJECT)
        delete_one_node_reject(node);
    } else if (node_type[node] == USEREJECT) {
      if (prev_node_type[node] == USEONLINE)
        one_node_reject(node);
      else if (prev_node_type[node] == USEALIAS) {
        delete_one_node_edge_alias(node);
        one_node_fac_reject(node);
      }
    } else { // node_type[node] == USEALIAS,
      if (prev_node_type[node] == USEONLINE) {
        one_node_alias_reject(node);
        one_node_edge_alias(node);
      } else if (prev_node_type[node] == USEREJECT) {
        delete_one_node_fac_reject(node);
        one_node_edge_alias(node);
      }
    }
  }

  auto end2 = chrono::steady_clock::now();

  if (verbosity > 0) {
    cout << "Finished updating memory, using "
         << chrono::duration_cast<chrono::duration<float>>(end2 - begin2).count()
         << " s to run" << endl;
    cout << "Memory usage: " << (double)space_now / 1024 / 1024
         << " Mb" << endl;
    cout << "(increased "<< double(space_now - prev_space_now) / 1024 / 1024 << " Mb)" << endl;
    if (recursive)
      cout << "Another " << slope_cnt - skipped - full_slope_pointer
           << " slopes used after first skip" << endl;
  }
}

void update_auto() {
  slope_cnt = 0;
  auto begin2 = chrono::steady_clock::now();
  LL tot_mem = 0;
  for (int node = 0; node < nv; node++) {
    node_type[node] = USEONLINE;
    int d = degree(node);
    LL alias_space = (LL)d * (d+1) //(LL)degrees[node] * (degrees[node] + 1)
                    * (sizeof(int) + sizeof(float));
    tot_mem += alias_space;
        // here + 1 is sample_start_edge
    LL reject_space = degree(node) * (sizeof(int) + sizeof(float)); // + sizeof(float));

    float alias_time = (2 + log(d) / log(2)); //alias table is stored in dst node, need binary search to find the reverse edge.
    float reject_time = (2 + log(d) / log(2)) * rej_time[node];
    float online_time = (2 + log(d) / log(2)) * d;
    float slope_online_reject = (online_time - reject_time)/(0LL - reject_space);
    float slope_online_alias = (online_time - alias_time)/(0LL - alias_space);
    float slope_reject_alias = (reject_time - alias_time)/(reject_space - alias_space);

    // NOTICE: different from before
    if (another_cost) {
      if (alias_time < online_time)
        slopes[slope_cnt++].set_slope(d * slope_online_alias, node, alias_space, ONLINE2ALIAS);
      if (reject_time < online_time)
        slopes[slope_cnt++].set_slope(d * slope_online_reject, node, reject_space, ONLINE2REJECT);
      if (alias_time < reject_time)
        slopes[slope_cnt++].set_slope(d * slope_reject_alias, node, alias_space - reject_space, REJECT2ALIAS);
    } else {
      if (alias_time <= online_time)
        slopes[slope_cnt++].set_slope(slope_online_alias, node, alias_space, ONLINE2ALIAS);
      if (reject_time < online_time)
        slopes[slope_cnt++].set_slope(slope_online_reject, node, reject_space, ONLINE2REJECT);
      if (alias_time <= reject_time)
        slopes[slope_cnt++].set_slope(slope_reject_alias, node, alias_space - reject_space, REJECT2ALIAS);
    }
  }
  free(rej_time);
  sort(slopes, slopes + slope_cnt, slope_cmp);
  auto end2 = chrono::steady_clock::now();
  //TODO: Here we should compute the accumlative distribution of memory increasing.
  // char const *fp = "slop_dist";
  cout <<"TOT_MEM (ALIAS):" << tot_mem <<endl;
  // compute_mem_accumlative_dist(fp);
  if (verbosity > 0)
    cout << "Finished computing slopes and sorting, using "
         << chrono::duration_cast<chrono::duration<float>>(end2 - begin2).count()
         << " s to run" << endl;

  // change memory_size to rest size!
  space_now = 0;
  update_memory();
  compute_node_type_dist(nullptr);
}

// NOTICE: only once!
inline void init_auto() {
  init_reject();
  // init_alias();
  init_online();
  node_type = static_cast<char *>(malloc(nv * sizeof(char)));
  prev_node_type = static_cast<char *>(malloc(nv * sizeof(char)));
  slopes = static_cast<slope *>(malloc(nv * 3 * sizeof(slope)));

  cout << "Before executing the cost-based optimizer " <<endl;
  // cout << sizeof(slopes) << endl;
  showMemoryInfo();
  // memory_size = memory_size - get_memory();
  if (memory_size <= 0) {
    cout << "Memory not enough! Aborting now..." << endl;
    exit(0);
  }

  // slope_pointer = 0;
  rej_time = static_cast<float *>(malloc(nv * sizeof(float)));
  memset(rej_time, 0, nv * sizeof(float));
  cout <<"beging computing reject time: " << rej_bound << endl;
  auto begin1 = chrono::steady_clock::now();
  if (rej_bound == 0)
    compute_rej_time(rej_time);
  else
    compute_rej_time_with_bound(rej_time, rej_bound);
  auto end1 = chrono::steady_clock::now();
  if (verbosity > 0)
    cout << "Finished computing reject time, using "
         << chrono::duration_cast<chrono::duration<float>>(end1 - begin1).count()
         << " s to run" << endl;

  update_auto();
}


inline LL sample_start_edge_auto(int n0, myrandom &r, int tid) {
  if (node_type[n0] == USEONLINE)
    return sample_start_edge_online(n0, r, tid);
  return sample_start_edge(n0, r);
}

inline LL sample_edge_auto(int n0, LL lastedgeidx, myrandom &r, int tid) {
  int n1 = edges[lastedgeidx];
  switch(node_type[n1]) {
    case USEALIAS:
      return sample_edge_alias(n0, lastedgeidx, r);
    case USEREJECT:
      // for reject, node alias table is built in all_node_auto
      return sample_edge_reject(n0, lastedgeidx, r);
    default: // USEONLINE:
      return sample_edge_online(n0, lastedgeidx, r, tid);
  }
}
