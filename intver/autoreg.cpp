/*****
main() and other functions for node2vec
including acc for reject and Generate_edge_alias for alias
*/

#include "myauto.h"
//#include "myauto_desc.h"
//#include "myauto_asc.h"

using namespace std;

/**
Related to node2vec task
*/
// ADD whether compute accept_rate and file name of that
bool get_reject_cnt = false;
char *reject_cnt_name;

// ADD output node types for auto
char *auto_types_name = nullptr;

int pi_div_n = 4;     // number of random walks / n
float dump_fac = 0.85; // dump factor of pagerank
int max_len = 20; // max walk length

// alpha for auto regression
float alpha = 0.2;
// sum of weight of each node
float *sum_weight;

// count the number of stopping at each node
int *node_cnt;

// ADD what method to use
int mymethod = -1;

string network_file, embedding_file;

// ADD my random generator
myrandom mainrandom(time(nullptr));
// ADD whether use time as random seed
// If fix seed in argv, then false; otherwise true
bool truerandom = true;

int ArgPos(char *str, int argc, char **argv) {
  for (int a = 1; a < argc; a++)
    if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        cout << "Argument missing for " << str << endl;
        exit(1);
      }
      return a;
    }
  return -1;
}

// DIFFERENT
/**
True weight
*/
// src -> edges[lastedgeidx] -> edges[dstadji]
inline float new_weight(int src, int lastedgeidx, int dstadji) {
  int dst = edges[lastedgeidx];
  int dstadj = edges[dstadji];
  LL ed = find_edge(src, dstadj);
  if (ed == -1)
    return (1 - alpha) * weights[dstadji] / sum_weight[dst];
  else
    return (1 - alpha) * weights[dstadji] / sum_weight[dst] + alpha * weights[ed] / sum_weight[src];
//return 1.0;
}


int  Generate_walks(int start_point) {
  int total_steps = pi_div_n * nv;
//  int ret = 0;
#pragma omp parallel
{
  int tid = omp_get_thread_num();
  myrandom *trandom, one_random(time(nullptr) + mainrandom.irand(nv));
  if (truerandom)
    // trandom = &one_random[tid];
    trandom = &one_random;
  else
    trandom = &mainrandom;
  
//  auto begin = chrono::steady_clock::now();
#pragma omp for schedule(dynamic)
  for (int step = 0; step < total_steps; step++) {
    if (degrees[start_point] == 0 || trandom->drand() > dump_fac) {
#pragma omp atomic
      node_cnt[start_point]++;
      continue;
    }
//      auto now = chrono::steady_clock::now();
//      float during = chrono::duration_cast<chrono::duration<float>>(now-begin).count();
//      if (during > 3600) {
//          cout <<"TLE!!!"<<endl;
//          ret = -1;
//          break;
//      }

    int lastedgeidx;
    switch (mymethod) {
    case USEALIAS:
    case USEREJECT:
      lastedgeidx = sample_start_edge(start_point, *trandom);
      break;
    case USEONLINE:
      lastedgeidx = sample_start_edge_online(start_point, *trandom, tid);
      break;
    default:
      lastedgeidx = sample_start_edge_auto(start_point, *trandom, tid);
    }
    int prev = start_point;
    int nxt = edges[lastedgeidx];
    for (int i = 1; i < max_len; i++) {
      if (degrees[nxt] == 0 || trandom->drand() > dump_fac)
        break;
      switch (mymethod) {
      case USEALIAS:
        lastedgeidx = sample_edge_alias(prev, lastedgeidx, *trandom);
        break;
      case USEREJECT:
        if (get_reject_cnt) // this needs tid!
          lastedgeidx = sample_edge_reject(prev, lastedgeidx, *trandom, tid);
        else
          lastedgeidx = sample_edge_reject(prev, lastedgeidx, *trandom);
        break;
      case USEONLINE:
        lastedgeidx = sample_edge_online(prev, lastedgeidx, *trandom, tid);
        break;
      default: // USEAUTO
        lastedgeidx = sample_edge_auto(prev, lastedgeidx, *trandom, tid);
      }
      prev = nxt;
      nxt = edges[lastedgeidx];
    }
#pragma omp atomic
    node_cnt[nxt]++;
  }
}
   return 0;
}

void testUpdate(string logfile) {
  ifstream inputFile(logfile, ios::in | ios::binary);
  string line;
  init_auto();
  memory_size = 0;
  int cnt = 0;
  if (inputFile.is_open()) {
    while(getline(inputFile, line)){
        int delta = atoi(line.c_str());
        memory_size += delta;
        cout <<logfile<<": "<< cnt <<", Current Mem: " << memory_size << " delta: " << delta <<endl;
        cnt ++;
        if(memory_size >= 92160){
          cout << "Memory Exceeded" << endl;
           continue;
        }
        update_memory(0.0);
        cout <<endl;
        // showMemoryInfo();
    }
  } else {
    cout << "Failed to open files." << endl;
  }
}

void print(float init_time, float app_time, float used_mem_size, int query_num, float graph_size, float init2_time) {
  cout << "#LOG: "<< mymethod
        <<" "<< AUTO_FLAG
        << " (" << recursive<<","<<another_cost <<","<<rej_bound
        << ") " << memory_size
        << " " << n_threads
        << " " << alpha
        << " " << network_file
        << " " << graph_size
        << " :"
        << " " << init_time
        << " " << init2_time
        << " " << app_time
        << " " << used_mem_size 
        << " " << query_num
        << endl;
}

int main(int argc, char **argv) {
  auto ini = chrono::steady_clock::now();
  int a;
  char *start_file, *result_file = nullptr;
  init_sigmoid_table();

  if ((a = ArgPos(const_cast<char *>("-recursive"), argc, argv)) > 0)
    recursive = true;

  if ((a = ArgPos(const_cast<char *>("-anothercost"), argc, argv)) > 0)
    another_cost = true;

  if ((a = ArgPos(const_cast<char *>("-rejbound"), argc, argv)) > 0)
    rej_bound = atoi(argv[a + 1]);;

  if ((a = ArgPos(const_cast<char *>("-memory"), argc, argv)) > 0)
    memory_size = atoi(argv[a + 1]);

  if ((a = ArgPos(const_cast<char *>("-verbose"), argc, argv)) > 0)
    verbosity = atoi(argv[a + 1]);

  if ((a = ArgPos(const_cast<char *>("-input"), argc, argv)) > 0)
    network_file = argv[a + 1];
  else {
    if (verbosity > 0)
      cout << "Input file not given! Aborting now.." << endl;
    return 1;
  }
  if ((a = ArgPos(const_cast<char *>("-output"), argc, argv)) > 0)
    result_file = argv[a + 1];

  if ((a = ArgPos(const_cast<char *>("-startpoints"), argc, argv)) > 0)
    start_file = argv[a + 1];
  else {
    if (verbosity > 0)
      cout << "Start points file not given! Aborting now.." << endl;
    return 1;
  }
  if ((a = ArgPos(const_cast<char *>("-seed"), argc, argv)) > 0) {
    mainrandom.reinit(atoi(argv[a + 1]));
    truerandom = false;
  }
  if ((a = ArgPos(const_cast<char *>("-threads"), argc, argv)) > 0)
    n_threads = atoi(argv[a + 1]);

  if ((a = ArgPos(const_cast<char *>("-pidivn"), argc, argv)) > 0)
    pi_div_n = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-dumpfac"), argc, argv)) > 0)
    dump_fac = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-maxlen"), argc, argv)) > 0)
    max_len = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-alpha"), argc, argv)) > 0)
    alpha = atof(argv[a + 1]);

  if ((a = ArgPos(const_cast<char *>("-alias"), argc, argv)) > 0)
    mymethod = USEALIAS;
  else if ((a = ArgPos(const_cast<char *>("-reject"), argc, argv)) > 0)
    mymethod = USEREJECT;
  else if ((a = ArgPos(const_cast<char *>("-online"), argc, argv)) > 0)
    mymethod = USEONLINE;
  else // default: USEAUTO
    mymethod = USEAUTO;
  if (verbosity > 1)
    cout << "type: " << mymethod << endl;

  // ADD whether to compute accept rate
  if (mymethod == USEREJECT && (a = ArgPos(const_cast<char *>("-rejectcnt"), argc, argv)) > 0)
    get_reject_cnt = true;
  if ((a = ArgPos(const_cast<char *>("-cntname"), argc, argv)) > 0)
    reject_cnt_name = argv[a + 1];
  else if (get_reject_cnt) {
    if (verbosity > 0)
      cout << "Reject count file not given! Aborting now.." << endl;
    return 1;
  }

  // ADD whether to output node type (alias/reject/online)
  if (mymethod == USEAUTO && (a = ArgPos(const_cast<char *>("-autotypes"), argc, argv)) > 0)
    auto_types_name = argv[a + 1];

  omp_set_num_threads(n_threads);
  if (init_graph(network_file) < 0)
    return 1;

  // DIFFERENT
  sum_weight = static_cast<float *>(malloc(nv * sizeof(float)));
#pragma omp parallel for schedule(dynamic)
  for (int node = 0; node < nv; node++) {
    float one_sum_weight = 0.0;
    for (LL ed = offsets[node]; ed < offsets[node + 1]; ed++)
      one_sum_weight += weights[ed];
    sum_weight[node] = one_sum_weight;
  }
  node_cnt = static_cast<int *>(malloc(nv * sizeof(int)));

  FILE *fp = fopen(start_file, "r");
  int start_cnt, *start_points;
  fscanf(fp, "%d", &start_cnt);
  start_points = static_cast<int *>(malloc(start_cnt * sizeof(int)));
  for (int i = 0; i < start_cnt; i++)
    fscanf(fp, "%d", &start_points[i]);
  fclose(fp);

  cout << "Show memory info. before init. sampler methods" <<endl;
  float graph_size = showMemoryInfo();

  // ADD memorychange
  if ((a = ArgPos(const_cast<char *>("-memorychange"), argc, argv)) > 0) {
    string logfile = argv[a+1];
    testUpdate(logfile);
    return 0;
  }

  auto ini2 = chrono::steady_clock::now();

  if (mymethod == USEREJECT) {
    init_reject();
    all_node_reject();
  }
  if (mymethod == USEALIAS) {
    init_reject();
    all_node_alias_reject();
    init_alias();
    all_edge_alias();
  }
  if (mymethod == USEONLINE)
    init_online();

  if (get_reject_cnt)
    init_reject_cnt();

  if (mymethod == USEAUTO) {
    init_auto();
  }

  //The total memory should not exceed 94GB.
  if (get_memory() > 94*1024) {
    cout <<"Memory exceeded: " << get_memory() <<endl;
    return 0;
  }


  if (verbosity > 0)
#if VECTORIZE
    cout << "Using vectorized operations" << endl;
#else
    cout << "Not using vectorized operations (!)" << endl;
#endif
  auto inied = chrono::steady_clock::now();
  if (verbosity > 0)
    cout << "Initialization took "
         << chrono::duration_cast<chrono::duration<float>>(inied - ini).count()
         << " s to run" << endl;

  int true_start_cnt = 0;
  float time_sum = 0.0;
  for (int i = 0; i < start_cnt; i++) {
    if (degrees[start_points[i]] == 0)
      continue;
    memset(node_cnt, 0, nv * sizeof(int));
    auto begin = chrono::steady_clock::now();
    int status = Generate_walks(start_points[i]);
    auto end = chrono::steady_clock::now();
    time_sum += chrono::duration_cast<chrono::duration<float>>(end - begin).count();
    true_start_cnt++;
    if(status == -1) {
       time_sum = 1e20;
       break;
    }
  }

  float init_time = chrono::duration_cast<chrono::duration<float>>(inied - ini).count();
  float init2_time = chrono::duration_cast<chrono::duration<float>>(inied - ini2).count();
  float app_time = ((true_start_cnt > 0) ? (time_sum / true_start_cnt) : 0);
  float used_mem_size = -1;

  if (verbosity > 1)
    cout << endl;
  if (verbosity > 0) {
    cout << "Finished generation" << endl;
    cout << "Walks generation average time of "
         << true_start_cnt << " start points: "
         << ((true_start_cnt > 0) ? (time_sum / true_start_cnt) : 0)
         << " s to run" << endl;
    used_mem_size = showMemoryInfo();
    if (get_reject_cnt)
      output_reject_cnt(reject_cnt_name);
    if (auto_types_name != nullptr)
      output_auto_types(auto_types_name);
  }
  print(init_time, app_time, used_mem_size, true_start_cnt, graph_size, init2_time);
  if (result_file != nullptr) {
    fp = fopen(result_file, "w");
    for (int i = 0; i < nv; i++)
        fprintf(fp, "%d\n", node_cnt[i]);
    fclose(fp);
  }
  return 0;
}
