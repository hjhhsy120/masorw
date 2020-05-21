/*****
main() and other functions for node2vec
including acc for reject and Generate_edge_alias for alias
*/

#include "myauto.h"

using namespace std;

/**
Related to node2vec task
*/
// ADD whether only generate walks
bool onlywalk = false;
// ADD whether compute accept_rate and file name of that
bool get_reject_cnt = false;
char *reject_cnt_name;

// ADD output node types for auto
char *auto_types_name = nullptr;

float initial_lr = 0.025f; // initial learning rate
int n_hidden = 128;   // DeepWalk parameter "d" = embedding dimensionality aka
                      // number of nodes in the hidden layer
int n_walks = 10;     // DeepWalk parameter "\gamma" = walks per vertex
int walk_length = 80; // DeepWalk parameter "t" = length of the walk
int window_size = 10; // DeepWalk parameter "w" = window size
int n_neg_samples = 5;
float p = 1;
float q = 1;

LL step = 0; // global atomically incremented step counter

// negative sampling: using uniform distribution instead
/*
int *neg_js;
float *neg_qs;
float *node_cnts;
*/

int *train_order;   // We shuffle the nodes for better performance

float *wVtx; // Vertex embedding, aka DeepWalk's \Phi
float *wCtx; // Context embedding

/*
// this part is useless now
// ADD accept probability for pnode, qnode, 1node
float accp, accq, acc1;
*/

// ADD what method to use
int mymethod = -1;

// ADD walk result file name, should be followed by tid
char *walkname;
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



/**
NOTICE: this is useless now
Accept judgment for rejection

inline bool acc(int n0, int lastedgeidx, int nextedgeidx, myrandom &r) {
  int n2 = edges[nextedgeidx];
  if (n0 == n2)
    return r.drand() <= accp;
  if (has_edge(n0, n2))
    return r.drand() <= acc1;
  return r.drand() <= accq;
}
*/

/**
True weight
*/
// src -> edges[lastedgeidx] -> edges[dstadji]
inline float new_weight(int src, LL lastedgeidx, LL dstadji) {
  int dstadj = edges[dstadji];
  if (dstadj == src)
    return 1.0 / p;
  else if (has_edge(dstadj, src))
    return 1.0;
  else
    return 1.0 / q;
}

inline float new_weight(int src, float lastedgeweight, int dst, LL dstadji) {
  int dstadj = edges[dstadji];
  if (dstadj == src)
    return 1.0 / p;
  else if (has_edge(dstadj, src))
    return 1.0;
  else
    return 1.0 / q;
}

/*
// this part has been moved to rej.h
// NOTICE: init first!
inline void compute_rej_time(float * rej_time) {
#pragma omp parallel for
  for (int dst = 0; dst < nv; dst++) {
    double sum = 0;
    // from who
    for (int edge = offsets[dst]; edge < offsets[dst + 1]; edge++) {
      int src = edges[edge];
      double sum2 = 0;
      double totw = 0;
      // before accept or reject, which one was chosen
      for (int edge2 = offsets[dst]; edge2 < offsets[dst + 1]; edge2++) {
        int ed = edges[edge2];
        double w = weights[edge2];
        totw += w;
        if (ed == src)
          sum2 += w * accp;
        else if (has_edge(src, ed)) {
          sum2 += w * acc1;
        }
        else
          sum2 += w * accq;
      }
      // NOTICE: sum2 / totw is acc_rate for this src, but we need avg_time!
      sum += totw / sum2;
    }
    sum /= degrees[dst];
    rej_time[dst] = sum;
  }
}
*/


/**
Functions for generating and training
*/
inline void update( // update the embedding, putting w_t gradient in w_t_cache
    float *w_s, float *w_t, float *w_t_cache, float lr, const int label) {
  float score = 0; // score = dot(w_s, w_t)
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    score += w_s[c] * w_t[c];
  score = (label - fast_sigmoid(score)) * lr;
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_t_cache[c] += score * w_s[c]; // w_t gradient
  AVX_LOOP
  for (int c = 0; c < n_hidden; c++)
    w_s[c] += score * w_t[c]; // w_s gradient
}

int Generate_walks() {
  LL total_steps = n_walks * nv;
//  int ret = 0;
#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    char mywalkname[100];
    strcpy(mywalkname, walkname);
    sprintf(mywalkname, "%s%d", mywalkname, tid);
    FILE *fp = fopen(mywalkname, "wb");

    const int trnd = mainrandom.irand(nv);
    myrandom one_random(time(nullptr) + trnd), *trandom;
    if (truerandom)
      trandom = &one_random;
    else
      trandom = &mainrandom;
    int *dw_rw = static_cast<int *>(
        malloc(walk_length * sizeof(int)));

    auto begin = chrono::steady_clock::now();

#pragma omp for schedule(dynamic)
    for (LL step = 0; step < total_steps; step++) {
      dw_rw[0] = train_order[step % nv];
      if (degree(dw_rw[0]) == 0)
        continue;

 //     auto now = chrono::steady_clock::now();
 //     float during = chrono::duration_cast<chrono::duration<float>>(now-begin).count();
 //     if (during > 3600) {
 //         cout <<"TLE!!!"<<endl;
 //         ret = -1;
 //         break;
 //     }
      LL lastedgeidx;
      switch (mymethod) {
      case USEALIAS:
      case USEREJECT:
        lastedgeidx = sample_start_edge(dw_rw[0], *trandom);
        break;
      case USEONLINE:
        lastedgeidx = sample_start_edge_online(dw_rw[0], *trandom, tid);
        break;
      default:
        lastedgeidx = sample_start_edge_auto(dw_rw[0], *trandom, tid);
      }
      dw_rw[1] = edges[lastedgeidx];

      for (int i = 2; i < walk_length; i++) {
        if (degree(dw_rw[i - 1]) == 0) {
          dw_rw[i] = -2;
          break;
        }
        switch (mymethod) {
        case USEALIAS:
          lastedgeidx = sample_edge_alias(dw_rw[i - 2], lastedgeidx, *trandom);
          break;
        case USEREJECT:
          if (get_reject_cnt) // this needs tid!
            lastedgeidx = sample_edge_reject(dw_rw[i - 2], lastedgeidx, *trandom, tid);
          else
            lastedgeidx = sample_edge_reject(dw_rw[i - 2], lastedgeidx, *trandom);
          break;
        case USEONLINE:
          lastedgeidx = sample_edge_online(dw_rw[i - 2], lastedgeidx, *trandom, tid);
          break;
        default: // USEAUTO
          lastedgeidx = sample_edge_auto(dw_rw[i - 2], lastedgeidx, *trandom, tid);
        }
        dw_rw[i] = edges[lastedgeidx];
      }

      fwrite(dw_rw, sizeof(int), walk_length, fp);
    }
    fclose(fp);
    free(dw_rw);
  }
  return 0; //ret;
}

void Train() {
  LL total_steps = (LL)n_walks * nv;
  step = 0;
#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    char mywalkname[100];
    strcpy(mywalkname, walkname);
    sprintf(mywalkname, "%s%d", mywalkname, tid);
    FILE *fp = fopen(mywalkname, "rb");

    const int trnd = mainrandom.irand(nv);
    myrandom one_random(time(nullptr) + trnd), *trandom;
    if (truerandom)
      trandom = &one_random;
    else
      trandom = &mainrandom;

    LL ncount = 0;
    // LL local_step = 0;
    float lr = initial_lr;
    int *dw_rw = static_cast<int *>(
        malloc(walk_length * sizeof(int))); // we cache one random walk per thread
    float *cache = static_cast<float *>(malloc(
        n_hidden * sizeof(float))); // cache for updating the gradient of a node
#pragma omp barrier
    while (fread(dw_rw, sizeof(int), walk_length, fp) > 0) {
      if (ncount > 10) { // update progress every now and then
#pragma omp atomic
        step += ncount;
        /*
        if (step > total_steps) // note than we may train for a little longer
                                // than user requested
          break;*/
        if (tid == 0)
          if (verbosity > 1)
            cout << fixed << setprecision(6) << "\rlr " << lr << ", Progress "
                 << setprecision(2) << step * 100.f / (total_steps + 1) << "%";
        ncount = 0;
        // local_step = step;
        lr =
            initial_lr *
            (1 - step / static_cast<float>(total_steps + 1)); // linear LR decay
        if (lr < initial_lr * 0.0001)
          lr = initial_lr * 0.0001;
      }

      for (int dwi = 0; dwi < walk_length; dwi++) {
        int b = trandom->irand(window_size); // subsample window size
        int n1 = dw_rw[dwi];
        if (n1 < 0)
          break;
        // DELETE randomly subsample frequent nodes
        // I am not sure about the line below ...
        for (int dwj = max(0, dwi - window_size + b);
             dwj < min(dwi + window_size - b + 1, walk_length); dwj++) {
          if (dwi == dwj)
            continue;
          int n2 = dw_rw[dwj];
          if (n2 < 0)
            break;

          memset(cache, 0, n_hidden * sizeof(float)); // clear cache
          update(&wCtx[n1 * n_hidden], &wVtx[n2 * n_hidden], cache, lr, 1);
          for (int i = 0; i < n_neg_samples; i++) {
            int neg = trandom->irand(nv);
            while (neg == n2)
              neg = trandom->irand(nv);
            update(&wCtx[neg * n_hidden], &wVtx[n2 * n_hidden], cache, lr, 0);
          }
          AVX_LOOP
          for (int c = 0; c < n_hidden; c++)
            wVtx[n2 * n_hidden + c] += cache[c];
        }
      }
      ncount++;
    }
    fclose(fp);
    remove(mywalkname);
  }
}

// NOTICE: after training, walk has already been deleted
// this is for onlywalk tasks
void delete_walks() {
  char mywalkname[100];
  for (int tid = 0; tid < n_threads; tid++) {
    strcpy(mywalkname, walkname);
    sprintf(mywalkname, "%s%d", mywalkname, tid);
    remove(mywalkname);
  }
}

void print(float init_time, float app_time, float used_mem_size, float graph_size, float init2_time) {
  cout << "#LOG: "<<mymethod
        << " (" << recursive<<","<<another_cost <<","<<rej_bound
        << ") " << memory_size
        << " " << n_threads
        << " " << p <<" "<< q << " " << walk_length << " "<<n_walks
        << " " << network_file
        << " " << graph_size
        << " :"
        << " " << init_time
        << " " << init2_time
        << " " << app_time
        << " " << used_mem_size
        << endl;
}

int main(int argc, char **argv) {
  is_node2vec = true;
  auto ini = chrono::steady_clock::now();
  int a;
  init_sigmoid_table();

  if ((a = ArgPos(const_cast<char *>("-recursive"), argc, argv)) > 0)
    recursive = true;

  if ((a = ArgPos(const_cast<char *>("-anothercost"), argc, argv)) > 0)
    another_cost = true;

  if ((a = ArgPos(const_cast<char *>("-rejbound"), argc, argv)) > 0)
    rej_bound = atoi(argv[a + 1]);

  if ((a = ArgPos(const_cast<char *>("-memory"), argc, argv)) > 0)
    memory_size = atoi(argv[a + 1]);

  if ((a = ArgPos(const_cast<char *>("-verbose"), argc, argv)) > 0)
    verbosity = atoi(argv[a + 1]);

  if ((a = ArgPos(const_cast<char *>("-onlywalk"), argc, argv)) > 0)
    onlywalk = true;
  if ((a = ArgPos(const_cast<char *>("-input"), argc, argv)) > 0)
    network_file = argv[a + 1];
  else {
    if (verbosity > 0)
      cout << "Input file not given! Aborting now.." << endl;
    return 1;
  }
  if ((a = ArgPos(const_cast<char *>("-output"), argc, argv)) > 0)
    embedding_file = argv[a + 1];
  else if (!onlywalk) {
    if (verbosity > 0)
      cout << "Output file not given! Aborting now.." << endl;
    return 1;
  }
  if ((a = ArgPos(const_cast<char *>("-walkname"), argc, argv)) > 0)
    walkname = argv[a + 1];
  else {
    if (verbosity > 0)
      cout << "Walk result file not given! Aborting now.." << endl;
    return 1;
  }
  if ((a = ArgPos(const_cast<char *>("-dim"), argc, argv)) > 0)
    n_hidden = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-seed"), argc, argv)) > 0) {
    mainrandom.reinit(atoi(argv[a + 1]));
    truerandom = false;
  }
  if ((a = ArgPos(const_cast<char *>("-threads"), argc, argv)) > 0)
    n_threads = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-lr"), argc, argv)) > 0)
    initial_lr = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-p"), argc, argv)) > 0)
    p = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-q"), argc, argv)) > 0)
    q = atof(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-nwalks"), argc, argv)) > 0)
    n_walks = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-walklen"), argc, argv)) > 0)
    walk_length = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-window"), argc, argv)) > 0)
    window_size = atoi(argv[a + 1]);
  if ((a = ArgPos(const_cast<char *>("-nsamples"), argc, argv)) > 0)
    n_neg_samples = atoi(argv[a + 1]);

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

  node_fac_const = max(1.0, 1.0 / p);
  node_fac_const = max(node_fac_const, 1.0 / q);

  // ADD memorychange
  if ((a = ArgPos(const_cast<char *>("-memorychange"), argc, argv)) > 0) {
    cout << "Memory Usage: 0" << endl;
    showMemoryInfo();
    cout << endl;
    init_auto();
    showMemoryInfo();

    for (int i = a + 1; ; i++) {
      int change_size = atoi(argv[i]);
      if (change_size == 0)
        break;
      cout << endl << "Increase memory " << change_size << "Mb" << endl;
      memory_size += change_size;
      update_memory();
      showMemoryInfo();
    }
    return 0;
  }



/*
  wVtx = static_cast<float *>(
      malloc(nv * n_hidden * sizeof(float)));
  for (int i = 0; i < nv * n_hidden; i++)
    wVtx[i] = (mainrandom.drand() - 0.5) / n_hidden;
  wCtx = static_cast<float *>(
      malloc(nv * n_hidden * sizeof(float)));
  memset(wCtx, 0, nv * n_hidden * sizeof(float));*/
  train_order =
      static_cast<int *>(malloc(nv * sizeof(int)));
  for (int i = 0; i < nv; i++)
    train_order[i] = i;
  shuffle(train_order, nv, mainrandom);

  /*
  // this part is useless now
  if (mymethod == USEREJECT || mymethod == USEAUTO) {
    float k_for_drop = 1.0;
    float recp = 1.0 / p;
    float recq = 1.0 / q;
    if (k_for_drop < recp)
    k_for_drop = recp;
    if (k_for_drop < recq)
    k_for_drop = recq;
    accp = recp / k_for_drop;
    accq = recq / k_for_drop;
    acc1 = 1.0 / k_for_drop;
  }
  */

  cout << "Show memory info. before init. sampler methods" <<endl;
  float graph_size = showMemoryInfo();

  auto init2 = chrono::steady_clock::now();

  if (mymethod == USEREJECT) {
    init_reject();
    all_node_reject();
  }
  if (mymethod == USEALIAS) {
    init_reject();
    all_node_alias_reject();
    //init_alias();
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
//  if (get_memory() > 94*1024) {
//    cout <<"Memory exceeded: " << get_memory() <<endl;
//    return 0;
//  }

// DELETE part of subsampling
/*
  cout << endl << "Generating a corpus for negative samples.." << endl;
  neg_qs =
      static_cast<float *>(malloc(nv * sizeof(float), DEFAULT_ALIGN));
  neg_js = static_cast<int *>(malloc(nv * sizeof(int), DEFAULT_ALIGN));
  node_cnts =
      static_cast<float *>(malloc(nv * sizeof(float), DEFAULT_ALIGN));
  memset(neg_qs, 0, nv * sizeof(float));
#pragma omp parallel for num_threads(n_threads)
  for (int i = 0; i < nv * n_walks; i++) {
    int src = train_order[i % nv];
#pragma omp atomic
    neg_qs[src]++;
    if (degrees[src] == 0)
      continue;
    int lastedgeidx = irand(offsets[src], offsets[src + 1]);
    int lastnode = edges[lastedgeidx];
#pragma omp atomic
    neg_qs[lastnode]++;
    for (int j = 2; j < walk_length; j++) {
      if (degrees[lastnode] == 0)
        break;
      lastedgeidx =
          offsets[lastnode] + walker_draw(degrees[lastnode],
                                          &n2v_qs[edge_offsets[lastedgeidx]],
                                          &n2v_js[edge_offsets[lastedgeidx]]);
      lastnode = edges[lastedgeidx];
#pragma omp atomic
      neg_qs[lastnode]++;
    }
  }
  for (int i = 0; i < nv; i++)
    node_cnts[i] = neg_qs[i];
  float sum = 0;
  for (int i = 0; i < nv; i++) {
    neg_qs[i] = pow(neg_qs[i], 0.75f);
    sum += neg_qs[i];
  }
  for (int i = 0; i < nv; i++)
    neg_qs[i] *= nv / sum;
  init_walker(nv, neg_js, neg_qs); */

  if (verbosity > 0)
#if VECTORIZE
    cout << "Using vectorized operations" << endl;
#else
    cout << "Not using vectorized operations (!)" << endl;
#endif
  auto begin = chrono::steady_clock::now();
  int status = Generate_walks();
  auto mid = chrono::steady_clock::now();
  if (verbosity > 1)
    cout << endl;
  if (verbosity > 0)
    cout << "Finished generation" << endl;
  if (!onlywalk)
    Train();
  else
    delete_walks();
  auto end = chrono::steady_clock::now();


  float init_time = chrono::duration_cast<chrono::duration<float>>(begin - ini).count();
  float init2_time = chrono::duration_cast<chrono::duration<float>>(begin - init2).count();
  float app_time = chrono::duration_cast<chrono::duration<float>>(mid - begin).count();
  float used_mem_size = -1;

  if (verbosity > 1 && !onlywalk)
    cout << endl;
  if (verbosity > 0) {
    cout << "Initialization took "
         << chrono::duration_cast<chrono::duration<float>>(begin - ini).count()
         << " s to run" << endl
         << "Walks generation took "
         << chrono::duration_cast<chrono::duration<float>>(mid - begin).count()
         << " s to run" << endl;
    if (!onlywalk)
      cout << "Training took "
           << chrono::duration_cast<chrono::duration<float>>(end - mid).count()
           << " s to run" << endl;
    used_mem_size = showMemoryInfo();
    if (get_reject_cnt)
      output_reject_cnt(reject_cnt_name);
    if (auto_types_name != nullptr)
      output_auto_types(auto_types_name);
  }
  print(init_time, app_time, used_mem_size, graph_size, init2_time);
  if (onlywalk)
    return 0;
  ofstream output(embedding_file, ios::binary);
  output.write(reinterpret_cast<char *>(wVtx), sizeof(float) * n_hidden * nv);
  output.close();
  return 0;
}
