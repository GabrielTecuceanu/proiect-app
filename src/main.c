/* Pipeline independent de implementarea KNN:
 * load MNIST -> split stratificat train/val/test -> alegere k pe val ->
 * evaluare pe test -> metrici + timpi. Implementarea se alege la link. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "knn.h"

#ifndef KNN_IMPL
#define KNN_IMPL "serial"
#endif

typedef struct {
  const char *data;
  size_t ntrain, nval, ntest;
  int k; /* 0 = alege din {1,3,5,7} pe validare */
  unsigned seed;
  int selftest;
} Args;

static void parse(int argc, char **argv, Args *a) {
  a->data = "data"; a->ntrain = 20000; a->nval = 2000; a->ntest = 2000; a->k = 0; a->seed = 42; a->selftest = 0;
  for (int i = 1; i < argc; ++i) {
    const char *s = argv[i];
    if (!strcmp(s, "--selftest")) { a->selftest = 1; continue; }
    if (i + 1 >= argc) { fprintf(stderr, "lipseste valoarea pentru %s\n", s); exit(2); }
    const char *v = argv[++i];
    if (!strcmp(s, "--data")) a->data = v;
    else if (!strcmp(s, "--ntrain")) a->ntrain = strtoul(v, NULL, 10);
    else if (!strcmp(s, "--nval")) a->nval = strtoul(v, NULL, 10);
    else if (!strcmp(s, "--ntest")) a->ntest = strtoul(v, NULL, 10);
    else if (!strcmp(s, "--k")) a->k = atoi(v);
    else if (!strcmp(s, "--seed")) a->seed = (unsigned)strtoul(v, NULL, 10);
    else { fprintf(stderr, "argument necunoscut: %s\n", s); exit(2); }
  }
}

int main(int argc, char **argv) {
  Args a;
  parse(argc, argv, &a);
  double t_start = now_sec(), t;
  printf("KNN MNIST - implementare: %s\n", KNN_IMPL);

  char p[4][512];
  const char *names[4] = {"train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte",
                          "t10k-labels-idx1-ubyte"};
  for (int i = 0; i < 4; ++i) snprintf(p[i], sizeof p[i], "%s/%s", a.data, names[i]);
  t = now_sec();
  Dataset tr, te;
  if (load_idx(p[0], p[1], &tr) || load_idx(p[2], p[3], &te)) return 1;
  Dataset pool = ds_concat(&tr, &te);
  ds_free(&tr); ds_free(&te);
  double t_load = now_sec() - t;

  t = now_sec();
  Split sp;
  if (stratified_split(&pool, a.ntrain, a.nval, a.ntest, a.seed, &sp)) {
    fprintf(stderr, "ntrain+nval+ntest depaseste %zu\n", pool.n);
    return 1;
  }
  Dataset train = ds_subset(&pool, sp.train, sp.ntrain), val = ds_subset(&pool, sp.val, sp.nval),
          test = ds_subset(&pool, sp.test, sp.ntest);
  double t_split = now_sec() - t;
  printf("pool=%zu d=%zu | train=%zu val=%zu test=%zu | seed=%u\n", pool.n, pool.d, train.n, val.n, test.n, a.seed);

  if (a.selftest) { /* k=1 pe propriul train => vecinul e el insusi (dist 0) */
    size_t nq = train.n < 200 ? train.n : 200, ok = 0;
    Neighbor *nb = malloc(nq * sizeof *nb);
    knn_search(&train, train.X, nq, 1, nb);
    for (size_t q = 0; q < nq; ++q) ok += (nb[q].idx == (int32_t)q && nb[q].dist == 0.0f);
    printf("selftest: %zu/%zu vecini exacti\n", ok, nq);
    return ok == nq ? 0 : 1;
  }

  const int cand[4] = {1, 3, 5, 7};
  int kmax = a.k > 0 ? a.k : 7, best_k = a.k;
  if ((size_t)kmax > train.n) { fprintf(stderr, "k > ntrain\n"); return 1; }
  double t_search = 0; /* doar timpul in knn_search conteaza pentru speedup */

  if (a.k == 0) {
    Neighbor *nb = malloc(val.n * kmax * sizeof *nb);
    int *pred = malloc(val.n * sizeof *pred);
    t = now_sec();
    knn_search(&train, val.X, val.n, kmax, nb);
    t_search += now_sec() - t;
    double best_acc = -1;
    for (int c = 0; c < 4; ++c) {
      knn_predict(nb, val.n, kmax, cand[c], train.y, pred);
      double acc = compute_metrics(val.y, pred, val.n).accuracy;
      printf("validare: k=%d accuracy=%.4f\n", cand[c], acc);
      if (acc > best_acc) { best_acc = acc; best_k = cand[c]; }
    }
    printf("k ales (val): %d\n", best_k);
    free(nb); free(pred);
  }

  Neighbor *nb = malloc(test.n * best_k * sizeof *nb);
  int *pred = malloc(test.n * sizeof *pred);
  t = now_sec();
  knn_search(&train, test.X, test.n, best_k, nb);
  t_search += now_sec() - t;
  knn_predict(nb, test.n, best_k, best_k, train.y, pred);
  Metrics m = compute_metrics(test.y, pred, test.n);
  print_metrics(&m, "TEST");

  size_t nq = test.n + (a.k == 0 ? val.n : 0);
  printf("== timpi (s) ==\nload=%.3f split=%.3f search=%.3f (%.0f interogari/s) total=%.3f\n", t_load, t_split,
         t_search, nq / t_search, now_sec() - t_start);
  return 0;
}
