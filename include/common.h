#ifndef COMMON_H
#define COMMON_H
#include <stddef.h>
#include <stdint.h>

#define NUM_CLASSES 10

/* Dataset dens, row-major (n x d), contiguu -> simplu de trimis prin
 * MPI_Scatter / cudaMemcpy. Etichete 0..9. */
typedef struct {
  size_t n, d;
  float *X;
  uint8_t *y;
} Dataset;

static inline const float *ds_row(const Dataset *ds, size_t i) { return ds->X + i * ds->d; }

void ds_free(Dataset *ds);
/* Citeste IDX3 (imagini) + IDX1 (etichete); pixeli normalizati /255. 0 = ok. */
int load_idx(const char *images_path, const char *labels_path, Dataset *out);
Dataset ds_concat(const Dataset *a, const Dataset *b);
Dataset ds_subset(const Dataset *ds, const size_t *idx, size_t m);

/* ---- split stratificat, reproductibil ---- */
typedef struct {
  size_t *train, *val, *test;
  size_t ntrain, nval, ntest;
} Split;
/* Returneaza 0 daca ok, -1 daca ntrain+nval+ntest > pool->n. */
int stratified_split(const Dataset *pool, size_t ntrain, size_t nval, size_t ntest, unsigned seed, Split *out);
void split_free(Split *s);

/* ---- metrici (independente de implementarea KNN) ---- */
typedef struct {
  long conf[NUM_CLASSES][NUM_CLASSES]; /* [adevarat][prezis] */
  double accuracy, macro_f1;
  double precision[NUM_CLASSES], recall[NUM_CLASSES], f1[NUM_CLASSES];
} Metrics;
Metrics compute_metrics(const uint8_t *truth, const int *pred, size_t n);
void print_metrics(const Metrics *m, const char *title);

/* ---- timer wall-clock (secunde) ---- */
double now_sec(void);

#endif
