#ifndef KNN_H
#define KNN_H
#include "common.h"

/* Interfata comuna a implementarilor (serial / OpenMP / MPI / CUDA).
 * Doar CAUTAREA vecinilor (knn_search) e paralelizata si se defineste in
 * src/knn_<varianta>.c. Votul (knn_vote) e comun si determinist => toate
 * variantele trebuie sa dea aceleasi predictii. */

typedef struct {
  float dist;   /* distanta euclidiana la patrat */
  int32_t idx;  /* indexul in setul de antrenare */
} Neighbor;

/* Ordine totala: distanta, apoi index. */
static inline int neighbor_less(const Neighbor *a, const Neighbor *b) {
  return a->dist < b->dist || (a->dist == b->dist && a->idx < b->idx);
}

/* L2^2 cu 4 acumulatori; ordinea sumarii e fixata aici ca toate variantele
 * pe CPU sa dea exact aceleasi valori. */
static inline float l2sq(const float *a, const float *b, size_t d) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  size_t i = 0;
  for (; i + 4 <= d; i += 4) {
    float t0 = a[i] - b[i], t1 = a[i + 1] - b[i + 1], t2 = a[i + 2] - b[i + 2], t3 = a[i + 3] - b[i + 3];
    s0 += t0 * t0; s1 += t1 * t1; s2 += t2 * t2; s3 += t3 * t3;
  }
  for (; i < d; ++i) { float t = a[i] - b[i]; s0 += t * t; }
  return (s0 + s1) + (s2 + s3);
}

/* KERNEL: pentru fiecare din cele nq interogari (row-major nq x train->d)
 * scrie in out[q*kmax .. +kmax) cei mai apropiati kmax vecini, crescator
 * dupa neighbor_less. Preconditie: 1 <= kmax <= train->n. */
void knn_search(const Dataset *train, const float *queries, size_t nq, int kmax, Neighbor *out);

/* Vot majoritar pe primii k vecini. Egalitate: suma distantelor mai mica,
 * apoi eticheta mai mica. */
static inline int knn_vote(const Neighbor *nb, int k, const uint8_t *train_labels) {
  int cnt[NUM_CLASSES] = {0};
  float dsum[NUM_CLASSES] = {0};
  for (int i = 0; i < k; ++i) {
    int c = train_labels[nb[i].idx];
    cnt[c]++;
    dsum[c] += nb[i].dist;
  }
  int best = -1;
  for (int c = 0; c < NUM_CLASSES; ++c) {
    if (cnt[c] == 0) continue;
    if (best < 0 || cnt[c] > cnt[best] || (cnt[c] == cnt[best] && dsum[c] < dsum[best])) best = c;
  }
  return best;
}

/* pred[q] pentru un k <= kmax, din vecinii calculati cu kmax. */
static inline void knn_predict(const Neighbor *nbs, size_t nq, int kmax, int k, const uint8_t *train_labels,
                               int *pred) {
  for (size_t q = 0; q < nq; ++q) pred[q] = knn_vote(nbs + q * kmax, k, train_labels);
}

#endif
