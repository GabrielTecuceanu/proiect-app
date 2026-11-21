/* Implementarea SERIALA a kernel-ului knn_search (baseline pentru speedup).
 * Mentine top-kmax prin insertie (kmax mic) - O(n*d + n*kmax), fara sort. */
#include "knn.h"

void knn_search(const Dataset *train, const float *queries, size_t nq, int kmax, Neighbor *out) {
  const size_t n = train->n, d = train->d;
  for (size_t q = 0; q < nq; ++q) {
    const float *x = queries + q * d;
    Neighbor *best = out + q * kmax;
    int cnt = 0;
    for (size_t i = 0; i < n; ++i) {
      Neighbor c = {l2sq(x, ds_row(train, i), d), (int32_t)i};
      if (cnt == kmax && !neighbor_less(&c, &best[kmax - 1])) continue;
      int p = cnt < kmax ? cnt++ : kmax - 1;
      while (p > 0 && neighbor_less(&c, &best[p - 1])) { best[p] = best[p - 1]; --p; }
      best[p] = c;
    }
  }
}
