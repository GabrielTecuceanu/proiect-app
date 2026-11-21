/* Varianta OpenMP: interogarile sunt independente => paralelizam bucla peste
 * q. Fiecare fir scrie doar in out[q*kmax..], deci nu sunt cursuri. Rezultatul
 * e identic bit-cu-bit cu varianta seriala (acelasi l2sq, aceeasi ordine). */
#include <omp.h>
#include "knn.h"

void knn_search(const Dataset *train, const float *queries, size_t nq, int kmax, Neighbor *out) {
  const size_t n = train->n, d = train->d;
  #pragma omp parallel for schedule(static)
  for (long q = 0; q < (long)nq; ++q) {
    const float *x = queries + (size_t)q * d;
    Neighbor *best = out + (size_t)q * kmax;
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
