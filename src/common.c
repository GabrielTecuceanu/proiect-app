#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "common.h"

double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ------------------------------ dataset ------------------------------ */
void ds_free(Dataset *ds) {
  free(ds->X);
  free(ds->y);
  ds->X = NULL; ds->y = NULL; ds->n = ds->d = 0;
}

static int read_be32(FILE *f, uint32_t *v) {
  unsigned char b[4];
  if (fread(b, 1, 4, f) != 4) return -1;
  *v = ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | b[3];
  return 0;
}

int load_idx(const char *images_path, const char *labels_path, Dataset *out) {
  FILE *fi = fopen(images_path, "rb"), *fl = fopen(labels_path, "rb");
  uint8_t *raw = NULL;
  int rc = -1;
  if (!fi) { fprintf(stderr, "Nu pot deschide %s\n", images_path); goto done; }
  if (!fl) { fprintf(stderr, "Nu pot deschide %s\n", labels_path); goto done; }
  uint32_t magic, n, rows, cols, magic_l, nl;
  if (read_be32(fi, &magic) || magic != 0x803 || read_be32(fi, &n) || read_be32(fi, &rows) || read_be32(fi, &cols)) {
    fprintf(stderr, "IDX3 invalid: %s\n", images_path); goto done;
  }
  if (read_be32(fl, &magic_l) || magic_l != 0x801 || read_be32(fl, &nl) || nl != n) {
    fprintf(stderr, "IDX1 invalid: %s\n", labels_path); goto done;
  }
  size_t d = (size_t)rows * cols, total = (size_t)n * d;
  raw = malloc(total);
  out->X = malloc(total * sizeof(float));
  out->y = malloc(n);
  if (!raw || !out->X || !out->y) { fprintf(stderr, "memorie insuficienta\n"); goto done; }
  if (fread(raw, 1, total, fi) != total || fread(out->y, 1, n, fl) != n) {
    fprintf(stderr, "IDX trunchiat\n"); goto done;
  }
  for (size_t i = 0; i < total; ++i) out->X[i] = raw[i] * (1.0f / 255.0f);
  out->n = n; out->d = d;
  rc = 0;
done:
  if (rc) { free(out->X); free(out->y); out->X = NULL; out->y = NULL; out->n = out->d = 0; }
  free(raw);
  if (fi) fclose(fi);
  if (fl) fclose(fl);
  return rc;
}

Dataset ds_concat(const Dataset *a, const Dataset *b) {
  Dataset r = {a->n + b->n, a->d, malloc((a->n + b->n) * a->d * sizeof(float)), malloc(a->n + b->n)};
  memcpy(r.X, a->X, a->n * a->d * sizeof(float));
  memcpy(r.X + a->n * a->d, b->X, b->n * b->d * sizeof(float));
  memcpy(r.y, a->y, a->n);
  memcpy(r.y + a->n, b->y, b->n);
  return r;
}

Dataset ds_subset(const Dataset *ds, const size_t *idx, size_t m) {
  Dataset s = {m, ds->d, malloc(m * ds->d * sizeof(float)), malloc(m)};
  for (size_t i = 0; i < m; ++i) {
    memcpy(s.X + i * ds->d, ds_row(ds, idx[i]), ds->d * sizeof(float));
    s.y[i] = ds->y[idx[i]];
  }
  return s;
}

/* ------------------------------- split ------------------------------- */
/* splitmix64: PRNG propriu => acelasi split pe orice platforma. */
static uint64_t sm64(uint64_t *s) {
  uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

/* Fiecare clasa e amestecata separat (Fisher-Yates), apoi clasele sunt
 * intercalate round-robin => orice prefix e echilibrat pe clase. Primele
 * ntrain -> train, urmatoarele nval -> val, apoi ntest -> test (disjuncte). */
int stratified_split(const Dataset *pool, size_t ntrain, size_t nval, size_t ntest, unsigned seed, Split *out) {
  if (ntrain + nval + ntest > pool->n) return -1;
  uint64_t rng = seed;
  size_t cnt[NUM_CLASSES] = {0}, pos[NUM_CLASSES] = {0};
  for (size_t i = 0; i < pool->n; ++i) cnt[pool->y[i]]++;
  size_t *byc[NUM_CLASSES];
  for (int c = 0; c < NUM_CLASSES; ++c) byc[c] = malloc((cnt[c] ? cnt[c] : 1) * sizeof(size_t));
  size_t fill[NUM_CLASSES] = {0};
  for (size_t i = 0; i < pool->n; ++i) byc[pool->y[i]][fill[pool->y[i]]++] = i;
  for (int c = 0; c < NUM_CLASSES; ++c)
    for (size_t i = cnt[c]; i > 1; --i) {
      size_t j = sm64(&rng) % i, t = byc[c][i - 1];
      byc[c][i - 1] = byc[c][j]; byc[c][j] = t;
    }
  size_t *order = malloc(pool->n * sizeof(size_t)), m = 0;
  while (m < pool->n)
    for (int c = 0; c < NUM_CLASSES; ++c)
      if (pos[c] < cnt[c]) order[m++] = byc[c][pos[c]++];
  out->ntrain = ntrain; out->nval = nval; out->ntest = ntest;
  out->train = malloc(ntrain * sizeof(size_t) + 1);
  out->val = malloc(nval * sizeof(size_t) + 1);
  out->test = malloc(ntest * sizeof(size_t) + 1);
  memcpy(out->train, order, ntrain * sizeof(size_t));
  memcpy(out->val, order + ntrain, nval * sizeof(size_t));
  memcpy(out->test, order + ntrain + nval, ntest * sizeof(size_t));
  for (int c = 0; c < NUM_CLASSES; ++c) free(byc[c]);
  free(order);
  return 0;
}

void split_free(Split *s) { free(s->train); free(s->val); free(s->test); }

/* ------------------------------ metrici ------------------------------ */
Metrics compute_metrics(const uint8_t *truth, const int *pred, size_t n) {
  Metrics m;
  memset(&m, 0, sizeof m);
  for (size_t i = 0; i < n; ++i) m.conf[truth[i]][pred[i]]++;
  long correct = 0, total = 0;
  for (int c = 0; c < NUM_CLASSES; ++c) {
    long tp = m.conf[c][c], row = 0, col = 0;
    for (int j = 0; j < NUM_CLASSES; ++j) { row += m.conf[c][j]; col += m.conf[j][c]; }
    correct += tp; total += row;
    m.precision[c] = col ? (double)tp / col : 0;
    m.recall[c] = row ? (double)tp / row : 0;
    double s = m.precision[c] + m.recall[c];
    m.f1[c] = s > 0 ? 2 * m.precision[c] * m.recall[c] / s : 0;
    m.macro_f1 += m.f1[c] / NUM_CLASSES;
  }
  m.accuracy = total ? (double)correct / total : 0;
  return m;
}

void print_metrics(const Metrics *m, const char *title) {
  printf("== %s ==\n", title);
  printf("accuracy: %.4f   macro-F1: %.4f\n", m->accuracy, m->macro_f1);
  printf("clasa  precision  recall     f1\n");
  for (int c = 0; c < NUM_CLASSES; ++c)
    printf("  %d     %.4f     %.4f   %.4f\n", c, m->precision[c], m->recall[c], m->f1[c]);
  printf("matrice de confuzie (rand=adevarat, coloana=prezis):\n     ");
  for (int j = 0; j < NUM_CLASSES; ++j) printf("%5d", j);
  printf("\n");
  for (int i = 0; i < NUM_CLASSES; ++i) {
    printf("  %d: ", i);
    for (int j = 0; j < NUM_CLASSES; ++j) printf("%5ld", m->conf[i][j]);
    printf("\n");
  }
}
