// ==========================================================================
// KIVI fused decode attention — hot loops. Included twice by
// fused_attention.cpp: once with KIVI_TARGET = AVX2+FMA, once generic, each
// inside its own namespace; the caller picks one at runtime.
//
// Shapes (one (batch, kv-head) slice; n_rep query heads share it):
//   keys   packed  [D, T/4]  (row per channel, codes along tokens)
//          scales/zeros [D, T/G]  (float — converted by the caller)
//   values packed  [T, D/4]  (row per token, codes along channels)
//          scales/zeros [T, D/G]
//   2-bit code -> float through LUT: lut[byte*4 + j] = (byte >> 2j) & 3.
// ==========================================================================

// Keys, tokens [t0, t1) (multiples of G and of 4):
//   logits[r][t] = scaling * ( sum_d (q_r[d] * s[d,g]) * c[d,t]  +  sum_d q_r[d] * z[d,g] )
// The second term is one constant per token group, accumulated alongside.
KIVI_TARGET static void key_logits_chunk(
    const float* __restrict__ q,          // [n_rep, D]
    int n_rep, int D, int G,
    const uint8_t* __restrict__ pk, int64_t pk_row,      // row stride (bytes) per channel
    const float* __restrict__ sk, const float* __restrict__ zk, int64_t meta_row,
    int64_t t0, int64_t t1, float scaling,
    float* __restrict__ logits,           // [n_rep, ld] — writes [t0, t1)
    int64_t ld, const float* __restrict__ lut,
    float* __restrict__ acc,              // scratch [n_rep, t1-t0]
    float* __restrict__ bias)             // scratch [n_rep, (t1-t0)/G]
{
    const int64_t n = t1 - t0;
    const int64_t ng = n / G;
    for (int64_t i = 0; i < n_rep * n; ++i) acc[i] = 0.0f;
    for (int64_t i = 0; i < n_rep * ng; ++i) bias[i] = 0.0f;

    for (int d = 0; d < D; ++d) {
        const uint8_t* row = pk + d * pk_row + t0 / 4;
        const float* srow = sk + d * meta_row + t0 / G;
        const float* zrow = zk + d * meta_row + t0 / G;
        for (int r = 0; r < n_rep; ++r) {
            const float qd = q[r * D + d];
            float* a = acc + r * n;
            float* b = bias + r * ng;
            for (int64_t g = 0; g < ng; ++g) {
                const float w = qd * srow[g];
                b[g] += qd * zrow[g];
                const uint8_t* bytes = row + g * (G / 4);
                float* ag = a + g * G;
                for (int j = 0; j < G / 4; ++j) {
                    const float* c = lut + 4 * bytes[j];
                    ag[4 * j + 0] += w * c[0];
                    ag[4 * j + 1] += w * c[1];
                    ag[4 * j + 2] += w * c[2];
                    ag[4 * j + 3] += w * c[3];
                }
            }
        }
    }
    for (int r = 0; r < n_rep; ++r) {
        const float* a = acc + r * n;
        const float* b = bias + r * ng;
        float* out = logits + r * ld + t0;
        for (int64_t t = 0; t < n; ++t) out[t] = scaling * (a[t] + b[t / G]);
    }
}

// Values, tokens [t0, t1):
//   out_r[d] += sum_t (p_r[t] * s[t,gd]) * c[t,d]  +  sum_t p_r[t] * z[t,gd]
// Accumulates into partial [n_rep, D] (caller zeroes and reduces it).
KIVI_TARGET static void value_accum_chunk(
    const float* __restrict__ p,          // [n_rep, ld] softmax probs
    int64_t ld, int n_rep, int D, int G,
    const uint8_t* __restrict__ pv, int64_t pv_row,      // bytes per token row
    const float* __restrict__ sv, const float* __restrict__ zv, int64_t meta_row,
    int64_t t0, int64_t t1,
    float* __restrict__ partial,          // [n_rep, D]
    const float* __restrict__ lut)
{
    const int ngd = D / G;
    for (int64_t t = t0; t < t1; ++t) {
        const uint8_t* row = pv + t * pv_row;
        const float* srow = sv + t * meta_row;
        const float* zrow = zv + t * meta_row;
        for (int r = 0; r < n_rep; ++r) {
            const float pt = p[r * ld + t];
            float* o = partial + r * D;
            for (int gd = 0; gd < ngd; ++gd) {
                const float a = pt * srow[gd];
                const float z = pt * zrow[gd];
                const uint8_t* bytes = row + gd * (G / 4);
                float* og = o + gd * G;
                for (int j = 0; j < G / 4; ++j) {
                    const float* c = lut + 4 * bytes[j];
                    og[4 * j + 0] += a * c[0] + z;
                    og[4 * j + 1] += a * c[1] + z;
                    og[4 * j + 2] += a * c[2] + z;
                    og[4 * j + 3] += a * c[3] + z;
                }
            }
        }
    }
}
