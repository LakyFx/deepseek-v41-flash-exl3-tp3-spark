/* SPDX-License-Identifier: MIT
 * A1: stock parallel I/O followed by exact LUT conversion to raw BF16 bits.
 * No file I/O, worker creation, global writable state or CUDA operations.
 */
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int dgx_engram_dequant_abi(void) { return 1; }

int dgx_engram_dequant(int unique_rows, int dim, int sb, int rows,
                      const uint8_t *weights, const uint8_t *scales,
                      const int64_t *inverse, const uint8_t *owned,
                      const uint16_t *lut, uint16_t *out) {
    if (unique_rows < 0 || unique_rows > 16384 || rows < 0 || rows > 16384
            || dim != 256 || sb != 8) return -EINVAL;
    if (!rows) return 0;
    if (!unique_rows || !weights || !scales || !inverse || !owned || !lut || !out)
        return -EINVAL;
    /* Reject all invalid metadata before writing any output. */
    for (int r = 0; r < rows; r++)
        if (owned[r] > 1 || inverse[r] < 0 || inverse[r] >= unique_rows) return -EINVAL;
    int *first = malloc((size_t)unique_rows * sizeof *first);
    if (!first) return -ENOMEM;
    for (int u = 0; u < unique_rows; u++) first[u] = -1;
    for (int r = 0; r < rows; r++) {
        uint16_t *dst = out + (size_t)r * dim;
        if (!owned[r]) { memset(dst, 0, (size_t)dim * sizeof *dst); continue; }
        int u = (int)inverse[r];
        if (first[u] >= 0) {
            memcpy(dst, out + (size_t)first[u] * dim, (size_t)dim * sizeof *dst);
            continue;
        }
        const uint8_t *w = weights + (size_t)inverse[r] * dim;
        const uint8_t *s = scales + (size_t)inverse[r] * sb;
        for (int c = 0; c < dim; c++) dst[c] = lut[(unsigned)s[c / 32] * 256 + w[c]];
        first[u] = r;
    }
    free(first);
    return 0;
}
