/* SPDX-License-Identifier: MIT
 * Original TP3 adapter: rank-local ids and original per-rank file offsets.
 * Lookup values are supplied by the serving torch's exhaustive FP8*UE8M0->BF16
 * table, so NaN payload/rounding rules are those of that runtime, not this CPU.
 * This is a CPU helper; no CUDA, GPU synchronisation or persistent row cache.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>

int dgx_engram_abi(void) { return 1; }

static int read_full(int fd, uint8_t *dst, size_t n, int64_t offset, int64_t *calls) {
    size_t done=0;
    while (done<n) {
        ssize_t got=pread(fd,dst+done,n-done,(off_t)(offset+(int64_t)done));
        (*calls)++;
        if (got<0) { if(errno==EINTR) continue; return -errno; }
        if (got==0) return -EIO;
        done+=(size_t)got;
    }
    return 0;
}

/* rel[] is already rank-local; w_off/s_off already include row_start.
 * owned[] is 0/1. The caller allocates out[rows*dim]. No host hash prediction.
 * stats={owned rows,unique rows,read syscalls}. Invalid inputs never read files.
 */
int dgx_engram_gather(int wfd,int64_t woff,int sfd,int64_t soff,
    int64_t table_rows,int dim,int sb,int rows,const int64_t *rel,
    const uint8_t *owned,const uint16_t *lut,uint16_t *out,int64_t *stats) {
    if (woff<0 || soff<0 || table_rows<0 || dim<=0 || sb<=0 || dim%sb || rows<0 || rows>16384
        || dim>4096 || !rel || !owned || !lut || !out || !stats) return -EINVAL;
    stats[0]=stats[1]=stats[2]=0;
    if (table_rows > (INT64_MAX-woff)/dim || table_rows > (INT64_MAX-soff)/sb) return -EOVERFLOW;
    for(int r=0;r<rows;r++) {
        if(owned[r]>1 || (owned[r] && (rel[r]<0 || rel[r]>=table_rows))) return -EINVAL;
    }
    if (!rows) return 0;
    size_t cap=1; while(cap<(size_t)rows*2) cap*=2;
    int64_t *keys=malloc(cap*sizeof *keys);
    int *first=malloc(cap*sizeof *first);
    uint8_t *w=malloc((size_t)dim), *s=malloc((size_t)sb);
    if(!keys || !first || !w || !s) { free(keys);free(first);free(w);free(s);return -ENOMEM; }
    for(size_t i=0;i<cap;i++) keys[i]=-1;
    int rc=0, qb=dim/sb;
    for(int r=0;r<rows;r++) {
        uint16_t *dst=out+(size_t)r*dim;
        if(!owned[r]) { memset(dst,0,(size_t)dim*sizeof *dst);continue; }
        stats[0]++;
        uint64_t hash=(uint64_t)rel[r]*UINT64_C(11400714819323198485);
        size_t pos=(size_t)hash&(cap-1);
        while(keys[pos]!=-1 && keys[pos]!=rel[r]) pos=(pos+1)&(cap-1);
        if(keys[pos]!=-1) {
            memcpy(dst,out+(size_t)first[pos]*dim,(size_t)dim*sizeof *dst);continue;
        }
        rc=read_full(wfd,w,(size_t)dim,woff+rel[r]*dim,&stats[2]);
        if(!rc) rc=read_full(sfd,s,(size_t)sb,soff+rel[r]*sb,&stats[2]);
        if(rc) break;
        for(int c=0;c<dim;c++) dst[c]=lut[(unsigned)s[c/qb]*256+w[c]];
        keys[pos]=rel[r];first[pos]=r;stats[1]++;
    }
    free(keys);free(first);free(w);free(s);
    return rc;
}

