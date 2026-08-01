// anon_hog — anonymous (dirty) memory-pressure generator for the OS-paging
// full-eviction endpoint.
//
// Allocates <total_mb> MiB of private anonymous memory and *writes* every page,
// dirtying it so the kernel cannot simply drop it — it must keep the page in RAM
// or swap it to zram. Holding this working set hot (re-writing once per second
// for <hold_sec>) makes the idle KV cache the coldest anonymous memory in the
// system, so the kernel swaps the KV out to zram. Unlike file-cache pressure
// (mmap_touch), whose reclaimable clean pages the kernel just recycles, dirty
// anon competes directly with the KV for RAM+swap and drives its resident Rss to
// ~0 — reaching the eviction floor that file pressure plateaus above.
//
// Usage: anon_hog <total_mb> [hold_sec=3]
//   <total_mb>  MiB of dirty anonymous memory to hold
//   [hold_sec]  keep it resident and hot this many seconds (0 = touch once)
//
// Ramp <total_mb> upward across invocations and stop once the target's resident
// KV Rss is low enough; sizing it past RAM+swap will invoke the OOM killer.

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <total_mb> [hold_sec=3]\n", argv[0]);
        return 1;
    }
    size_t mb   = (size_t) atol(argv[1]);
    int    hold = (argc > 2) ? atoi(argv[2]) : 3;
    size_t len  = mb * 1024UL * 1024UL;
    long   page = sysconf(_SC_PAGESIZE);

    // MAP_NORESERVE: don't fail commit accounting up front; pages fault in as we
    // write them, forcing reclaim/swap of colder anon (the KV) at that point.
    unsigned char * m = mmap(NULL, len, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (m == MAP_FAILED) { perror("mmap"); return 1; }

    for (size_t off = 0; off < len; off += (size_t) page)
        m[off] = (unsigned char) (off >> 12);          // dirty every page
    printf("anon_hog: dirtied %zu MiB (pid %d)\n", mb, getpid());
    fflush(stdout);

    for (int s = 0; s < hold; s++) {
        sleep(1);
        for (size_t off = 0; off < len; off += (size_t) page)
            m[off] += 1;                                // keep the set hot + dirty
    }
    munmap(m, len);
    return 0;
}
