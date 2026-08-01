// mmap_touch — file-cache memory-pressure generator for the OS-paging baseline.
//
// Randomly picks <count> distinct files matching "stress_*.bin" in the current
// directory (or $MEMSTRESS_DIR), mmaps them MAP_SHARED, and touches every page
// so they enter the page cache. Under swappiness=100 the kernel reclaims file
// pages (the mmapped model weights) before anonymous pages, and then swaps the
// anonymous KV cache to zram — reproducing the weights-first eviction order
// without explicit zram off/on cycling (see EVALUATION.md, Figure 9, OS Paging).
//
// Usage: mmap_touch <count> [size_mb=128] [hold_sec=3]
//   <count>    number of stress files to touch this invocation
//   [size_mb]  expected file size (bytes actually mapped = min(file size, size_mb))
//   [hold_sec] keep the mappings resident for this many seconds, re-touching
//              once per second; 0 = touch once and exit immediately
//
// NOTE: this is a clean reimplementation of the interface documented for the
// original (unreleased) pressure tool used in the paper's experiments.
// TODO(author): diff behavior against the original deployed binary.

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define MAX_FILES 1024

static volatile unsigned long g_sink;  // defeat dead-load elimination

static void touch_pages(const unsigned char * base, size_t len, long page) {
    unsigned long sum = 0;
    for (size_t off = 0; off < len; off += (size_t) page) {
        sum += base[off];
    }
    g_sink = sum;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <count> [size_mb=128] [hold_sec=3]\n", argv[0]);
        return 1;
    }
    int    count    = atoi(argv[1]);
    size_t size_mb  = (argc > 2) ? (size_t) atol(argv[2]) : 128;
    int    hold_sec = (argc > 3) ? atoi(argv[3]) : 3;

    const char * dir = getenv("MEMSTRESS_DIR");
    if (!dir) dir = ".";

    // enumerate stress_*.bin
    char * names[MAX_FILES];
    int n_files = 0;
    DIR * d = opendir(dir);
    if (!d) { perror("opendir"); return 1; }
    struct dirent * e;
    while ((e = readdir(d)) && n_files < MAX_FILES) {
        if (strncmp(e->d_name, "stress_", 7) == 0 && strstr(e->d_name, ".bin")) {
            names[n_files++] = strdup(e->d_name);
        }
    }
    closedir(d);
    if (n_files == 0) {
        fprintf(stderr, "no stress_*.bin files in %s (run setup_memstress.sh first)\n", dir);
        return 1;
    }
    if (count > n_files) {
        fprintf(stderr, "count %d > available files %d, clamping\n", count, n_files);
        count = n_files;
    }

    // shuffle and take the first <count>
    srand((unsigned) time(NULL) ^ (unsigned) getpid());
    for (int i = n_files - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        char * t = names[i]; names[i] = names[j]; names[j] = t;
    }

    const long page = sysconf(_SC_PAGESIZE);
    unsigned char * maps[MAX_FILES];
    size_t lens[MAX_FILES];
    size_t total = 0;

    for (int i = 0; i < count; i++) {
        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir, names[i]);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { perror(path); return 1; }
        struct stat st;
        fstat(fd, &st);
        size_t len = (size_t) st.st_size;
        if (len > size_mb * 1024 * 1024) len = size_mb * 1024 * 1024;
        unsigned char * m = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (m == MAP_FAILED) { perror("mmap"); return 1; }
        touch_pages(m, len, page);
        maps[i] = m;
        lens[i] = len;
        total  += len;
    }
    printf("mmap_touch: touched %d files, %.1f MiB (pid %d)\n",
           count, (double) total / (1024.0 * 1024.0), getpid());
    fflush(stdout);

    for (int s = 0; s < hold_sec; s++) {
        sleep(1);
        for (int i = 0; i < count; i++) {
            touch_pages(maps[i], lens[i], page);
        }
    }
    for (int i = 0; i < count; i++) {
        munmap(maps[i], lens[i]);
    }
    return 0;
}
