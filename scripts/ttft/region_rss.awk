# region_rss.awk — sum resident (Rss) kB of the weight and KV mappings from
# /proc/<pid>/smaps, classified by their backing (not by size alone):
#
#   weight = the mmapped GGUF   — file-backed, pathname ends in ".gguf"
#   KV     = large anonymous    — "[anon:...]" or unnamed (numeric inode, no
#                                 pathname) mappings whose Size >= MIN kB
#
# The size floor only gates the anonymous class, so it excludes small anon noise
# (thread stacks, heap arenas) while the file-backed weight is matched by its
# pathname regardless of size. When the KV anon pages swap out to zram their Rss
# drops (the bytes move to the region's Swap: field), so this metric reaches ~0
# under full eviction — unlike VmSwap, which keeps counting zram-resident pages.
#
# Prints: "<weight_rss_kB> <kv_rss_kB>".  Pass -v MIN=<kB> (default 200000).
BEGIN { if (MIN == "") MIN = 200000 }
/^[0-9a-f]+-[0-9a-f]+ / {
    if (cls == 1) wr += rss; else if (cls == 2 && size >= MIN) kr += rss
    cls = 0; rss = 0; size = 0
    if ($NF ~ /\.gguf$/)                          cls = 1   # file-backed weights
    else if ($NF ~ /^\[anon/ || $NF ~ /^[0-9]+$/) cls = 2   # anonymous (KV)
    next
}
$1 == "Size:" { size = $2 }
$1 == "Rss:"  { rss  = $2 }
END { if (cls == 1) wr += rss; else if (cls == 2 && size >= MIN) kr += rss
      print wr + 0, kr + 0 }
