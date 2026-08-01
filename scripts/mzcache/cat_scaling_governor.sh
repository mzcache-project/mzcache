#!/system/bin/sh
for i in /sys/devices/system/cpu/cpu[0-9]*; do
    gov_file="$i/cpufreq/scaling_governor"
    if [ -f "$gov_file" ]; then
        cpu_name=$(basename $i)
        echo "$cpu_name: $(cat $gov_file)"
    fi
done