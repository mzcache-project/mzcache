#!/system/bin/sh

BINARY="/data/local/tmp/mskim/android-build/bin/mz_profile_test"
MODEL_NAME="Qwen3-0.6B-FP16.gguf"
LOG_PATH="/data/local/tmp/mskim/experiment"
LOG_FILE_MMAP="20250707_load_log_mmap"
LOG_FILE_NOMMAP="20250707_load_log_nommap"

tensor_num=1
while [ "$tensor_num" -le 308 ]
do
    echo "====== Running tensor_num: $tensor_num with mmap: true ======"
    $BINARY $tensor_num $MODEL_NAME $LOG_PATH $LOG_FILE_MMAP true

    echo "====== Running tensor_num: $tensor_num with mmap: false ======"
    $BINARY $tensor_num $MODEL_NAME $LOG_PATH $LOG_FILE_NOMMAP false

    tensor_num=$((tensor_num + 1))
done
