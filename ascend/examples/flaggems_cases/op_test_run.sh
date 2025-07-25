#!/bin/bash

# 设备内多线程并行 + 跨设备动态负载均衡执行脚本（修复进度显示）

# 获取传入参数
param="$1"
input_ops="$2"
device_count="${3:-1}"           # 默认使用1个设备
threads_per_device="${4:-64}"    # 每个设备线程数，默认64

# 定义路径
DIR_TESTS="tests"
DIR_BENCHMARK="benchmark"
DAILY_LOG_DIR="/home/daily_log"
TIMESTAMP=$(date +"%Y%m%d")
LOG_ARCHIVE="test_flaggems_logs_${TIMESTAMP}.tar.gz"

# 检查日志目录
mkdir -p "$DAILY_LOG_DIR" || { echo "无法创建日志目录 $DAILY_LOG_DIR"; exit 1; }

# 中央计数器文件定义
COUNTER_FILE=$(mktemp)
LOCK_FILE="/tmp/op_test_run.lock"
touch $LOCK_FILE

# 任务队列管理函数
init_task_queue() {
    local -n arr_ref=$1
    TASK_FILE=$(mktemp)
    printf "%s\n" "${arr_ref[@]}" > "$TASK_FILE"
    echo 0 > "$TASK_FILE.counter"
    echo "${#arr_ref[@]}" > "$COUNTER_FILE.total"
    echo 0 > "$COUNTER_FILE.completed"
}

get_next_task() {
    (
        # 文件锁保证原子操作
        flock -x 9
        counter=$(< $TASK_FILE.counter)
        total_tasks=$(wc -l < $TASK_FILE)

        if (( counter >= total_tasks )); then
            echo ""
            return
        fi

        task_name=$(sed -n "$((counter+1))p" $TASK_FILE)
        echo $((counter+1)) > "$TASK_FILE.counter"
        echo "$task_name"
    ) 9> "$TASK_FILE.lock"
}

# 原子更新完成计数器
update_progress() {
    (
        flock -x 11
        local current=$(< $COUNTER_FILE.completed)
        echo $((current + 1)) > $COUNTER_FILE.completed
        echo $((current + 1))
    ) 11> $LOCK_FILE
}

# 获取进度信息
get_progress() {
    (
        flock -s 11   # 共享锁（只读）
        completed=$(< $COUNTER_FILE.completed)
        total=$(< $COUNTER_FILE.total)
        echo "$completed $total"
    ) 11> $LOCK_FILE
}

cleanup_tasks() {
    rm -f "$TASK_FILE" "$TASK_FILE.counter" "$TASK_FILE.lock" $LOCK_FILE $COUNTER_FILE*
}

# 算子列表定义
OPS=("abs" "add" "addmm" "all" "amax" "arange" "argmax" "scaled_dot_product_attention" "apply_rotary_pos_emb" "bitwise_and" "bitwise_not" "bitwise_or" "bmm" "cat" "clamp" \
     "contiguous" "cos" "CrossEntropyLoss" "cumsum" "count_nonzero" "diag_embed" "diagonal_backward" "div" "dot" "dropout" "elu" "embedding" "eq" "erf" \
     "exp" "exponential_" "fill" "flip" "full_like" "full" "gather" "ge" "gelu" \
     "gelu_and_mul" "group_norm" "gt" "hstack" "index_select" "isinf" "isnan" "rsub" "le" "linspace" "log_sigmoid" "log" "logical_and" "logical_not" \
     "logical_or" "logical_xor" "lt" "masked_fill" "max" "maximum" "mean" "min" "minimum" "mm" "mse_loss" "mul" "mv" "native_dropout" "nan_to_num" "ne" \
     "neg" "normal" "ones_like" "pow" "prod" "rand_like" "rand" "randn_like" "randn" "reciprocal" "relu" "resolve_conj" "resolve_neg" "rms_norm" "rsqrt" \
     "select_scatter" "sigmoid" "silu" "silu_and_mul" "sin" "slice_scatter" "softmax" "stack" "sub" "sum" "tanh" "topk" "triu" "uniform_" "where" "threshold" \
     "zeros_like" "zeros")

OPS_EXT=( "instance_norm" "skip_rms_norm" "weight_norm" "angle" "any" "argmin" \
          "batch_norm" "conv_depthwise2d" "conv1d" "copy" "cummin" "diag" \
          "index_add" "index_put" "isclose" "isfinite" "isin" "outer" "kron" "layer_norm" \
          "log" "logical_and" "log_softmax" "masked_select" \
          "multinomial" "nllloss" "nonzero" "pad" "polar" "quantile" \
          "randperm" "repeat_interleave" "repeat" "sort" "tile" "unique" \
           "upsample_bicubic2d_aa" "upsample_nearest2d" "vdot" "vector_norm" "vstack")

OPS_EXCLUDE=( "conv2d" "var_mean" )

# 处理算子列表
if [ "$input_ops" == "fullop" ]; then
    OPS+=("${OPS_EXT[@]}")
elif [ "$input_ops" == "extop" ]; then
    OPS=("${OPS_EXT[@]}")
elif [[ -n "$input_ops" ]]; then
    OPS=()
    IFS=',' read -ra OPS <<< "$input_ops"
fi

total_ops=${#OPS[@]}
echo "======================================"
echo "测试算子列表: ${OPS[@]}"
echo "算子总数: $total_ops"
echo "使用设备数量: $device_count"
echo "每设备线程数: $threads_per_device"
echo "======================================"

# 初始化性能计数器
start_time=$SECONDS

# 线程执行函数 - 正确性测试
run_tests_thread() {
    local device_id=$1
    local thread_id=$2
    local device_log_dir=$3
    local thread_log_dir="$device_log_dir/thread_${thread_id}"
    mkdir -p "$thread_log_dir"

    while true; do
        task_name=$(get_next_task)
        [[ -z "$task_name" ]] && break

        echo "[设备 $device_id-线程 $thread_id] 正在执行: pytest -m $task_name --ref cpu -sv"
        log_file="${thread_log_dir}/result_${task_name}.log"

        # 执行正确性测试并记录时间
        start_op=$(date +%s)
        pytest -m $task_name --ref cpu -sv &> "$log_file"
        exit_code=$?
        duration=$(( $(date +%s) - start_op ))

        # 原子更新完成计数
        new_completed=$(update_progress)

        # 获取最新进度状态
        read completed total < <(get_progress)
        progress=$(( completed * 100 / total ))

        # 输出结果
        if [ $exit_code -ne 0 ]; then
            echo "[错误] [$device_id-$thread_id] $task_name 失败! (用时 ${duration}s, 进度: $completed/$total)"
        else
            echo "[成功] [$device_id-$thread_id] $task_name 完成! (用时 ${duration}s, 进度: $completed/$total)"
        fi
    done
}

# 线程执行函数 - 性能测试
run_benchmark_thread() {
    local device_id=$1
    local thread_id=$2
    local device_log_dir=$3
    local thread_log_dir="$device_log_dir/thread_${thread_id}"
    mkdir -p "$thread_log_dir"

    while true; do
        task_name=$(get_next_task)
        [[ -z "$task_name" ]] && break

        echo "[设备 $device_id-线程 $thread_id] 正在执行: pytest -m $task_name --level core --record log"
        log_file="${thread_log_dir}/benchmark_${task_name}.log"
        perf_file="${thread_log_dir}/perf_${task_name}.log"

        # 执行性能测试并记录时间
        start_op=$(date +%s)
        pytest -m $task_name --level core --record "$perf_file" &> "$log_file"
        exit_code=$?
        duration=$(( $(date +%s) - start_op ))

        # 原子更新完成计数
        new_completed=$(update_progress)

        # 获取最新进度状态
        read completed total < <(get_progress)
        progress=$(( completed * 100 / total ))

        # 输出结果
        if [ $exit_code -ne 0 ]; then
            echo "[错误] [$device_id-$thread_id] $task_name 性能测试失败! (用时 ${duration}s, 进度: $completed/$total)"
        else
            echo "[成功] [$device_id-$thread_id] $task_name 性能测试完成! (用时 ${duration}s, 进度: $completed/$total)"
        fi
    done
}

# 设备主函数
run_device() {
    local device_id=$1
    local mode=$2
    local device_log_dir="device_${device_id}_logs"
    mkdir -p "$device_log_dir"

    # 创建设备内的线程池
    for ((thread_id=0; thread_id < threads_per_device; thread_id++)); do
        if [ "$mode" == "tests" ]; then
            run_tests_thread $device_id $thread_id "$device_log_dir" &
        elif [ "$mode" == "benchmark" ]; then
            run_benchmark_thread $device_id $thread_id "$device_log_dir" &
        fi
    done

    # 等待设备内所有线程完成
    wait
    echo "======== 设备 $device_id 上所有任务完成 ========"
}

# 根据参数执行测试
if [ "$param" == "tests" ]; then
    cd "$DIR_TESTS" || { echo "无法进入目录 $DIR_TESTS"; exit 1; }

    # 创建全局任务队列
    init_task_queue OPS

    # 启动设备主进程
    for ((device_id=0; device_id < device_count; device_id++)); do
        (
            export ASCEND_RT_VISIBLE_DEVICES=$device_id
            run_device $device_id "tests"
        ) &
    done

elif [ "$param" == "benchmark" ]; then
    cd "$DIR_BENCHMARK" || { echo "无法进入目录 $DIR_BENCHMARK"; exit 1; }

    # 性能测试使用单线程模式（保证准确性）
    if [ "$threads_per_device" -gt 1 ]; then
        echo "警告：性能测试模式下自动设置为单线程模式（每个设备1个线程）"
        threads_per_device=1
    fi

    # 创建全局任务队列
    init_task_queue OPS

    # 启动设备主进程
    for ((device_id=0; device_id < device_count; device_id++)); do
        (
            export ASCEND_RT_VISIBLE_DEVICES=$device_id
            run_device $device_id "benchmark"
        ) &
    done

else
    echo "参数错误! 用法:"
    echo "正确性测试: $0 tests \"算子列表\" [设备数量] [线程数]"
    echo "性能测试:   $0 benchmark \"算子列表\" [设备数量] [线程数]"
    cleanup_tasks
    exit 1
fi

# 等待所有设备完成
wait
cleanup_tasks

# 计算总耗时
total_time=$(( SECONDS - start_time ))
hours=$(( total_time / 3600 ))
minutes=$(( (total_time % 3600) / 60 ))
seconds=$(( total_time % 60 ))
time_str=$(printf "%02dh %02dm %02ds" $hours $minutes $seconds)

# 归档所有日志文件
log_dirs=($(find . -maxdepth 1 -type d -name "device_*_logs" 2>/dev/null))
if [ ${#log_dirs[@]} -gt 0 ]; then
    echo "归档日志文件到 $LOG_ARCHIVE"
    tar -czf "$LOG_ARCHIVE" "${log_dirs[@]}"

    if mv "$LOG_ARCHIVE" "$DAILY_LOG_DIR"; then
        echo "日志已保存到: $DAILY_LOG_DIR/$LOG_ARCHIVE"
    else
        echo "警告：日志移动到 $DAILY_LOG_DIR 失败"
    fi

    # 清理临时日志
    rm -rf "${log_dirs[@]}"
else
    echo "警告：未找到任何日志目录，跳过归档"
fi

# 获取最终完成情况
read completed total < <(get_progress 2>/dev/null || echo "0 $total_ops")

# 计算平均耗时
if [ $total -gt 0 ]; then
    avg_time=$(( total_time * 1.0 / total | bc -l ))
    avg_time_seconds=$(printf "%.0f" $avg_time)
    avg_str=$(printf "%02d:%02d" $((avg_time_seconds / 60)) $((avg_time_seconds % 60)))
else
    avg_time=0
    avg_str="N/A"
fi

# 输出统计信息
echo -e "\n=============== 测试统计 ==============="
echo "执行类型:       ${param^}"
echo "总算子数:       $total"
echo "完成算子数:     $completed"
echo "使用设备数:     $device_count"
echo "每设备线程数:   $threads_per_device"
echo "总耗时:         $time_str"
echo "平均耗时/算子:   $avg_str (mm:ss)"
echo "日志位置:       $DAILY_LOG_DIR/$LOG_ARCHIVE"
echo "开始时间:       $(date -d @$start_time '+%Y-%m-%d %H:%M:%S')"
echo "结束时间:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

echo "所有算子测试执行完成!"
exit 0