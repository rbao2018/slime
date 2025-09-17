import copy
import asyncio
from concurrent.futures import as_completed, Future, ThreadPoolExecutor
import requests
from pebble import ProcessPool
from tqdm import tqdm
from transformers import AutoTokenizer

from slime.utils.data import Dataset
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.types import Sample
from slime.rollout.rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout"]

def sync_rm_wrapper(args, sample):
    """Synchronous wrapper to run the async_rm function."""
    return asyncio.run(async_rm(args, sample))

def batched_sync_rm_wrapper(args, samples):
    """Synchronous wrapper to run the batched_async_rm function."""
    return asyncio.run(batched_async_rm(args, samples))

class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args):
        # persistent state for the generation process
        self.args = args
        # self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        self.max_workers = args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        self.sampling_params = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        self.reset()

    def reset(self):
        self.remaining_batch_size = 0
        self.pendings = []
        # 新增：用于跟踪未完成任务的原始样本
        self.pending_samples: dict[Future, list[Sample]] = {}
        self.aborted = False
        self.pool = None

    def submit_generate_tasks(self, samples: list[list[Sample]], pool: ProcessPool):
        for group in samples:
            future = pool.schedule(
                generate_and_rm_group_worker,
                args=[self.args, group, self.sampling_params.copy(), False]
            )
            self.pendings.append(future)
            # 关键修改：将 future 和它对应的输入样本关联起来
            self.pending_samples[future] = group
        self.remaining_batch_size += len(samples)


def generate_worker(args, sample: Sample, sampling_params) -> Sample:
    """Worker function for generate"""
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    if len(sample.response) > 0:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(
            tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        )

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Token-based mode: use tokens directly
    if len(sample.response) > 0:
        input_token_ids = sample.tokens
    else:
        # First turn: initialize with prompt tokens
        prompt_token_ids = tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        input_token_ids = prompt_token_ids
        # Initialize sample.tokens with prompt for subsequent turns
        if not sample.tokens:  # Only set if empty
            sample.tokens = prompt_token_ids

    # Prepare payload - shared structure
    payload = {
        "input_ids": input_token_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    response = requests.post(url, json=payload, timeout=3600)
    output = response.json()

    # Extract new response tokens
    if "output_token_logprobs" in output["meta_info"]:
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        # abort
        new_response_tokens = []
        new_response_log_probs = []

    # Update sample with tokens directly - avoiding re-tokenization
    sample.tokens = sample.tokens + new_response_tokens
    sample.response_length += len(new_response_tokens)
    sample.response += output["text"]
    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs += new_response_log_probs

    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


def generate_and_rm_worker(args, sample: Sample, sampling_params: dict, evaluation=False) -> Sample:
    """Worker function for generate_and_rm"""
    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    # generate
    if args.custom_generate_function_path is not None:
        custom_generate_func = load_function(args.custom_generate_function_path)
        sample = custom_generate_func(args, sample, sampling_params)
    else:
        sample = generate_worker(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = batched_sync_rm_wrapper(args, samples_need_reward)
        # rewards = [0 for sample in samples_need_reward]
        for sample, reward in zip(samples_need_reward, rewards):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        sample.reward = sync_rm_wrapper(args, sample)

    return sample

def generate_and_rm_group_worker(args, group: list[Sample], sampling_params: dict, evaluation=False) -> list[Sample]:
    """
    Worker function for generate_and_rm_group.
    Processes samples within the group in parallel using a ThreadPool.
    """
    processed_group = [None] * len(group) # Pre-allocate list to maintain order
    
    # 使用 ThreadPoolExecutor 并行处理组内的所有 sample
    # max_workers 可以根据实际情况调整，设为 None 通常会使用一个合理的默认值
    # 对于IO密集型任务，可以设置得比 CPU 核心数多，例如等于 group 的大小
    with ThreadPoolExecutor(max_workers=len(group)) as executor:
        # 提交所有任务，并用一个字典来映射 future 和它对应的原始索引
        future_to_index = {
            executor.submit(generate_and_rm_worker, args, sample, sampling_params.copy(), evaluation): i
            for i, sample in enumerate(group)
        }

        # 等待任务完成并收集结果
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                # 获取任务结果
                processed_sample = future.result()
                # 按照原始顺序放入结果列表
                processed_group[index] = processed_sample
            except Exception as e:
                print(f"A task in generate_and_rm_group_worker failed for sample index {index}: {e}", flush=True)
                # 你可以在这里决定如何处理失败的 sample，例如用 None 填充或重新抛出异常
                # 为了保持健壮性，我们暂时不打断整个流程
                pass # 或者 processed_group[index] = None

    # 过滤掉可能失败的任务（如果它们返回None）
    processed_group = [s for s in processed_group if s is not None]

    # for the rm that need the whole group, we will do the rm here
    if args.group_rm and processed_group:
        # 确保在计算奖励之前，所有样本都已经成功生成
        rewards = batched_sync_rm_wrapper(args, processed_group) # 假设你已应用了上一个回答的修改
        for sample, reward in zip(processed_group, rewards):
            sample.reward = reward

    return processed_group

def abort(args, state):
    """
    Aborts pending tasks by cancelling their futures and returns the original
    input samples of all tasks that did not complete.
    """
    state.aborted = True

    # 1. 尝试取消所有仍在 pendings 列表中的 future
    # 这是为了尽早停止工作进程中排队的任务
    print(f"Aborting... Attempting to cancel {len(state.pendings)} futures.", flush=True)
    for future in state.pendings:
        if not future.done():
            future.cancel()

    # 2. 关键修改：直接收集 `pending_samples` 字典中剩余的所有样本
    # 这些就是所有已提交但从未收到返回结果的任务的原始输入
    aborted_samples_groups = list(state.pending_samples.values())
    
    print(f"Collected {len(aborted_samples_groups)} aborted sample groups into the data buffer.", flush=True)

    return aborted_samples_groups


def generate_rollout_sync(args, rollout_id: int, data_source) -> tuple[list[list[Sample]], list]:
    """Synchronous version of generate_rollout_async"""
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    over_sampling_filter = (
        load_function(args.over_sampling_filter_path) if args.over_sampling_filter_path is not None else None
    )

    # target_data_size is the total number of valid samples to get
    target_data_size = args.over_sampling_batch_size if over_sampling_filter is not None else args.rollout_batch_size

    data = []
    do_print = True
    submit_generate_tasks_sum = 0 

    # pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")

    # 手动创建进程池，不再使用 `with` 语句
    pool = ProcessPool(max_workers=state.max_workers)
    state.pool = pool

    try:
        # 初始提交第一批任务
        if not state.pendings:
            initial_samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(initial_samples, pool)
            submit_generate_tasks_sum += args.over_sampling_batch_size
            print(f"submit_generate_tasks_sum: {submit_generate_tasks_sum} with length state.pendings: {len(state.pendings)} and data length {len(data)}", flush=True)

        # 只要数据没集齐，就持续处理
        while len(data) < target_data_size:
            # 如果待处理的任务太少，就补充一些
            # 这里的判断条件为保持一定数量的正在运行的任务
            if len(state.pendings) < state.max_workers * 2:
                 samples = data_source(args.over_sampling_batch_size)
                 state.submit_generate_tasks(samples, pool)
                 submit_generate_tasks_sum += args.over_sampling_batch_size
                 print(f"submit_generate_tasks_sum: {submit_generate_tasks_sum} with length state.pendings: {len(state.pendings)} and data length {len(data)}", flush=True)
            
            try:
                # 从已完成的任务中取出一个
                future = next(as_completed(state.pendings))

                # 从 pendings 列表中移除，表示已处理
                state.pendings.remove(future)

                # 获取结果
                group: list[Sample] = future.result()
                
                # 任务已成功完成，从跟踪字典中移除
                if future in state.pending_samples:
                    del state.pending_samples[future]
                
                # ... (处理 group 的逻辑不变) ...
                if do_print:
                    sample = group[0][0] if isinstance(group[0], list) else group[0]
                    print(
                        f"First rollout sample: {[sample.prompt + sample.response]}, label: {sample.label}, reward: {sample.reward}",
                        flush=True,
                    )
                    do_print = False

                if dynamic_filter is not None and dynamic_filter(args, group):
                    continue

                data.append(group)
                # pbar.update(args.n_samples_per_prompt)

            except StopIteration:
                # 所有提交的任务都已完成，但数据仍不足
                print("All submitted tasks completed, but target data size not reached. Breaking.", flush=True)
                break 
            except Exception as e:
                print(f"A task failed with error: {e}", flush=True)
                # 任务失败了，也要从跟踪字典中移除
                if future in state.pending_samples:
                    del state.pending_samples[future]

    finally:
        # 无论 try 块如何退出（包括 break），这里都会被执行
        print("Data collection goal reached or process interrupted. Forcefully stopping the pool.", flush=True)
        # 关键：强制停止所有工作进程
        pool.stop()
        # 等待进程完全终止
        pool.join()
        # 清理池资源
        pool.close()

    # 此时，所有工作进程都已被终止
    # pbar.close()
    
    if data:
        sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
        print(
            f"Finish rollout: {[sample.prompt + sample.response]}, label: {sample.label}, reward: {sample.reward}",
            flush=True,
        )

    # 调用我们之前设计的、现在可以完美工作的 abort 函数
    aborted_samples = abort(args, state)

    # ... (后续数据处理和返回逻辑不变) ...
    if over_sampling_filter is not None:
        data = over_sampling_filter(args, data)[: args.rollout_batch_size]

    assert len(data) <= args.rollout_batch_size, f"Got {len(data)} samples, expected at most {args.rollout_batch_size}"
    
    # 因为可能提前中断，所以不一定能严格等于batch_size，这里调整断言
    if len(data) < args.rollout_batch_size:
        print(f"Warning: Final data size {len(data)} is less than target {args.rollout_batch_size}", flush=True)

    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)

    state.reset()
    return data, aborted_samples



EVAL_PROMPT_DATASET = {}


def eval_rollout(args, rollout_id):
    """Synchronous version of eval_rollout"""
    assert not args.group_rm, "Group RM is not supported for eval rollout"
    results = {}
    for i in range(0, len(args.eval_prompt_data), 2):
        name, path = args.eval_prompt_data[i : i + 2]
        results.update(eval_rollout_single_dataset(args, rollout_id, name, path))
    return results, []


def eval_rollout_single_dataset(args, rollout_id, name, path):
    """Synchronous version of eval_rollout_single_dataset"""
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    if name not in EVAL_PROMPT_DATASET:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[name] = Dataset(
            path,
            tokenizer=tokenizer,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key if args.eval_input_key is None else args.eval_input_key,
            label_key=args.label_key if args.eval_label_key is None else args.eval_label_key,
            metadata_key=args.metadata_key,
            tool_key=args.tool_key if args.eval_tool_key is None else args.eval_tool_key,
            apply_chat_template=args.apply_chat_template,
        )
    dataset = EVAL_PROMPT_DATASET[name]

    sampling_params = dict(
        temperature=args.rollout_temperature if args.eval_temperature is None else args.eval_temperature,
        top_p=args.rollout_top_p if args.eval_top_p is None else args.eval_top_p,
        top_k=args.rollout_top_k if args.eval_top_k is None else args.eval_top_k,
        max_new_tokens=(
            args.rollout_max_response_len if args.eval_max_response_len is None else args.eval_max_response_len
        ),
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    # Prepare tasks
    task_args = []
    sample_index = 0
    for i, prompt_sample in enumerate(dataset.samples):
        for j in range(args.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            task_args.append((args, sample, sampling_params, True))

    data = []
    do_print = True
    pbar = tqdm(total=len(task_args), desc="Rollout generation", disable=not do_print)
    
    state = GenerateState(args)
    with ProcessPool(max_workers=state.max_workers) as pool:
        # Submit all tasks
        futures = []
        for task_arg in task_args:
            future = pool.schedule(generate_and_rm_worker, args=task_arg, timeout=None)
            futures.append(future)
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                sample = future.result()
                if do_print:
                    print([sample.prompt + sample.response], sample.reward, flush=True)
                    do_print = False
                data.append(sample)
                pbar.update(1)
            except Exception as e:
                print(f"Task failed with error: {e}")
                pbar.update(1)
    
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.reward_key or args.eval_reward_key
    return {
        name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
        }
    }


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Main function - synchronous version"""
    completed_samples, aborted_samples = generate_abortable_samples(
        args, rollout_id, data_buffer.get_samples, evaluation=evaluation
    )
    data_buffer.add_samples(aborted_samples)
    return completed_samples


def generate_abortable_samples(args, rollout_id, data_source, evaluation=False):
    """Synchronous version of generate_abortable_samples"""
    assert args.rollout_global_dataset
    if evaluation:
        return eval_rollout(args, rollout_id)
    return generate_rollout_sync(args, rollout_id, data_source)