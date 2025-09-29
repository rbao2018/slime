# Copyright 2025 rbao2018. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import aiohttp

# from dataclasses import dataclass

# @dataclass
# class Sample:
#     response: str
#     label: str
#     prompt: str = ""

async def request_api_wrapper_async(data: dict, url: str = "http://localhost:11111/get_reward", result_key: str = "reward", max_retries: int = 3) -> float:
    """Make async API request with retry logic."""
    async with aiohttp.ClientSession() as session:
        for _ in range(max_retries):
            try:
                async with session.post(url=url, json=data, timeout=10) as response:
                    response.raise_for_status()
                    result = await response.json()
                    if result_key not in result:
                        raise KeyError(f"{result_key} not in response")
                    return result[result_key]
            except Exception as e:
                # print(f"API request error: {e}", flush=True)
                # await asyncio.sleep(3)
                continue
        return 0.0

async def compute_score(args, samples, **kwargs):
    # 处理单个样本的情况
    if hasattr(samples, 'response'):  # 单个Sample对象
        sample = samples
        solution_str = sample.response
        ground_truth = str(sample.label)
        return await request_api_wrapper_async(
            {"predictions": solution_str, "answers": ground_truth}
        )
    
    # 处理批量样本的情况
    tasks = []
    for sample in samples:
        solution_str = sample.response
        ground_truth = str(sample.label)
        task = request_api_wrapper_async(
            {"predictions": solution_str, "answers": ground_truth}
        )
        tasks.append(task)
    
    return await asyncio.gather(*tasks)

# async def main():
#     # 创建测试数据
#     args = None  # 这里不需要args
    
#     # 测试单个样本
#     print("Testing single sample...")
#     single_sample = Sample(response="0.5", label="1/2")
#     try:
#         single_result = await compute_score(args, single_sample)
#         print(f"Single sample result: {single_result}")
#     except Exception as e:
#         print(f"Single sample error: {e}")
    
#     # 测试批量样本
#     print("\nTesting batch samples...")
#     batch_samples = [
#         Sample(response=text, label="7.0"),
#         Sample(response="0.25", label="1/4"),
#         Sample(response="0.75", label="3/4"),
#     ]
#     try:
#         batch_results = await compute_score(args, batch_samples)
#         print(f"Batch results: {batch_results}")
#     except Exception as e:
#         print(f"Batch samples error: {e}")

# if __name__ == "__main__":
#     asyncio.run(main())
