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
import os
import aiohttp

# from dataclasses import dataclass

# @dataclass
# class Sample:
#     response: str
#     label: str
#     prompt: str = ""

DEFAULT_URL = os.environ.get("MATH_VERIFY_URL", "http://localhost:11111/get_reward")
DEFAULT_RESULT_KEY = os.environ.get("MATH_VERIFY_RESULT_KEY", "reward")
DEFAULT_TIMEOUT_S = float(os.environ.get("MATH_VERIFY_TIMEOUT", "10"))
DEFAULT_MAX_RETRIES = int(os.environ.get("MATH_VERIFY_MAX_RETRIES", "3"))
DEFAULT_CONCURRENCY = int(os.environ.get("MATH_VERIFY_CONCURRENCY", "128"))


async def request_api_wrapper_async(
    data: dict,
    *,
    session: aiohttp.ClientSession,
    url: str,
    result_key: str = DEFAULT_RESULT_KEY,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> float:
    """Make async API request with retry logic using provided session."""
    for _ in range(max_retries):
        try:
            async with session.post(url=url, json=data) as response:
                response.raise_for_status()
                result = await response.json()
                if result_key not in result:
                    raise KeyError(f"{result_key} not in response")
                return float(result[result_key])
        except Exception:
            continue
    return 0.0

async def compute_score(args, samples, **kwargs):
    # URL/字段与并发设置
    url = getattr(args, "rm_url", None) or DEFAULT_URL
    result_key = getattr(args, "reward_key", None) or DEFAULT_RESULT_KEY
    concurrency = DEFAULT_CONCURRENCY
    timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT_S)
    connector = aiohttp.TCPConnector(limit=concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def _score_one(sample):
            solution_str = sample.response
            ground_truth = str(sample.label)
            payload = {"predictions": solution_str, "answers": ground_truth}
            async with semaphore:
                return await request_api_wrapper_async(
                    payload,
                    session=session,
                    url=url,
                    result_key=result_key,
                )

        # 单个样本
        if hasattr(samples, 'response'):
            return await _score_one(samples)

        # 批量样本
        tasks = [_score_one(sample) for sample in samples]
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
