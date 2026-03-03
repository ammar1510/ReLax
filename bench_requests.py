"""Send concurrent chat completion requests to the local server."""

import asyncio
import time
import aiohttp

URL = "http://localhost:8081/v1/chat/completions"
NUM_REQUESTS = 300
MAX_CONCURRENT = 30

PROMPTS = [
    "Explain the theory of general relativity and its major implications for modern physics and cosmology.",
    "Describe the complete process of photosynthesis, including the light-dependent and light-independent reactions.",
    "Write a comprehensive explanation of how neural networks are trained, including backpropagation and gradient descent.",
    "Discuss the major causes, key events, and long-term consequences of the Industrial Revolution.",
    "Explain the structure and function of DNA, and describe how genetic information is passed from parent to offspring.",
    "Describe the water cycle in detail, including evaporation, condensation, precipitation, and infiltration.",
    "Analyze the main factors that led to World War II and discuss its global impact on society and geopolitics.",
    "Explain the concept of entropy, its significance in thermodynamics, and how it relates to the arrow of time.",
    "Write a detailed overview of the evolution of human communication technologies from ancient times to the internet.",
    "Describe the major cloud computing models and discuss the trade-offs between infrastructure, platform, and software as a service.",
]


async def send_request(session: aiohttp.ClientSession, i: int, sem: asyncio.Semaphore):
    prompt = PROMPTS[i % len(PROMPTS)]
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
    }
    start = time.perf_counter()
    async with sem:
        try:
            async with session.post(URL, json=payload) as resp:
                data = await resp.json()
                elapsed = time.perf_counter() - start
                if resp.status == 200:
                    tokens = data["usage"]["completion_tokens"]
                    text = data["choices"][0]["message"]["content"][:80]
                    print(f"[{i:3d}] {elapsed:6.2f}s  {tokens:4d} tok  | {text}")
                else:
                    print(f"[{i:3d}] {elapsed:6.2f}s  ERROR {resp.status}: {data}")
        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"[{i:3d}] {elapsed:6.2f}s  EXCEPTION: {e}")


async def main():
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        t0 = time.perf_counter()
        tasks = [send_request(session, i, sem) for i in range(NUM_REQUESTS)]
        await asyncio.gather(*tasks)
        total = time.perf_counter() - t0
        print(f"\n--- {NUM_REQUESTS} requests in {total:.2f}s ---")


if __name__ == "__main__":
    asyncio.run(main())
