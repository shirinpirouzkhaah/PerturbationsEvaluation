import concurrent.futures
from tqdm import tqdm
import types
from caching import get_cached_values
import concurrent
from openai import OpenAI

n_threads = 100
cache = {}

def instruct_model(client, prompts, model,max_tokens, n=1, temperature=0.2, frequency_penalty=0, presence_penalty=0, with_cache=True, **kwargs):
    def fetch_fn(missing_prompt):
        messages = [{"role": "user", "content": missing_prompt}]
        adjusted_max_tokens = max_tokens - 2 * len(missing_prompt.split(' '))
        if adjusted_max_tokens < 1:
            return missing_prompt, None
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=adjusted_max_tokens,
                temperature=temperature,
                top_p=0.5,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                **kwargs
            )
            return missing_prompt, [choice.message.content.strip() for choice in response.choices if choice.message.content]
        except Exception as e:
            print(f"Model returned this error: {e}")
            return missing_prompt, None

    def parallel_fetch_fn(missing_prompt_list):
        inputs = []
        outputs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(fetch_fn, prompt) for prompt in missing_prompt_list]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(prompts), desc="Sending prompts to model"):
                i, o = future.result()
                inputs.append(i)
                outputs.append(o)
        return inputs, outputs
    if not with_cache:
        if isinstance(prompts, types.GeneratorType):
            prompts = tuple(prompts)
        return parallel_fetch_fn(prompts)[-1]
    return get_cached_values(prompts, cache, parallel_fetch_fn, key_fn=lambda x: (x, model, 1, temperature, frequency_penalty, presence_penalty))


