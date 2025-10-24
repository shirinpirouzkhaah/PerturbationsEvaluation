import os
import pickle
from more_itertools import unique_everseen 
import types

def create_cache(file_name, create_fn):
    print(f'Creating cache <{file_name}>..')
    result = create_fn()
    with open(file_name, 'wb') as f:
        pickle.dump(result, f)
    return result

def load_cache(file_name):
    if os.path.isfile(file_name):
        print(f'Loading cache <{file_name}>..')
        with open(file_name,'rb') as f:
            return pickle.load(f)
    return None

def load_or_create_cache(file_name, create_fn):
    result = load_cache(file_name)
    if result is None:
        result = create_cache(file_name, create_fn)
    return result

def get_cached_value(q, cache, fetch_fn, key_fn=lambda x:x):
    key_q = key_fn(q)
    cached = key_q in cache
    if not cached:
        new_q = fetch_fn(q)
        cache.update({
            key_q:new_q 
        })
    return cache[key_q], cached

def get_cached_values(value_list, cache, fetch_fn, key_fn=lambda x:x):
	if isinstance(value_list, types.GeneratorType):
		value_list = tuple(value_list)
	missing_values = tuple(
		q 
		for q in unique_everseen(filter(lambda x:x, value_list), key=key_fn) 
		if key_fn(q) not in cache
	)
	if len(missing_values) > 0:
		old_values, new_values = fetch_fn(missing_values)
		cache.update({
			key_fn(q):v 
			for q,v in zip(old_values,new_values)
		})
	return [cache[key_fn(q)] if q else None for q in value_list]
