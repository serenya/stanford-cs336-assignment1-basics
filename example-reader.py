import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import logging
from typing import Callable, Iterable, Tuple
import traceback


logger = logging.getLogger('reader')
logger.setLevel(logging.INFO)
CHUNK_SIZE = 1 << 20

def chunks_iterator(f_size: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    pos = 0
    while pos < f_size:
        end = min(pos + chunk_size, f_size)
        yield pos, end
        pos = end

def process_chunk(f_path: str, start: int, end: int, fn: Callable[[bytes], float | int]):
    size = end - start
    try:
        with open(f_path, "rb") as f:
            f.seek(start)
            data = f.read(size)
        return fn(data)

    except Exception as e:
        logger.error(f"[IO error] failed to open file {f_path} {e}")

def run(
        f_path: str, 
        chunk_map_fn: Callable[[bytes], float|int], 
        reducer_fn: Callable[[Iterable[float|int]], float | int], 
        chunk_size: int, 
        max_workers: int):
    
    try:
        f_size = os.path.getsize(f_path)
    except OSError as e:
        logger.error(f"[IO error] Failed to access '{f_path}': {e}")

    try:
        tasks = list(chunks_iterator(f_size, chunk_size))
        worker = partial(process_chunk, f_path, fn = chunk_map_fn)
        results = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, start, end) for start, end in tasks]
            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as e:
                    logger.error(f"[Worker error] {e}")
                
            return reducer_fn(results)
    except Exception as e:
        logger.error(f"[Executor error] {e}")
        traceback.print_exc()
        return None

# chunk map functions could be any aggregation
def count_ab_sequences(b: bytes):
    return b.count(b"ab")

if __name__ == '__main__':

    path = "/mnt/d/ISO/debian-12.10.0-amd64-netinst.iso"

    try:
        result = run(path, chunk_map_fn=count_ab_sequences, reducer_fn=sum, chunk_size=CHUNK_SIZE, max_workers=8)
        print(result)
        
    except Exception as e:
        print(f"[Fatal] unhandled exception in main")
        traceback.print_exc()
