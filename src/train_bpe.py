import os
import time
from multiprocessing.pool import Pool
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
from tests.common import FIXTURES_PATH

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocabulary = {}
    merges = []
    chunks_of_pretokenized_dict = []
    paired_tokens = {}
    for i in range(len(special_tokens)):
        vocabulary[i] = special_tokens[i].encode("utf-8")

    all_bytes = range(256)
    special_tokens_shift = len(special_tokens)
    for index in all_bytes:
        vocabulary[special_tokens_shift + index] = bytes([index])

    start_time = time.time()
    ## Usage
    boundaries = []
    with open(input_path, "rb") as f:
        num_processes = 200
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.

        multiple_results = []
        process_id = 1
        with Pool(processes=10) as pool:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                chunks_of_pretokenized_dict = []
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                
                
                print(f"Scheduling chunk from {start} to {end} with process_id {process_id}...")
                #items.append((f, start, end, special_tokens))
                #items.append((input_path, process_id, start, end, special_tokens))
                
                result = pool.apply_async(read_file_in_chunks, (chunk, process_id, special_tokens))#(input_path, process_id, start, end, special_tokens))
                multiple_results.append(result)

                process_id += 1

                if len(multiple_results) >= 10:
                    for result in multiple_results:
                        #print("Getting result...")
                        get_result = result.get()
                        print(f"Got result of length {len(get_result)}.")
                        chunks_of_pretokenized_dict += get_result

                    multiple_results = []
                    process_id = 1

                """ if len(multiple_results) >= 10:
                    process_id = 1

                    

                    multiple_results = []

                    print(f"Processed 10% chunks so far...") """
    
    #items = []
    
        



    

        """ multiple_results = []
        for item in items:
            result = pool.apply_async(read_file_in_chunks, item)
            multiple_results.append(result)
        
        for result in multiple_results:
            chunks_of_pretokenized_dict += result.get() """

    end_time = time.time()
    print(f"Time to read and pretokenize: {end_time - start_time} seconds")

    """ start_time = time.time()

    for pretokenized_dict in chunks_of_pretokenized_dict:
        paired_tokens = build_token_pairs(pretokenized_dict, paired_tokens)

    end_time = time.time()
    print(f"Time to build token pairs: {end_time - start_time} seconds")

    start_time = time.time()

    for i in range(vocab_size - len(vocabulary)):
        if len(paired_tokens) == 0:
            break

        most_frequent_pair = find_most_frequent_pair(paired_tokens)
        token_id = len(vocabulary)
        vocabulary[token_id] = most_frequent_pair[0] + most_frequent_pair[1]
        merges.append(most_frequent_pair)

        for merged_tokens in chunks_of_pretokenized_dict:
            merge_tokens(merged_tokens, most_frequent_pair, paired_tokens)

    end_time = time.time()
    print(f"Time to build BPE merges: {end_time - start_time} seconds") """

    return (vocabulary, merges)

def read_file_in_chunks(
    chunk: str,
    #f: BinaryIO,
    #input_path: str | os.PathLike,
    process_id: int,
    #start: int,
    #end: int,
    special_tokens: list[str],
) -> list[dict[tuple[bytes], int]]:
    i = 0
    chunks_of_pretokenized_dict = []
    """ with open(input_path, "rb") as f:
        
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
 """        # Run pre-tokenization on your chunk and store the counts for each pre-token
        
    for content in re.split("|".join([re.escape(st) for st in special_tokens]), chunk):
        if content.strip() == "":
            continue

        """ if i > 0 and i % 10000 == 0:
            print(f"With process_id {process_id}, processed {i} chunks...") """

        pretokenized_dict = build_pretokenized_dict(content)
        chunks_of_pretokenized_dict.append(pretokenized_dict)
        i += 1

    print(f"Scheduled chunk with process_id {process_id} finished with length {len(chunks_of_pretokenized_dict)}.")
    return chunks_of_pretokenized_dict

def build_pretokenized_dict(
    content: str
) -> dict[tuple[bytes], int]:
    pretokenized_dict = {}
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    for match in re.finditer(PAT, content):
        matched_tuple = tuple(map(int.to_bytes, match.group().encode("utf-8")))

        existing_pretokenized_item_count = pretokenized_dict.get(matched_tuple, 0)

        pretokenized_dict[matched_tuple] = existing_pretokenized_item_count + 1

    return pretokenized_dict
    

def build_token_pairs(
    pretokenized_dict: dict[tuple[bytes], int],
    paired_tokens: dict[tuple[bytes, bytes], int]
) -> dict[tuple[bytes, bytes], int]: 
    for token_tuple, count in pretokenized_dict.items():
        if len(token_tuple) == 1:
            continue

        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            existing_pair_count = paired_tokens.get(pair, 0)
            paired_tokens[pair] = existing_pair_count + count

    return paired_tokens

def find_most_frequent_pair(
    paired_tokens: dict[tuple[bytes, bytes], int],    
) -> tuple[bytes, bytes]:
    max_entry_count = max(paired_tokens.values()) # max count of any pair

    return max([k for k, v in paired_tokens.items() if v == max_entry_count]) # lexicographically greater pair if multiple max

def merge_tokens(
    byte_pair_frequency: dict[tuple[bytes], int],
    most_frequent_pair: tuple[bytes, bytes],
    paired_tokens: dict[tuple[bytes], int]
):
    new_merged_tokens = {}
    old_merged_tokens = []

    for token_tuple, count in byte_pair_frequency.items():
        any_merges = False
        skip_next_pair = False
        is_last_token_merged = False
        token_tuple_values = []
        for i in range(len(token_tuple) - 1):
            if skip_next_pair:
                skip_next_pair = False
                continue

            current_pair = (token_tuple[i], token_tuple[i + 1])
            if current_pair == most_frequent_pair:
                token_tuple_values.append(token_tuple[i] + token_tuple[i + 1])
                skip_next_pair = True
                any_merges = True
                is_last_token_merged = i + 1 == len(token_tuple) - 1

                pair_count = paired_tokens.get(current_pair, 0)
                if pair_count > 0:
                    del paired_tokens[current_pair]

                # Update the paired_tokens dictionary to remove pairs that are no longer valid
                if i - 1 >= 0:
                    left_pair = (token_tuple[i - 1], token_tuple[i])
                    existing_left_pair_count = paired_tokens.get(left_pair, 0)
                    if existing_left_pair_count > 0:
                        paired_tokens[left_pair] -= count
                        if paired_tokens[left_pair] <= 0:
                            del paired_tokens[left_pair]

                    pair = (token_tuple[i - 1], token_tuple[i] + token_tuple[i + 1])
                    existing_pair_count = paired_tokens.get(pair, 0)
                    paired_tokens[pair] = existing_pair_count + count

                    

                if i + 2 < len(token_tuple):
                    right_pair = (token_tuple[i + 1], token_tuple[i + 2])
                    existing_right_pair_count = paired_tokens.get(right_pair, 0)
                    if existing_right_pair_count > 0:
                        paired_tokens[right_pair] -= count
                        if paired_tokens[right_pair] <= 0:
                            del paired_tokens[right_pair]

                    pair = (token_tuple[i] + token_tuple[i + 1], token_tuple[i + 2])
                    existing_pair_count = paired_tokens.get(pair, 0)
                    paired_tokens[pair] = existing_pair_count + count
            else:
                token_tuple_values.append(token_tuple[i])

        if any_merges:
            if not is_last_token_merged:
                token_tuple_values.append(token_tuple[-1])

            old_merged_tokens.append(token_tuple)
            new_merged_tokens[tuple(token_tuple_values)] = count

    for old_token_tuple in old_merged_tokens:
        del byte_pair_frequency[old_token_tuple]

    for new_token_tuple, count in new_merged_tokens.items():
        byte_pair_frequency[new_token_tuple] = count

train_bpe(
    input_path=FIXTURES_PATH / "data/TinyStoriesV2-GPT4-train.txt",
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
)