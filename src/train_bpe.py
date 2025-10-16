import os
import time
from collections import Counter
import concurrent.futures
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
from tests.common import FIXTURES_PATH

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocabulary = build_initial_vocabulary(special_tokens)
    pretokenized_dict = pretokenize(input_path, special_tokens)
    paired_tokens = build_token_pairs(pretokenized_dict)
    merges = compute_merges(vocabulary, vocab_size, pretokenized_dict, paired_tokens)
    return (vocabulary, merges)

def build_initial_vocabulary(
    special_tokens: list[str]
) -> dict[int, bytes]:
    vocabulary = {}
    for i in range(len(special_tokens)):
        vocabulary[i] = special_tokens[i].encode("utf-8")

    all_bytes = range(256)
    special_tokens_shift = len(special_tokens)
    for index in all_bytes:
        vocabulary[special_tokens_shift + index] = bytes([index])

    return vocabulary

def pretokenize(
    input_path: str,
    special_tokens: list[str]
) -> dict[tuple[bytes], int]:
    start_time = time.time()
    pretokenized_counter = Counter()
    try:
        num_processes = 16
        futures = []
        with concurrent.futures.ProcessPoolExecutor(num_processes) as executor:
            with open(input_path, "rb") as f: # Move to the end of the file to ensure it's accessible
                boundaries = find_chunk_boundaries(f, 4 * num_processes, b"<|endoftext|>")
            
            special_tokens_PAT = re.compile("|".join([re.escape(st) for st in special_tokens]))
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            regex_instance = re.compile(PAT)
            futures = [executor.submit(read_file_in_chunks, input_path, start, end, special_tokens_PAT, regex_instance) for start, end in zip(boundaries[:-1], boundaries[1:])]
            for future in concurrent.futures.as_completed(futures):
                pretokenized_counter += future.result()
            
            print(f"Length of words counter: {len(pretokenized_counter)}")
           
        print(f"Finished processing all chunks with {num_processes} processes.")
    except Exception as e:
        print(f"Main process error: {e}")

    end_time = time.time()

    print(f"Time to read and pretokenize: {end_time - start_time} seconds")

    return { tuple(map(int.to_bytes, key.encode("utf-8"))):value for key, value in pretokenized_counter.items() }

def read_file_in_chunks(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens_PAT: re.Pattern[str],
    regex_instance: re.Pattern[str]
) -> Counter[str]:
    try:
        counter = Counter()
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            for content in special_tokens_PAT.split(chunk):
                if content.strip() == "":
                    continue

                counter = build_pretokenized_counter(content, counter, regex_instance)
        return counter
    except Exception as e:
        return f"Custom Error: {e}"

def build_pretokenized_counter(
    content: str,
    counter: Counter[str],
    pattern: re.Pattern[str]
) -> Counter[str]:
    for match in pattern.finditer(content):
        counter[match.group()] += 1

    return counter
    
def build_token_pairs(
    pretokenized_dict: dict[tuple[bytes], int]
) -> dict[tuple[bytes, bytes], int]: 
    start_time = time.time()

    paired_tokens = {}
    for token_tuple, count in pretokenized_dict.items():
        if len(token_tuple) == 1:
            continue

        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            existing_pair_count = paired_tokens.get(pair, 0)
            paired_tokens[pair] = existing_pair_count + count

    end_time = time.time()

    print(f"Time to build token pairs: {end_time - start_time} seconds")

    return paired_tokens

def find_most_frequent_pair(
    paired_tokens: dict[tuple[bytes, bytes], int],    
) -> tuple[bytes, bytes]:
    max_entry_count = max(paired_tokens.values()) # max count of any pair

    return max([k for k, v in paired_tokens.items() if v == max_entry_count]) # lexicographically greater pair if multiple max

def compute_merges(
    vocabulary: dict[int, bytes],
    vocab_size: int,
    pretokenized_dict: dict[tuple[bytes], int],
    paired_tokens: tuple[bytes, bytes],
) -> list[tuple[bytes, bytes]]:
    start_time = time.time()

    merges = []

    for i in range(vocab_size - len(vocabulary)):
        """ if len(paired_tokens) == 0:
            return (vocabulary, merges)
         """
        most_frequent_pair = find_most_frequent_pair(paired_tokens)
        token_id = len(vocabulary)
        vocabulary[token_id] = most_frequent_pair[0] + most_frequent_pair[1]
        merges.append(most_frequent_pair)
        merge_tokens(pretokenized_dict, most_frequent_pair, paired_tokens)

    end_time = time.time()

    print(f"Time to build BPE merges: {end_time - start_time} seconds")

    return merges

def merge_tokens(
    merged_tokens: dict[tuple[bytes], int],
    most_frequent_pair: tuple[bytes, bytes],
    paired_tokens: dict[tuple[bytes], int]
):
    new_merged_tokens = {}
    old_merged_tokens = []

    for token_tuple, count in merged_tokens.items():
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
        del merged_tokens[old_token_tuple]

    for new_token_tuple, count in new_merged_tokens.items():
        merged_tokens[new_token_tuple] = count