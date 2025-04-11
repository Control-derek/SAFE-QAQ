# SPDX-License-Identifier: Apache-2.0
"""
This script performs batch inference with audio LLM using vLLM.
It reads inputs from a JSON file, generates multiple outputs for each input,
and saves the results to another JSON file.
"""
import os
import json
import argparse
import librosa
import torch
import random
import time
import numpy as np
import pickle
import tempfile
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from datetime import datetime
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import gc

# Disable Dynamo to avoid potential issues
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference for audio LLM")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the model")
    parser.add_argument("--input_json", type=str, required=True, 
                        help="Path to input JSON file")
    parser.add_argument("--output_json", type=str, required=True, 
                        help="Path to output JSON file")
    parser.add_argument("--preprocess_cache", type=str, default="", 
                        help="Path to save/load preprocessed data cache")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for inference")
    parser.add_argument("--num_samples", type=int, default=3, 
                        help="Number of samples to generate for each input")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Temperature for sampling (0.7-1.0 for more randomness)")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Top-p (nucleus) sampling parameter (0.9-1.0 for more diversity)")
    parser.add_argument("--top_k", type=int, default=50, 
                        help="Top-k sampling parameter (higher for more diversity)")
    parser.add_argument("--max_tokens", type=int, default=1024, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Maximum model length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95,
                        help="GPU memory utilization")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip_preprocessing", action="store_true",
                        help="Skip preprocessing if cache exists")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Interval (in batches) to save intermediate results")
    parser.add_argument("--cpu_workers", type=int, default=4,
                        help="Number of CPU workers for preprocessing")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Number of batches to prefetch for inference")
    parser.add_argument("--chunk_size", type=int, default=50,
                        help="Number of items to process in each preprocessing chunk")
    return parser.parse_args()

def load_json_data(json_path):
    """Load data from JSON file"""
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON file: {str(e)}")
            print("Attempting to repair JSON file...")
            return repair_json_file(json_path)
    return []

def repair_json_file(json_path):
    """
    Attempt to repair a corrupted JSON file
    Returns as many valid items as possible
    """
    items = []
    corrupted_file = os.path.basename(json_path)
    backup_path = f"{json_path}.bak"
    
    # First create a backup of the corrupted file
    if not os.path.exists(backup_path):
        print(f"Creating backup of corrupted file at: {backup_path}")
        try:
            import shutil
            shutil.copy2(json_path, backup_path)
        except Exception as e:
            print(f"Warning: Could not create backup: {str(e)}")
    
    # Try to read file line by line to find valid JSON objects
    print("Searching for valid JSON items...")
    with open(json_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for complete JSON objects
    if content.strip().startswith('[') and ']' in content:
        # This looks like a JSON array
        try:
            # Try to find the last valid item (before any corruption)
            item_count = 0
            depth = 0
            in_string = False
            escape_next = False
            last_valid_pos = 1  # Skip the opening '['
            
            for i, char in enumerate(content):
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\' and in_string:
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if char == '{':
                        if depth == 0:
                            item_start = i
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            # Found a complete object
                            item_count += 1
                            last_valid_pos = i + 1
                            
                            # Check if followed by comma or closing bracket
                            next_chars = content[i+1:i+10].strip()
                            if next_chars.startswith(',') or next_chars.startswith(']'):
                                last_valid_pos = i + 2 if next_chars.startswith(',') else i + 1
            
            # Extract the valid portion of the array
            if last_valid_pos > 1:
                valid_json = content[:last_valid_pos].rstrip(',') + ']'
                try:
                    items = json.loads(valid_json)
                    print(f"Successfully recovered {len(items)} items from the corrupted JSON file")
                except json.JSONDecodeError as e:
                    print(f"Error parsing recovered JSON: {str(e)}")
                    # Try recovering individual items if array parsing fails
                    items = recover_individual_items(content)
            else:
                items = recover_individual_items(content)
        except Exception as e:
            print(f"Error during JSON repair: {str(e)}")
            items = recover_individual_items(content)
    
    # Save the repaired data
    if items:
        repaired_path = f"{json_path}.repaired"
        print(f"Saving {len(items)} repaired items to: {repaired_path}")
        with open(repaired_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        
        # Replace the original file with the repaired version
        print(f"Replacing corrupted file with repaired version")
        os.replace(repaired_path, json_path)
    else:
        print("Could not recover any valid items from the corrupted JSON file")
    
    return items

def recover_individual_items(content):
    """
    Try to recover individual JSON objects from content
    by searching for patterns like complete {...} objects
    """
    items = []
    import re
    
    # Find all potential JSON objects (anything between { and })
    # This is a simplified approach and might not work for all cases
    pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
    potential_objects = re.findall(pattern, content)
    
    for obj_str in potential_objects:
        try:
            obj = json.loads(obj_str)
            # Verify this looks like a valid item with required fields
            if isinstance(obj, dict) and "prompt" in obj:
                items.append(obj)
        except json.JSONDecodeError:
            pass
    
    print(f"Recovered {len(items)} individual items from content")
    return items

def save_json_data(data, json_path):
    """Save data to JSON file safely to prevent corruption"""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # First write to a temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=os.path.dirname(json_path) or ".", suffix='.json')
    try:
        with temp_file:
            json.dump(data, temp_file, ensure_ascii=False, indent=2)
        
        # If successful, rename to the target file (atomic operation on most file systems)
        if os.path.exists(json_path):
            # On Windows, we need to remove the target file first
            if os.name == 'nt' and os.path.exists(json_path):
                os.remove(json_path)
        os.rename(temp_file.name, json_path)
    except Exception as e:
        print(f"Error saving JSON data: {str(e)}")
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        raise

def process_single_item(item, processor, sampling_rate=16000):
    """Process a single input item - suitable for parallel processing"""
    # Apply chat template to create text prompt
    text_prompt = processor.apply_chat_template(item['prompt'], add_generation_prompt=True, tokenize=False)
    
    # Extract audio data from messages
    audio_data = []
    audio_files = []  # Just for tracking file paths
    
    for message in item["prompt"]:
        if isinstance(message.get("content"), list):
            for ele in message["content"]:
                if ele.get("type") == "audio" and "audio_url" in ele:
                    try:
                        audio_path = ele['audio_url']
                        # Use stereo (mono=False) to preserve channel information
                        audio, _ = librosa.load(
                            audio_path, 
                            sr=sampling_rate,
                            mono=False  # Preserve stereo channels
                        )
                        audio_data.append(audio)
                        audio_files.append(audio_path)
                    except Exception as e:
                        print(f"Error loading audio file: {ele['audio_url']}, error: {e}")
    
    # Create multimedia data dictionary
    mm_data = {}
    if audio_data:
        mm_data["audio"] = audio_data if len(audio_data) > 1 else audio_data[0]
    
    # Store preprocessed item
    return {
        "prompt": text_prompt,
        "mm_data": mm_data,
        "original": item,
        "audio_files": audio_files  # For debugging/logging
    }

def process_batch_items(item_batch, processor, sampling_rate=16000):
    """Process a batch of input items in parallel"""
    results = []
    for i, item in enumerate(item_batch):
        try:
            processed = process_single_item(item, processor, sampling_rate)
            processed["index"] = i
            results.append(processed)
        except Exception as e:
            print(f"Error processing item {i}: {e}")
    return results

def preprocess_inputs_in_chunks(input_data, processor, num_workers=4, chunk_size=50, cache_path=None, skip_if_exists=True):
    """
    Preprocess inputs in chunks to reduce memory usage, using parallel processing for each chunk
    Returns a list of preprocessed items with prompts and multimedia data
    """
    # Check if preprocessed cache exists
    if cache_path and os.path.exists(cache_path) and skip_if_exists:
        print(f"Loading preprocessed data from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"Preprocessing input data with {num_workers} parallel workers in chunks of {chunk_size}...")
    
    # Get sampling rate once to avoid loading it repeatedly
    sampling_rate = processor.feature_extractor.sampling_rate
    
    preprocessed_items = []
    total_chunks = (len(input_data) + chunk_size - 1) // chunk_size
    
    # Process chunks
    for chunk_idx in range(total_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, len(input_data))
        current_chunk = input_data[chunk_start:chunk_end]
        
        print(f"Processing chunk {chunk_idx+1}/{total_chunks} with {len(current_chunk)} items")
        
        # Process the current chunk in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Split the chunk into mini-batches for the workers
            batch_size = max(1, len(current_chunk) // num_workers)
            mini_batches = [current_chunk[i:i+batch_size] for i in range(0, len(current_chunk), batch_size)]
            
            # Process each mini-batch
            chunk_results = []
            for result_batch in executor.map(
                    lambda batch: process_batch_items(batch, processor, sampling_rate), 
                    mini_batches):
                chunk_results.extend(result_batch)
        
        # Add global index to each preprocessed item
        for item in chunk_results:
            item["index"] = chunk_start + item["index"]
            preprocessed_items.append(item)
        
        # Free memory after each chunk
        gc.collect()
    
    # Save preprocessed data to cache if path is provided
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"Saving {len(preprocessed_items)} preprocessed items to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(preprocessed_items, f)
    
    return preprocessed_items

def get_processed_items(output_data, num_samples):
    """
    Get a set of already processed item indices from the output data
    Returns a dictionary mapping original item hash to a set of processed sample_ids
    """
    processed_items = {}
    
    for item in output_data:
        if "prompt" in item:
            # Create a hash of the prompt to identify the original item
            prompt_str = json.dumps(item["prompt"], sort_keys=True)
            prompt_hash = hash(prompt_str)
            
            if prompt_hash not in processed_items:
                processed_items[prompt_hash] = set()
            
            if "sample_id" in item:
                processed_items[prompt_hash].add(item["sample_id"])
    
    return processed_items

def is_item_fully_processed(item_hash, sample_ids, num_samples):
    """Check if an item has all its samples processed"""
    return len(sample_ids) >= num_samples

def get_audio_data_complexity(mm_data):
    """Calculate audio complexity for more intelligent batching"""
    if not mm_data or "audio" not in mm_data:
        return 0
    
    audio_data = mm_data["audio"]
    
    # If it's a list of audio files
    if isinstance(audio_data, list):
        # Get both total length and number of files into consideration
        total_length = sum(audio.shape[-1] for audio in audio_data)
        num_files = len(audio_data)
        return (total_length, num_files)
    
    # If it's a single audio file
    elif hasattr(audio_data, 'shape'):
        return (audio_data.shape[-1], 1)
    
    # Fallback
    return 0

def create_optimized_batches(preprocessed_items, processed_items, num_samples, batch_size):
    """
    Create inference batches optimized for similar audio lengths and complexity
    Each batch contains items with similar audio characteristics for better performance
    """
    pending_items = []
    
    # First gather all items that need processing
    for item in preprocessed_items:
        original_item = item["original"]
        prompt_str = json.dumps(original_item["prompt"], sort_keys=True)
        item_hash = hash(prompt_str)
        processed_samples = processed_items.get(item_hash, set())
        
        # Only process samples that haven't been generated yet
        needed_samples = [j for j in range(num_samples) if j not in processed_samples]
        
        for sample_id in needed_samples:
            # Calculate audio complexity for better batching
            audio_complexity = get_audio_data_complexity(item["mm_data"])
            
            batch_item = {
                "index": item["index"],
                "sample_id": sample_id,
                "prompt": item["prompt"],
                "mm_data": item["mm_data"],
                "original": item["original"],
                "audio_complexity": audio_complexity
            }
            pending_items.append(batch_item)
    
    # Sort by audio complexity (total length, then number of files)
    # This creates more homogeneous batches with less padding wasted
    pending_items.sort(key=lambda x: x["audio_complexity"])
    
    # Create batches
    inference_batches = []
    for i in range(0, len(pending_items), batch_size):
        batch = pending_items[i:i+batch_size]
        inference_batches.append(batch)
    
    return inference_batches

def get_batch_prefetcher(batches, prefetch_factor=2):
    """
    Creates a batch prefetcher that pre-loads the next few batches
    while the current batch is being processed
    """
    import queue
    import threading
    
    prefetch_queue = queue.Queue(maxsize=prefetch_factor)
    
    def prefetch_worker():
        for batch in batches:
            # Create a copy of batch data to avoid modification issues
            prefetch_queue.put(batch.copy())
        # Signal end of batches
        prefetch_queue.put(None)
    
    # Start prefetching thread
    threading.Thread(target=prefetch_worker, daemon=True).start()
    
    # Generator that yields prefetched batches
    while True:
        batch = prefetch_queue.get()
        if batch is None:  # End signal
            break
        yield batch

def main():
    args = parse_args()
    set_seed(args.seed)
    
    print(f"=== Batch inference started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"=== Model path: {args.model_path} ===")
    print(f"=== Input data: {args.input_json} ===")
    print(f"=== Output path: {args.output_json} ===")
    
    # Set default preprocess cache path if not provided
    if not args.preprocess_cache:
        cache_dir = os.path.dirname(args.output_json) or "."
        args.preprocess_cache = os.path.join(cache_dir, 
                                          f"preprocess_cache_{os.path.basename(args.input_json)}.pkl")
    
    # Check if output JSON exists for resuming
    resume_mode = os.path.exists(args.output_json)
    if resume_mode:
        print(f"\nOutput file {args.output_json} exists. Resuming from previous run...")
        output_data = load_json_data(args.output_json)
        print(f"Loaded {len(output_data)} existing outputs")
    else:
        print("\nStarting new run...")
        output_data = []
    
    # First, load processor and do preprocessing with CPU resources
    print("\nLoading processor...")
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
        processor.pad_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.pad_token)
        processor.eos_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.eos_token)
        print("Processor loaded successfully!")
    except Exception as e:
        print(f"Error loading processor: {str(e)}")
        return
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>"],  # Qwen model's stop token
    )
    
    # Load input data
    print(f"\nLoading data from {args.input_json}...")
    input_data = load_json_data(args.input_json)
    print(f"Loaded {len(input_data)} input items")
    
    # CPU-based preprocessing with chunking to reduce memory usage
    # Process in smaller chunks to avoid memory issues with large datasets
    preprocessed_items = preprocess_inputs_in_chunks(
        input_data, 
        processor, 
        num_workers=args.cpu_workers,
        chunk_size=args.chunk_size,
        cache_path=args.preprocess_cache,
        skip_if_exists=args.skip_preprocessing
    )
    print(f"Preprocessing complete. {len(preprocessed_items)} items ready for inference")
    
    # Clean up processor to free memory
    del processor
    gc.collect()
    
    # If resuming, determine which items have been processed
    if resume_mode:
        processed_items = get_processed_items(output_data, args.num_samples)
        print(f"Found {len(processed_items)} items with some processed samples")
    else:
        processed_items = {}
    
    # Create batches for inference with optimized grouping
    inference_batches = create_optimized_batches(
        preprocessed_items, 
        processed_items, 
        args.num_samples, 
        args.batch_size
    )
    
    total_samples_needed = sum(len(batch) for batch in inference_batches)
    print(f"Created {len(inference_batches)} inference batches with {total_samples_needed} total samples to generate")
    
    if not inference_batches:
        print("All items are already fully processed.")
        return
    
    # Use CUDA empty cache to free up fragmented memory
    torch.cuda.empty_cache()
    
    # Load the model (only when we actually need to do inference)
    print(f"\nLoading model from {args.model_path}...")
    try:
        # Configure vLLM with performance-optimized settings
        llm = LLM(model=args.model_path,
                max_model_len=args.max_model_len,
                max_num_seqs=args.batch_size * 2,  # Allow more sequences for better GPU utilization
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=args.tensor_parallel_size,
                limit_mm_per_prompt={"audio": 15},  # Allow more audio files
                dtype=torch.bfloat16,  # Use half-precision for better performance
                trust_remote_code=True,  # Enable custom model code
                enforce_eager=True)  # Enforce eager mode for some operations for better reliability
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # INFERENCE PHASE - now the GPU should be consistently busy
    # Record start time
    start_time = time.time()
    samples_generated = 0
    last_save_time = start_time
    batch_save_counter = 0
    
    # Set up batch prefetcher
    batch_iterator = get_batch_prefetcher(inference_batches, args.prefetch_factor)
    
    # Process batches with progress bar
    progress_bar = tqdm(
        range(len(inference_batches)),
        desc="Processing batches",
        total=len(inference_batches),
        ncols=100
    )
    
    try:
        for batch_idx in progress_bar:
            # Get next prefetched batch
            try:
                current_batch = next(batch_iterator)
            except StopIteration:
                break
            
            # Prepare vLLM inputs (all preprocessing is already done)
            vllm_inputs = [
                {"prompt": item["prompt"], "multi_modal_data": item["mm_data"]}
                for item in current_batch
            ]
            
            # Generate responses - this is where GPU utilization should be maximized
            try:
                outputs = llm.generate(vllm_inputs, use_tqdm=False, sampling_params=sampling_params)
                
                # Process outputs
                for i, output in enumerate(outputs):
                    batch_item = current_batch[i]
                    original_item = batch_item["original"]
                    sample_id = batch_item["sample_id"]
                    
                    result_item = {
                        "prompt": original_item["prompt"],
                        "answer": original_item.get("answer", ""),
                        "generated": output.outputs[0].text,
                        "sample_id": sample_id,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    output_data.append(result_item)
                    samples_generated += 1
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                # Log the error and continue with next batch
                continue
            
            # Save intermediate results at specified intervals or if it's been more than 5 minutes
            batch_save_counter += 1
            current_time = time.time()
            time_since_last_save = current_time - last_save_time
            
            if batch_save_counter >= args.save_interval or time_since_last_save > 300:  # 5 minutes
                # Save to temporary file first, then replace the main file
                save_json_data(output_data, args.output_json)
                last_save_time = current_time
                batch_save_counter = 0
            
            # Update progress information
            elapsed_time = current_time - start_time
            items_per_second = samples_generated / elapsed_time if elapsed_time > 0 else 0
            
            progress_bar.set_postfix(
                samples=samples_generated,
                speed=f"{items_per_second:.2f} items/s",
                elapsed=f"{elapsed_time:.1f}s"
            )
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current results...")
    finally:
        # Always save final results, even if interrupted
        save_json_data(output_data, args.output_json)
    
    # Calculate and display stats
    elapsed_time = time.time() - start_time
    items_per_second = samples_generated / elapsed_time if elapsed_time > 0 and samples_generated > 0 else 0
    
    print(f"\nFinished processing. Total outputs: {len(output_data)}")
    print(f"Newly generated outputs: {samples_generated}")
    print(f"Results saved to {args.output_json}")
    print(f"Total inference time: {elapsed_time:.2f} seconds")
    if samples_generated > 0:
        print(f"Processing speed: {items_per_second:.2f} items/second")

if __name__ == "__main__":
    main()