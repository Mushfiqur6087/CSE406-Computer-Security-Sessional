#!/usr/bin/env python3
"""
Dataset Merger Script with Normalization

This script reads all dataset.json files from subfolders, applies sklearn normalization
to each dataset, and merges them into a single normalized dataset.json file.
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from sklearn.preprocessing import StandardScaler

DATA_ROOT = "./individual-data"


def load_json_file(filepath: str) -> Any:
    """Load and parse a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error reading {filepath}: {e}")


def normalize_trace_data(trace_data: List[float]) -> List[float]:
    """
    Normalize a single trace_data array using StandardScaler.
    
    Args:
        trace_data: List of numerical values
        
    Returns:
        Normalized trace_data as a list
    """
    if not trace_data:
        return trace_data
    
    # Convert to numpy array and reshape for sklearn
    data_array = np.array(trace_data).reshape(-1, 1)
    
    # Apply StandardScaler
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_array)
    
    # Convert back to list and flatten
    return normalized_data.flatten().tolist()


def process_dataset(data: List[Dict], source_folder: str) -> List[Dict]:
    """
    Process a dataset by normalizing trace data and standardizing format.
    
    Args:
        data: Raw dataset
        source_folder: Source folder name
        
    Returns:
        Processed dataset with normalized trace_data
    """
    processed_data = []
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"   ⚠️  Skipping item {i} in {source_folder}: not a JSON object")
            continue
        
        # Check required fields
        if 'website' not in item:
            print(f"   ⚠️  Skipping item {i} in {source_folder}: missing 'website' field")
            continue
        
        # Get trace data (could be 'trace_data' or 'traces')
        trace_field = None
        if 'trace_data' in item:
            trace_field = 'trace_data'
        elif 'traces' in item:
            trace_field = 'traces'
        else:
            print(f"   ⚠️  Skipping item {i} in {source_folder}: missing trace data field")
            continue
        
        raw_trace_data = item[trace_field]
        
        if not isinstance(raw_trace_data, list):
            print(f"   ⚠️  Skipping item {i} in {source_folder}: trace data is not an array")
            continue
        
        # Check if all trace data elements are numbers
        try:
            numeric_trace_data = [float(x) for x in raw_trace_data]
        except (ValueError, TypeError):
            print(f"   ⚠️  Skipping item {i} in {source_folder}: trace data contains non-numeric values")
            continue
        
        # Normalize the trace data
        try:
            normalized_trace_data = normalize_trace_data(numeric_trace_data)
        except Exception as e:
            print(f"   ⚠️  Skipping item {i} in {source_folder}: normalization failed - {e}")
            continue
        
        # Create processed item with standardized format
        processed_item = {
            "website": item["website"],
            "trace_data": normalized_trace_data,
            "source": source_folder
        }
        
        processed_data.append(processed_item)
    
    return processed_data


def create_website_mapping(data_folder: str) -> Dict[str, int]:
    """
    Create a mapping from website URLs to indices by scanning all files without loading full datasets.
    
    Args:
        data_folder: Path to the data folder containing subfolders with datasets
        
    Returns:
        Dictionary mapping website URL to index
    """
    unique_websites = set()
    
    # Find all subfolders
    subfolders = [item for item in os.listdir(data_folder) 
                  if os.path.isdir(os.path.join(data_folder, item))]
    
    # Scan all datasets to collect unique websites
    for subfolder in subfolders:
        dataset_path = os.path.join(data_folder, subfolder, "dataset.json")
        
        if not os.path.exists(dataset_path):
            continue
            
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                raw_dataset = json.load(f)
                
            if not isinstance(raw_dataset, list):
                continue
                
            # Extract websites without loading full trace data
            for item in raw_dataset:
                if isinstance(item, dict) and 'website' in item:
                    unique_websites.add(item['website'])
                    
        except Exception:
            continue
    
    # Sort websites for consistent ordering
    sorted_websites = sorted(unique_websites)
    
    # Create mapping
    website_to_index = {website: idx for idx, website in enumerate(sorted_websites)}
    
    return website_to_index


def merge_all_datasets(data_folder: str = DATA_ROOT) -> None:
    """
    Main function to merge all datasets with normalization using streaming approach.
    
    Args:
        data_folder: Path to the data folder containing subfolders with datasets
    """
    print("🚀 Dataset Merger with Normalization Starting...")
    print("-" * 60)
    
    if not os.path.exists(data_folder):
        print(f"\n❌ Error: Data folder '{data_folder}' not found\n")
        sys.exit(1)
    
    # Find all subfolders
    subfolders = [item for item in os.listdir(data_folder) 
                  if os.path.isdir(os.path.join(data_folder, item))]
    
    if not subfolders:
        print(f"\n❌ Error: No subfolders found in '{data_folder}'\n")
        sys.exit(1)
    
    print(f"📁 Found {len(subfolders)} source folders")
    print()
    
    # Create website mapping by scanning files without loading full datasets
    print("🌐 Creating website mapping...")
    website_to_index = create_website_mapping(data_folder)
    
    print(f"Found {len(website_to_index)} unique websites:")
    for website, idx in sorted(website_to_index.items(), key=lambda x: x[1]):
        print(f"  [{idx}] {website}")
    print()
    
    # Statistics tracking
    total_original_items = 0
    total_processed_items = 0
    source_counts = defaultdict(int)
    website_counts = defaultdict(int)
    
    # Open output file for streaming write
    output_path = "merged_dataset.json"
    print(f"💾 Streaming merged dataset to {output_path}...")
    print("🔄 Processing datasets...")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as output_file:
            # Write JSON array opening
            output_file.write('[\n')
            
            first_item = True
            
            # Process each dataset one by one
            for subfolder in sorted(subfolders):
                subfolder_path = os.path.join(data_folder, subfolder)
                dataset_path = os.path.join(subfolder_path, "dataset.json")
                
                print(f"📂 Processing {subfolder}...")
                
                # Check if dataset.json exists
                if not os.path.exists(dataset_path):
                    print(f"   ❌ dataset.json not found - skipping")
                    continue
                
                try:
                    # Load dataset
                    raw_dataset = load_json_file(dataset_path)
                    
                    if not isinstance(raw_dataset, list):
                        print(f"   ❌ Dataset is not an array - skipping")
                        continue
                    
                    original_count = len(raw_dataset)
                    total_original_items += original_count
                    
                    # Process and normalize dataset
                    processed_dataset = process_dataset(raw_dataset, subfolder)
                    processed_count = len(processed_dataset)
                    total_processed_items += processed_count
                    
                    # Stream write processed items to output file
                    for item in processed_dataset:
                        # Add website index to merged item
                        merged_item = {
                            "website": item["website"],
                            "website_index": website_to_index[item["website"]],
                            "trace_data": item["trace_data"],
                            "source": item["source"]
                        }
                        
                        # Update statistics
                        source_counts[item['source']] += 1
                        website_counts[item['website']] += 1
                        
                        # Write to output file
                        if not first_item:
                            output_file.write(',\n')
                        else:
                            first_item = False
                        
                        json.dump(merged_item, output_file, ensure_ascii=False)
                    
                    print(f"   ✅ Processed {processed_count}/{original_count} items")
                    
                    if processed_count < original_count:
                        print(f"   ⚠️  {original_count - processed_count} items were skipped due to errors")
                    
                    # Clear processed data from memory immediately
                    del raw_dataset
                    del processed_dataset
                    
                except Exception as e:
                    print(f"   ❌ Error processing {subfolder}: {e}")
                    continue
                
                print()
            
            # Write JSON array closing
            output_file.write('\n]\n')
        
        print(f"✅ Successfully created {output_path}")
        print(f"📈 Final dataset contains {total_processed_items} normalized items")
        print(f"🌐 Covering {len(website_to_index)} unique websites")
        
        # Print summary statistics
        print(f"\n📊 Items per source:")
        for source, count in sorted(source_counts.items()):
            print(f"  {source}: {count} items")
        
        print(f"\n🌐 Items per website:")
        for website, count in sorted(website_counts.items(), key=lambda x: website_to_index[x[0]]):
            website_idx = website_to_index[website]
            print(f"  [{website_idx}] {website}: {count} items")
        
        print(f"\n📊 Processing Summary:")
        print(f"  Original items: {total_original_items}")
        print(f"  Successfully processed: {total_processed_items}")
        if total_original_items > 0:
            print(f"  Success rate: {(total_processed_items/total_original_items*100):.1f}%")
            
    except Exception as e:
        print(f"\n❌ Error writing output file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    merge_all_datasets()
