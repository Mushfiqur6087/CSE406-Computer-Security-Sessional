#!/usr/bin/env python3
"""
Dataset Validation Script

This script validates the merged dataset.json file to ensure:
1. It's valid JSON
2. It has the correct data format
3. All required fields are present
4. Data types are correct
5. Provides statistics and integrity checks
"""

import json
import os
import sys
from typing import Dict, List, Any, Set
from collections import defaultdict, Counter


def validate_json_file(filepath: str) -> Any:
    """
    Validate that the file is valid JSON and load it.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded JSON data
        
    Raises:
        Exception if validation fails
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Valid JSON file: {filepath}")
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")
    except Exception as e:
        raise Exception(f"Error reading {filepath}: {e}")


def validate_dataset_structure(data: Any) -> List[Dict]:
    """
    Validate that the dataset has the correct top-level structure.
    
    Args:
        data: Loaded JSON data
        
    Returns:
        Dataset as list of dictionaries
        
    Raises:
        Exception if structure is invalid
    """
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON array")
    
    if len(data) == 0:
        raise ValueError("Dataset is empty")
    
    print(f"✅ Dataset is a valid array with {len(data)} items")
    return data


def validate_item_format(item: Any, index: int) -> Dict[str, Any]:
    """
    Validate individual item format.
    
    Args:
        item: Individual dataset item
        index: Item index for error reporting
        
    Returns:
        Validated item
        
    Raises:
        Exception if item format is invalid
    """
    if not isinstance(item, dict):
        raise ValueError(f"Item {index}: Must be a JSON object, got {type(item)}")
    
    # Check required fields
    required_fields = ["website", "website_index", "trace_data", "source"]
    missing_fields = [field for field in required_fields if field not in item]
    
    if missing_fields:
        raise ValueError(f"Item {index}: Missing required fields: {missing_fields}")
    
    # Validate field types
    if not isinstance(item["website"], str):
        raise ValueError(f"Item {index}: 'website' must be a string, got {type(item['website'])}")
    
    if not isinstance(item["website_index"], int):
        raise ValueError(f"Item {index}: 'website_index' must be an integer, got {type(item['website_index'])}")
    
    if not isinstance(item["trace_data"], list):
        raise ValueError(f"Item {index}: 'trace_data' must be an array, got {type(item['trace_data'])}")
    
    if not isinstance(item["source"], str):
        raise ValueError(f"Item {index}: 'source' must be a string, got {type(item['source'])}")
    
    # Validate trace_data contents
    if len(item["trace_data"]) == 0:
        raise ValueError(f"Item {index}: 'trace_data' cannot be empty")
    
    for i, value in enumerate(item["trace_data"]):
        if not isinstance(value, (int, float)):
            raise ValueError(f"Item {index}: 'trace_data[{i}]' must be a number, got {type(value)}")
    
    # Validate website format (basic URL check)
    website = item["website"]
    if not (website.startswith("http://") or website.startswith("https://")):
        raise ValueError(f"Item {index}: 'website' should be a valid URL, got '{website}'")
    
    return item


def validate_dataset_integrity(dataset: List[Dict]) -> Dict[str, Any]:
    """
    Validate dataset integrity and consistency.
    
    Args:
        dataset: List of validated items
        
    Returns:
        Dictionary with integrity statistics
    """
    print("\n🔍 Checking dataset integrity...")
    
    # Collect statistics
    websites = set()
    website_indices = set()
    sources = set()
    trace_lengths = []
    website_to_indices = defaultdict(set)
    index_to_websites = defaultdict(set)
    
    for i, item in enumerate(dataset):
        website = item["website"]
        website_index = item["website_index"]
        source = item["source"]
        trace_data = item["trace_data"]
        
        websites.add(website)
        website_indices.add(website_index)
        sources.add(source)
        trace_lengths.append(len(trace_data))
        
        website_to_indices[website].add(website_index)
        index_to_websites[website_index].add(website)
    
    # Check for consistency issues
    issues = []
    
    # Check website-to-index mapping consistency
    for website, indices in website_to_indices.items():
        if len(indices) > 1:
            issues.append(f"Website '{website}' maps to multiple indices: {sorted(indices)}")
    
    # Check index-to-website mapping consistency
    for index, websites_for_index in index_to_websites.items():
        if len(websites_for_index) > 1:
            issues.append(f"Index {index} maps to multiple websites: {sorted(websites_for_index)}")
    
    # Check for gaps in website indices
    expected_indices = set(range(len(websites)))
    if website_indices != expected_indices:
        missing = expected_indices - website_indices
        extra = website_indices - expected_indices
        if missing:
            issues.append(f"Missing website indices: {sorted(missing)}")
        if extra:
            issues.append(f"Extra website indices: {sorted(extra)}")
    
    if issues:
        print("⚠️  Integrity issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Dataset integrity checks passed")
    
    # Calculate statistics
    stats = {
        "total_items": len(dataset),
        "unique_websites": len(websites),
        "unique_sources": len(sources),
        "website_indices_range": f"{min(website_indices)}-{max(website_indices)}",
        "trace_length_stats": {
            "min": min(trace_lengths),
            "max": max(trace_lengths),
            "avg": sum(trace_lengths) / len(trace_lengths)
        },
        "issues": issues
    }
    
    return stats


def print_dataset_summary(dataset: List[Dict], stats: Dict[str, Any]) -> None:
    """
    Print row counts per website and trace data information.
    
    Args:
        dataset: Validated dataset
        stats: Integrity statistics
    """
    print("\n📊 Dataset Row Count Summary:")
    print(f"  Total rows: {stats['total_items']:,}")
    
    # Count rows per website
    website_counts = defaultdict(int)
    
    for item in dataset:
        website_counts[item['website']] += 1
    
    print(f"\n🌐 Rows per website:")
    for website, count in sorted(website_counts.items()):
        print(f"  {website}: {count:,} rows")
    
    print(f"\n� Trace Data Information:")
    trace_stats = stats['trace_length_stats']
    print(f"  Each row contains: {trace_stats['min']} trace data points")
    if trace_stats['min'] != trace_stats['max']:
        print(f"  Trace data length varies from {trace_stats['min']} to {trace_stats['max']} points")
    else:
        print(f"  All rows have consistent trace data length: {trace_stats['min']} points")


def validate_dataset(filepath: str = "merged_dataset.json") -> None:
    """
    Main validation function.
    
    Args:
        filepath: Path to the dataset file to validate
    """
    print("🔍 Dataset Validation Starting...")
    print("-" * 60)
    
    try:
        # Step 1: Validate JSON
        print("1. Validating JSON format...")
        data = validate_json_file(filepath)
        
        # Step 2: Validate structure
        print("\n2. Validating dataset structure...")
        dataset = validate_dataset_structure(data)
        
        # Step 3: Validate individual items
        print(f"\n3. Validating individual items...")
        validated_items = []
        validation_errors = []
        
        for i, item in enumerate(dataset):
            try:
                validated_item = validate_item_format(item, i)
                validated_items.append(validated_item)
                
                # Progress indicator for large datasets
                if (i + 1) % 10000 == 0:
                    print(f"   Validated {i + 1:,} items...")
                    
            except Exception as e:
                validation_errors.append(f"Item {i}: {e}")
                if len(validation_errors) >= 10:  # Limit error reporting
                    validation_errors.append(f"... and more errors (stopping at 10)")
                    break
        
        if validation_errors:
            print(f"\n❌ Validation errors found:")
            for error in validation_errors:
                print(f"  - {error}")
            sys.exit(1)
        
        print(f"✅ All {len(validated_items):,} items validated successfully")
        
        # Step 4: Validate integrity
        print(f"\n4. Validating dataset integrity...")
        stats = validate_dataset_integrity(validated_items)
        
        # Step 5: Print summary
        print_dataset_summary(validated_items, stats)
        
        # Final result
        if stats['issues']:
            print(f"\n⚠️  Dataset validation completed with {len(stats['issues'])} integrity issues")
        else:
            print(f"\n✅ Dataset validation completed successfully!")
            print(f"   File: {filepath}")
            print(f"   Size: {os.path.getsize(filepath) / (1024*1024):.1f} MB")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Allow custom file path as command line argument
    filepath = sys.argv[1] if len(sys.argv) > 1 else "merged_dataset.json"
    validate_dataset(filepath)
