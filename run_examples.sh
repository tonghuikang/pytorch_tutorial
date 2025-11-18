#!/bin/bash

# Exit immediately if any command fails
set -e

# Run all Python files in the examples directory
for file in examples/*.py; do
    if [ -f "$file" ]; then
        echo "Running $file..."
        uv run "$file"
        echo "Completed $file"
        echo "---"
    fi
done

echo "All examples completed!"
