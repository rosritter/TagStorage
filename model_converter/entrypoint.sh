#!/bin/bash
set -e

echo "Starting model conversion..."
python exportbert.py

echo "Running model tests..."
python test.py

# If both scripts complete successfully, keep the container running
echo "Model conversion and testing completed successfully."
exit 0