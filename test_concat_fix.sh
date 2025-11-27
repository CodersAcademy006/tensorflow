#!/bin/bash

# Test script to validate concat fix
echo "Testing dynamic concat fix..."

# Test 1: Run the original repro
echo "=== Test 1: Original repro ==="
python3 -u concat_fix_repro.py

echo ""
echo "=== Test 2: Regression test ==="
python3 -u tensorflow/compiler/tf2xla/kernels/concat_dynamic_test.py

echo ""
echo "Test completed."