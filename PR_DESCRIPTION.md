# PR: Fix XLA shape inference for concat with dynamic partition outputs

## Root Cause

**Matrix size-incompatible: In[0]: [20,64], In[1]: [2,32]** error occurs because the tf2xla `concat_op.cc` kernel doesn't preserve dynamic dimensions from `DynamicPartition` outputs.

### The Problem Chain
1. **DynamicPartition** correctly outputs tensors with dynamic first dimension using `SetDimensionSize(reshape, length, 0)`
2. **Concat** calls `xla::ConcatInDim()` directly, losing the dynamic dimension info 
3. **TopK** gets a static shape `[2, 64]` instead of `[dynamic, 64]`
4. **MatMul** fails with wrong inferred shapes `[20,64]` vs `[2,32]`

## Fix Details

### Files Modified
- `tensorflow/compiler/tf2xla/kernels/concat_op.cc` - Lines ~97-117
- `tensorflow/compiler/tf2xla/kernels/concat_dynamic_test.py` - New regression test

### Conceptual Change
Modified `ConcatBaseOp::Compile()` to:
1. Check if any input has `is_dynamic_dimension(axis)` 
2. Sum dynamic sizes using `GetDimensionSize()` for dynamic inputs + constants for static inputs
3. Call `SetDimensionSize(result, dynamic_size_sum, axis)` on concat result if any input was dynamic

### Code Change
```cpp
// Before (line 97):
ctx->SetOutput(0, xla::ConcatInDim(ctx->builder(), input_data, axis));

// After (lines 97-122):
xla::XlaOp result = xla::ConcatInDim(ctx->builder(), input_data, axis);

// Check if any input has dynamic dimension on concat axis and preserve it
bool any_input_dynamic = false;
xla::XlaOp dynamic_size_sum;
bool size_sum_initialized = false;

for (int i = 0; i < N; ++i) {
  auto input_shape_or = ctx->InputXlaShape(i);
  OP_REQUIRES_OK(ctx, input_shape_or.status());
  const xla::Shape& input_shape = *input_shape_or;
  
  xla::XlaOp input_size;
  if (input_shape.is_dynamic_dimension(axis)) {
    any_input_dynamic = true;
    input_size = xla::GetDimensionSize(input_data[i], axis);
  } else {
    input_size = xla::ConstantR0<int32_t>(ctx->builder(), 
                                          input_shape.dimensions(axis));
  }
  
  if (!size_sum_initialized) {
    dynamic_size_sum = input_size;
    size_sum_initialized = true;
  } else {
    dynamic_size_sum = xla::Add(dynamic_size_sum, input_size);
  }
}

// Only set dynamic size if any input was dynamic
if (any_input_dynamic && N > 0) {
  result = xla::SetDimensionSize(result, dynamic_size_sum, axis);
}

ctx->SetOutput(0, result);
```

## Tests Added

### Python Regression Test
`tensorflow/compiler/tf2xla/kernels/concat_dynamic_test.py`:
- `test_dynamic_partition_concat_topk_matmul_xla()` - Full pipeline test
- `test_concat_preserves_dynamic_dimensions()` - Direct concat test  

Both tests use `@tf.function(jit_compile=True)` to verify XLA compilation succeeds.

## Risk & Mitigation

**Performance**: Only computes dynamic size sum when inputs have dynamic dimensions. Static-only concat paths unchanged.

**Correctness**: Follows same pattern as `unique_op.cc` and `dynamic_partition_op.cc` for dynamic dimension handling.

## Build & Test Commands

```bash
# Build tf2xla kernels
bazel build //tensorflow/compiler/tf2xla/kernels:concat_op

# Run regression test
python3 tensorflow/compiler/tf2xla/kernels/concat_dynamic_test.py

# Run original repro (should pass)
python3 concat_fix_repro.py
```

## Before/After HLO

**Before**: Concat produces static shape `[2, 64]` 
**After**: Concat produces dynamic shape `[?, 64]` with proper size tracking

## Reviewers

- XLA Team (@tensorflow/xla)
- tf2xla owners
- Issue assignee: @Venkat6871

## References

- Issue #105143: XLA/GPU Shape inference inconsistency with dynamic operations
- Similar fixes in `unique_op.cc` and `dynamic_partition_op.cc` 
- XLA dynamic shape documentation