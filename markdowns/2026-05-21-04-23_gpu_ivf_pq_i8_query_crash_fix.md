# Fix: GPU_IVF_PQ crashes with `cudaErrorInvalidAddressSpace` when --quantization I8

## Root cause

`GPU_IVF_PQ` uses Product Quantization to compress *stored* vectors server-side, but its
field type is still `FLOAT_VECTOR`. Passing int8 query vectors (from `--quantization I8`)
causes RAFT/cuVS to call `cudaMemcpyAsync` with an int8 pointer into a float32 buffer →
`cudaErrorInvalidAddressSpace`.

The collection `v14_gpu_pc_i8` has `_i8` in the name to indicate 8-bit PQ compression of
the index, **not** that the field type is `INT8_VECTOR`.

Stack trace key frame:
```
void raft::copy<signed char>(signed char*, signed char const*, unsigned long, ...)
cuvs_knowhere::cuvs_knowhere_index<GPU_IVF_PQ, signed char>::impl::search(...)
=> cudaErrorInvalidAddressSpace
```

## Fix

Auto-detect the embedding field's `DataType` from the collection schema during the
index-type auto-detection pass, and override `--quantization` if it doesn't match:

- `FLOAT_VECTOR` → F32 (float32 queries)
- `FLOAT16_VECTOR` → FP16
- `INT8_VECTOR` → I8

File changed: `contact_embedding/scripts/milvus/milvus_eval.py` (~line 614)

## How to run

Since the field is `FLOAT_VECTOR`, drop `--quantization I8` or let auto-detection fix it:

```bash
python contact_embedding/scripts/milvus/milvus_eval.py \
  --host milvus-gpu \
  --checkpoint small_model_v14 \
  --collection v14_gpu_pc_i8 \
  --test-file /mnt/data/mpd_extract_2026_05_05/contacts_0.parquet \
  --id-column id \
  --workers 32 64 128 \
  --sample-size 100 \
  --batch-sizes 100 \
  --no-shuffle
```
(omit `--quantization I8`; auto-detection will set it to F32 and log a warning)

---
Tokens used: ~3,500
