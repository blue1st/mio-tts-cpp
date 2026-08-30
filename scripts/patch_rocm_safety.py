#!/usr/bin/env python3
import os
import sys
import re

def patch_getrows(cuda_dir):
    p = os.path.join(cuda_dir, "getrows.cu")
    if not os.path.exists(p):
        return
    with open(p, "r", encoding="utf-8") as f:
        src = f.read()
    if "src1_dev" not in src:
        target = "get_rows_cuda(src0->data, src0->type, (const int32_t *) src1->data, dst->data, dst->type,"
        repl = ("const int32_t * src1_d = (const int32_t *) src1->data;\n"
                "    ggml_cuda_pool_alloc<int32_t> src1_dev(ctx.pool());\n"
                "    if (src1->buffer && ggml_backend_buffer_is_host(src1->buffer)) {\n"
                "        src1_dev.alloc(ggml_nelements(src1));\n"
                "        CUDA_CHECK(cudaMemcpyAsync(src1_dev.ptr, src1->data, ggml_nbytes(src1), cudaMemcpyHostToDevice, stream));\n"
                "        src1_d = src1_dev.ptr;\n"
                "    }\n"
                "    get_rows_cuda(src0->data, src0->type, src1_d, dst->data, dst->type,")
        if target in src:
            src = src.replace(target, repl)
            with open(p, "w", encoding="utf-8") as f:
                f.write(src)
            print(f"[patch_rocm] Patched {p}")

def patch_rope(cuda_dir):
    p = os.path.join(cuda_dir, "rope.cu")
    if not os.path.exists(p):
        return
    with open(p, "r", encoding="utf-8") as f:
        src = f.read()
    
    modified = False
    if "pos_dev" not in src:
        target = "const int32_t * pos = (const int32_t *) src1_d;"
        repl = ("const int32_t * pos = (const int32_t *) src1_d;\n"
                "    ggml_cuda_pool_alloc<int32_t> pos_dev(ctx.pool());\n"
                "    if (src1->buffer && ggml_backend_buffer_is_host(src1->buffer)) {\n"
                "        pos_dev.alloc(ggml_nelements(src1));\n"
                "        CUDA_CHECK(cudaMemcpyAsync(pos_dev.ptr, src1->data, ggml_nbytes(src1), cudaMemcpyHostToDevice, stream));\n"
                "        pos = pos_dev.ptr;\n"
                "    }")
        if target in src:
            src = src.replace(target, repl)
            modified = True

    if "row_indices_dev" not in src:
        target_rows = "row_indices     = (const int64_t *) set_rows->src[1]->data;"
        repl_rows = ("const ggml_tensor * sr1 = set_rows->src[1];\n"
                     "        row_indices     = (const int64_t *) sr1->data;\n"
                     "        if (sr1->buffer && ggml_backend_buffer_is_host(sr1->buffer)) {\n"
                     "            row_indices_dev.alloc(ggml_nelements(sr1));\n"
                     "            CUDA_CHECK(cudaMemcpyAsync(row_indices_dev.ptr, sr1->data, ggml_nbytes(sr1), cudaMemcpyHostToDevice, stream));\n"
                     "            row_indices = row_indices_dev.ptr;\n"
                     "        }")
        if target_rows in src:
            src = src.replace("const int64_t * row_indices     = nullptr;",
                              "const int64_t * row_indices     = nullptr;\n    ggml_cuda_pool_alloc<int64_t> row_indices_dev(ctx.pool());")
            src = src.replace(target_rows, repl_rows)
            modified = True

    if modified:
        with open(p, "w", encoding="utf-8") as f:
            f.write(src)
        print(f"[patch_rocm] Patched {p}")

def patch_set_rows(cuda_dir):
    p = os.path.join(cuda_dir, "set-rows.cu")
    if not os.path.exists(p):
        return
    with open(p, "r", encoding="utf-8") as f:
        src = f.read()
    if "src1_dev" not in src:
        target = "cudaStream_t stream = ctx.stream();"
        repl = ("cudaStream_t stream = ctx.stream();\n\n"
                "    ggml_cuda_pool_alloc<idx_t> src1_dev(ctx.pool());\n"
                "    if (src1->buffer && ggml_backend_buffer_is_host(src1->buffer)) {\n"
                "        src1_dev.alloc(ggml_nelements(src1));\n"
                "        CUDA_CHECK(cudaMemcpyAsync(src1_dev.ptr, src1->data, ggml_nbytes(src1), cudaMemcpyHostToDevice, stream));\n"
                "        src1_d = src1_dev.ptr;\n"
                "    }")
        if target in src:
            src = src.replace(target, repl, 1)
            with open(p, "w", encoding="utf-8") as f:
                f.write(src)
            print(f"[patch_rocm] Patched {p}")

def patch_disable_fusions(cuda_dir):
    # Safely disable unsafe RMS_NORM + ROPE fusions in ggml-cuda files
    for fname in os.listdir(cuda_dir):
        if not (fname.endswith(".cu") or fname.endswith(".cpp") or fname.endswith(".cuh")):
            continue
        p = os.path.join(cuda_dir, fname)
        with open(p, "r", encoding="utf-8") as f:
            src = f.read()
        
        modified = False
        
        # Replace { GGML_OP_RMS_NORM, ..., GGML_OP_ROPE } with { GGML_OP_NONE }
        patterns = [
            r'\{\s*GGML_OP_RMS_NORM\s*,\s*GGML_OP_MUL\s*,\s*GGML_OP_ROPE\s*\}',
            r'\{\s*GGML_OP_RMS_NORM\s*,\s*GGML_OP_ROPE\s*\}',
            r'\{\s*GGML_OP_RMS_NORM\s*,\s*GGML_OP_MUL\s*,\s*GGML_OP_ROPE\s*,\s*GGML_OP_VIEW\s*,\s*GGML_OP_SET_ROWS\s*\}',
            r'\{\s*GGML_OP_RMS_NORM\s*,\s*GGML_OP_ROPE\s*,\s*GGML_OP_VIEW\s*,\s*GGML_OP_SET_ROWS\s*\}'
        ]
        for pat in patterns:
            if re.search(pat, src):
                src = re.sub(pat, '{ GGML_OP_NONE }', src)
                modified = True

        if modified:
            with open(p, "w", encoding="utf-8") as f:
                f.write(src)
            print(f"[patch_rocm] Safely disabled RMS_NORM+ROPE fusion in {fname}")

def main():
    if len(sys.argv) < 2:
        print("Usage: patch_rocm_safety.py <llama_cpp_source_dir>")
        sys.exit(1)
    llama_dir = sys.argv[1]
    cuda_dir = os.path.join(llama_dir, "ggml", "src", "ggml-cuda")
    if not os.path.isdir(cuda_dir):
        print(f"Directory not found: {cuda_dir}")
        sys.exit(0)
    patch_getrows(cuda_dir)
    patch_rope(cuda_dir)
    patch_set_rows(cuda_dir)
    patch_disable_fusions(cuda_dir)

if __name__ == "__main__":
    main()
