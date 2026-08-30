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
            with open(p, "w", encoding="utf-8") as f:
                f.write(src)
            print(f"[patch_rocm] Patched {p}")

def patch_fusion_disable(cuda_dir):
    # Disable unsafe RMS_NORM + ROPE / RMS_NORM_MUL_ROPE fusion in all ggml-cuda files
    for fname in os.listdir(cuda_dir):
        if not (fname.endswith(".cu") or fname.endswith(".cpp") or fname.endswith(".cuh")):
            continue
        p = os.path.join(cuda_dir, fname)
        with open(p, "r", encoding="utf-8") as f:
            src = f.read()
        
        modified = False
        
        # 1. Disable can_fuse pattern for RMS_NORM + ... + ROPE
        pattern1 = r'(\bggml_cuda_can_fuse\s*\([^;]+GGML_OP_RMS_NORM[^;]+GGML_OP_ROPE[^;]+\))'
        if re.search(pattern1, src):
            src = re.sub(pattern1, r'(false /* disabled for ROCm host pointer safety */)', src)
            modified = True

        # 2. Disable can_fuse pattern for rms_norm_mul_rope
        pattern2 = r'(\bggml_cuda_can_fuse_rms_norm_mul_rope\s*\([^)]+\))'
        if re.search(pattern2, src):
            src = re.sub(pattern2, r'(false)', src)
            modified = True

        # 3. Disable any "if (fused_rms_norm_mul_rope)" or similar check
        pattern3 = r'(\bif\s*\(\s*ggml_cuda_can_fuse\s*\(\s*cgraph\s*,\s*i\s*,\s*\{\s*GGML_OP_RMS_NORM\s*,\s*GGML_OP_MUL\s*,\s*GGML_OP_ROPE\s*\}\s*,\s*\{\s*\}\s*\)\s*\))'
        if re.search(pattern3, src):
            src = re.sub(pattern3, r'if (false)', src)
            modified = True

        if modified:
            with open(p, "w", encoding="utf-8") as f:
                f.write(src)
            print(f"[patch_rocm] Disabled RMS_NORM+ROPE fusion in {fname}")

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
    patch_fusion_disable(cuda_dir)

if __name__ == "__main__":
    main()
