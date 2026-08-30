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

def patch_norm(cuda_dir):
    p = os.path.join(cuda_dir, "norm.cu")
    if not os.path.exists(p):
        return
    with open(p, "r", encoding="utf-8") as f:
        src = f.read()
    
    modified = False
    if "rms_norm_mul_rope" in src and "pos_dev" not in src:
        src = re.sub(
            r'(\bconst\s+int32_t\s*\*\s*(\w+)\s*=\s*\(const\s+int32_t\s*\*\)\s*(\w+)->data;)',
            r'\1\n    ggml_cuda_pool_alloc<int32_t> \2_dev(ctx.pool());\n    if (\3->buffer && ggml_backend_buffer_is_host(\3->buffer)) {\n        \2_dev.alloc(ggml_nelements(\3));\n        CUDA_CHECK(cudaMemcpyAsync(\2_dev.ptr, \3->data, ggml_nbytes(\3), cudaMemcpyHostToDevice, stream));\n        \2 = \2_dev.ptr;\n    }',
            src
        )
        modified = True

    if modified:
        with open(p, "w", encoding="utf-8") as f:
            f.write(src)
        print(f"[patch_rocm] Patched {p}")

def patch_ggml_cuda(cuda_dir):
    p = os.path.join(cuda_dir, "ggml-cuda.cu")
    if not os.path.exists(p):
        return
    with open(p, "r", encoding="utf-8") as f:
        src = f.read()
    
    modified = False
    if "rms_norm_mul_rope" in src and "pos_dev" not in src:
        src = re.sub(
            r'(\bconst\s+int32_t\s*\*\s*(\w+)\s*=\s*\(const\s+int32_t\s*\*\)\s*(\w+)->data;)',
            r'\1\n    ggml_cuda_pool_alloc<int32_t> \2_dev(ctx.pool());\n    if (\3->buffer && ggml_backend_buffer_is_host(\3->buffer)) {\n        \2_dev.alloc(ggml_nelements(\3));\n        CUDA_CHECK(cudaMemcpyAsync(\2_dev.ptr, \3->data, ggml_nbytes(\3), cudaMemcpyHostToDevice, stream));\n        \2 = \2_dev.ptr;\n    }',
            src
        )
        modified = True
    
    if modified:
        with open(p, "w", encoding="utf-8") as f:
            f.write(src)
        print(f"[patch_rocm] Patched {p}")

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
    patch_norm(cuda_dir)
    patch_ggml_cuda(cuda_dir)

if __name__ == "__main__":
    main()
