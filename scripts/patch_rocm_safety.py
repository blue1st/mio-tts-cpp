#!/usr/bin/env python3
import os
import sys
import re

def patch_backend(llama_dir):
    # Patch ggml-backend.cpp to place input tensors (pos, tokens, KQ_mask, etc.) on GPU VRAM (backend 0)
    # when op_offload is enabled, except out_ids which expects host buffer.
    candidates = [
        os.path.join(llama_dir, "ggml", "src", "ggml-backend.cpp"),
        os.path.join(llama_dir, "src", "ggml-backend.cpp"),
    ]
    for p in candidates:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            src = f.read()
        target = 'cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)'
        repl = 'cur_backend_id = (sched->n_backends > 1 && sched->op_offload && strcmp(tensor->name, "out_ids") != 0) ? 0 : (sched->n_backends - 1);'
        if target in src:
            src = src.replace(target, repl, 1)
            with open(p, "w", encoding="utf-8") as f:
                f.write(src)
            print(f"[patch_rocm] Patched input tensor placement in {p}")

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
            src = src.replace(target, repl, 1)
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
            src = src.replace(target, repl, 1)
            modified = True

    if "row_indices_dev" not in src:
        target_pat = r'void\s*\*\s*dst_d\s*=\s*dst->data;\s*\n\s*const\s+int64_t\s*\*\s*row_indices\s*=\s*nullptr;\s*\n\s*ggml_type\s+dst_type\s*=\s*dst->type;\s*\n\s*int\s+set_rows_stride\s*=\s*0;\s*\n\s*if\s*\(set_rows\s*!=\s*nullptr\)\s*\{\s*\n\s*GGML_ASSERT\(forward\);\s*\n\s*dst_d\s*=\s*set_rows->data;\s*\n\s*row_indices\s*=\s*\(const\s+int64_t\s*\*\)\s*set_rows->src\[1\]->data;\s*\n\s*dst_type\s*=\s*set_rows->type;\s*\n\s*set_rows_stride\s*=\s*set_rows->nb\[1\]\s*/\s*ggml_type_size\(set_rows->type\);\s*\n\s*\}\s*\n\s*cudaStream_t\s+stream\s*=\s*ctx\.stream\(\);'
        
        repl_block = """cudaStream_t stream = ctx.stream();
    void *          dst_d           = dst->data;
    const int64_t * row_indices     = nullptr;
    ggml_cuda_pool_alloc<int64_t> row_indices_dev(ctx.pool());
    ggml_type       dst_type        = dst->type;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d           = set_rows->data;
        const ggml_tensor * sr1 = set_rows->src[1];
        row_indices     = (const int64_t *) sr1->data;
        if (sr1->buffer && ggml_backend_buffer_is_host(sr1->buffer)) {
            row_indices_dev.alloc(ggml_nelements(sr1));
            CUDA_CHECK(cudaMemcpyAsync(row_indices_dev.ptr, sr1->data, ggml_nbytes(sr1), cudaMemcpyHostToDevice, stream));
            row_indices = row_indices_dev.ptr;
        }
        dst_type        = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }"""
        if re.search(target_pat, src):
            src = re.sub(target_pat, repl_block, src, count=1)
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
        target_pat = r'(static\s+void\s+set_rows_cuda\s*\([^)]+\)\s*\{\s*\n\s*const\s+src_t\s*\*\s*src0_d\s*=\s*\([^)]+\)src0->data;\s*\n\s*const\s+idx_t\s*\*\s*src1_d\s*=\s*\([^)]+\)src1->data;\s*\n\s*GGML_TENSOR_BINARY_OP_LOCALS\s*\n\s*cudaStream_t\s+stream\s*=\s*ctx\.stream\(\);)'
        repl = r'''\1

    ggml_cuda_pool_alloc<idx_t> src1_dev(ctx.pool());
    if (src1->buffer && ggml_backend_buffer_is_host(src1->buffer)) {
        src1_dev.alloc(ggml_nelements(src1));
        CUDA_CHECK(cudaMemcpyAsync(src1_dev.ptr, src1->data, ggml_nbytes(src1), cudaMemcpyHostToDevice, stream));
        src1_d = src1_dev.ptr;
    }'''
        if re.search(target_pat, src):
            src = re.sub(target_pat, repl, src, count=1)
            with open(p, "w", encoding="utf-8") as f:
                f.write(src)
            print(f"[patch_rocm] Patched {p}")

def main():
    if len(sys.argv) < 2:
        print("Usage: patch_rocm_safety.py <llama_cpp_source_dir>")
        sys.exit(1)
    llama_dir = sys.argv[1]
    patch_backend(llama_dir)
    cuda_dir = os.path.join(llama_dir, "ggml", "src", "ggml-cuda")
    if os.path.isdir(cuda_dir):
        patch_getrows(cuda_dir)
        patch_rope(cuda_dir)
        patch_set_rows(cuda_dir)

if __name__ == "__main__":
    main()
