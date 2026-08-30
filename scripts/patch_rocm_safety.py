#!/usr/bin/env python3
import os
import sys
import re

def patch_backend(llama_dir):
    # Patch ggml-backend.cpp to place input tensors (pos, tokens, embd) on GPU VRAM (backend 0)
    # when op_offload is enabled. Other inputs (masks, out_ids, buckets) stay on host.
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
        repl = '''bool offload_to_gpu = false;
        if (sched->n_backends > 1 && sched->op_offload) {
            if (strcmp(tensor->name, "inp_pos") == 0 ||
                strcmp(tensor->name, "inp_tokens") == 0 ||
                strcmp(tensor->name, "inp_embd") == 0 ||
                strcmp(tensor->name, "pos") == 0 ||
                strcmp(tensor->name, "tokens") == 0 ||
                strcmp(tensor->name, "embd") == 0) {
                offload_to_gpu = true;
            }
        }
        cur_backend_id = offload_to_gpu ? 0 : (sched->n_backends - 1);'''
        if target in src:
            src = src.replace(target, repl, 1)
            with open(p, "w", encoding="utf-8") as f:
                f.write(src)
            print(f"[patch_rocm] Patched input tensor placement in {p}")

def patch_llama_graph(llama_dir):
    # Ensure inp_pos tensor has name set so backend scheduler can identify it
    p = os.path.join(llama_dir, "src", "llama-graph.cpp")
    if not os.path.exists(p):
        return
    with open(p, "r", encoding="utf-8") as f:
        src = f.read()
    if 'cb(cur, "inp_pos", -1);' not in src:
        target = 'cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());\n    ggml_set_input(cur);'
        repl = 'cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());\n    cb(cur, "inp_pos", -1);\n    ggml_set_name(cur, "inp_pos");\n    ggml_set_input(cur);'
        if target in src:
            src = src.replace(target, repl, 1)
            with open(p, "w", encoding="utf-8") as f:
                f.write(src)
            print(f"[patch_rocm] Patched {p}")

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
            src = src.replace(target, repl, 1)
            with open(p, "w", encoding="utf-8") as f:
                f.write(src)
            print(f"[patch_rocm] Patched {p}")

def patch_fusions(cuda_dir):
    # In ggml-cuda.cu / norm.cu, disable fusing set_rows or rope when input indices are on host
    candidates = [
        os.path.join(cuda_dir, "ggml-cuda.cu"),
        os.path.join(cuda_dir, "norm.cu"),
    ]
    for p in candidates:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            src = f.read()
        
        modified = False
        
        # In should_fuse_rope_set_rows, check host buffer
        target1 = "static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,\n                                                const ggml_tensor * view,\n                                                const ggml_tensor * set_rows) {\n\n    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {\n        return false;\n    }"
        repl1 = target1 + "\n\n    if (set_rows->src[1]->buffer && ggml_backend_buffer_is_host(set_rows->src[1]->buffer)) {\n        return false;\n    }"
        if target1 in src and "ggml_backend_buffer_is_host(set_rows->src[1]->buffer)" not in src:
            src = src.replace(target1, repl1, 1)
            modified = True

        # In any other should_fuse with set_rows
        # If any function has should_fuse...set_rows, disallow host buffer
        if modified:
            with open(p, "w", encoding="utf-8") as f:
                f.write(src)
            print(f"[patch_rocm] Patched fusions in {p}")

def main():
    if len(sys.argv) < 2:
        print("Usage: patch_rocm_safety.py <llama_cpp_source_dir>")
        sys.exit(1)
    llama_dir = sys.argv[1]
    patch_backend(llama_dir)
    patch_llama_graph(llama_dir)
    cuda_dir = os.path.join(llama_dir, "ggml", "src", "ggml-cuda")
    if os.path.isdir(cuda_dir):
        patch_getrows(cuda_dir)
        patch_set_rows(cuda_dir)
        patch_rope(cuda_dir)
        patch_fusions(cuda_dir)

if __name__ == "__main__":
    main()
