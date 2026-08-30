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

def patch_all_cuda_files(cuda_dir):
    # Scan and patch row_indices and pos in all cuda backend files (norm.cu, rope.cu, etc.)
    for fname in os.listdir(cuda_dir):
        if not (fname.endswith(".cu") or fname.endswith(".cpp")):
            continue
        p = os.path.join(cuda_dir, fname)
        with open(p, "r", encoding="utf-8") as f:
            src = f.read()
        
        modified = False
        
        # 1. Patch set_rows->src[1]->data or sr1->data for row_indices
        if "row_indices_dev" not in src:
            # Pattern: row_indices = (const int64_t *) set_rows->src[1]->data;
            pat1 = r'(\brow_indices\s*=\s*\(const\s+int64_t\s*\*\)\s*([a-zA-Z0-9_>.-]+->src\[1\])->data;)'
            if re.search(pat1, src):
                src = re.sub(
                    r'(\bconst\s+int64_t\s*\*\s*row_indices\s*=\s*nullptr;)',
                    r'\1\n    ggml_cuda_pool_alloc<int64_t> row_indices_dev(ctx.pool());',
                    src
                )
                src = re.sub(
                    pat1,
                    r'''const ggml_tensor * _sr1 = \2;
        row_indices = (const int64_t *) _sr1->data;
        if (_sr1->buffer && ggml_backend_buffer_is_host(_sr1->buffer)) {
            row_indices_dev.alloc(ggml_nelements(_sr1));
            CUDA_CHECK(cudaMemcpyAsync(row_indices_dev.ptr, _sr1->data, ggml_nbytes(_sr1), cudaMemcpyHostToDevice, stream));
            row_indices = row_indices_dev.ptr;
        }''',
                    src
                )
                # Ensure stream is declared before row_indices assignment if needed
                if "cudaStream_t stream = ctx.stream();" in src:
                    # Move stream to top of function if it was after
                    pass
                modified = True

        # 2. Patch pos in any kernel launcher (rms_norm_mul_rope / rope)
        if "pos_dev" not in src and ("rms_norm_mul_rope" in src or "rope" in fname):
            pat2 = r'(\bconst\s+int32_t\s*\*\s*pos\s*=\s*\(const\s+int32_t\s*\*\)\s*([a-zA-Z0-9_>.-]+)->data;)'
            if re.search(pat2, src):
                src = re.sub(
                    pat2,
                    r'''\1\n    ggml_cuda_pool_alloc<int32_t> pos_dev(ctx.pool());\n    if (\2 && \2->buffer && ggml_backend_buffer_is_host(\2->buffer)) {\n        pos_dev.alloc(ggml_nelements(\2));\n        CUDA_CHECK(cudaMemcpyAsync(pos_dev.ptr, \2->data, ggml_nbytes(\2), cudaMemcpyHostToDevice, stream));\n        pos = pos_dev.ptr;\n    }''',
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
    patch_backend(llama_dir)
    patch_llama_graph(llama_dir)
    cuda_dir = os.path.join(llama_dir, "ggml", "src", "ggml-cuda")
    if os.path.isdir(cuda_dir):
        patch_getrows(cuda_dir)
        patch_set_rows(cuda_dir)
        patch_all_cuda_files(cuda_dir)

if __name__ == "__main__":
    main()
