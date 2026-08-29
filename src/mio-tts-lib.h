#pragma once

#if __has_include(<llama/llama.h>)
#include <llama/llama.h>
#else
#include "llama.h"
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct mio_tts_context;
struct mio_tts_vocab_map;

struct mio_tts_params {
    int32_t n_threads;
    float max_reference_seconds;
    enum llama_flash_attn_type wavlm_flash_attn_type;
    enum llama_flash_attn_type miocodec_flash_attn_type;
    bool trim_trailing_silence;
    float silence_threshold_db;
    float fade_out_ms;
};

LLAMA_API struct mio_tts_params mio_tts_default_params(void);

LLAMA_API struct mio_tts_context * mio_tts_init_from_file(
        const char * model_vocoder,
        const char * wavlm_model,
        char * err,
        size_t err_size);

// Create a new context that shares read-only model weights with `ctx`.
LLAMA_API struct mio_tts_context * mio_tts_context_clone(
        const struct mio_tts_context * ctx,
        char * err,
        size_t err_size);

LLAMA_API void mio_tts_free(struct mio_tts_context * ctx);

LLAMA_API bool mio_tts_context_is_dynamic_global(const struct mio_tts_context * ctx);
LLAMA_API int32_t mio_tts_context_sample_rate(const struct mio_tts_context * ctx);
LLAMA_API int32_t mio_tts_context_samples_per_token(const struct mio_tts_context * ctx);
LLAMA_API bool mio_tts_context_set_backend_device(
        struct mio_tts_context * ctx,
        const char * backend_name,
        char * err,
        size_t err_size);

LLAMA_API struct mio_tts_vocab_map * mio_tts_vocab_map_init(
        const struct llama_vocab * vocab,
        char * err,
        size_t err_size);

LLAMA_API void mio_tts_vocab_map_free(struct mio_tts_vocab_map * map);

LLAMA_API bool mio_tts_tokens_to_codes(
        const struct mio_tts_vocab_map * map,
        const llama_token * tokens,
        size_t n_tokens,
        int32_t ** codes_out,
        size_t * n_codes_out,
        char * err,
        size_t err_size);

// Try to convert a single LLM token to a MioCodec audio code.
// Returns true and sets *code_out if the token maps to a code.
// Returns false (no error) if the token is not an audio code (e.g. text token).
LLAMA_API bool mio_tts_token_to_code(
        const struct mio_tts_vocab_map * map,
        llama_token token,
        int32_t * code_out);

LLAMA_API bool mio_tts_codes_load(
        const char * path,
        int32_t ** codes_out,
        size_t * n_codes_out,
        char * err,
        size_t err_size);

LLAMA_API bool mio_tts_codes_save(
        const char * path,
        const int32_t * codes,
        size_t n_codes,
        char * err,
        size_t err_size);

LLAMA_API void mio_tts_codes_free(int32_t * codes);

LLAMA_API bool mio_tts_codes_trim_trailing(
        int32_t * codes,
        size_t * n_codes,
        size_t max_repeat_tail);

LLAMA_API bool mio_tts_reference_to_embedding(
        struct mio_tts_context * ctx,
        const char * reference_audio,
        struct mio_tts_params params,
        float ** embedding_out,
        size_t * n_embedding_out,
        char * err,
        size_t err_size);

LLAMA_API bool mio_tts_embedding_load_gguf(
        const char * path,
        float ** embedding_out,
        size_t * n_embedding_out,
        char * err,
        size_t err_size);

LLAMA_API bool mio_tts_embedding_save_gguf(
        const char * path,
        const float * embedding,
        size_t n_embedding,
        char * err,
        size_t err_size);

LLAMA_API void mio_tts_embedding_free(float * embedding);

LLAMA_API bool mio_tts_synthesize(
        struct mio_tts_context * ctx,
        const int32_t * codes,
        size_t n_codes,
        const float * embedding,
        size_t n_embedding,
        struct mio_tts_params params,
        float ** audio_out,
        size_t * n_audio_out,
        int32_t * sample_rate_out,
        char * err,
        size_t err_size);

// Reserve MioCodec decode workspace for up to `n_codes` code tokens.
LLAMA_API bool mio_tts_reserve_workspace(
        struct mio_tts_context * ctx,
        int32_t n_codes,
        char * err,
        size_t err_size);

// Estimate MioCodec decode workspace size in bytes for up to `n_codes`.
LLAMA_API bool mio_tts_estimate_workspace_bytes(
        struct mio_tts_context * ctx,
        int32_t n_codes,
        uint64_t * out_bytes,
        char * err,
        size_t err_size);

// Reserve WavLM + global-embedding workspaces for reference generation.
LLAMA_API bool mio_tts_reserve_reference_workspace(
        struct mio_tts_context * ctx,
        float max_reference_seconds,
        char * err,
        size_t err_size);

// Estimate WavLM + global-embedding workspace size in bytes for reference generation.
LLAMA_API bool mio_tts_estimate_reference_workspace_bytes(
        struct mio_tts_context * ctx,
        float max_reference_seconds,
        uint64_t * out_bytes,
        char * err,
        size_t err_size);

LLAMA_API bool mio_tts_audio_save_wav16(
        const char * path,
        const float * audio,
        size_t n_audio,
        int32_t sample_rate,
        char * err,
        size_t err_size);

LLAMA_API void mio_tts_audio_free(float * audio);

#ifdef __cplusplus
}

namespace mio_tts_compat {
    template <typename T>
    inline auto set_use_mmap_helper(T & p, bool val, int) -> decltype((void) (p.use_mmap = val)) {
        p.use_mmap = val;
    }
    template <typename T>
    inline void set_use_mmap_helper(T &, bool, long) {}

    inline void set_use_mmap(llama_model_params & p, bool val) {
        set_use_mmap_helper(p, val, 0);
    }

    template <typename T>
    inline auto set_use_mlock_helper(T & p, bool val, int) -> decltype((void) (p.use_mlock = val)) {
        p.use_mlock = val;
    }
    template <typename T>
    inline void set_use_mlock_helper(T &, bool, long) {}

    inline void set_use_mlock(llama_model_params & p, bool val) {
        set_use_mlock_helper(p, val, 0);
    }

    template <typename T>
    inline auto set_check_tensors_helper(T & p, bool val, int) -> decltype((void) (p.check_tensors = val)) {
        p.check_tensors = val;
    }
    template <typename T>
    inline void set_check_tensors_helper(T &, bool, long) {}

    inline void set_check_tensors(llama_model_params & p, bool val) {
        set_check_tensors_helper(p, val, 0);
    }

    template <typename T>
    inline auto set_offload_kqv_helper(T & p, bool val, int) -> decltype((void) (p.offload_kqv = val)) {
        p.offload_kqv = val;
    }
    template <typename T>
    inline void set_offload_kqv_helper(T &, bool, long) {}

    inline void set_offload_kqv(llama_context_params & p, bool val) {
        set_offload_kqv_helper(p, val, 0);
    }

    template <typename T>
    inline auto set_no_perf_helper(T & p, bool val, int) -> decltype((void) (p.no_perf = val)) {
        p.no_perf = val;
    }
    template <typename T>
    inline void set_no_perf_helper(T &, bool, long) {}

    inline void set_no_perf(llama_context_params & p, bool val) {
        set_no_perf_helper(p, val, 0);
    }

    template <typename T>
    inline auto set_op_offload_helper(T & p, bool val, int) -> decltype((void) (p.op_offload = val)) {
        p.op_offload = val;
    }
    template <typename T>
    inline void set_op_offload_helper(T &, bool, long) {}

    inline void set_op_offload(llama_context_params & p, bool val) {
        set_op_offload_helper(p, val, 0);
    }

    inline struct llama_batch make_llama_batch_with_pos(
            const llama_token * tokens,
            int32_t n_tokens,
            int32_t start_pos,
            bool is_last_chunk) {
        struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (int32_t i = 0; i < n_tokens; ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = start_pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = is_last_chunk && (i == n_tokens - 1);
        }
        batch.n_tokens = n_tokens;
        return batch;
    }

    template <typename Fn>
    inline auto call_penalties_5(Fn fn, int32_t n_vocab, int32_t last_n, float repeat, float freq, float present, int)
        -> decltype(fn(n_vocab, last_n, repeat, freq, present)) {
        return fn(n_vocab, last_n, repeat, freq, present);
    }

    template <typename Fn>
    inline auto call_penalties_5(Fn fn, int32_t n_vocab, int32_t last_n, float repeat, float freq, float present, long)
        -> decltype(fn(last_n, repeat, freq, present)) {
        (void) n_vocab;
        return fn(last_n, repeat, freq, present);
    }

    inline struct llama_sampler * compat_llama_sampler_init_penalties(
            int32_t n_vocab,
            int32_t penalty_last_n,
            float penalty_repeat,
            float penalty_freq,
            float penalty_present) {
        return call_penalties_5(&::llama_sampler_init_penalties, n_vocab, penalty_last_n, penalty_repeat, penalty_freq, penalty_present, 0);
    }
}
#endif
