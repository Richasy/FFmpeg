/*
 * Copyright (c) 2025 Vittorio Palmisano
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with FFmpeg; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <whisper.h>
#include <ggml-backend.h>

#include "libavutil/avutil.h"
#include "libavutil/opt.h"
#include "libavutil/channel_layout.h"
#include "libavutil/samplefmt.h"
#include "libavfilter/avfilter.h"
#include "libavfilter/audio.h"
#include "libavutil/mem.h"
#include "libavutil/avstring.h"
#include "libavutil/internal.h"
#include "libavformat/avio.h"
#include "libavutil/thread.h"

#include "formats.h"

#define WHISPER_HAS_THREADS (HAVE_PTHREADS || HAVE_W32THREADS || HAVE_OS2THREADS)

typedef struct WhisperContext {
    const AVClass *class;
    char *model_path;
    const char *language;
    char *language_str;
    bool translate;
    bool use_gpu;
    int gpu_device;
    char *backend_path;
    int n_processors;
    char *vad_model_path;
    float vad_threshold;
    int64_t vad_min_speech_duration;
    int64_t vad_min_silence_duration;

    int64_t queue;
    char *destination;
    char *format;
    int max_len;

    struct whisper_context *ctx_wsp;
    struct whisper_vad_context *ctx_vad;
    struct whisper_vad_params vad_params;

    float *audio_buffer;
    int audio_buffer_queue_size;
    int audio_buffer_fill_size;
    int audio_buffer_vad_size;
    int64_t audio_buffer_start_ms;

    int eof;
    int64_t next_pts;

    AVIOContext *avio_context;
    int index;

#if WHISPER_HAS_THREADS
    float *infer_buffer;
    int infer_buffer_size;
    int64_t infer_buffer_start_ms;

    pthread_t infer_thread;
    AVMutex infer_mutex;
    AVCond infer_cond;
    int infer_pending;
    int infer_done;
    int shutdown;
    int thread_created;

    char *infer_result_text;
    int64_t infer_vad_speech_start_ms;  /* VAD speech start offset within buffer, -1 if no VAD */
    int infer_n_threads;

    /* All VAD-detected speech ranges within the inference buffer, used to
     * snap whisper's per-segment timestamps onto real speech windows.
     * Whisper's t0/t1 for individual segments are notoriously unreliable
     * when the input contains silence around the speech (it tends to place
     * the timestamp earlier than the actual onset, which makes subtitles
     * pop up several seconds before the corresponding voice).  By clamping
     * each whisper segment to its best-overlapping VAD range we get
     * subtitles that align with what the user actually hears.
     * Offsets are in ms, relative to infer_buffer_start_ms. */
    int64_t *infer_vad_starts_ms;
    int64_t *infer_vad_ends_ms;
    int infer_vad_n;
    int infer_vad_cap;

    AVFilterContext *filter_ctx;
#endif
} WhisperContext;

static void cb_log(enum ggml_log_level level, const char *text, void *user_data)
{
    AVFilterContext *ctx = user_data;
    int av_log_level = AV_LOG_DEBUG;
    switch (level) {
    case GGML_LOG_LEVEL_ERROR:
        av_log_level = AV_LOG_ERROR;
        break;
    case GGML_LOG_LEVEL_WARN:
        av_log_level = AV_LOG_WARNING;
        break;
    }
    av_log(ctx, av_log_level, "%s", text);
}

/* Common whisper hallucination patterns.  When the audio contains no clear
 * speech (silence, background music, ambient noise), whisper tends to
 * "hallucinate" these phrases — they originate from its YouTube training
 * data rather than the actual audio.  We filter them out to avoid spurious
 * subtitles. Case-insensitive substring matching is used. */
static const char *const hallucination_patterns[] = {
    "thanks for watching",
    "thank you for watching",
    "subscribe",
    "like and share",
    "please like",
    "感谢观看",
    "感谢您的观看",
    "感谢收看",
    "谢谢观看",
    "谢谢收看",
    "请订阅",
    "ご視聴ありがとう",
    "チャンネル登録",
    "視聴ありがとう",
};

static int is_hallucination(const char *text)
{
    for (int i = 0; i < FF_ARRAY_ELEMS(hallucination_patterns); i++) {
        if (av_stristr(text, hallucination_patterns[i]))
            return 1;
    }
    return 0;
}

/* Escape a string for JSON: \ → \\, " → \", control chars → \uXXXX */
static char *json_escape(const char *s)
{
    size_t len = strlen(s);
    /* Worst case: every char needs \uXXXX (6 bytes) + NUL */
    char *out = av_malloc(len * 6 + 1);
    if (!out)
        return NULL;
    char *p = out;
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"')       { *p++ = '\\'; *p++ = '"'; }
        else if (c == '\\') { *p++ = '\\'; *p++ = '\\'; }
        else if (c == '\n') { *p++ = '\\'; *p++ = 'n'; }
        else if (c == '\r') { *p++ = '\\'; *p++ = 'r'; }
        else if (c == '\t') { *p++ = '\\'; *p++ = 't'; }
        else if (c < 0x20)  { p += snprintf(p, 7, "\\u%04x", c); }
        else                { *p++ = c; }
    }
    *p = '\0';
    return out;
}

#if WHISPER_HAS_THREADS
static void *whisper_infer_thread(void *arg)
{
    WhisperContext *wctx = arg;
    AVFilterContext *ctx = wctx->filter_ctx;

    ff_thread_setname("whisper-infer");

    ff_mutex_lock(&wctx->infer_mutex);
    while (!wctx->shutdown) {
        while (!wctx->infer_pending && !wctx->shutdown)
            ff_cond_wait(&wctx->infer_cond, &wctx->infer_mutex);

        if (wctx->shutdown)
            break;

        /* Copy inference parameters while locked */
        float *samples = wctx->infer_buffer;
        int n_samples = wctx->infer_buffer_size;
        int64_t timestamp_ms = wctx->infer_buffer_start_ms;
        int64_t vad_speech_start_ms = wctx->infer_vad_speech_start_ms;
        int n_threads = wctx->infer_n_threads;

        /* Snapshot the VAD segment list onto the stack/heap so we can read
         * it without holding the lock. Filter_frame won't touch the array
         * while infer_pending is set, but a stable snapshot keeps the
         * snapping code straightforward. */
        int n_vad = wctx->infer_vad_n;
        int64_t *vad_starts = NULL;
        int64_t *vad_ends   = NULL;
        if (n_vad > 0) {
            vad_starts = av_malloc_array(n_vad, sizeof(*vad_starts));
            vad_ends   = av_malloc_array(n_vad, sizeof(*vad_ends));
            if (vad_starts && vad_ends) {
                memcpy(vad_starts, wctx->infer_vad_starts_ms, n_vad * sizeof(*vad_starts));
                memcpy(vad_ends,   wctx->infer_vad_ends_ms,   n_vad * sizeof(*vad_ends));
            } else {
                av_freep(&vad_starts);
                av_freep(&vad_ends);
                n_vad = 0;
            }
        }
        ff_mutex_unlock(&wctx->infer_mutex);

        /* Run whisper inference without lock held */
        const float duration = (float)n_samples / WHISPER_SAMPLE_RATE;

        av_log(ctx, AV_LOG_INFO,
               "async transcription at %" PRId64 " ms, %d samples (%.2f seconds)...\n",
               timestamp_ms, n_samples, duration);

        struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        params.language = wctx->language;
        params.translate = wctx->translate;
        params.n_threads = n_threads;
        params.print_special = 0;
        params.print_progress = 0;
        params.print_realtime = 0;
        params.print_timestamps = 0;
        params.max_len = wctx->max_len;
        params.token_timestamps = (wctx->max_len > 0);

        /* Adaptive audio_ctx: whisper's encoder always processes 30 s of mel
         * spectrogram regardless of input length, so short VAD segments waste
         * 5–6× the encoder compute.  Cap audio_ctx to roughly the actual
         * audio length (50 mel frames per second, plus a small margin), with
         * the upstream max of 1500 (= 30 s).  Setting audio_ctx = 0 means
         * "use full 30 s", which is the default. */
        const int n_samples_ms = (int)((int64_t)n_samples * 1000 / WHISPER_SAMPLE_RATE);
        params.audio_ctx = FFMIN(1500, (n_samples_ms / 20) + 16);

        char *segments_json = NULL;

        if (whisper_full_parallel(wctx->ctx_wsp, params, samples, n_samples,
                                  n_samples >= 5 * WHISPER_SAMPLE_RATE ? wctx->n_processors : 1) != 0) {
            av_log(ctx, AV_LOG_ERROR, "Failed to process audio with whisper.cpp (async)\n");
        } else {
            const int n_segments = whisper_full_n_segments(wctx->ctx_wsp);

            /* Build JSON array: [{"s":ms,"e":ms,"t":"text"}, ...] */
            segments_json = av_strdup("[");

            for (int i = 0; i < n_segments; ++i) {
                const char *text = whisper_full_get_segment_text(wctx->ctx_wsp, i);
                if (av_isspace(text[0]))
                    text++;
                char *text_cleaned = av_strireplace(text, "[BLANK_AUDIO]", "");

                if (av_strnlen(text_cleaned, 1) == 0) {
                    av_freep(&text_cleaned);
                    continue;
                }

                if (wctx->max_len > 0 && (strcmp(text_cleaned, "[") == 0 || strcmp(text_cleaned, "]") == 0 ||
                                          strcmp(text_cleaned, "BLANK") == 0 || strcmp(text_cleaned, "_") == 0 ||
                                          strcmp(text_cleaned, "AUDIO") == 0)) {
                    av_freep(&text_cleaned);
                    continue;
                }

                if (is_hallucination(text_cleaned)) {
                    av_log(ctx, AV_LOG_INFO, "filtered hallucination: \"%s\"\n", text_cleaned);
                    av_freep(&text_cleaned);
                    continue;
                }

                int64_t t0_ms = whisper_full_get_segment_t0(wctx->ctx_wsp, i) * 10;
                int64_t t1_ms = whisper_full_get_segment_t1(wctx->ctx_wsp, i) * 10;

                /* Snap [t0, t1] (relative to inference buffer start) to the
                 * VAD speech segment with the largest overlap.  This fixes
                 * the "subtitle appears seconds before the speech" problem:
                 * whisper's per-segment timestamps are computed from its
                 * decoder's attention and frequently drift earlier than the
                 * actual voice onset when the input contains leading or
                 * inter-segment silence.  We trust VAD's edges instead. */
                if (n_vad > 0) {
                    /* Snap to VAD speech windows.  Strategy: whisper's
                     * timestamps are *locally* accurate inside a single
                     * speech window (the decoder attention behaves well
                     * around real voiced frames) but routinely drift
                     * earlier when whisper extends into silence.  So:
                     *
                     *   - If t0 lies inside a VAD window, KEEP it (this
                     *     allows several whisper segments to coexist
                     *     within one window without collapsing onto the
                     *     same start time).
                     *   - If t0 lies in a silence gap before window i,
                     *     push it forward to window[i].start.
                     *   - If t0 is past every window, fall back to the
                     *     last window's end.
                     *
                     * Symmetric clamp for t1: cap it at the latest window
                     * that begins at or before t1, plus a 200 ms tail. */
                    int idx0 = -1;
                    for (int v = 0; v < n_vad; v++) {
                        if (t0_ms < vad_ends[v]) { idx0 = v; break; }
                    }
                    if (idx0 < 0) {
                        t0_ms = vad_ends[n_vad - 1];
                    } else if (t0_ms < vad_starts[idx0]) {
                        t0_ms = vad_starts[idx0];
                    }

                    int idx1 = -1;
                    for (int v = n_vad - 1; v >= 0; v--) {
                        if (vad_starts[v] <= t1_ms) { idx1 = v; break; }
                    }
                    if (idx1 < 0) {
                        t1_ms = t0_ms + 200;
                    } else {
                        int64_t cap = vad_ends[idx1] + 200;
                        if (t1_ms > cap) t1_ms = cap;
                    }

                    if (t1_ms < t0_ms + 200) t1_ms = t0_ms + 200;
                } else if (vad_speech_start_ms >= 0 && t0_ms < vad_speech_start_ms) {
                    /* Backward-compat path (no segment list available) */
                    t0_ms = vad_speech_start_ms;
                }

                const int64_t abs_start = timestamp_ms + t0_ms;
                const int64_t abs_end   = timestamp_ms + t1_ms;

                av_log(ctx, AV_LOG_DEBUG, "  [%" PRId64 "-%" PRId64 "]: \"%s\"\n",
                       abs_start, abs_end, text_cleaned);

                char *escaped = json_escape(text_cleaned);
                char *entry = av_asprintf("%s{\"s\":%" PRId64 ",\"e\":%" PRId64 ",\"t\":\"%s\"}",
                                          strcmp(segments_json, "[") == 0 ? "" : ",",
                                          abs_start, abs_end, escaped);
                av_freep(&escaped);

                char *new_json = av_asprintf("%s%s", segments_json, entry);
                av_freep(&segments_json);
                av_freep(&entry);
                segments_json = new_json;

                if (wctx->avio_context) {
                    const int64_t start_t = abs_start;
                    const int64_t end_t = abs_end;
                    char *buf = NULL;

                    if (!av_strcasecmp(wctx->format, "srt")) {
                        buf = av_asprintf(
                            "%d\n%02" PRId64 ":%02" PRId64 ":%02" PRId64 ",%03" PRId64
                            " --> %02" PRId64 ":%02" PRId64 ":%02" PRId64 ",%03" PRId64 "\n%s\n\n",
                            wctx->index, start_t / 3600000,
                            (start_t / 60000) % 60, (start_t / 1000) % 60,
                            start_t % 1000, end_t / 3600000, (end_t / 60000) % 60,
                            (end_t / 1000) % 60, end_t % 1000, text_cleaned);
                        wctx->index++;
                    } else if (!av_strcasecmp(wctx->format, "json")) {
                        buf = av_asprintf("{\"start\":%" PRId64 ",\"end\":%" PRId64 ",\"text\":\"%s\"}\n",
                                          start_t, end_t, text_cleaned);
                    } else {
                        buf = av_asprintf("%s\n", text_cleaned);
                    }

                    if (buf) {
                        avio_write(wctx->avio_context, buf, strlen(buf));
                        av_freep(&buf);
                    }
                }

                av_freep(&text_cleaned);
            }

            /* Close JSON array */
            char *closed = av_asprintf("%s]", segments_json);
            av_freep(&segments_json);
            segments_json = closed;

            /* If only "[]", set to NULL (no valid segments) */
            if (strcmp(segments_json, "[]") == 0)
                av_freep(&segments_json);
        }

        ff_mutex_lock(&wctx->infer_mutex);
        av_freep(&wctx->infer_result_text);
        wctx->infer_result_text = segments_json;
        wctx->infer_pending = 0;
        wctx->infer_done = 1;
        ff_cond_signal(&wctx->infer_cond);

        /* Wake the libavfilter scheduler so activate() runs even if no new
         * input frame is currently arriving — the result must be flushed
         * downstream as soon as possible, otherwise it sits unharvested
         * until the next audio frame eventually shows up. */
        ff_filter_set_ready(ctx, 100);

        av_freep(&vad_starts);
        av_freep(&vad_ends);
    }
    ff_mutex_unlock(&wctx->infer_mutex);

    return NULL;
}

static void collect_result_locked(WhisperContext *wctx, AVFrame *frame)
{
    if (!wctx->infer_done)
        return;

    if (wctx->infer_result_text && frame) {
        AVDictionary **metadata = &frame->metadata;
        if (metadata)
            av_dict_set(metadata, "lavfi.whisper.segments", wctx->infer_result_text, 0);
    }

    av_freep(&wctx->infer_result_text);
    wctx->infer_done = 0;
}

/* Run Silero VAD on [audio_buffer .. audio_buffer + n_samples) and store
 * every detected speech window into wctx->infer_vad_starts_ms/ends_ms.
 * Returns the start of the first speech window in ms (suitable for the
 * legacy vad_speech_start_ms parameter), or -1 if no segments. */
static int64_t populate_vad_segments_locked(AVFilterContext *ctx, int n_samples)
{
    WhisperContext *wctx = ctx->priv;

    wctx->infer_vad_n = 0;
    if (!wctx->ctx_vad || n_samples <= 0)
        return -1;

    struct whisper_vad_segments *segments =
        whisper_vad_segments_from_samples(wctx->ctx_vad, wctx->vad_params,
                                          wctx->audio_buffer, n_samples);
    if (!segments)
        return -1;

    int n = whisper_vad_segments_n_segments(segments);
    int64_t first_start = -1;
    if (n > 0) {
        if (n > wctx->infer_vad_cap) {
            int64_t *ns = av_realloc_array(wctx->infer_vad_starts_ms, n, sizeof(*ns));
            int64_t *ne = av_realloc_array(wctx->infer_vad_ends_ms,   n, sizeof(*ne));
            if (ns) wctx->infer_vad_starts_ms = ns;
            if (ne) wctx->infer_vad_ends_ms   = ne;
            if (ns && ne) wctx->infer_vad_cap = n;
        }
        if (wctx->infer_vad_cap >= n) {
            for (int v = 0; v < n; v++) {
                wctx->infer_vad_starts_ms[v] = (int64_t)(whisper_vad_segments_get_segment_t0(segments, v) * 10.0);
                wctx->infer_vad_ends_ms[v]   = (int64_t)(whisper_vad_segments_get_segment_t1(segments, v) * 10.0);
            }
            wctx->infer_vad_n = n;
            first_start = wctx->infer_vad_starts_ms[0];
        }
    }
    whisper_vad_free_segments(segments);
    return first_start;
}

static void submit_inference_locked(AVFilterContext *ctx, int samples, int64_t vad_speech_start_ms)
{
    WhisperContext *wctx = ctx->priv;

    if (samples <= 0 || samples > wctx->audio_buffer_fill_size)
        return;

    /* Ensure infer_buffer is large enough */
    if (samples > wctx->infer_buffer_size) {
        float *new_buf = av_realloc_array(wctx->infer_buffer, samples, sizeof(*wctx->infer_buffer));
        if (!new_buf) {
            av_log(ctx, AV_LOG_ERROR, "Failed to allocate infer_buffer\n");
            return;
        }
        wctx->infer_buffer = new_buf;
    }

    memcpy(wctx->infer_buffer, wctx->audio_buffer, samples * sizeof(*wctx->audio_buffer));
    wctx->infer_buffer_size = samples;
    wctx->infer_buffer_start_ms = wctx->audio_buffer_start_ms;
    wctx->infer_vad_speech_start_ms = vad_speech_start_ms;
    wctx->infer_n_threads = ff_filter_get_nb_threads(ctx);

    /* Compress audio_buffer: remove consumed samples */
    const float duration = (float)samples / WHISPER_SAMPLE_RATE;
    if (wctx->audio_buffer_fill_size > samples) {
        memmove(wctx->audio_buffer, wctx->audio_buffer + samples,
                (wctx->audio_buffer_fill_size - samples) * sizeof(*wctx->audio_buffer));
        wctx->audio_buffer_start_ms += (int64_t)(duration * 1000);
    }
    wctx->audio_buffer_fill_size -= samples;
    wctx->audio_buffer_vad_size = wctx->audio_buffer_fill_size;

    wctx->infer_pending = 1;
    wctx->infer_done = 0;
    ff_cond_signal(&wctx->infer_cond);
}
#endif

static char g_backend_load_dir[1024];

static void load_backends_from_dir_once_cb(void)
{
    ggml_backend_load_all_from_path(g_backend_load_dir);
}

static int init(AVFilterContext *ctx)
{
    WhisperContext *wctx = ctx->priv;

    /* ggml's backend registry is a process-wide singleton.  Calling
     * ggml_backend_load_all / ggml_backend_load_all_from_path more than
     * once corrupts it: previously-registered backends keep callbacks
     * pointing into now-freed loggers/contexts (whisper_log_set was bound
     * to the previous AVFilterContext), and the next log emission
     * dereferences a stale pointer.  This manifests as a crash inside
     * ggml_log_internal the *second* time the user enables Whisper
     * subtitles in the same process.  Guard every load path with
     * AV_ONCE_INIT so backend registration runs exactly once. */
    static AVOnce init_static_once       = AV_ONCE_INIT;
    static AVOnce init_static_once_dir   = AV_ONCE_INIT;
    static AVOnce init_static_once_file  = AV_ONCE_INIT;

    if (wctx->backend_path && wctx->backend_path[0]) {
        // Caller provided an explicit ggml backend dll path. Treat its
        // PARENT DIRECTORY as the search dir for ALL ggml-*.dll backends
        // (CPU + GPU): with GGML_BACKEND_DL=ON, ggml-cpu.dll is no longer
        // statically linked into ggml-base.dll and must be loaded too,
        // otherwise whisper has no compute backend at all.
        // ggml_backend_load_all_from_path scans the directory for known
        // backends (cpu/vulkan/cuda/...) and registers each one found.
        char dir_buf[1024];
        av_strlcpy(dir_buf, wctx->backend_path, sizeof(dir_buf));
        char *sep = strrchr(dir_buf, '/');
#ifdef _WIN32
        char *sep2 = strrchr(dir_buf, '\\');
        if (sep2 && (!sep || sep2 > sep)) sep = sep2;
#endif
        if (sep) {
            *sep = '\0';
            av_log(ctx, AV_LOG_INFO,
                   "Loading ggml backends from directory '%s' (derived from "
                   "backend_path='%s').\n", dir_buf, wctx->backend_path);
            av_strlcpy(g_backend_load_dir, dir_buf, sizeof(g_backend_load_dir));
            ff_thread_once(&init_static_once_dir, load_backends_from_dir_once_cb);
        } else {
            // No directory separator: fall back to single-dll load.
            // ggml_backend_load is also a one-shot global registration —
            // the same crash applies if it runs more than once.
            ff_thread_once(&init_static_once_file, ggml_backend_load_all);
            (void)ggml_backend_load(wctx->backend_path);
            av_log(ctx, AV_LOG_INFO,
                   "Loaded ggml backend from '%s' (or fell back to auto).\n",
                   wctx->backend_path);
        }
    } else {
        ff_thread_once(&init_static_once, ggml_backend_load_all);
    }

    whisper_log_set(cb_log, ctx);

    // Init whisper context
    if (!wctx->model_path) {
        av_log(ctx, AV_LOG_ERROR, "No whisper model path specified. Use the 'model' option.\n");
        return AVERROR(EINVAL);
    }

    struct whisper_context_params params = whisper_context_default_params();
    params.use_gpu = wctx->use_gpu;
    params.gpu_device = wctx->gpu_device;

    wctx->ctx_wsp = whisper_init_from_file_with_params(wctx->model_path, params);
    if (wctx->ctx_wsp == NULL) {
        av_log(ctx, AV_LOG_ERROR, "Failed to initialize whisper context from model: %s\n", wctx->model_path);
        return AVERROR(EIO);
    }

    // Init buffer
    wctx->audio_buffer_queue_size = av_rescale(wctx->queue, WHISPER_SAMPLE_RATE, AV_TIME_BASE);
    wctx->audio_buffer = av_malloc_array(wctx->audio_buffer_queue_size, sizeof(*wctx->audio_buffer));
    if (!wctx->audio_buffer)
        return AVERROR(ENOMEM);

    // Init VAD model context
    if (wctx->vad_model_path) {
        struct whisper_vad_context_params ctx_params = whisper_vad_default_context_params();
        ctx_params.n_threads = ff_filter_get_nb_threads(ctx);
        // ctx_params.use_gpu = wctx->use_gpu; TODO (see: whisper_vad_init_context)
        ctx_params.gpu_device = wctx->gpu_device;
        wctx->ctx_vad = whisper_vad_init_from_file_with_params(wctx->vad_model_path, ctx_params);

        wctx->vad_params = whisper_vad_default_params();
        wctx->vad_params.threshold = wctx->vad_threshold;
        wctx->vad_params.min_speech_duration_ms = av_rescale(wctx->vad_min_speech_duration, 1000, AV_TIME_BASE);
        wctx->vad_params.min_silence_duration_ms = av_rescale(wctx->vad_min_silence_duration, 1000, AV_TIME_BASE);
        wctx->vad_params.max_speech_duration_s = av_rescale(wctx->queue, 1, AV_TIME_BASE);
        wctx->vad_params.speech_pad_ms = 0;
        wctx->vad_params.samples_overlap = 0;
    }

    wctx->next_pts = AV_NOPTS_VALUE;

    if (wctx->destination && strcmp("", wctx->destination)) {
        const char *dst = wctx->destination;
        if (!strcmp("-", dst))
            dst = "pipe:1";
        int ret = avio_open(&wctx->avio_context, dst, AVIO_FLAG_WRITE);

        if (ret < 0) {
            av_log(ctx, AV_LOG_ERROR, "Could not open %s: %s\n", wctx->destination, av_err2str(ret));
            return ret;
        }

        wctx->avio_context->direct = AVIO_FLAG_DIRECT;
    }

    if (!whisper_is_multilingual(wctx->ctx_wsp)) {
        if (!wctx->translate && strcmp(wctx->language_str, "auto") == 0) {
            av_log(ctx, AV_LOG_WARNING,
                   "Multilingual model not provided. Non-English audio may not be correctly transcribed.\n");
        } else if (wctx->translate || (strcmp(wctx->language_str, "auto") != 0 && strcmp(wctx->language_str, "en") != 0)) {
            av_log(ctx, AV_LOG_ERROR,
                   "%s requested but multilingual model not provided.\n", wctx->translate ? "Translation" : "Transcription");
            return AVERROR(ENOSYS);
        }
        wctx->language = "en";
    } else
        wctx->language = wctx->language_str;

#if WHISPER_HAS_THREADS
    wctx->filter_ctx = ctx;
    wctx->infer_buffer = av_malloc_array(wctx->audio_buffer_queue_size, sizeof(*wctx->infer_buffer));
    if (!wctx->infer_buffer)
        return AVERROR(ENOMEM);
    wctx->infer_buffer_size = wctx->audio_buffer_queue_size;

    ff_mutex_init(&wctx->infer_mutex, NULL);
    ff_cond_init(&wctx->infer_cond, NULL);

    if (pthread_create(&wctx->infer_thread, NULL, whisper_infer_thread, wctx) != 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create whisper inference thread\n");
        return AVERROR(ENOMEM);
    }
    wctx->thread_created = 1;
#endif

    av_log(ctx, AV_LOG_INFO,
           "Whisper filter initialized: model: %s lang: %s queue: %" PRId64 " ms async: %s\n",
           wctx->model_path, wctx->language, wctx->queue / 1000,
#if WHISPER_HAS_THREADS
           "yes"
#else
           "no"
#endif
           );

    return 0;
}

static void uninit(AVFilterContext *ctx)
{
    WhisperContext *wctx = ctx->priv;

#if WHISPER_HAS_THREADS
    if (wctx->thread_created) {
        ff_mutex_lock(&wctx->infer_mutex);
        wctx->shutdown = 1;
        ff_cond_signal(&wctx->infer_cond);
        ff_mutex_unlock(&wctx->infer_mutex);

        pthread_join(wctx->infer_thread, NULL);
        wctx->thread_created = 0;
    }

    ff_mutex_destroy(&wctx->infer_mutex);
    ff_cond_destroy(&wctx->infer_cond);
    av_freep(&wctx->infer_buffer);
    av_freep(&wctx->infer_result_text);
    av_freep(&wctx->infer_vad_starts_ms);
    av_freep(&wctx->infer_vad_ends_ms);
#endif

    if (wctx->audio_buffer_fill_size > 0) {
        av_log(ctx, AV_LOG_WARNING,
               "Remaining audio buffer %d samples (%d seconds) after stopping\n",
               wctx->audio_buffer_fill_size, wctx->audio_buffer_fill_size / WHISPER_SAMPLE_RATE);
    }

    if (wctx->ctx_vad) {
        whisper_vad_free(wctx->ctx_vad);
        wctx->ctx_vad = NULL;
    }

    if (wctx->ctx_wsp) {
        whisper_free(wctx->ctx_wsp);
        wctx->ctx_wsp = NULL;
    }

    av_freep(&wctx->audio_buffer);

    if (wctx->avio_context)
        avio_closep(&wctx->avio_context);
}

static void run_transcription(AVFilterContext *ctx, AVFrame *frame, int samples)
{
    WhisperContext *wctx = ctx->priv;
    samples = FFMAX(0, FFMIN(samples, wctx->audio_buffer_fill_size));

    if (!wctx->ctx_wsp || samples == 0)
        return;

    const int64_t timestamp_ms = wctx->audio_buffer_start_ms;
    const float duration = (float) samples / WHISPER_SAMPLE_RATE;

    av_log(ctx, AV_LOG_INFO,
           "run transcription at %" PRId64 " ms, %d/%d samples (%.2f seconds)...\n",
           timestamp_ms, samples, wctx->audio_buffer_fill_size, duration);

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.language = wctx->language;
    params.translate = wctx->translate;
    params.n_threads = ff_filter_get_nb_threads(ctx);
    params.print_special = 0;
    params.print_progress = 0;
    params.print_realtime = 0;
    params.print_timestamps = 0;
    params.max_len = wctx->max_len;
    params.token_timestamps = (wctx->max_len > 0);
    params.split_on_word = (wctx->max_len > 0);

    /* Adaptive audio_ctx (see whisper_infer_thread for rationale). */
    {
        const int n_samples_ms = (int)((int64_t)samples * 1000 / WHISPER_SAMPLE_RATE);
        params.audio_ctx = FFMIN(1500, (n_samples_ms / 20) + 16);
    }

    if (whisper_full_parallel(wctx->ctx_wsp, params, wctx->audio_buffer, samples,
                              samples >= 5 * WHISPER_SAMPLE_RATE ? wctx->n_processors : 1) != 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to process audio with whisper.cpp\n");
        return;
    }

    const int n_segments = whisper_full_n_segments(wctx->ctx_wsp);
    char *segments_json = av_strdup("[");

    for (int i = 0; i < n_segments; ++i) {
        const char *text = whisper_full_get_segment_text(wctx->ctx_wsp, i);
        if (av_isspace(text[0]))
            text++;
        char *text_cleaned = av_strireplace(text, "[BLANK_AUDIO]", "");

        if (av_strnlen(text_cleaned, 1) == 0) {
            av_freep(&text_cleaned);
            continue;
        }

        // Skip segments that are parts of [BLANK_AUDIO] when max_len splits them
        if (wctx->max_len > 0 && (strcmp(text_cleaned, "[") == 0 || strcmp(text_cleaned, "]") == 0 ||
                                  strcmp(text_cleaned, "BLANK") == 0 || strcmp(text_cleaned, "_") == 0 ||
                                  strcmp(text_cleaned, "AUDIO") == 0)) {
            av_freep(&text_cleaned);
            continue;
        }

        if (is_hallucination(text_cleaned)) {
            av_log(ctx, AV_LOG_INFO, "filtered hallucination: \"%s\"\n", text_cleaned);
            av_freep(&text_cleaned);
            continue;
        }

        const int64_t t0_ms = whisper_full_get_segment_t0(wctx->ctx_wsp, i) * 10;
        const int64_t t1_ms = whisper_full_get_segment_t1(wctx->ctx_wsp, i) * 10;
        const int64_t abs_start = timestamp_ms + t0_ms;
        const int64_t abs_end   = timestamp_ms + t1_ms;

        av_log(ctx, AV_LOG_DEBUG, "  [%" PRId64 "-%" PRId64 "]: \"%s\"\n",
               abs_start, abs_end, text_cleaned);

        char *escaped = json_escape(text_cleaned);
        char *entry = av_asprintf("%s{\"s\":%" PRId64 ",\"e\":%" PRId64 ",\"t\":\"%s\"}",
                                  strcmp(segments_json, "[") == 0 ? "" : ",",
                                  abs_start, abs_end, escaped);
        av_freep(&escaped);

        char *new_json = av_asprintf("%s%s", segments_json, entry);
        av_freep(&segments_json);
        av_freep(&entry);
        segments_json = new_json;

        if (wctx->avio_context) {
            const int64_t start_t = abs_start;
            const int64_t end_t = abs_end;
            char *buf = NULL;

            if (!av_strcasecmp(wctx->format, "srt")) {
                buf =
                    av_asprintf
                    ("%d\n%02" PRId64 ":%02" PRId64 ":%02" PRId64 ",%03" PRId64 " --> %02" PRId64 ":%02" PRId64 ":%02" PRId64 ",%03" PRId64 "\n%s\n\n",
                     wctx->index, start_t / 3600000,
                     (start_t / 60000) % 60, (start_t / 1000) % 60,
                     start_t % 1000, end_t / 3600000, (end_t / 60000) % 60,
                     (end_t / 1000) % 60, end_t % 1000, text_cleaned);

                wctx->index++;
            } else if (!av_strcasecmp(wctx->format, "json")) {
                buf = av_asprintf("{\"start\":%" PRId64 ",\"end\":%" PRId64 ",\"text\":\"%s\"}\n", start_t, end_t, text_cleaned);
            } else
                buf = av_asprintf("%s\n", text_cleaned);

            if (buf) {
                avio_write(wctx->avio_context, buf, strlen(buf));
                av_freep(&buf);
            }
        }

        av_freep(&text_cleaned);
    }

    /* Close JSON array */
    char *closed = av_asprintf("%s]", segments_json);
    av_freep(&segments_json);
    segments_json = closed;

    AVDictionary **metadata = &frame->metadata;
    if (metadata && strcmp(segments_json, "[]") != 0) {
        av_dict_set(metadata, "lavfi.whisper.segments", segments_json, 0);
    }
    av_freep(&segments_json);

    if (wctx->audio_buffer_fill_size > samples) {
        memcpy(wctx->audio_buffer, wctx->audio_buffer + samples,
               (wctx->audio_buffer_fill_size - samples) * sizeof(*wctx->audio_buffer));
        wctx->audio_buffer_start_ms += duration * 1000;
    }
    wctx->audio_buffer_fill_size -= samples;
    wctx->audio_buffer_vad_size = wctx->audio_buffer_fill_size;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    WhisperContext *wctx = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];

    const int samples = frame->nb_samples;
    const float *input_data = (const float *) frame->data[0];

#if WHISPER_HAS_THREADS
    ff_mutex_lock(&wctx->infer_mutex);

    /* Harvest previous inference result */
    collect_result_locked(wctx, frame);

    /* If buffer would overflow, try to submit or drop oldest samples */
    if (wctx->audio_buffer_fill_size + samples > wctx->audio_buffer_queue_size) {
        if (!wctx->infer_pending) {
            int64_t fs = populate_vad_segments_locked(ctx, wctx->audio_buffer_fill_size);
            submit_inference_locked(ctx, wctx->audio_buffer_fill_size, fs);
        } else {
            /* Inference still running — wait for it to complete, then submit
             * current buffer.  Blocking here is fine: the caller is either in
             * sync mode or a background lookahead thread, so stalling does not
             * affect real-time playback. */
            while (wctx->infer_pending)
                ff_cond_wait(&wctx->infer_cond, &wctx->infer_mutex);
            collect_result_locked(wctx, frame);
            int64_t fs = populate_vad_segments_locked(ctx, wctx->audio_buffer_fill_size);
            submit_inference_locked(ctx, wctx->audio_buffer_fill_size, fs);
        }
    }

    /* Record start timestamp if buffer was empty */
    if (!wctx->audio_buffer_fill_size)
        wctx->audio_buffer_start_ms = av_rescale_q(frame->pts,
                                                   (AVRational) {1000, 1},
                                                   (AVRational) {inlink->time_base.den, inlink->time_base.num});

    /* Append new samples */
    memcpy(wctx->audio_buffer + wctx->audio_buffer_fill_size, input_data, samples * sizeof(*wctx->audio_buffer));
    wctx->audio_buffer_fill_size += samples;

    /* Check VAD or buffer-full trigger */
    if (!wctx->infer_pending) {
        if (wctx->ctx_vad
            && (wctx->audio_buffer_fill_size - wctx->audio_buffer_vad_size) >=
            av_rescale(wctx->vad_min_speech_duration + wctx->vad_min_silence_duration, WHISPER_SAMPLE_RATE, AV_TIME_BASE)) {
            struct whisper_vad_segments *segments = whisper_vad_segments_from_samples(wctx->ctx_vad,
                                                                                      wctx->vad_params,
                                                                                      wctx->audio_buffer,
                                                                                      wctx->audio_buffer_fill_size);
            wctx->audio_buffer_vad_size = wctx->audio_buffer_fill_size;

            if (!segments) {
                av_log(ctx, AV_LOG_ERROR, "failed to detect VAD\n");
            } else {
                int n_segments = whisper_vad_segments_n_segments(segments);

                if (n_segments > 0) {
                    const float start_ms = whisper_vad_segments_get_segment_t0(segments, 0) * 10.0;
                    const float end_ms = whisper_vad_segments_get_segment_t1(segments, n_segments - 1) * 10.0;
                    int end_pos = (int) (end_ms * WHISPER_SAMPLE_RATE / 1000);

                    if (end_pos <= wctx->audio_buffer_fill_size -
                        av_rescale(wctx->vad_min_silence_duration, WHISPER_SAMPLE_RATE, AV_TIME_BASE)) {
                        av_log(ctx, AV_LOG_INFO,
                                "VAD detected %d segments, start: %.0f ms, end: %.0f ms (buffer: %d ms)\n",
                                n_segments, start_ms, end_ms, 1000 * wctx->audio_buffer_fill_size / WHISPER_SAMPLE_RATE);

                        /* Snapshot every VAD speech window so the inference
                         * thread can later snap each whisper segment to
                         * a real speech range. */
                        if (n_segments > wctx->infer_vad_cap) {
                            int64_t *ns = av_realloc_array(wctx->infer_vad_starts_ms, n_segments, sizeof(*ns));
                            int64_t *ne = av_realloc_array(wctx->infer_vad_ends_ms,   n_segments, sizeof(*ne));
                            if (ns) wctx->infer_vad_starts_ms = ns;
                            if (ne) wctx->infer_vad_ends_ms   = ne;
                            if (ns && ne) wctx->infer_vad_cap = n_segments;
                        }
                        if (wctx->infer_vad_cap >= n_segments) {
                            for (int v = 0; v < n_segments; v++) {
                                wctx->infer_vad_starts_ms[v] = (int64_t)(whisper_vad_segments_get_segment_t0(segments, v) * 10.0);
                                wctx->infer_vad_ends_ms[v]   = (int64_t)(whisper_vad_segments_get_segment_t1(segments, v) * 10.0);
                            }
                            wctx->infer_vad_n = n_segments;
                        } else {
                            wctx->infer_vad_n = 0;
                        }

                        submit_inference_locked(ctx, end_pos, (int64_t)start_ms);
                    }
                }

                whisper_vad_free_segments(segments);
            }
        } else if (wctx->audio_buffer_fill_size >= wctx->audio_buffer_queue_size) {
            /* Buffer-full trigger: continuous speech kept the VAD-edge
             * trigger above from firing.  Run VAD anyway so the inference
             * thread can snap whisper's per-segment timestamps; without
             * VAD info, every subtitle in this 30-second batch falls back
             * to whisper's heuristic t0/t1 which routinely drifts earlier
             * than the actual voice onset. */
            int64_t fs = populate_vad_segments_locked(ctx, wctx->audio_buffer_fill_size);
            submit_inference_locked(ctx, wctx->audio_buffer_fill_size, fs);
        }
    }

    ff_mutex_unlock(&wctx->infer_mutex);

#else /* !WHISPER_HAS_THREADS — synchronous fallback */

    if (wctx->audio_buffer_fill_size + samples > wctx->audio_buffer_queue_size) {
        run_transcription(ctx, frame, wctx->audio_buffer_fill_size);
    }

    if (!wctx->audio_buffer_fill_size)
        wctx->audio_buffer_start_ms = av_rescale_q(frame->pts,
                                                   (AVRational) {1000, 1},
                                                   (AVRational) {inlink->time_base.den, inlink->time_base.num});
    memcpy(wctx->audio_buffer + wctx->audio_buffer_fill_size, input_data, samples * sizeof(*wctx->audio_buffer));
    wctx->audio_buffer_fill_size += samples;

    if (wctx->ctx_vad
        && (wctx->audio_buffer_fill_size - wctx->audio_buffer_vad_size) >=
        av_rescale(wctx->vad_min_speech_duration + wctx->vad_min_silence_duration, WHISPER_SAMPLE_RATE, AV_TIME_BASE)) {
        struct whisper_vad_segments *segments = whisper_vad_segments_from_samples(wctx->ctx_vad,
                                                                                  wctx->vad_params,
                                                                                  wctx->audio_buffer,
                                                                                  wctx->audio_buffer_fill_size);
        wctx->audio_buffer_vad_size = wctx->audio_buffer_fill_size;

        if (!segments) {
            av_log(ctx, AV_LOG_ERROR, "failed to detect VAD\n");
        } else {
            int n_segments = whisper_vad_segments_n_segments(segments);

            if (n_segments > 0) {
                const float start_ms = whisper_vad_segments_get_segment_t0(segments, 0) * 10.0;
                const float end_ms = whisper_vad_segments_get_segment_t1(segments, n_segments - 1) * 10.0;
                int end_pos = (int) (end_ms * WHISPER_SAMPLE_RATE / 1000);

                if (end_pos <= wctx->audio_buffer_fill_size -
                    av_rescale(wctx->vad_min_silence_duration, WHISPER_SAMPLE_RATE, AV_TIME_BASE)) {
                    av_log(ctx, AV_LOG_INFO,
                            "VAD detected %d segments, start: %.0f ms, end: %.0f ms (buffer: %d ms)\n",
                            n_segments, start_ms, end_ms, 1000 * wctx->audio_buffer_fill_size / WHISPER_SAMPLE_RATE);
                    run_transcription(ctx, frame, end_pos);
                }
            }

            whisper_vad_free_segments(segments);
        }
    } else if (wctx->audio_buffer_fill_size >= wctx->audio_buffer_queue_size)
        run_transcription(ctx, frame, wctx->audio_buffer_fill_size);

#endif /* WHISPER_HAS_THREADS */

    wctx->next_pts = frame->pts + av_rescale_q(samples, (AVRational) {
                                               1, inlink->sample_rate}
                                               , inlink->time_base);
    return ff_filter_frame(outlink, frame);
}

static int push_last_frame(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    WhisperContext *wctx = ctx->priv;
    AVFrame *frame;
    int n_out = 1;

    if (ctx->is_disabled)
        return 0;

#if WHISPER_HAS_THREADS
    {
        int has_work;
        ff_mutex_lock(&wctx->infer_mutex);
        has_work = wctx->audio_buffer_fill_size > 0 || wctx->infer_pending || wctx->infer_done;
        ff_mutex_unlock(&wctx->infer_mutex);
        if (!has_work)
            return 0;
    }
#else
    if (wctx->audio_buffer_fill_size == 0)
        return 0;
#endif

    frame = ff_get_audio_buffer(outlink, n_out);
    if (!frame)
        return AVERROR(ENOMEM);

    av_samples_set_silence(frame->extended_data, 0, n_out, frame->ch_layout.nb_channels, frame->format);

    frame->pts = wctx->next_pts;
    if (wctx->next_pts != AV_NOPTS_VALUE)
        wctx->next_pts += av_rescale_q(n_out, (AVRational) {
                                       1, outlink->sample_rate}
                                       , outlink->time_base);

#if WHISPER_HAS_THREADS
    /* At EOF, we can afford to block — wait for any pending inference */
    ff_mutex_lock(&wctx->infer_mutex);
    while (wctx->infer_pending)
        ff_cond_wait(&wctx->infer_cond, &wctx->infer_mutex);

    /* Collect the last async result if any */
    collect_result_locked(wctx, frame);
    ff_mutex_unlock(&wctx->infer_mutex);

    /* Process remaining audio_buffer synchronously */
    if (wctx->audio_buffer_fill_size > 0)
        run_transcription(ctx, frame, wctx->audio_buffer_fill_size);
#else
    run_transcription(ctx, frame, wctx->audio_buffer_fill_size);
#endif

    return ff_filter_frame(outlink, frame);
}

static int activate(AVFilterContext *ctx)
{
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    WhisperContext *wctx = ctx->priv;
    int64_t pts;
    int status;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

#if WHISPER_HAS_THREADS
    /* If async inference produced a result but no fresh input frame is
     * available right now, emit a synthetic 1-sample silent frame just to
     * carry the metadata downstream.  Without this, completed
     * transcriptions sit in infer_result_text until the next audio frame
     * happens to arrive (which can be seconds away when the producer is
     * temporarily idle, e.g. demux cache not yet filled past playback,
     * paused player, or GPU-bound while the rest of the pipeline drained). */
    {
        int have_result;
        ff_mutex_lock(&wctx->infer_mutex);
        have_result = wctx->infer_done && wctx->infer_result_text;
        ff_mutex_unlock(&wctx->infer_mutex);

        if (have_result && !wctx->eof &&
            !ff_inlink_queued_frames(inlink) &&
            ff_outlink_frame_wanted(outlink))
        {
            AVFrame *frame = ff_get_audio_buffer(outlink, 1);
            if (frame) {
                av_samples_set_silence(frame->extended_data, 0, 1,
                                       frame->ch_layout.nb_channels,
                                       frame->format);
                frame->pts = wctx->next_pts;
                if (wctx->next_pts != AV_NOPTS_VALUE)
                    wctx->next_pts += av_rescale_q(1,
                                                   (AVRational){1, outlink->sample_rate},
                                                   outlink->time_base);

                ff_mutex_lock(&wctx->infer_mutex);
                collect_result_locked(wctx, frame);
                ff_mutex_unlock(&wctx->infer_mutex);

                return ff_filter_frame(outlink, frame);
            }
        }
    }
#endif

    if (!wctx->eof && ff_inlink_queued_frames(inlink)) {
        AVFrame *frame = NULL;
        int ret;

        ret = ff_inlink_consume_frame(inlink, &frame);
        if (ret < 0)
            return ret;
        if (ret > 0)
            return filter_frame(inlink, frame);
    }

    if (!wctx->eof && ff_inlink_acknowledge_status(inlink, &status, &pts))
        wctx->eof = status == AVERROR_EOF;

    if (wctx->eof) {
        push_last_frame(outlink);

        ff_outlink_set_status(outlink, AVERROR_EOF, wctx->next_pts);
        return 0;
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);

    return FFERROR_NOT_READY;
}

static int query_formats(const AVFilterContext *ctx,
                         AVFilterFormatsConfig **cfg_in,
                         AVFilterFormatsConfig **cfg_out)
{
    static const enum AVSampleFormat sample_fmts[] = { AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_NONE };
    AVChannelLayout chlayouts[] = { FF_COUNT2LAYOUT(1), { 0 } };
    int sample_rates[] = { WHISPER_SAMPLE_RATE, -1 };
    int ret;

    ret = ff_set_sample_formats_from_list2(ctx, cfg_in, cfg_out, sample_fmts);
    if (ret < 0)
        return ret;

    ret = ff_set_common_channel_layouts_from_list2(ctx, cfg_in, cfg_out, chlayouts);
    if (ret < 0)
        return ret;

    return ff_set_common_samplerates_from_list2(ctx, cfg_in, cfg_out, sample_rates);
}

#define OFFSET(x) offsetof(WhisperContext, x)
#define FLAGS AV_OPT_FLAG_AUDIO_PARAM | AV_OPT_FLAG_FILTERING_PARAM
#define HOURS 3600000000

static const AVOption whisper_options[] = {
    { "model", "Path to the whisper.cpp model file", OFFSET(model_path), AV_OPT_TYPE_STRING,.flags = FLAGS },
    { "language", "Language for transcription ('auto' for auto-detect)", OFFSET(language_str), AV_OPT_TYPE_STRING, {.str = "auto"}, .flags = FLAGS },
    { "translate", "Translate from source language to English", OFFSET(translate), AV_OPT_TYPE_BOOL, {.i64 = 0}, 0, 1, .flags = FLAGS },
    { "queue", "Audio queue size", OFFSET(queue), AV_OPT_TYPE_DURATION, {.i64 = 10000000}, 20000, HOURS, .flags = FLAGS },
    { "use_gpu", "Use GPU for processing", OFFSET(use_gpu), AV_OPT_TYPE_BOOL, {.i64 = 1}, 0, 1, .flags = FLAGS },
    { "gpu_device", "GPU device to use", OFFSET(gpu_device), AV_OPT_TYPE_INT, {.i64 = 0}, 0, INT_MAX, .flags = FLAGS },
    { "backend_path", "Explicit path to a ggml backend dynamic library to load (skips auto-scan)", OFFSET(backend_path), AV_OPT_TYPE_STRING, {.str = NULL}, .flags = FLAGS },
    { "n_processors", "Number of parallel processors for transcription (>1 duplicates the encoder pass and is usually slower; kept for compatibility)", OFFSET(n_processors), AV_OPT_TYPE_INT, {.i64 = 1}, 1, 16, .flags = FLAGS },
    { "destination", "Output destination", OFFSET(destination), AV_OPT_TYPE_STRING, {.str = ""}, .flags = FLAGS },
    { "format", "Output format (text|srt|json)", OFFSET(format), AV_OPT_TYPE_STRING, {.str = "text"},.flags = FLAGS },
    { "max_len", "Max segment length in characters", OFFSET(max_len), AV_OPT_TYPE_INT, {.i64 = 0}, 0, INT_MAX, .flags = FLAGS },
    { "vad_model", "Path to the VAD model file", OFFSET(vad_model_path), AV_OPT_TYPE_STRING,.flags = FLAGS },
    { "vad_threshold", "VAD threshold", OFFSET(vad_threshold), AV_OPT_TYPE_FLOAT, {.dbl = 0.5}, 0.0, 1.0, .flags = FLAGS },
    { "vad_min_speech_duration", "Minimum speech duration for VAD", OFFSET(vad_min_speech_duration), AV_OPT_TYPE_DURATION, {.i64 = 100000}, 20000, HOURS, .flags = FLAGS },
    { "vad_min_silence_duration", "Minimum silence duration for VAD", OFFSET(vad_min_silence_duration), AV_OPT_TYPE_DURATION, {.i64 = 500000}, 0, HOURS, .flags = FLAGS },
    { NULL }
};

static const AVClass whisper_class = {
    .class_name = "whisper",
    .item_name = av_default_item_name,
    .option = whisper_options,
    .version = LIBAVUTIL_VERSION_INT,
};

const FFFilter ff_af_whisper = {
    .p.name = "whisper",
    .p.description = NULL_IF_CONFIG_SMALL("Transcribe audio using whisper.cpp."),
    .p.priv_class = &whisper_class,
    .p.flags = AVFILTER_FLAG_METADATA_ONLY,
    .init = init,
    .uninit = uninit,
    .activate = activate,
    .priv_size = sizeof(WhisperContext),
    FILTER_INPUTS(ff_audio_default_filterpad),
    FILTER_OUTPUTS(ff_audio_default_filterpad),
    FILTER_QUERY_FUNC2(query_formats),
};
