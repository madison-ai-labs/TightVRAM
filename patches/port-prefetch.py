#!/usr/bin/env python3
"""Port the scheduler part of thecodacus/llama.cpp fable5/prefetch-experts (5e7f627) onto llama.cpp 45b6604b2.

Input:  a llama.cpp worktree at that commit; the path to ggml/src/ggml-backend.cpp.
Output: the file is patched in place (prints "ok"; aborts if an anchor is not found exactly once).
Run (inside the worktree): python3 port-prefetch.py ggml/src/ggml-backend.cpp

Upstream has since moved the input copy into ggml_backend_sched_copy_input (with callback_copy for the
selective expert copy from llama-context). The prefetch path therefore sits in the host-weights loop of
compute_splits and bypasses copy_input and its callback (no ids readback). The logic is otherwise unchanged:
GGML_SCHED_PREFETCH_EXPERTS=N (N<=1 -> 3 slots, max 8), slots sized for the largest expert tensor, a second
backend instance on the same device, ready/free events per slot, only for splits whose first node is MUL_MAT_ID
and n_used*n_tokens >= 2*n_expert. Addition: GGML_SCHED_PREFETCH_DEBUG=1 reports slot size/count once."""
import sys

p = sys.argv[1]
s = open(p).read()


def replace_anchor(alt, neu, n=1):
    global s
    if s.count(alt) != n:
        raise SystemExit(f"expected anchor {n}x, found {s.count(alt)}x:\n{alt}")
    s = s.replace(alt, neu)


replace_anchor("""#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif
""", """#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

#ifndef GGML_SCHED_MAX_PREFETCH_SLOTS
#define GGML_SCHED_MAX_PREFETCH_SLOTS 8
#endif
""")

replace_anchor("""    bool op_offload;

    int debug;
""", """    bool op_offload;

    // full-tensor prefetch of offloaded MUL_MAT_ID weights (GGML_SCHED_PREFETCH_EXPERTS), port of
    // thecodacus/llama.cpp fable5/prefetch-experts: with a large batch virtually every expert is used, so the
    // routing ids are not worth waiting for; uploads run through a second backend instance on the same device
    bool prefetch_experts;
    bool prefetch_debug;
    ggml_backend_t prefetch_backend;
    int prefetch_n_slots;
    ggml_backend_buffer_t prefetch_slots[GGML_SCHED_MAX_PREFETCH_SLOTS];
    ggml_backend_event_t prefetch_ready[GGML_SCHED_MAX_PREFETCH_SLOTS];
    ggml_backend_event_t prefetch_free[GGML_SCHED_MAX_PREFETCH_SLOTS];
    bool prefetch_used[GGML_SCHED_MAX_PREFETCH_SLOTS];
    int prefetch_cur;

    int debug;
""")

replace_anchor("""static bool ggml_backend_sched_is_host_weight(const struct ggml_tensor * t) {
    return t->buffer != NULL &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}
""", """static bool ggml_backend_sched_is_host_weight(const struct ggml_tensor * t) {
    return t->buffer != NULL &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

static void ggml_backend_sched_prefetch_disable(ggml_backend_sched_t sched, ggml_backend_t split_backend) {
    sched->prefetch_experts = false;
    if (sched->prefetch_backend) {
        ggml_backend_synchronize(split_backend);
        ggml_backend_synchronize(sched->prefetch_backend);
    }
    for (int i = 0; i < sched->prefetch_n_slots; i++) {
        ggml_backend_buffer_free(sched->prefetch_slots[i]);
        sched->prefetch_slots[i] = NULL;
        sched->prefetch_used[i] = false;
    }
}

// slots are sized once for the largest offloaded expert tensor in the current graph so that they never need to
// grow mid-eval
static size_t ggml_backend_sched_prefetch_max_size(ggml_backend_sched_t sched) {
    size_t max_size = 0;
    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &sched->splits[split_id];
        if (split->graph.n_nodes == 0 || split->graph.nodes[0]->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            const struct ggml_tensor * input = split->inputs[input_id];
            if (ggml_backend_sched_is_host_weight(input)) {
                max_size = std::max(max_size, ggml_nbytes(input));
            }
        }
    }
    return max_size;
}

static bool ggml_backend_sched_prefetch_init(ggml_backend_sched_t sched, ggml_backend_t split_backend, size_t size) {
    if (sched->prefetch_backend == NULL) {
        ggml_backend_dev_t dev = split_backend->device;
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.async || !props.caps.events) {
            sched->prefetch_experts = false;
            return false;
        }
        sched->prefetch_backend = ggml_backend_dev_init(dev, NULL);
        if (sched->prefetch_backend == NULL) {
            sched->prefetch_experts = false;
            return false;
        }
        for (int i = 0; i < sched->prefetch_n_slots; i++) {
            sched->prefetch_ready[i] = ggml_backend_event_new(dev);
            sched->prefetch_free[i]  = ggml_backend_event_new(dev);
            if (sched->prefetch_ready[i] == NULL || sched->prefetch_free[i] == NULL) {
                sched->prefetch_experts = false;
                return false;
            }
        }
    }

    size = std::max(size, ggml_backend_sched_prefetch_max_size(sched));

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(split_backend);
    for (int i = 0; i < sched->prefetch_n_slots; i++) {
        if (sched->prefetch_slots[i] == NULL || ggml_backend_buffer_get_size(sched->prefetch_slots[i]) < size) {
            // allocate before freeing so a failure leaves the old slot intact
            ggml_backend_buffer_t new_buf = ggml_backend_buft_alloc_buffer(buft, size);
            if (new_buf == NULL) {
                // overlap needs at least 2 slots, otherwise run with what fits
                if (i >= 2 && sched->prefetch_slots[0] != NULL &&
                    ggml_backend_buffer_get_size(sched->prefetch_slots[0]) >= size) {
                    sched->prefetch_n_slots = i;
                    sched->prefetch_cur = 0;
                    return true;
                }
                ggml_backend_sched_prefetch_disable(sched, split_backend);
                return false;
            }
            if (sched->prefetch_slots[i] != NULL) {
                ggml_backend_synchronize(split_backend);
                ggml_backend_synchronize(sched->prefetch_backend);
                ggml_backend_buffer_free(sched->prefetch_slots[i]);
            }
            sched->prefetch_slots[i] = new_buf;
            sched->prefetch_used[i] = false;
            if (sched->prefetch_debug) {
                GGML_LOG_INFO("%s: prefetch slot %d: %.1f MiB\\n", __func__, i, size / 1024.0 / 1024.0);
            }
        }
    }
    return true;
}

// returns the slot used, or -1 if the regular copy path has to be taken
static int ggml_backend_sched_prefetch_input(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split,
        struct ggml_tensor * input, struct ggml_tensor ** saved_cpy, ggml_backend_buffer_t * saved_buffer, void ** saved_data) {
    if (!sched->prefetch_experts || sched->callback_eval || split->graph.n_nodes == 0) {
        return -1;
    }
    const int split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
    struct ggml_tensor * node = split->graph.nodes[0];
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] != input_cpy) {
        return -1;
    }
    const struct ggml_tensor * ids = node->src[2];
    const int64_t n_expert = input->ne[2];
    if (ids->ne[0]*ids->ne[1] < 2*n_expert || !ggml_backend_sched_prefetch_init(sched, split_backend, ggml_nbytes(input))) {
        return -1;
    }
    const int slot = sched->prefetch_cur;
    sched->prefetch_cur = (sched->prefetch_cur + 1) % sched->prefetch_n_slots;
    // wait for the previous user of this slot to finish computing
    if (sched->prefetch_used[slot]) {
        ggml_backend_event_wait(sched->prefetch_backend, sched->prefetch_free[slot]);
    }
    // point the staging copy at the slot only for the duration of this split
    *saved_cpy    = input_cpy;
    *saved_buffer = input_cpy->buffer;
    *saved_data   = input_cpy->data;
    input_cpy->buffer = sched->prefetch_slots[slot];
    input_cpy->data   = ggml_backend_buffer_get_base(sched->prefetch_slots[slot]);
    ggml_backend_tensor_set_async(sched->prefetch_backend, input_cpy, input->data, 0, ggml_nbytes(input));
    ggml_backend_event_record(sched->prefetch_ready[slot], sched->prefetch_backend);
    ggml_backend_event_wait(split_backend, sched->prefetch_ready[slot]);
    return slot;
}
""")

replace_anchor("""        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            if (ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id]);
            }
        }

        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
""", """        int split_prefetch_slot = -1;
        struct ggml_tensor * prefetch_input_cpy = NULL;
        ggml_backend_buffer_t prefetch_saved_buffer = NULL;
        void * prefetch_saved_data = NULL;
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            if (ggml_backend_sched_is_host_weight(split->inputs[input_id])) {
                if (split_prefetch_slot == -1) {
                    split_prefetch_slot = ggml_backend_sched_prefetch_input(sched, split, split->inputs[input_id],
                            &prefetch_input_cpy, &prefetch_saved_buffer, &prefetch_saved_data);
                    if (split_prefetch_slot != -1) {
                        continue;
                    }
                }
                ggml_backend_sched_copy_input(sched, split, split->inputs[input_id]);
            }
        }

        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (split_prefetch_slot != -1) {
                // the kernels have captured the slot address at launch, safe to restore
                ggml_backend_event_record(sched->prefetch_free[split_prefetch_slot], split_backend);
                sched->prefetch_used[split_prefetch_slot] = true;
                prefetch_input_cpy->buffer = prefetch_saved_buffer;
                prefetch_input_cpy->data   = prefetch_saved_data;
            }
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
""")

replace_anchor("""    sched->op_offload = op_offload;
""", """    sched->op_offload = op_offload;

    // GGML_SCHED_PREFETCH_EXPERTS=1 enables the default slot count (3: gate/up/down of one MoE layer), higher
    // values set it directly; each slot costs one max-sized expert tensor of device memory
    {
        const char * env = getenv("GGML_SCHED_PREFETCH_EXPERTS");
        const int n = env ? atoi(env) : 0;
        sched->prefetch_experts = op_offload && n > 0;
        sched->prefetch_n_slots = n <= 1 ? 3 : std::min(n, GGML_SCHED_MAX_PREFETCH_SLOTS);
        const char * dbg = getenv("GGML_SCHED_PREFETCH_DEBUG");
        sched->prefetch_debug = dbg && atoi(dbg) > 0;
        if (sched->prefetch_experts) {
            GGML_LOG_INFO("%s: expert prefetch on, %d slots\\n", __func__, sched->prefetch_n_slots);
        }
    }
""")

replace_anchor("""            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
""", """            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    if (sched->prefetch_backend) {
        ggml_backend_synchronize(sched->prefetch_backend);
        // the slot count may have been reduced after a failed allocation, free everything
        for (int i = 0; i < GGML_SCHED_MAX_PREFETCH_SLOTS; i++) {
            ggml_backend_event_free(sched->prefetch_ready[i]);
            ggml_backend_event_free(sched->prefetch_free[i]);
            ggml_backend_buffer_free(sched->prefetch_slots[i]);
        }
        ggml_backend_free(sched->prefetch_backend);
    }
    ggml_gallocr_free(sched->galloc);
""")

replace_anchor("""    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {""", """    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (sched->prefetch_backend) {
        ggml_backend_synchronize(sched->prefetch_backend);
    }
    if (!sched->is_alloc) {""")

open(p, "w").write(s)
print("ok")
