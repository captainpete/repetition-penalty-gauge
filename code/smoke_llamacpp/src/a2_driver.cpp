// A2 JSON-validity probe in llama.cpp's OWN repeat_penalty sampler.
//
// Replicates code/run_a2_downstream.py JSON task: greedy, stock repeat_penalty,
// Qwen2.5-Coder-7B. Chain: penalties(last_n, theta) -> greedy. We emit the
// generated TEXT so the VERBATIM first_json + json_valid validators (copied into
// the Python analysis script) score it, exactly as in the HF run.
//
// Runs both seen-set conditions in one process (holds the GPU lock once):
//   (a) penalty_last_n = whole context   (matches HF all-seen semantics)
//   (b) penalty_last_n = 64              (llama.cpp default)
//
// Output JSON: {model, records:[{schema_idx,theta,penalty_last_n,rep_idx,text}]}.
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

// The 6 JSON_TASKS descriptions, VERBATIM from run_a2_downstream.py.
static const char * DESCS[] = {
    "a user with fields name (string), age (integer), email (string)",
    "a product with fields title (string), price (number), in_stock (boolean)",
    "a book with fields title (string), author (string), year (integer), tags (array of strings)",
    "a city with fields name (string), population (integer), country (string), capital (boolean)",
    "an event with fields name (string), date (string), attendees (integer), virtual (boolean)",
    "a car with fields make (string), model (string), year (integer), electric (boolean)",
};
static const int N_SCHEMAS = 6;

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = -llama_tokenize(vocab, text.c_str(), text.size(), NULL, 0, true, true);
    std::vector<llama_token> toks(n);
    llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), true, true);
    return toks;
}

static std::string piece(const llama_vocab * vocab, llama_token t) {
    char buf[256];
    int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
    if (n < 0) return "";
    return std::string(buf, n);
}

int main(int argc, char ** argv) {
    std::string model_path, out_path = "a2_raw.json";
    int max_new = 160, ngl = 99, json_reps = 8;
    std::vector<float> thetas = {1.0f, 1.1f, 1.3f};
    std::vector<int>   conds  = {1024, 64};   // penalty_last_n: whole-context, default
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m"))        model_path = argv[++i];
        else if (!strcmp(argv[i], "--out"))     out_path   = argv[++i];
        else if (!strcmp(argv[i], "--max-new")) max_new    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ngl"))      ngl        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps"))    json_reps  = atoi(argv[++i]);
    }
    if (model_path.empty()) { fprintf(stderr, "need -m model.gguf\n"); return 1; }

    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const char * STOPS[] = {"\n\n", "```"};

    FILE * f = fopen(out_path.c_str(), "w");
    fprintf(f, "{\"model\":\"%s\",\"max_new\":%d,\"json_reps\":%d,\"records\":[",
            model_path.c_str(), max_new, json_reps);
    bool first_rec = true;

    for (int si = 0; si < N_SCHEMAS; si++) {
        std::string prompt = std::string("Output ONLY a single JSON object describing ")
                             + DESCS[si] + ". JSON:\n";
        std::vector<llama_token> ptoks = tokenize(vocab, prompt);
        for (float theta : thetas) {
            for (int last_n : conds) {
                for (int rep = 0; rep < json_reps; rep++) {
                    llama_context_params cp = llama_context_default_params();
                    cp.n_ctx   = ptoks.size() + max_new + 8;
                    cp.n_batch = ptoks.size() + max_new + 8;
                    cp.no_perf = true;
                    llama_context * ctx = llama_init_from_model(model, cp);

                    auto sp = llama_sampler_chain_default_params();
                    sp.no_perf = true;
                    llama_sampler * chain = llama_sampler_chain_init(sp);
                    llama_sampler_chain_add(chain, llama_sampler_init_penalties(last_n, theta, 0.0f, 0.0f));
                    llama_sampler_chain_add(chain, llama_sampler_init_greedy());

                    // prime penalty window with prompt tokens (HF unique(input_ids) includes prompt)
                    for (llama_token t : ptoks) llama_sampler_accept(chain, t);

                    llama_batch batch = llama_batch_get_one(ptoks.data(), ptoks.size());
                    llama_decode(ctx, batch);

                    std::string text;
                    std::vector<llama_token_data> cur(n_vocab);
                    for (int step = 0; step < max_new; step++) {
                        const float * logits = llama_get_logits_ith(ctx, -1);
                        for (int t = 0; t < n_vocab; t++) cur[t] = { t, logits[t], 0.0f };
                        llama_token_data_array cur_p = { cur.data(), (size_t) n_vocab, -1, false };
                        llama_sampler_apply(chain, &cur_p);
                        llama_token next = cur_p.data[cur_p.selected].id;
                        llama_sampler_accept(chain, next);
                        if (llama_vocab_is_eog(vocab, next)) break;
                        text += piece(vocab, next);
                        bool stop = false;
                        for (const char * s : STOPS) if (text.find(s) != std::string::npos) stop = true;
                        if (stop) break;
                        batch = llama_batch_get_one(&next, 1);
                        llama_decode(ctx, batch);
                    }

                    // JSON-escape the text
                    fprintf(f, "%s{\"schema_idx\":%d,\"theta\":%.4g,\"penalty_last_n\":%d,\"rep_idx\":%d,\"text\":\"",
                            first_rec ? "" : ",", si, theta, last_n, rep);
                    first_rec = false;
                    for (char ch : text) {
                        switch (ch) {
                            case '"':  fputs("\\\"", f); break;
                            case '\\': fputs("\\\\", f); break;
                            case '\n': fputs("\\n", f);  break;
                            case '\r': fputs("\\r", f);  break;
                            case '\t': fputs("\\t", f);  break;
                            default:
                                if ((unsigned char) ch < 0x20) fprintf(f, "\\u%04x", ch);
                                else fputc(ch, f);
                        }
                    }
                    fprintf(f, "\"}");

                    llama_sampler_free(chain);
                    llama_free(ctx);
                }
            }
        }
        fprintf(stderr, "schema %d/%d done\n", si + 1, N_SCHEMAS);
    }
    fprintf(f, "]}\n");
    fclose(f);
    llama_model_free(model);
    fprintf(stderr, "wrote %s\n", out_path.c_str());
    return 0;
}
