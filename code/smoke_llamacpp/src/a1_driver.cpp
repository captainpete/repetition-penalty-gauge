// A1 gauge probe in llama.cpp's OWN sampler stack.
//
// Replicates code/run_a1.py: greedy decoding, add a constant gauge shift c to
// EVERY vocab logit BEFORE the repeat penalty, compare c=+5 vs c=-5 token-by-token.
//
// Chain (applied in order by llama_sampler_chain_apply):
//     logit_bias(all n_vocab tokens, bias=c)  ->  penalties(last_n, theta)  ->  greedy
//
// We build cur_p over the FULL vocab ourselves and call llama_sampler_apply +
// llama_sampler_accept manually (not llama_sampler_sample), so (a) the logit-bias
// covers every token and (b) we never hit any backend/pre-sampled-logits path.
//
// Output: JSON {model, max_new, penalty_last_n, records:[{prompt_idx,theta,c,gen_ids}]}.
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

static const char * PROMPTS[] = {
    "The history of the Roman Empire began",
    "Here is a list of my favorite things:",
    "My morning routine is simple. First,",
    "The most important rule of cooking is",
    "In the small town where I grew up,",
    "To whom it may concern, I am writing to",
    "The weather today is",
    "She opened the door and saw",
    "The best advice I ever received was",
    "Once upon a time, in a land far away,",
    "The instructions were clear:",
    "Breaking news this morning:",
    "I have always believed that",
    "The recipe calls for the following ingredients:",
    "After many years of research, scientists have",
    "Dear diary, today was",
};
static const int N_PROMPTS = 16;

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    // add_special=false, parse_special=false  -> matches HF tok(text)["input_ids"] for GPT-2
    int n = -llama_tokenize(vocab, text.c_str(), text.size(), NULL, 0, false, false);
    std::vector<llama_token> toks(n);
    llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), false, false);
    return toks;
}

int main(int argc, char ** argv) {
    std::string model_path, out_path = "a1_raw.json";
    int max_new = 200, ngl = 0, penalty_last_n = 1024;
    std::vector<float> thetas = {1.0f, 1.15f, 1.3f};
    std::vector<float> cs     = {-5.0f, 5.0f};
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m"))              model_path     = argv[++i];
        else if (!strcmp(argv[i], "--out"))           out_path       = argv[++i];
        else if (!strcmp(argv[i], "--max-new"))       max_new        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ngl"))            ngl            = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--penalty-last-n"))penalty_last_n = atoi(argv[++i]);
    }
    if (model_path.empty()) { fprintf(stderr, "need -m model.gguf\n"); return 1; }

    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    FILE * f = fopen(out_path.c_str(), "w");
    fprintf(f, "{\"model\":\"%s\",\"max_new\":%d,\"penalty_last_n\":%d,\"n_vocab\":%d,\"records\":[",
            model_path.c_str(), max_new, penalty_last_n, n_vocab);
    bool first_rec = true;

    for (int pi = 0; pi < N_PROMPTS; pi++) {
        std::vector<llama_token> ptoks = tokenize(vocab, PROMPTS[pi]);
        for (float theta : thetas) {
            for (float c : cs) {
                // fresh context per run for clean KV + deterministic state
                llama_context_params cp = llama_context_default_params();
                cp.n_ctx   = ptoks.size() + max_new + 8;
                cp.n_batch = ptoks.size() + max_new + 8;
                cp.no_perf = true;
                llama_context * ctx = llama_init_from_model(model, cp);

                // sampler chain: logit_bias(all vocab, c) -> penalties(last_n,theta) -> greedy
                std::vector<llama_logit_bias> bias(n_vocab);
                for (int t = 0; t < n_vocab; t++) { bias[t].token = t; bias[t].bias = c; }
                auto sp = llama_sampler_chain_default_params();
                sp.no_perf = true;
                llama_sampler * chain = llama_sampler_chain_init(sp);
                llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n_vocab, n_vocab, bias.data()));
                llama_sampler_chain_add(chain, llama_sampler_init_penalties(penalty_last_n, theta, 0.0f, 0.0f));
                llama_sampler_chain_add(chain, llama_sampler_init_greedy());

                // prime penalty window with the prompt tokens (HF seen = set(prompt_ids))
                for (llama_token t : ptoks) llama_sampler_accept(chain, t);

                // decode the prompt
                llama_batch batch = llama_batch_get_one(ptoks.data(), ptoks.size());
                llama_decode(ctx, batch);

                std::vector<llama_token> gen;
                gen.reserve(max_new);
                std::vector<llama_token_data> cur(n_vocab);
                llama_token next = 0;
                for (int step = 0; step < max_new; step++) {
                    const float * logits = llama_get_logits_ith(ctx, -1);
                    for (int t = 0; t < n_vocab; t++) cur[t] = { t, logits[t], 0.0f };
                    llama_token_data_array cur_p = { cur.data(), (size_t) n_vocab, -1, false };
                    llama_sampler_apply(chain, &cur_p);       // logit_bias -> penalties -> greedy
                    next = cur_p.data[cur_p.selected].id;
                    llama_sampler_accept(chain, next);        // update penalty window
                    gen.push_back(next);
                    // continue past EOG so positions align (original counted a fixed 200)
                    batch = llama_batch_get_one(&next, 1);
                    llama_decode(ctx, batch);
                }

                fprintf(f, "%s{\"prompt_idx\":%d,\"theta\":%.4g,\"c\":%.4g,\"gen_ids\":[",
                        first_rec ? "" : ",", pi, theta, c);
                first_rec = false;
                for (size_t k = 0; k < gen.size(); k++) fprintf(f, "%s%d", k ? "," : "", gen[k]);
                fprintf(f, "]}");

                llama_sampler_free(chain);
                llama_free(ctx);
            }
        }
        fprintf(stderr, "prompt %d/%d done\n", pi + 1, N_PROMPTS);
    }
    fprintf(f, "]}\n");
    fclose(f);
    llama_model_free(model);
    fprintf(stderr, "wrote %s\n", out_path.c_str());
    return 0;
}
