#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-model.h"
#include "json.hpp"

#include <chrono>
#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <algorithm>
#include <cctype>
#include <regex>
#include <iomanip>
#include <unordered_map>

using json = nlohmann::json;

// ========== Utility functions ==========
static char * copy_cstr(const std::string & s) {
    char * p = (char *) std::malloc(s.size() + 1);
    std::memcpy(p, s.data(), s.size());
    p[s.size()] = '\0';
    return p;
}

// Normalize a string (lowercase, strip whitespace/punctuation)
static std::string normalize_answer(const std::string& s) {
    std::string result;
    result.reserve(s.size());
    
    for (char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            result.push_back(std::tolower(static_cast<unsigned char>(c)));
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            if (!result.empty() && result.back() != ' ') {
                result.push_back(' ');
            }
        }
    }
    
    // Strip leading/trailing whitespace
    while (!result.empty() && result.back() == ' ') result.pop_back();
    while (!result.empty() && result.front() == ' ') result.erase(0, 1);
    
    return result;
}

// Compute F1 score (token-based)
static double compute_f1(const std::string& pred, const std::string& gold) {
    std::string norm_pred = normalize_answer(pred);
    std::string norm_gold = normalize_answer(gold);
    
    if (norm_pred.empty() && norm_gold.empty()) return 1.0;
    if (norm_pred.empty() || norm_gold.empty()) return 0.0;
    
    auto tokenize = [](const std::string& s) -> std::vector<std::string> {
        std::vector<std::string> tokens;
        std::istringstream iss(s);
        std::string word;
        while (iss >> word) {
            tokens.push_back(word);
        }
        return tokens;
    };
    
    auto pred_tokens = tokenize(norm_pred);
    auto gold_tokens = tokenize(norm_gold);
    
    std::unordered_map<std::string, int> pred_counts, gold_counts;
    for (const auto& t : pred_tokens) pred_counts[t]++;
    for (const auto& t : gold_tokens) gold_counts[t]++;
    
    int common = 0;
    for (const auto& [word, count] : pred_counts) {
        if (gold_counts.count(word)) {
            common += std::min(count, gold_counts[word]);
        }
    }
    
    if (common == 0) return 0.0;
    
    double precision = (double)common / pred_tokens.size();
    double recall = (double)common / gold_tokens.size();
    double f1 = 2.0 * precision * recall / (precision + recall);
    
    return f1;
}

// Compute Exact Match
static bool compute_em(const std::string& pred, const std::string& gold) {
    return normalize_answer(pred) == normalize_answer(gold);
}

// Max F1/EM across multiple gold answers
static std::pair<double, bool> compute_metrics(const std::string& pred, 
                                                const std::vector<std::string>& golds) {
    double max_f1 = 0.0;
    bool em = false;
    
    for (const auto& gold : golds) {
        double f1 = compute_f1(pred, gold);
        max_f1 = std::max(max_f1, f1);
        if (compute_em(pred, gold)) {
            em = true;
        }
    }
    
    return {max_f1, em};
}

// Build the state filename (includes mzcache_id)
static std::string get_state_name_pattern(llama_model* model, const common_params& params) {
    std::string state_name;
    
    switch (model->arch) {
        case LLM_ARCH_LLAMA:   state_name += "llama3";   break;
        case LLM_ARCH_QWEN3:   state_name += "qwen3";    break;
        case LLM_ARCH_EXAONE4: state_name += "exaone4";  break;
        case LLM_ARCH_GEMMA3:  state_name += "gemma3";   break;
        default:               state_name += "unknown";  break;
    }
    
    state_name += "_" + model->type_name();
    
    if (params.cache_type_k == GGML_TYPE_BF16) {
        state_name += "_bf16";
    } else if (params.cache_type_k == GGML_TYPE_Q8_0) {
        state_name += "_q8_0";
    } else if (params.cache_type_k == GGML_TYPE_Q4_0) {
        state_name += "_q4_0";
    }
    
    if (params.flash_attn) {
        state_name += "_fa";
    }
    
    state_name += "_";  // where the mzcache_id will be appended
    
    return state_name;
}

// Scan all .kv files in the directory and extract the mzcache_id
static std::unordered_map<int, std::string> scan_state_files(
    const std::string& state_dir, 
    const std::string& pattern) {
    
    std::unordered_map<int, std::string> id_to_path;
    
    if (!std::filesystem::exists(state_dir)) {
        std::cerr << "ERROR: State directory does not exist: " << state_dir << "\n";
        return id_to_path;
    }
    
    // Build the regex pattern: pattern + digits + .kv
    std::string regex_pattern = pattern + "(\\d+)\\.kv$";
    std::regex state_regex(regex_pattern);
    
    for (const auto& entry : std::filesystem::directory_iterator(state_dir)) {
        if (!entry.is_regular_file()) continue;
        
        std::string filename = entry.path().filename().string();
        std::smatch match;
        
        if (std::regex_search(filename, match, state_regex)) {
            int mzcache_id = std::stoi(match[1].str());
            id_to_path[mzcache_id] = entry.path().string();
            std::cout << "[Scan] Found state: " << filename 
                      << " -> mzcache_id=" << mzcache_id << "\n";
        }
    }
    
    return id_to_path;
}

// Generation function (greedy decoding)
static std::string generate_answer(llama_context* ctx, 
                                   const llama_vocab* vocab,
                                   llama_sampler* sampler,
                                   int max_tokens = 64) {
    std::string answer;
    
    for (int i = 0; i < max_tokens; ++i) {
        const float* logits = llama_get_logits(ctx);
        if (!logits) break;
        
        llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
        
        if (new_token == llama_vocab_eos(vocab)) {
            break;
        }
        
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            answer.append(buf, n);
        }
        
        llama_batch batch = llama_batch_init(1, 0, 1);
        common_batch_add(batch, new_token, llama_kv_self_used_cells(ctx), {0}, true);
        
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            break;
        }
        llama_batch_free(batch);
    }
    
    while (!answer.empty() && std::isspace(answer.back())) answer.pop_back();
    while (!answer.empty() && std::isspace(answer.front())) answer.erase(0, 1);
    
    return answer;
}

// ========== Main function ==========

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, 
            "Usage: %s <model.gguf> <state_dir> <test_data.json> [max_gen_tokens]\n"
            "Example: %s model.gguf ./states narrativeqa.json 64\n",
            argv[0], argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string state_dir = argv[2];
    const std::string json_path = argv[3];
    const int max_gen_tokens = (argc >= 5) ? std::atoi(argv[4]) : 32;

    // ---- common_params setup ----
    common_params params;
    params.sampling.seed = 1234;
    params.n_ctx = 32768;
    params.n_batch = 32768;
    params.n_ubatch = 2048;
    params.n_predict = max_gen_tokens;
    params.flash_attn = true;
    params.n_gpu_layers = 100;

    {
        std::vector<char*> fixed_argv;
        fixed_argv.push_back(copy_cstr(argv[0]));
        fixed_argv.push_back(copy_cstr(std::string("--model")));
        fixed_argv.push_back(copy_cstr(model_path));

        int fixed_argc = (int)fixed_argv.size();
        if (!common_params_parse(fixed_argc, fixed_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
            std::fprintf(stderr, "ERROR: common_params_parse failed\n");
            return 1;
        }
        for (char *p : fixed_argv) std::free(p);
    }

    common_init();

    // ---- Model/context initialization ----
    auto init_res = common_init_from_params(params);
    llama_model* model = init_res.model.get();
    llama_context* ctx = init_res.context.get();
    
    if (!model || !ctx) {
        std::fprintf(stderr, "ERROR: model/context initialization failed\n");
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    llama_sampler* sampler = llama_sampler_init_greedy();

    // Get the chat template
    const char* tmpl = llama_model_chat_template(model, nullptr);
    if (tmpl) {
        std::cout << "[Info] Using model's chat template\n";
    } else {
        std::cout << "[Info] No chat template found\n";
    }

    // ---- Scan state files ----
    std::string pattern = get_state_name_pattern(model, params);
    std::cout << "[Info] State file pattern: " << pattern << "*.kv\n";
    
    auto id_to_path = scan_state_files(state_dir, pattern);
    std::cout << "[Info] Found " << id_to_path.size() << " state files\n\n";

    if (id_to_path.empty()) {
        std::cerr << "ERROR: No matching state files found in " << state_dir << "\n";
        return 1;
    }

    // ---- Load JSON ----
    std::ifstream fin(json_path);
    if (!fin.good()) {
        std::fprintf(stderr, "ERROR: Cannot open %s\n", json_path.c_str());
        return 1;
    }

    json data;
    try {
        fin >> data;
    } catch (const json::exception& e) {
        std::fprintf(stderr, "ERROR: JSON parse failed: %s\n", e.what());
        return 1;
    }

    if (!data.is_array()) {
        std::fprintf(stderr, "ERROR: JSON root must be an array\n");
        return 1;
    }

    std::cout << "[Info] Loaded " << data.size() << " documents from JSON\n";

    // ---- Evaluation loop ----
    double total_f1 = 0.0;
    int total_em = 0;
    int total_qa = 0;
    int processed_docs = 0;
    
    // Array to store per-document results
    std::vector<json> doc_results;

    for (size_t doc_idx = 0; doc_idx < data.size(); ++doc_idx) {
        const auto& doc_obj = data[doc_idx];
        
        // Check mzcache_id
        if (!doc_obj.contains("mzcache_id")) {
            std::cerr << "WARN: Document " << doc_idx << " missing mzcache_id, skipping\n";
            continue;
        }
        
        int mzcache_id = doc_obj["mzcache_id"].get<int>();
        
        // Find the matching state file
        if (id_to_path.find(mzcache_id) == id_to_path.end()) {
            std::cerr << "WARN: No state file for mzcache_id=" << mzcache_id 
                      << ", skipping\n";
            continue;
        }
        
        std::string state_path = id_to_path[mzcache_id];
        
        std::cout << "\n========== Document " << ++processed_docs 
                  << " [mzcache_id=" << mzcache_id << "] ==========\n";
        std::cout << "[State] Loading: " << state_path << "\n";

        // ---- Load state (for info inspection) ----
        llama_kv_self_clear(ctx);
        
        size_t n_token_count = 0;
        std::vector<llama_token> check_tokens(params.n_ctx);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        if (!llama_state_load_file(ctx, 
                                   state_path.c_str(),
                                   check_tokens.data(),
                                   check_tokens.size(),
                                   &n_token_count)) {
            std::cerr << "ERROR: Failed to load state from " << state_path << "\n";
            continue;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        
        std::cout << "[State] Loaded " << n_token_count << " tokens in " 
                  << load_ms << " ms\n";

        // Statistics for the current document
        double doc_f1_sum = 0.0;
        int doc_em_count = 0;
        int doc_qa_count = 0;

        // ---- Evaluate each question ----
        const auto& qas = doc_obj["qas"];
        if (!qas.is_array()) continue;

        std::cout << "[QA] Processing " << qas.size() << " questions\n\n";

        for (size_t qa_idx = 0; qa_idx < qas.size(); ++qa_idx) {
            const auto& qa = qas[qa_idx];
            
            std::string q_text = qa.value("input", "");
            if (q_text.empty()) continue;

            std::vector<std::string> gold_answers;
            if (qa.contains("answers") && qa["answers"].is_array()) {
                for (const auto& ans : qa["answers"]) {
                    if (ans.is_string()) {
                        gold_answers.push_back(ans.get<std::string>());
                    }
                }
            }
            if (gold_answers.empty()) continue;

            std::cout << "  Q" << qa_idx + 1 << ": " << q_text << "\n";

            // Reload state (start from the initial state for each question)
            // Important: use a fresh buffer each time to avoid contamination from previous data
            llama_kv_self_clear(ctx);
            size_t reloaded = 0;
            std::vector<llama_token> loaded_tokens(params.n_ctx);
            
            if (!llama_state_load_file(ctx,
                                      state_path.c_str(),
                                      loaded_tokens.data(),
                                      loaded_tokens.size(),
                                      &reloaded)) {
                std::cerr << "    ERROR: Failed to reload state\n";
                continue;
            }
            
            // Debug: check KV cache state after loading state
            int32_t kv_used = llama_kv_self_used_cells(ctx);
            std::cout << "    [DEBUG] After state load: reloaded=" << reloaded 
                      << ", kv_used=" << kv_used << "\n";
            
            // Print the first 5 tokens
            std::cout << "    [DEBUG] First 5 loaded tokens: ";
            for (size_t i = 0; i < std::min((size_t)5, reloaded); ++i) {
                std::cout << loaded_tokens[i] << " ";
            }
            std::cout << "\n";

            // Apply the chat template
            std::string prompt;
            if (tmpl) {
                std::cout << "    [Info] Applying chat template\n";
                std::vector<llama_chat_message> messages;
                
                // std::string sys_msg = "You are a helpful assistant. Answer the question based on the given context. Provide a concise, direct answer without explanations.";
                // messages.push_back({"system", sys_msg.c_str()});
                
                std::string user_msg = "Based on the context you have read, answer this question: " + q_text + "\n\nProvide only the answer, nothing else.";
                messages.push_back({"user", user_msg.c_str()});
                
                std::vector<char> formatted(8192);
                int new_len = llama_chat_apply_template(
                    tmpl, 
                    messages.data(), 
                    messages.size(), 
                    true,
                    formatted.data(), 
                    formatted.size()
                );
                
                if (new_len > (int)formatted.size()) {
                    formatted.resize(new_len);
                    new_len = llama_chat_apply_template(
                        tmpl, 
                        messages.data(), 
                        messages.size(), 
                        true, 
                        formatted.data(), 
                        formatted.size()
                    );
                }
                
                if (new_len < 0) {
                    std::cerr << "    ERROR: Failed to apply chat template\n";
                    continue;
                }
                
                prompt = std::string(formatted.begin(), formatted.begin() + new_len);
                
                if (prompt.find("<think>") == std::string::npos) {
                    prompt += "<think>\n\n</think>\n\n";
                }
            } else {
                prompt = "\n\n### Question:\n" + q_text + "\n\n### Answer:\n<think>\n\n</think>\n\n";
            }
            
            // Debug: print the prompt
            std::cout << "    [DEBUG] Prompt:\n---\n" << prompt << "\n---\n";

            // Tokenize the question
            int n_q_tokens = -llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                            nullptr, 0, false, true);
            
            if (n_q_tokens <= 0 || (int)reloaded + n_q_tokens > params.n_ctx - max_gen_tokens) {
                std::cerr << "    ERROR: Question too long (" << n_q_tokens << " tokens)\n";
                continue;
            }

            std::vector<llama_token> q_tokens(n_q_tokens);
            llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                          q_tokens.data(), n_q_tokens, false, true);
            
            // Debug: question token info
            std::cout << "    [DEBUG] Question tokens: " << n_q_tokens 
                      << ", position range: " << reloaded << " to " << (reloaded + n_q_tokens - 1) << "\n";
            std::cout << "    [DEBUG] First 5 question tokens: ";
            for (int i = 0; i < std::min(5, n_q_tokens); ++i) {
                std::cout << q_tokens[i] << " ";
            }
            std::cout << "\n";

            // Decode the question
            llama_batch q_batch = llama_batch_init(n_q_tokens, 0, 1);
            for (int i = 0; i < n_q_tokens; ++i) {
                common_batch_add(q_batch, q_tokens[i], reloaded + i, {0}, (i == n_q_tokens - 1));
            }

            if (llama_decode(ctx, q_batch) != 0) {
                std::cerr << "    ERROR: Question decode failed\n";
                llama_batch_free(q_batch);
                continue;
            }
            llama_batch_free(q_batch);
            
            // Debug: KV state after decoding the question
            int32_t kv_after_q = llama_kv_self_used_cells(ctx);
            std::cout << "    [DEBUG] After question decode: kv_used=" << kv_after_q << "\n";

            // Generate the answer
            std::string pred = generate_answer(ctx, vocab, sampler, max_gen_tokens);
            
            // Debug: KV state after generation
            int32_t kv_after_gen = llama_kv_self_used_cells(ctx);
            std::cout << "    [DEBUG] After generation: kv_used=" << kv_after_gen << "\n";
            
            // Compute metrics
            auto [f1, em] = compute_metrics(pred, gold_answers);
            
            total_f1 += f1;
            total_em += em ? 1 : 0;
            total_qa++;
            
            doc_f1_sum += f1;
            doc_em_count += em ? 1 : 0;
            doc_qa_count++;

            std::cout << "    Pred: [" << pred << "]\n";
            std::cout << "    Gold: [" << gold_answers[0] << "]";
            if (gold_answers.size() > 1) {
                std::cout << " (+" << (gold_answers.size() - 1) << " more)";
            }
            std::cout << "\n";
            std::cout << "    F1: " << std::fixed << std::setprecision(2) << (f1 * 100) 
                      << "%, EM: " << (em ? "✓" : "✗") << "\n\n";
        }
        
        // Store per-document results
        if (doc_qa_count > 0) {
            json doc_result = {
                {"doc_index", processed_docs},
                {"mzcache_id", mzcache_id},
                {"token_count", n_token_count},
                {"num_qas", doc_qa_count},
                {"avg_f1", doc_f1_sum / doc_qa_count},
                {"em_count", doc_em_count},
                {"em_rate", doc_em_count * 1.0 / doc_qa_count}
            };
            doc_results.push_back(doc_result);
            
            std::cout << "[Doc Summary] Tokens: " << n_token_count 
                      << ", Avg F1: " << std::fixed << std::setprecision(2) 
                      << (doc_f1_sum / doc_qa_count * 100) << "%, EM: " 
                      << doc_em_count << "/" << doc_qa_count << "\n";
        }
    }
    
    // ---- Final results ----
    std::cout << "\n========== Final Results ==========\n";
    std::cout << "Processed documents: " << processed_docs << "\n";
    std::cout << "Total QA pairs: " << total_qa << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Average F1: " << (total_qa > 0 ? (total_f1 / total_qa * 100) : 0) << "%\n";
    std::cout << "Exact Match: " << total_em << "/" << total_qa 
              << " (" << (total_qa > 0 ? (total_em * 100.0 / total_qa) : 0) << "%)\n";

    // Build per-document EM count and F1 score lists
    std::vector<int> em_counts_per_doc;
    std::vector<double> f1_scores_per_doc;
    
    for (const auto& doc : doc_results) {
        em_counts_per_doc.push_back(doc["em_count"]);
        f1_scores_per_doc.push_back(doc["avg_f1"]);
    }
    
    // Save results as JSON
    json result = {
        {"summary", {
            {"processed_docs", processed_docs},
            {"total_qa", total_qa},
            {"avg_f1", total_qa > 0 ? (total_f1 / total_qa) : 0},
            {"em_count", total_em},
            {"em_rate", total_qa > 0 ? (total_em * 1.0 / total_qa) : 0},
            {"em_counts_per_doc", em_counts_per_doc},
            {"f1_scores_per_doc", f1_scores_per_doc}
        }},
        {"per_document_results", doc_results}
    };
    
    std::ofstream result_file("mz_load_state_accuracy_results.json");
    result_file << result.dump(2);
    std::cout << "\n[Info] Results saved to mz_load_state_accuracy_results.json\n";

    // Cleanup
    llama_sampler_free(sampler);

    return 0;
}