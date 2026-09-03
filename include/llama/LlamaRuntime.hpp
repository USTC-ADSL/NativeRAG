#pragma once

#include "llama.h"

namespace mobile_rag {

class LlamaRuntime {
 public:
  static bool acquire();
  static void release();

  static bool apply_model_profile(llama_model_params& params);
  static void apply_context_profile(llama_context_params& params);
  static const char* accelerator_name();
  static const char* target_device_name();
  static bool uses_accelerator();
  static int embedding_batch_limit();
};

}  // namespace mobile_rag
