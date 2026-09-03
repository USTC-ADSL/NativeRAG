#include "llama/LlamaRuntime.hpp"

#include <cstdio>
#include <iostream>
#include <mutex>

#include "ggml-backend.h"
#include "llama.h"

namespace mobile_rag {
namespace {

std::mutex g_runtime_mutex;
int g_runtime_users = 0;
ggml_backend_dev_t g_selected_device = nullptr;
ggml_backend_dev_t g_selected_devices[] = {nullptr, nullptr};

#if defined(NATIVERAG_ACCEL_OPENCL)
constexpr const char* kAcceleratorName = "OPENCL";
constexpr const char* kTargetDeviceName = "GPUOpenCL";
constexpr bool kUsesAccelerator = true;
#elif defined(NATIVERAG_ACCEL_HEXAGON)
constexpr const char* kAcceleratorName = "HEXAGON";
constexpr const char* kTargetDeviceName = "HTP0";
constexpr bool kUsesAccelerator = true;
#else
constexpr const char* kAcceleratorName = "CPU";
constexpr const char* kTargetDeviceName = "CPU";
constexpr bool kUsesAccelerator = false;
#endif

const char* device_type_name(enum ggml_backend_dev_type type) {
  switch (type) {
    case GGML_BACKEND_DEVICE_TYPE_CPU:
      return "CPU";
    case GGML_BACKEND_DEVICE_TYPE_GPU:
      return "GPU";
    case GGML_BACKEND_DEVICE_TYPE_IGPU:
      return "IGPU";
    case GGML_BACKEND_DEVICE_TYPE_ACCEL:
      return "ACCEL";
    case GGML_BACKEND_DEVICE_TYPE_META:
      return "META";
  }
  return "UNKNOWN";
}

void llama_log_callback(ggml_log_level level, const char* text, void*) {
  if (level >= GGML_LOG_LEVEL_WARN ||
      (kUsesAccelerator && level >= GGML_LOG_LEVEL_INFO)) {
    std::fputs(text, stderr);
  }
}

}  // namespace

bool LlamaRuntime::acquire() {
  std::lock_guard<std::mutex> lock(g_runtime_mutex);
  if (g_runtime_users > 0) {
    ++g_runtime_users;
    return true;
  }

  llama_log_set(llama_log_callback, nullptr);
  llama_backend_init();

  const size_t device_count = ggml_backend_dev_count();
  std::cout << "[LlamaRuntime] accelerator=" << kAcceleratorName
            << " available_devices=" << device_count << '\n';
  for (size_t i = 0; i < device_count; ++i) {
    ggml_backend_dev_t device = ggml_backend_dev_get(i);
    std::cout << "[LlamaRuntime] device[" << i << "] name="
              << ggml_backend_dev_name(device) << " type="
              << device_type_name(ggml_backend_dev_type(device))
              << " description=" << ggml_backend_dev_description(device)
              << '\n';
  }

  if (kUsesAccelerator) {
    g_selected_device = ggml_backend_dev_by_name(kTargetDeviceName);
  } else {
    g_selected_device =
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  }
  if (!g_selected_device) {
    std::cerr << "[LlamaRuntime] required device '" << kTargetDeviceName
              << "' is unavailable; refusing CPU fallback\n";
    llama_backend_free();
    return false;
  }

  g_selected_devices[0] = g_selected_device;
  g_selected_devices[1] = nullptr;
  g_runtime_users = 1;
  std::cout << "[LlamaRuntime] selected_device="
            << ggml_backend_dev_name(g_selected_device)
            << " description="
            << ggml_backend_dev_description(g_selected_device)
            << " target_device_fallback=disabled\n";
  return true;
}

void LlamaRuntime::release() {
  std::lock_guard<std::mutex> lock(g_runtime_mutex);
  if (g_runtime_users <= 0) {
    return;
  }
  if (--g_runtime_users == 0) {
    g_selected_devices[0] = nullptr;
    g_selected_device = nullptr;
    llama_backend_free();
  }
}

bool LlamaRuntime::apply_model_profile(llama_model_params& params) {
  if (kUsesAccelerator) {
    if (!g_selected_device) {
      std::cerr << "[LlamaRuntime] model profile applied before the target "
                   "device was acquired\n";
      return false;
    }
    params.devices = g_selected_devices;
    params.n_gpu_layers = -1;
    params.split_mode = LLAMA_SPLIT_MODE_NONE;
    params.main_gpu = 0;
#if defined(NATIVERAG_ACCEL_HEXAGON)
    params.use_mmap = false;
#endif
  } else {
    params.devices = nullptr;
    params.n_gpu_layers = 0;
  }
  return true;
}

void LlamaRuntime::apply_context_profile(llama_context_params& params) {
  params.offload_kqv = kUsesAccelerator;
  params.op_offload = kUsesAccelerator;
#if defined(NATIVERAG_ACCEL_OPENCL)
  // Some Adreno drivers register only a subset of flash-attention kernels and
  // explicitly route the missing variants away from OpenCL. Standard
  // attention is slower, but keeps this single-backend profile deterministic.
  params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
#endif
}

const char* LlamaRuntime::accelerator_name() { return kAcceleratorName; }

const char* LlamaRuntime::target_device_name() { return kTargetDeviceName; }

bool LlamaRuntime::uses_accelerator() { return kUsesAccelerator; }

int LlamaRuntime::embedding_batch_limit() {
#if defined(NATIVERAG_ACCEL_OPENCL)
  return 512;
#elif defined(NATIVERAG_ACCEL_HEXAGON)
  return 1024;
#else
  return 2048;
#endif
}

}  // namespace mobile_rag
