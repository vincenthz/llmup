use llmup_llama_cpp_sys::llama;
use std::{ffi::CStr, sync::Once};

static LOG_CALLBACK: Once = Once::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogKey {
    KvCache,
    GraphReserve,
    Context,
    ModelLoader,
    CreateTensor,
    CreateMemory,
    GgufInitFile,
    Load,
    LoadTensors,
    InitTokenizer,
    ModelLoad,
    PrintInfo,
    SetAbortCallback,
    RegisterBackend,
    RegisterDevice,
    GgmlMetalDeviceInit,
    GgmlMetalLibraryInit,
    GgmlMetalInit,
    GgmlMetalLibraryCompilePipeline,
    GgmlGallocrReserveN,
    GgmlMetalLogAllocatedSize,
    Unknown,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl std::fmt::Debug for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trace => write!(f, "TRACE"),
            Self::Debug => write!(f, "DEBUG"),
            Self::Info => write!(f, "INFO"),
            Self::Warn => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as std::fmt::Debug>::fmt(self, f)
    }
}

impl From<llama::ggml_log_level> for LogLevel {
    fn from(value: llama::ggml_log_level) -> Self {
        match value {
            llama::ggml_log_level::GGML_LOG_LEVEL_NONE => Self::Trace,
            llama::ggml_log_level::GGML_LOG_LEVEL_DEBUG => Self::Debug,
            llama::ggml_log_level::GGML_LOG_LEVEL_INFO => Self::Info,
            llama::ggml_log_level::GGML_LOG_LEVEL_WARN => Self::Warn,
            llama::ggml_log_level::GGML_LOG_LEVEL_ERROR => Self::Error,
            llama::ggml_log_level::GGML_LOG_LEVEL_CONT => Self::Trace,
            _ => Self::Trace,
        }
    }
}

impl<'a> From<&'a str> for LogKey {
    fn from(value: &'a str) -> Self {
        match value {
            "llama_kv_cache" => LogKey::KvCache,
            "llama_context" => LogKey::Context,
            "llama_model_loader" => LogKey::ModelLoader,
            "llama_model_load" => LogKey::ModelLoad,
            "llama_model_load_from_file_impl" => LogKey::ModelLoad,
            "init_tokenizer" => LogKey::InitTokenizer,
            "gguf_init_from_file_impl" => LogKey::GgufInitFile,
            "ggml_gallocr_reserve_n" => LogKey::GgmlGallocrReserveN,
            "set_abort_callback" => LogKey::SetAbortCallback,
            "load" => LogKey::Load,
            "print_info" => LogKey::PrintInfo,
            "load_tensors" => LogKey::LoadTensors,
            "graph_reserve" => LogKey::GraphReserve,
            "create_tensor" => LogKey::CreateTensor,
            "create_memory" => LogKey::CreateMemory,
            "register_backend" => LogKey::RegisterBackend,
            "register_device" => LogKey::RegisterDevice,
            "ggml_metal_device_init" => LogKey::GgmlMetalDeviceInit,
            "ggml_metal_library_init" => LogKey::GgmlMetalLibraryInit,
            "ggml_metal_init" => LogKey::GgmlMetalInit,
            "ggml_metal_library_compile_pipeline" => LogKey::GgmlMetalLibraryCompilePipeline,
            "ggml_metal_log_allocated_size" => LogKey::GgmlMetalLogAllocatedSize,
            _ => {
                //eprintln!("unknown logkey {}", value);
                LogKey::Unknown
            }
        }
    }
}

unsafe extern "C" fn _llama_bindings_log_callback_internal(
    level: llama::ggml_log_level,
    text: *const ::std::os::raw::c_char,
    user_data: *mut ::std::os::raw::c_void,
) {
    if level == llama::ggml_log_level::GGML_LOG_LEVEL_CONT {
        return;
    }
    let level = LogLevel::from(level);
    let t = unsafe { CStr::from_ptr(text) };
    let s = t.to_string_lossy();
    let x = s.strip_suffix("\n").unwrap_or(&s);
    let f = unsafe { &mut *(user_data as *mut LogCallback) };
    if let Some((cat, content)) = x.split_once(':') {
        let k = LogKey::from(cat);
        f.0(level, k, content)
    } else {
        f.0(level, LogKey::Unknown, x)
    }
}

struct LogCallback<'a>(Box<dyn FnMut(LogLevel, LogKey, &'a str) + Send + 'static>);

pub fn llama_logging<'a, F: 'a>(f: Box<F>)
where
    F: FnMut(LogLevel, LogKey, &'a str) + Send + 'static,
{
    let cb = LogCallback(Box::new(f));
    let boxed = Box::into_raw(Box::new(cb));
    LOG_CALLBACK.call_once(|| unsafe {
        llama::llama_log_set(
            Some(_llama_bindings_log_callback_internal),
            boxed as *mut ::std::os::raw::c_void,
        )
    })
}
