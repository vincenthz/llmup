use llmup_llama_cpp_sys;

use llmup_llama_cpp_sys::llama;

pub struct Sampler {
    pub(crate) ptr: *mut llama::llama_sampler,
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { llama::llama_sampler_free(self.ptr) }
    }
}

impl Sampler {
    pub fn new() -> Self {
        unsafe {
            let params = llama::llama_sampler_chain_default_params();
            let smpl = llama::llama_sampler_chain_init(params);
            llama::llama_sampler_chain_add(smpl, llama::llama_sampler_init_min_p(0.05, 1));
            llama::llama_sampler_chain_add(smpl, llama::llama_sampler_init_temp(0.8));
            llama::llama_sampler_chain_add(smpl, llama::llama_sampler_init_dist(0xFFFFFFFF));
            Self { ptr: smpl }
        }
    }
}
