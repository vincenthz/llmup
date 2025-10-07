use llmup_llama_cpp_sys;

use llmup_llama_cpp_sys::llama;

use crate::Context;
use crate::token::Token;

pub struct SamplerChain {
    pub(crate) ptr: *mut llama::llama_sampler,
}

pub trait Sampler {
    unsafe fn as_mut(&mut self) -> *mut llama::llama_sampler;

    fn accept(&mut self, token: Token) {
        unsafe {
            let m = self.as_mut();
            llama::llama_sampler_accept(m, token.0)
        }
    }

    fn apply(&mut self) {
        unsafe {
            let m = self.as_mut();
            llama::llama_sampler_apply(m, core::ptr::null_mut())
        }
    }

    fn reset(&mut self) {
        unsafe {
            let m = self.as_mut();
            llama::llama_sampler_reset(m)
        }
    }

    fn sample(&mut self, context: &Context) -> Token {
        Token(unsafe { llama::llama_sampler_sample(self.as_mut(), context.ptr, -1) })
    }
}

impl SamplerChain {
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

pub struct SamplerMinP {
    ptr: *mut llama::llama_sampler,
}

impl SamplerMinP {
    pub fn new(p: f32, min_keep: usize) -> Self {
        unsafe {
            Self {
                ptr: llama::llama_sampler_init_min_p(p, min_keep),
            }
        }
    }
}

pub struct SamplerTemperature {
    ptr: *mut llama::llama_sampler,
}

impl SamplerTemperature {
    pub fn new(temp: f32) -> Self {
        unsafe {
            Self {
                ptr: llama::llama_sampler_init_temp(temp),
            }
        }
    }
}

pub struct SamplerDistance {
    ptr: *mut llama::llama_sampler,
}

impl SamplerDistance {
    pub fn new(distance: u32) -> Self {
        unsafe {
            Self {
                ptr: llama::llama_sampler_init_dist(distance),
            }
        }
    }
}

macro_rules! impl_sampler {
    ($name:ident) => {
        impl Drop for $name {
            fn drop(&mut self) {
                unsafe { llama::llama_sampler_free(self.ptr) }
            }
        }
        impl Sampler for $name {
            unsafe fn as_mut(&mut self) -> *mut llama::llama_sampler {
                self.ptr
            }
        }
    };
}

impl_sampler!(SamplerMinP);
impl_sampler!(SamplerChain);
impl_sampler!(SamplerTemperature);
impl_sampler!(SamplerDistance);
