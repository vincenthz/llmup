use llmup_llama_cpp_sys;

use llmup_llama_cpp_sys::llama;

use crate::Context;
use crate::token::Token;

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

    fn sample(&mut self, context: &Context, idx: i32) -> Token {
        Token(unsafe { llama::llama_sampler_sample(self.as_mut(), context.ptr, idx) })
    }
}

pub struct SamplerChain {
    pub(crate) ptr: *mut llama::llama_sampler,
}

impl SamplerChain {
    pub fn new() -> Self {
        unsafe {
            let params = llama::llama_sampler_chain_default_params();
            let smpl = llama::llama_sampler_chain_init(params);
            Self { ptr: smpl }
        }
    }

    pub fn add<S: Sampler>(&mut self, mut s: S) {
        unsafe { llama::llama_sampler_chain_add(self.ptr, s.as_mut()) }
        core::mem::forget(s);
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

pub struct SamplerMirostatV1 {
    ptr: *mut llama::llama_sampler,
}

impl SamplerMirostatV1 {
    pub fn new(n_vocab: i32, seed: u32, tau: f32, eta: f32, m: i32) -> Self {
        unsafe {
            Self {
                ptr: llama::llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m),
            }
        }
    }
}

pub struct SamplerMirostatV2 {
    ptr: *mut llama::llama_sampler,
}

impl SamplerMirostatV2 {
    pub fn new(seed: u32, tau: f32, eta: f32) -> Self {
        unsafe {
            Self {
                ptr: llama::llama_sampler_init_mirostat_v2(seed, tau, eta),
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
impl_sampler!(SamplerMirostatV1);
impl_sampler!(SamplerMirostatV2);
