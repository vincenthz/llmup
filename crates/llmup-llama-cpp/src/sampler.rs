use llmup_llama_cpp_sys;

use llmup_llama_cpp_sys::llama;

use crate::token::Token;
use crate::{Context, TokenData, TokenDataArray};

pub trait SamplerC {
    unsafe fn as_mut(&mut self) -> *mut llama::llama_sampler;
    unsafe fn as_const(&self) -> *const llama::llama_sampler;

    /*
    fn sample_old(&mut self, context: &Context, idx: i32) -> Token {
        Token(unsafe { llama::llama_sampler_sample(self.as_mut(), context.ptr, idx) })
    }
    */
}

pub trait Sampler {
    fn accept(&mut self, token: Token);
    fn apply(&mut self, array: &mut TokenDataArray);
    fn reset(&mut self);

    fn sample(&mut self, context: &Context, idx: i32) -> Token {
        let vocab = context.model.clone().vocab();
        let mut data = Vec::with_capacity(vocab.n_tokens() as usize);

        let logits = context.get_logits(idx);
        for token in vocab.tokens() {
            data.push(TokenData::new(token, logits[token.as_index()], 0.0))
        }
        let mut array = TokenDataArray {
            data,
            selected: None,
            sorted: false,
        };

        self.apply(&mut array);
        let Some(sel) = array.selected else {
            panic!("sampler has no token selected");
        };

        let token = array.data[sel].id();

        self.accept(token);
        token
    }
}

impl<S: SamplerC> Sampler for S {
    fn accept(&mut self, token: Token) {
        unsafe {
            let m = self.as_mut();
            llama::llama_sampler_accept(m, token.0)
        }
    }

    fn apply(&mut self, array: &mut TokenDataArray) {
        unsafe {
            let m = self.as_mut();
            array.as_mut_ptr(|p| llama::llama_sampler_apply(m, p))
        }
    }

    fn reset(&mut self) {
        unsafe {
            let m = self.as_mut();
            llama::llama_sampler_reset(m)
        }
    }
}

pub trait SamplerRandom: SamplerC {
    fn get_seed(&self) -> u32 {
        unsafe { llama::llama_sampler_get_seed(self.as_const()) }
    }
}

pub struct SamplerChain {
    //pub(crate) ptr: *mut llama::llama_sampler,
    pub(crate) chain: Vec<Box<dyn Sampler>>,
}

impl SamplerChain {
    pub fn new() -> Self {
        Self { chain: Vec::new() }
    }

    pub fn add(&mut self, s: Box<dyn Sampler>) {
        self.chain.push(s)
    }
}

impl Sampler for SamplerChain {
    fn accept(&mut self, token: Token) {
        for sampler in self.chain.iter_mut() {
            sampler.accept(token)
        }
    }

    fn apply(&mut self, array: &mut TokenDataArray) {
        for sampler in self.chain.iter_mut() {
            sampler.apply(array)
        }
    }

    fn reset(&mut self) {
        for sampler in self.chain.iter_mut() {
            sampler.reset()
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
        impl SamplerC for $name {
            unsafe fn as_mut(&mut self) -> *mut llama::llama_sampler {
                self.ptr
            }
            unsafe fn as_const(&self) -> *const llama::llama_sampler {
                self.ptr
            }
        }
    };
}

impl_sampler!(SamplerMinP);
impl_sampler!(SamplerTemperature);
impl_sampler!(SamplerDistance);
impl_sampler!(SamplerMirostatV1);
impl_sampler!(SamplerMirostatV2);

impl SamplerRandom for SamplerMirostatV1 {}
impl SamplerRandom for SamplerMirostatV2 {}
impl SamplerRandom for SamplerDistance {}
