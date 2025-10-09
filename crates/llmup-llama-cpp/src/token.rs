#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Token(pub(crate) i32);

impl Token {
    pub fn as_index(self) -> usize {
        self.0 as usize
    }
}
