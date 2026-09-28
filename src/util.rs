use std::fmt;

use typed_index_collections::TiEnumerated;

pub type TiVec<K, V> = typed_index_collections::TiVec<K, V>;
pub type TiSlice<K, V> = typed_index_collections::TiSlice<K, V>;
pub type HashMap<K, V> = indexmap::IndexMap<K, V>;
pub type HashSet<K> = indexmap::IndexSet<K>;

pub fn log_dir() -> Option<String> {
    std::env::var("SILVER_OXIDE_VIZ")
        .ok()
        .filter(|s| !s.is_empty())
}

// fmt

pub struct Brackets<T, I: IntoIterator<Item = T> + Copy>(char, I, char);

impl<'a, K: From<usize>, V> IntoIterator for PairAdapter<'a, K, V> {
    type Item = Pair<'a, K, V>;
    type IntoIter = std::iter::Map<
        TiEnumerated<core::slice::Iter<'a, V>, K, &'a V>,
        fn((K, &'a V)) -> Self::Item,
    >;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_enumerated().map(Pair)
    }
}

pub trait AsBrackets<'a, T> {
    fn parenthesised(&'a self) -> Brackets<T, impl IntoIterator<Item = T> + Copy>;
    fn bracketed(&'a self) -> Brackets<T, impl IntoIterator<Item = T> + Copy>;
}

impl<'a, T> AsBrackets<'a, &'a T> for [T] {
    fn parenthesised(&'a self) -> Brackets<&'a T, impl IntoIterator<Item = &'a T> + Copy> {
        Brackets('(', self, ')')
    }
    fn bracketed(&'a self) -> Brackets<&'a T, impl IntoIterator<Item = &'a T> + Copy> {
        Brackets('[', self, ']')
    }
}

impl<'a, K: From<usize>, V> AsBrackets<'a, Pair<'a, K, V>> for TiSlice<K, V> {
    fn parenthesised(
        &'a self,
    ) -> Brackets<Pair<'a, K, V>, impl IntoIterator<Item = Pair<'a, K, V>> + Copy> {
        Brackets('(', PairAdapter(self), ')')
    }
    fn bracketed(
        &'a self,
    ) -> Brackets<Pair<'a, K, V>, impl IntoIterator<Item = Pair<'a, K, V>> + Copy> {
        Brackets('[', PairAdapter(self), ']')
    }
}

// fmt

impl<T, I: IntoIterator<Item = T> + Copy> Brackets<T, I> {
    pub fn print(
        &self,
        f: &mut fmt::Formatter<'_>,
        fmt: impl Fn(&T, &mut fmt::Formatter<'_>) -> fmt::Result,
    ) -> fmt::Result {
        write!(f, "{}", self.0)?;
        for (i, item) in self.1.into_iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            fmt(&item, f)?;
        }
        write!(f, "{}", self.2)
    }
}

impl<T: fmt::Debug, I: IntoIterator<Item = T> + Copy> fmt::Debug for Brackets<T, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.print(f, fmt::Debug::fmt)
    }
}

impl<T: fmt::Display, I: IntoIterator<Item = T> + Copy> fmt::Display for Brackets<T, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.print(f, fmt::Display::fmt)
    }
}

// pair adapter

pub struct Pair<'a, K, V>((K, &'a V));

pub struct PairAdapter<'a, K, V>(&'a TiSlice<K, V>);

impl<'a, K, V> Clone for PairAdapter<'a, K, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, K, V> Copy for PairAdapter<'a, K, V> {}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for Pair<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {:?}", self.0.0, self.0.1)
    }
}

impl<K: fmt::Display, V: fmt::Display> fmt::Display for Pair<'_, K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.0.0, self.0.1)
    }
}
