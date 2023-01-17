use super::bitmask::BitMask;
use core::mem;

pub type BitMaskWord = u16;
pub const BITMASK_STRIDE: usize = 1;
pub const BITMASK_MASK: BitMaskWord = 0xffff;

pub type HashWord = u16;
pub const HASH_MASK_HIGH_BIT: HashWord = 0b1000_0000_0000_0000;
pub const HASH_MASK_LOW_BIT: HashWord = 0b0111_1111_1111_1111;

/// Control byte value for an empty bucket.
pub const EMPTY: HashWord = 0b1111_1111_1111_1111;

/// Control byte value for a deleted bucket.
pub const DELETED: HashWord = 0b1000_0000_0000_0000;


/// Abstraction over a group of control bytes which can be scanned in
/// parallel.
///
/// This implementation uses a 128-bit SSE value.
#[derive(Copy, Clone)]
pub struct Group([HashWord; 16]);

// FIXME: https://github.com/rust-lang/rust-clippy/issues/3859
#[allow(clippy::use_self)]
impl Group {
    /// Number of bytes in the group.
    pub const BYTES: usize = mem::size_of::<Self>();
    pub const WIDTH: usize = 16;

    /// Returns a full group of empty bytes, suitable for use as the initial
    /// value for an empty hash table.
    ///
    /// This is guaranteed to be aligned to the group size.
    #[inline]
    #[allow(clippy::items_after_statements)]
    pub const fn static_empty() -> &'static [u16; Group::WIDTH] {
        #[repr(C)]
        struct AlignedBytes {
            _align: [Group; 0],
            bytes: [u16; Group::WIDTH],
        }
        const ALIGNED_BYTES: AlignedBytes = AlignedBytes {
            _align: [],
            bytes: [EMPTY; Group::WIDTH],
        };
        &ALIGNED_BYTES.bytes
    }

    /// Loads a group of bytes starting at the given address.
    #[inline]
    #[allow(clippy::cast_ptr_alignment)] // unaligned load
    pub unsafe fn load(ptr: *const u16) -> Self {
        Group(ptr.cast::<[u16; 16]>().read_unaligned())
    }

    /// Loads a group of bytes starting at the given address, which must be
    /// aligned to `mem::align_of::<Group>()`.
    #[inline]
    #[allow(clippy::cast_ptr_alignment)]
    pub unsafe fn load_aligned(ptr: *const u16) -> Self {
        // FIXME: use align_offset once it stabilizes
        debug_assert_eq!(ptr as usize & (mem::align_of::<Self>() - 1), 0);
        Group(ptr.cast::<[u16; 16]>().read())
    }

    /// Stores the group of bytes to the given address, which must be
    /// aligned to `mem::align_of::<Group>()`.
    #[inline]
    #[allow(clippy::cast_ptr_alignment)]
    pub unsafe fn store_aligned(self, ptr: *mut u16) {
        // FIXME: use align_offset once it stabilizes
        debug_assert_eq!(ptr as usize & (mem::align_of::<Self>() - 1), 0);
        ptr.cast::<[u16; 16]>().write(self.0);
    }

    /// Returns a `BitMask` indicating all bytes in the group which have
    /// the given value.
    #[inline]
    pub fn match_byte(self, byte: HashWord) -> BitMask {
        let mut mask = 0_u16;
        self.0.iter().enumerate().for_each(|(i, word)| {
            if byte == *word {
                mask |= 1 << i
            }
        });
        BitMask(mask)
    }

    /// Returns a `BitMask` indicating all bytes in the group which are
    /// `EMPTY`.
    #[inline]
    pub fn match_empty(self) -> BitMask {
        self.match_byte(EMPTY)
    }

    /// Returns a `BitMask` indicating all bytes in the group which are
    /// `EMPTY` or `DELETED`.
    #[inline]
    pub fn match_empty_or_deleted(self) -> BitMask {
        let mut mask = 0_u16;
        self.0.iter().enumerate().for_each(|(i, word)| {
            if *word == EMPTY || *word == DELETED {
                mask |= 1 << i
            }
        });
        BitMask(mask)
    }

    /// Returns a `BitMask` indicating all bytes in the group which are full.
    #[inline]
    pub fn match_full(&self) -> BitMask {
        self.match_empty_or_deleted().invert()
    }

    /// Performs the following transformation on all bytes in the group:
    /// - `EMPTY => EMPTY`
    /// - `DELETED => EMPTY`
    /// - `FULL => DELETED`
    #[inline]
    pub fn convert_special_to_empty_and_full_to_deleted(self) -> Self {
        let mut array = self.0;
        array.iter_mut().for_each(|word| {
            if *word & HASH_MASK_HIGH_BIT != 0 {
                *word = u16::MAX;
            } else {
                *word = HASH_MASK_HIGH_BIT;
            }
        });
        Group(array)
    }
}
