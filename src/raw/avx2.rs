use super::bitmask::BitMask;
use core::mem;

#[cfg(target_arch = "x86")]
use core::arch::x86;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as x86;

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
pub struct Group(x86::__m256i);

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
        Group(x86::_mm256_loadu_si256(ptr.cast()))
    }

    /// Loads a group of bytes starting at the given address, which must be
    /// aligned to `mem::align_of::<Group>()`.
    #[inline]
    #[allow(clippy::cast_ptr_alignment)]
    pub unsafe fn load_aligned(ptr: *const u16) -> Self {
        // FIXME: use align_offset once it stabilizes
        debug_assert_eq!(ptr as usize & (mem::align_of::<Self>() - 1), 0);
        Group(x86::_mm256_load_si256(ptr.cast()))
    }

    /// Stores the group of bytes to the given address, which must be
    /// aligned to `mem::align_of::<Group>()`.
    #[inline]
    #[allow(clippy::cast_ptr_alignment)]
    pub unsafe fn store_aligned(self, ptr: *mut u16) {
        // FIXME: use align_offset once it stabilizes
        debug_assert_eq!(ptr as usize & (mem::align_of::<Self>() - 1), 0);
        x86::_mm256_store_si256(ptr.cast(), self.0);
    }

    /// Returns a `BitMask` indicating all bytes in the group which have
    /// the given value.
    #[inline]
    pub fn match_byte(self, byte: HashWord) -> BitMask {
        #[allow(
            clippy::cast_possible_wrap, // byte: u8 as i8
            // byte: i32 as u16
            //   note: _mm_movemask_epi8 returns a 16-bit mask in a i32, the
            //   upper 16-bits of the i32 are zeroed:
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation
        )]
        unsafe {
            let cmp = x86::_mm256_cmpeq_epi16(self.0, x86::_mm256_set1_epi16(byte as i16));
            // let bm = x86::_mm256_movemask_epi8(cmp) as u32;
            // BitMask(x86::_pext_u32(bm, 0xAAAA_AAAA) as u16)

            let lo = x86::_mm256_extracti128_si256::<0>(cmp);
            let hi = x86::_mm256_extracti128_si256::<1>(cmp);
            BitMask(x86::_mm_movemask_epi8(x86::_mm_packs_epi16(lo, hi)) as u16)
        }
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
        #[allow(
            // byte: i32 as u16
            //   note: _mm_movemask_epi8 returns a 16-bit mask in a i32, the
            //   upper 16-bits of the i32 are zeroed:
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation
        )]
        unsafe {
            // A byte is EMPTY or DELETED iff the high bit is set
            let lo = x86::_mm256_extracti128_si256::<0>(self.0);
            let hi = x86::_mm256_extracti128_si256::<1>(self.0);
            BitMask(x86::_mm_movemask_epi8(x86::_mm_packs_epi16(lo, hi)) as u16)
        }
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
        // Map high_bit = 1 (EMPTY or DELETED) to 1111_1111
        // and high_bit = 0 (FULL) to 1000_0000
        //
        // Here's this logic expanded to concrete values:
        //   let special = 0 > byte = 1111_1111 (true) or 0000_0000 (false)
        //   1111_1111 | 1000_0000 = 1111_1111
        //   0000_0000 | 1000_0000 = 1000_0000
        #[allow(
            clippy::cast_possible_wrap, // byte: 0x80_u8 as i8
        )]
        unsafe {
            let zero = x86::_mm256_setzero_si256();
            let special = x86::_mm256_cmpgt_epi16(zero, self.0);
            Group(x86::_mm256_or_si256(
                special,
                x86::_mm256_set1_epi16(HASH_MASK_HIGH_BIT as i16),
            ))
        }
    }
}
