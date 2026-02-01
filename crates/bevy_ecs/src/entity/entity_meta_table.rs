//! Paged storage for entity metadata.
//!
//! This module provides [`EntityMetaTable`], a paged data structure for storing
//! [`EntityMeta`] indexed by [`EntityIndex`]. Instead of a flat `Vec<EntityMeta>`,
//! this uses a two-level structure: `pages[page_index][inner_index]`.
//!
//! # Design
//!
//! Inspired by [Flecs](https://www.flecs.dev/flecs/), this paged approach offers several benefits:
//!
//! - **Memory efficiency**: Pages are allocated on-demand, so sparse entity indices
//!   don't waste memory for unused ranges. This is critical for networked games where
//!   servers may allocate disjoint ID ranges to different clients.
//! - **Stability**: Page pointers remain stable when new pages are added, which is
//!   useful for concurrent access patterns.
//! - **Cache locality**: Entities with nearby indices share a page, improving cache
//!   performance for sequential access.
//!
//! # Page Size
//!
//! The page size is fixed at 1024 entities (2^10). This value was chosen as a balance
//! between memory overhead (larger pages waste more memory for sparse indices) and
//! lookup efficiency (smaller pages require more indirection).

use super::{EntityGeneration, EntityIndex, EntityLocation};
use crate::change_detection::{CheckChangeTicks, MaybeLocation, Tick};
use alloc::{boxed::Box, vec::Vec};

/// Number of bits used for the inner index within a page.
/// Page size = 2^PAGE_BITS = 1024 entities per page.
const PAGE_BITS: u32 = 10;

/// Number of entities per page.
const PAGE_SIZE: usize = 1 << PAGE_BITS;

/// Mask to extract the inner index from an entity index.
const PAGE_MASK: u32 = (PAGE_SIZE - 1) as u32;

/// Extracts the page index from an entity index.
#[inline(always)]
const fn page_index(index: u32) -> usize {
    (index >> PAGE_BITS) as usize
}

/// Extracts the inner index within a page from an entity index.
#[inline(always)]
const fn inner_index(index: u32) -> usize {
    (index & PAGE_MASK) as usize
}

/// Tracks when and where an entity was last spawned or despawned.
#[derive(Copy, Clone, Debug)]
pub(super) struct SpawnedOrDespawned {
    /// The source code location that triggered the spawn/despawn.
    pub by: MaybeLocation,
    /// The tick at which the spawn/despawn occurred.
    pub tick: Tick,
}

/// Metadata for a single entity.
#[derive(Copy, Clone, Debug)]
pub(super) struct EntityMeta {
    /// The current [`EntityGeneration`] of the [`EntityIndex`].
    pub generation: EntityGeneration,
    /// The current location of the [`EntityIndex`].
    pub location: Option<EntityLocation>,
    /// Location and tick of the last spawn/despawn.
    pub spawned_or_despawned: SpawnedOrDespawned,
}

impl EntityMeta {
    /// The metadata for a fresh entity: Never spawned/despawned, no location, etc.
    pub const FRESH: EntityMeta = EntityMeta {
        generation: EntityGeneration::FIRST,
        location: None,
        spawned_or_despawned: SpawnedOrDespawned {
            by: MaybeLocation::caller(),
            tick: Tick::new(0),
        },
    };
}

/// A single page of entity metadata.
///
/// Each page holds metadata for [`PAGE_SIZE`] entities.
#[derive(Clone)]
struct EntityMetaPage {
    /// The metadata entries in this page.
    /// Using a boxed array to avoid the page being stored inline in the pages vector.
    entries: Box<[EntityMeta; PAGE_SIZE]>,
}

impl EntityMetaPage {
    /// Creates a new page with all entries set to [`EntityMeta::FRESH`].
    fn new() -> Self {
        Self {
            entries: Box::new([EntityMeta::FRESH; PAGE_SIZE]),
        }
    }
}

impl core::fmt::Debug for EntityMetaPage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EntityMetaPage")
            .field("entries", &format_args!("[EntityMeta; {}]", PAGE_SIZE))
            .finish()
    }
}

/// A paged table storing metadata for all entities.
///
/// This structure maps [`EntityIndex`] to [`EntityMeta`] using a two-level
/// paging scheme: `pages[page_index][inner_index]`.
///
/// # Performance
///
/// - **Get**: O(1) - Two array lookups with bitwise operations
/// - **Insert/Ensure**: O(1) amortized - May allocate a new page
/// - **Memory**: Pages are allocated on-demand, each holding 1024 entries (~40KB per page)
#[derive(Debug, Clone)]
pub(super) struct EntityMetaTable {
    /// The pages of entity metadata.
    /// Indexed by `entity_index >> PAGE_BITS`.
    pages: Vec<Option<EntityMetaPage>>,
    /// The total number of entity indices that have been used.
    /// This tracks the high-water mark, not the count of live entities.
    len: usize,
}

impl EntityMetaTable {
    /// Creates an empty table with no pages allocated.
    pub const fn new() -> Self {
        Self {
            pages: Vec::new(),
            len: 0,
        }
    }

    /// Clears all entity metadata, deallocating all pages.
    pub fn clear(&mut self) {
        self.pages.clear();
        self.len = 0;
    }

    /// Returns the total number of entity indices that have been used.
    /// This is the high-water mark, not the count of live entities.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if no entity indices have been used.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Gets a reference to the metadata for the given index, if it exists.
    ///
    /// Returns `None` if the index has never been used.
    #[inline]
    pub fn get(&self, index: EntityIndex) -> Option<&EntityMeta> {
        let raw = index.index();
        let page_idx = page_index(raw);
        let inner_idx = inner_index(raw);

        self.pages
            .get(page_idx)?
            .as_ref()
            .map(|page| &page.entries[inner_idx])
    }

    /// Gets a mutable reference to the metadata for the given index, if it exists.
    ///
    /// Returns `None` if the index has never been used.
    #[inline]
    pub fn get_mut(&mut self, index: EntityIndex) -> Option<&mut EntityMeta> {
        let raw = index.index();
        let page_idx = page_index(raw);
        let inner_idx = inner_index(raw);

        self.pages
            .get_mut(page_idx)?
            .as_mut()
            .map(|page| &mut page.entries[inner_idx])
    }

    /// Gets a reference to the metadata for the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` has been previously ensured via
    /// [`ensure_index`](Self::ensure_index) or equivalent.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: EntityIndex) -> &EntityMeta {
        let raw = index.index();
        let page_idx = page_index(raw);
        let inner_idx = inner_index(raw);

        // SAFETY: Caller guarantees the index is valid.
        unsafe {
            self.pages
                .get_unchecked(page_idx)
                .as_ref()
                .unwrap_unchecked()
                .entries
                .get_unchecked(inner_idx)
        }
    }

    /// Gets a mutable reference to the metadata for the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` has been previously ensured via
    /// [`ensure_index`](Self::ensure_index) or equivalent.
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: EntityIndex) -> &mut EntityMeta {
        let raw = index.index();
        let page_idx = page_index(raw);
        let inner_idx = inner_index(raw);

        // SAFETY: Caller guarantees the index is valid.
        unsafe {
            self.pages
                .get_unchecked_mut(page_idx)
                .as_mut()
                .unwrap_unchecked()
                .entries
                .get_unchecked_mut(inner_idx)
        }
    }

    /// Ensures that the given index is valid, allocating pages as necessary.
    ///
    /// After calling this, `get(index)` and `get_mut(index)` will return `Some`.
    #[inline]
    pub fn ensure_index(&mut self, index: EntityIndex) {
        let raw = index.index();
        let page_idx = page_index(raw);

        // Ensure we have enough pages
        if page_idx >= self.pages.len() {
            self.pages.resize_with(page_idx + 1, || None);
        }

        // Ensure the page exists
        if self.pages[page_idx].is_none() {
            self.pages[page_idx] = Some(EntityMetaPage::new());
        }

        // Update len to track high-water mark
        let new_len = raw as usize + 1;
        if new_len > self.len {
            self.len = new_len;
        }
    }

    /// Gets metadata for the index, returning [`EntityMeta::FRESH`] if not present.
    ///
    /// This is useful when you want to treat unallocated indices as fresh entities.
    #[inline]
    pub fn get_or_fresh(&self, index: EntityIndex) -> &EntityMeta {
        self.get(index).unwrap_or(&EntityMeta::FRESH)
    }

    /// Iterates over all entity metadata, checking change ticks.
    pub fn check_change_ticks(&mut self, check: CheckChangeTicks) {
        for page in self.pages.iter_mut().flatten() {
            for meta in page.entries.iter_mut() {
                meta.spawned_or_despawned.tick.check_tick(check);
            }
        }
    }

    /// Iterates over all allocated pages and their metadata entries.
    /// Yields (index, metadata) pairs for all entries up to `len()`.
    pub fn iter(&self) -> impl Iterator<Item = &EntityMeta> {
        let len = self.len;
        self.pages
            .iter()
            .enumerate()
            .flat_map(move |(page_idx, page)| {
                let page_start = page_idx * PAGE_SIZE;
                let page_end = ((page_idx + 1) * PAGE_SIZE).min(len);
                let count = page_end.saturating_sub(page_start);

                page.as_ref()
                    .map(|p| p.entries[..count].iter())
                    .into_iter()
                    .flatten()
            })
    }

    /// Iterates mutably over all allocated pages and their metadata entries.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut EntityMeta> {
        let len = self.len;
        self.pages
            .iter_mut()
            .enumerate()
            .flat_map(move |(page_idx, page)| {
                let page_start = page_idx * PAGE_SIZE;
                let page_end = ((page_idx + 1) * PAGE_SIZE).min(len);
                let count = page_end.saturating_sub(page_start);

                page.as_mut()
                    .map(|p| p.entries[..count].iter_mut())
                    .into_iter()
                    .flatten()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index(raw: u32) -> EntityIndex {
        EntityIndex::from_raw_u32(raw).unwrap()
    }

    #[test]
    fn test_page_indexing() {
        // Page 0: indices 0-1023
        assert_eq!(page_index(0), 0);
        assert_eq!(inner_index(0), 0);
        assert_eq!(page_index(1023), 0);
        assert_eq!(inner_index(1023), 1023);

        // Page 1: indices 1024-2047
        assert_eq!(page_index(1024), 1);
        assert_eq!(inner_index(1024), 0);
        assert_eq!(page_index(2047), 1);
        assert_eq!(inner_index(2047), 1023);

        // Large sparse index
        assert_eq!(page_index(1_000_000), 976);
        assert_eq!(inner_index(1_000_000), 576);
    }

    #[test]
    fn test_empty_table() {
        let table = EntityMetaTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert!(table.get(make_index(0)).is_none());
    }

    #[test]
    fn test_ensure_and_get() {
        let mut table = EntityMetaTable::new();

        // Ensure index 0
        table.ensure_index(make_index(0));
        assert_eq!(table.len(), 1);
        assert!(table.get(make_index(0)).is_some());

        // Ensure index 1000 (same page)
        table.ensure_index(make_index(1000));
        assert_eq!(table.len(), 1001);
        assert!(table.get(make_index(1000)).is_some());

        // Ensure sparse index in different page
        table.ensure_index(make_index(100_000));
        assert_eq!(table.len(), 100_001);
        assert!(table.get(make_index(100_000)).is_some());

        // Original indices still accessible
        assert!(table.get(make_index(0)).is_some());
        assert!(table.get(make_index(1000)).is_some());
    }

    #[test]
    fn test_sparse_memory_efficiency() {
        let mut table = EntityMetaTable::new();

        // Only ensure two sparse indices
        table.ensure_index(make_index(0));
        table.ensure_index(make_index(1_000_000));

        // Should only have allocated 2 pages (page 0 and page 976)
        // Not 977 pages like a Vec would require
        let allocated_pages = table.pages.iter().filter(|p| p.is_some()).count();
        assert_eq!(allocated_pages, 2);
    }

    #[test]
    fn test_get_or_fresh() {
        let table = EntityMetaTable::new();

        // Non-existent index returns FRESH
        let meta = table.get_or_fresh(make_index(12345));
        assert_eq!(meta.generation, EntityGeneration::FIRST);
        assert!(meta.location.is_none());
    }

    #[test]
    fn test_mutation() {
        let mut table = EntityMetaTable::new();
        table.ensure_index(make_index(42));

        // Mutate the metadata
        if let Some(meta) = table.get_mut(make_index(42)) {
            meta.generation = EntityGeneration::FIRST.after_versions(5);
        }

        // Verify mutation persisted
        let meta = table.get(make_index(42)).unwrap();
        assert_eq!(meta.generation, EntityGeneration::FIRST.after_versions(5));
    }

    #[test]
    fn test_clear() {
        let mut table = EntityMetaTable::new();
        table.ensure_index(make_index(0));
        table.ensure_index(make_index(100_000));

        assert!(!table.is_empty());

        table.clear();

        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert!(table.get(make_index(0)).is_none());
    }
}
