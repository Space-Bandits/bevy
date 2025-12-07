//! Remote entity allocation for networked games.
//!
//! This module provides [`RemoteEntityAllocator`], an allocator designed for
//! client-server architectures where the server controls entity ID assignment.
//!
//! # Use Case
//!
//! In networked games, it's often desirable for the server to be authoritative
//! over entity IDs. This ensures that:
//! - Client and server can refer to the same entity with the same ID
//! - Entity IDs don't conflict between clients
//! - The server can pre-allocate ID ranges for clients
//!
//! # How It Works
//!
//! The remote allocator maintains a buffer of pre-allocated entity IDs that were
//! assigned by a remote authority (typically a server). When the buffer runs low,
//! the application should request more IDs from the server.
//!
//! ```ignore
//! // On the client:
//! let mut allocator = RemoteEntityAllocator::new();
//!
//! // Receive entity ID range from server
//! allocator.add_range(EntityIndex::from_raw_u32(1000).unwrap(), 100);
//!
//! // Now allocations will use IDs from the server-assigned range
//! let entity = allocator.alloc(); // Returns entity with index 1000
//! let entity = allocator.alloc(); // Returns entity with index 1001
//! // ...
//! ```
//!
//! # Differences from Local Allocator
//!
//! Unlike [`EntityAllocator`](super::EntityAllocator), the remote allocator:
//! - Does not generate new IDs on its own
//! - Requires external provisioning of ID ranges
//! - May run out of IDs if not replenished
//! - Does not support concurrent allocation (uses `&mut self`)

use super::{AllocateEntities, Entity, EntityIndex};
use alloc::{collections::VecDeque, vec::Vec};
use nonmax::NonMaxU32;

/// An entity allocator that uses pre-assigned entity IDs from a remote source.
///
/// This is useful for networked games where a server controls entity ID assignment.
/// See the [module documentation](self) for more details.
///
/// # Thread Safety
///
/// Unlike [`EntityAllocator`](super::EntityAllocator), this allocator is NOT
/// thread-safe. All operations require `&mut self`. This is intentional because:
/// - Remote allocators are typically used in single-threaded client contexts
/// - The added complexity of thread-safety isn't needed for the use case
/// - It simplifies the implementation and avoids atomic overhead
#[derive(Debug, Default)]
pub struct RemoteEntityAllocator {
    /// Queue of available entity IDs to hand out.
    /// Uses a deque for efficient push_back (when adding ranges) and pop_front (when allocating).
    available: VecDeque<Entity>,
    /// Entities that have been freed and can be reused.
    /// These take priority over fresh IDs from `available`.
    freed: Vec<Entity>,
}

impl RemoteEntityAllocator {
    /// Creates a new empty remote allocator.
    ///
    /// The allocator starts with no available IDs. You must call [`add_range`](Self::add_range)
    /// or [`add_entities`](Self::add_entities) before allocation will succeed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a range of consecutive entity indices to the allocator.
    ///
    /// This is typically called when receiving a range assignment from a server.
    /// All entities in the range will have [`EntityGeneration::FIRST`].
    ///
    /// # Arguments
    ///
    /// * `start` - The starting entity index
    /// * `count` - The number of consecutive indices to add
    ///
    /// # Panics
    ///
    /// Panics if `start + count` would overflow.
    pub fn add_range(&mut self, start: EntityIndex, count: u32) {
        let start_raw = start.index();
        for i in 0..count {
            let index_raw = start_raw.checked_add(i).expect("entity index overflow");
            if let Some(index) = NonMaxU32::new(index_raw) {
                let entity = Entity::from_index(EntityIndex::new(index));
                self.available.push_back(entity);
            }
        }
    }

    /// Adds specific entities to the allocator.
    ///
    /// This allows adding entities with specific generations, useful for
    /// synchronizing entity state from a server.
    pub fn add_entities(&mut self, entities: impl IntoIterator<Item = Entity>) {
        self.available.extend(entities);
    }

    /// Returns the number of entities available for allocation.
    ///
    /// This includes both fresh IDs from ranges and freed IDs.
    pub fn available_count(&self) -> usize {
        self.available.len() + self.freed.len()
    }

    /// Returns true if the allocator has no entities available.
    ///
    /// When this returns true, [`alloc`](Self::alloc) will panic.
    /// You should request more IDs from the server before this happens.
    pub fn is_empty(&self) -> bool {
        self.available.is_empty() && self.freed.is_empty()
    }

    /// Allocates an entity, consuming from available IDs.
    ///
    /// Freed entities are prioritized over fresh IDs to maximize ID reuse.
    ///
    /// # Panics
    ///
    /// Panics if no entities are available. Check [`is_empty`](Self::is_empty)
    /// or [`available_count`](Self::available_count) before calling.
    pub fn alloc_mut(&mut self) -> Entity {
        // Prefer freed entities for better ID reuse
        if let Some(entity) = self.freed.pop() {
            return entity;
        }

        self.available
            .pop_front()
            .expect("RemoteEntityAllocator: no entities available - request more from server")
    }

    /// Attempts to allocate an entity, returning `None` if none are available.
    ///
    /// This is the non-panicking version of [`alloc_mut`](Self::alloc_mut).
    pub fn try_alloc(&mut self) -> Option<Entity> {
        self.freed.pop().or_else(|| self.available.pop_front())
    }

    /// Allocates multiple entities efficiently.
    ///
    /// Returns an iterator that yields up to `count` entities.
    /// The iterator may yield fewer entities if not enough are available.
    pub fn alloc_many_mut(&mut self, count: u32) -> impl Iterator<Item = Entity> + '_ {
        (0..count).map_while(|_| self.try_alloc())
    }

    /// Returns an entity to the allocator for reuse.
    ///
    /// The entity will be available for future allocations.
    /// The generation should be incremented before calling this.
    pub fn free_entity(&mut self, entity: Entity) {
        self.freed.push(entity);
    }
}

impl AllocateEntities for RemoteEntityAllocator {
    /// Allocates an entity.
    ///
    /// # Note
    ///
    /// This implementation takes `&self` to satisfy the trait, but internally
    /// uses interior mutability patterns. For the remote allocator, prefer
    /// using [`alloc_mut`](Self::alloc_mut) directly when possible.
    ///
    /// # Panics
    ///
    /// Panics if no entities are available.
    fn alloc(&self) -> Entity {
        // This is a limitation - the trait requires &self but RemoteEntityAllocator
        // needs &mut self. For now, we panic with a helpful message.
        panic!(
            "RemoteEntityAllocator::alloc() called via trait - \
             use alloc_mut() directly or wrap in a mutex for concurrent access"
        );
    }

    fn alloc_many(&self, _count: u32) -> impl Iterator<Item = Entity> {
        // Same limitation as alloc()
        panic!(
            "RemoteEntityAllocator::alloc_many() called via trait - \
             use alloc_many_mut() directly or wrap in a mutex for concurrent access"
        );
        #[allow(unreachable_code)]
        core::iter::empty()
    }

    fn free(&mut self, entity: Entity) {
        self.free_entity(entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::EntityGeneration;
    use alloc::vec;

    #[test]
    fn basic_allocation() {
        let mut allocator = RemoteEntityAllocator::new();
        assert!(allocator.is_empty());

        // Add a range of IDs
        let start = EntityIndex::from_raw_u32(100).unwrap();
        allocator.add_range(start, 5);

        assert!(!allocator.is_empty());
        assert_eq!(allocator.available_count(), 5);

        // Allocate entities
        let e1 = allocator.alloc_mut();
        assert_eq!(e1.index().index(), 100);

        let e2 = allocator.alloc_mut();
        assert_eq!(e2.index().index(), 101);

        assert_eq!(allocator.available_count(), 3);
    }

    #[test]
    fn freed_entities_prioritized() {
        let mut allocator = RemoteEntityAllocator::new();

        // Add fresh IDs
        let start = EntityIndex::from_raw_u32(100).unwrap();
        allocator.add_range(start, 5);

        // Allocate and free one
        let e1 = allocator.alloc_mut();
        let freed_entity = Entity::from_index_and_generation(
            e1.index(),
            EntityGeneration::FIRST.after_versions(1),
        );
        allocator.free_entity(freed_entity);

        // Next allocation should return the freed entity, not a fresh one
        let e2 = allocator.alloc_mut();
        assert_eq!(e2.index(), e1.index());
        assert_eq!(e2.generation(), EntityGeneration::FIRST.after_versions(1));
    }

    #[test]
    fn try_alloc_returns_none_when_empty() {
        let mut allocator = RemoteEntityAllocator::new();
        assert!(allocator.try_alloc().is_none());

        allocator.add_range(EntityIndex::from_raw_u32(1).unwrap(), 1);
        assert!(allocator.try_alloc().is_some());
        assert!(allocator.try_alloc().is_none());
    }

    #[test]
    fn add_specific_entities() {
        let mut allocator = RemoteEntityAllocator::new();

        let entities = vec![
            Entity::from_index_and_generation(
                EntityIndex::from_raw_u32(50).unwrap(),
                EntityGeneration::FIRST.after_versions(3),
            ),
            Entity::from_index_and_generation(
                EntityIndex::from_raw_u32(100).unwrap(),
                EntityGeneration::FIRST.after_versions(7),
            ),
        ];

        allocator.add_entities(entities);
        assert_eq!(allocator.available_count(), 2);

        let e1 = allocator.alloc_mut();
        assert_eq!(e1.index().index(), 50);
        assert_eq!(e1.generation(), EntityGeneration::FIRST.after_versions(3));
    }

    #[test]
    fn alloc_many_partial() {
        let mut allocator = RemoteEntityAllocator::new();
        allocator.add_range(EntityIndex::from_raw_u32(1).unwrap(), 3);

        // Request more than available
        let allocated: Vec<_> = allocator.alloc_many_mut(5).collect();
        assert_eq!(allocated.len(), 3);
    }
}
