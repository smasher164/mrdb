#![feature(new_uninit)]

use dashmap::{mapref::one::Ref, DashMap};
use init_array::init_boxed_slice;
use std::{
    fs::File,
    io,
    ops::{Deref, DerefMut},
    path::Path,
    str::Utf8Error,
    sync::atomic::{AtomicU64, Ordering},
};

use thiserror::Error;

use tracing::{error};

#[cfg_attr(target_family = "unix", path = "os_unix.rs")]
#[cfg_attr(target_family = "windows", path = "os_windows.rs")]
mod os;

#[cfg_attr(target_os = "linux", path = "fsinfo_linux.rs")]
#[cfg_attr(
    all(target_family = "unix", not(target_os = "linux")),
    path = "fsinfo_unix.rs"
)]
#[cfg_attr(target_family = "windows", path = "fsinfo_windows.rs")]
mod fsinfo;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    Io(#[from] io::Error),

    #[error("{0}")]
    Utf8Error(#[from] Utf8Error),

    #[cfg(target_family = "windows")]
    #[error("{0}")]
    MissingTerminator(#[from] widestring::error::MissingNulTerminator),

    #[cfg(target_family = "windows")]
    #[error("{0}")]
    Utf16Error(#[from] widestring::error::Utf16Error),
}

// impl tracing::Value for Error {
//     fn record(&self, key: &tracing::field::Field, visitor: &mut dyn tracing::field::Visit) {
//         visitor.record_debug(key, &self)
//     }
// }
// impl tracing::Sea

type Result<T> = core::result::Result<T, Error>;

const KB: u64 = 1024;
const MB: u64 = 1024 * KB;
const GB: u64 = 1024 * MB;
const TB: u64 = 1024 * GB;
const PAGE_SIZE: u64 = 4 * KB;
const PAGE_COUNT_MASK: u64 = PAGE_SIZE - 1;
const PID_MASK: u64 = !PAGE_COUNT_MASK;
const OFFSET_MASK: u64 = PID_MASK;
const MAX_PAGES: u64 = 1 << 52;
const MAX_PAGES_LARGE: u64 = 1 << 40;
const DIRTY_BIT_MASK: u64 = 1 << (57 - 1);
const BATCH_SIZE: u64 = 64;
const BATCH_SPACE: u64 = BATCH_SIZE * 4096;
const DEFAULT_CACHE_CAPACITY_BYTES: u64 = 64 * MB;
const MAX_RESTARTS_OPTIMISTIC_READ: i32 = 100; // idk what to make this lol

// TODO: use async i/o for disk reads and writes instead of pread/pwrite.
// TODO: use parking lot to park threads instead of spinning
// TODO: replace cache with a ring. replace remove with pop

struct Mmap(*mut u8);

impl Mmap {
    unsafe fn to_ptr(&self, pid: PageId) -> *mut u8 {
        self.0.offset(pid.offset() as isize)
    }
    unsafe fn to_slice(&self, page_id: PageId) -> &[u8] {
        let buf = self.to_ptr(page_id);
        let size = page_id.size();
        std::slice::from_raw_parts(buf, size as usize)
    }
    unsafe fn to_slice_mut(&self, page_id: PageId) -> &mut [u8] {
        let buf = self.to_ptr(page_id);
        let size = page_id.size();
        std::slice::from_raw_parts_mut(buf, size as usize)
    }
    // not sure when this is needed. do we really want the pointer to be tagged?
    unsafe fn to_page_id(&self, ptr: *mut u8) -> PageId {
        // ptr should be tagged with the page_count.
        // Since we're 4K aligned, we have 12 bits free in the LSB.
        // requires masking before use.
        let page_count = (ptr as u64) & PAGE_COUNT_MASK;
        // determine offset into vm
        let offset = ptr.offset_from(self.0) as u64;
        PageId::new_offset(page_count, offset & OFFSET_MASK)
    }
    unsafe fn release(&self, pid: PageId) -> Result<()> {
        os::release(self.to_ptr(pid), pid.size())
    }
}

trait Pread {
    fn pread(&self, buf: *mut u8, offset: u64, size: u64) -> Result<u64>;
}

trait Pwrite {
    fn pwrite(&self, buf: *const u8, offset: u64, size: u64) -> Result<u64>;
}

// actually, bottom 12 bits will store it.
#[derive(Hash, PartialEq, Eq, Copy, Clone)]
pub struct PageId(u64);
impl PageId {
    pub fn is_empty(self) -> bool {
        self.page_count() == 0
    }
    pub fn is_tombstone(self) -> bool {
        self.page_count() == 0xFFF
    }
    // should this return a u16?
    pub fn page_count(self) -> u64 {
        let n = (self.0 & PAGE_COUNT_MASK); // no longer +1, because when cleared it's empty.
        n // for now, keep it simple.
          // maybe we can do n*n*n in the future (1, 8, 27, ...)
    }
    pub fn size(self) -> u64 {
        self.page_count() * PAGE_SIZE // 4K,32K,81K,256K,500K,864K,1372K,..256TB
    }
    pub fn pid(self) -> u64 {
        (self.0 & PID_MASK) >> 12
    }
    pub fn offset(self) -> u64 {
        self.pid() * PAGE_SIZE
    }
    pub fn new_pid(page_count: u64, pid: u64) -> Self {
        Self((pid << 12) | (page_count - 1))
    }
    pub fn new_size_pid(size: u64, pid: u64) -> Self {
        let page_count = size / PAGE_SIZE;
        Self::new_pid(page_count, pid)
    }
    pub fn new_offset(page_count: u64, offset: u64) -> Self {
        let pid = offset / PAGE_SIZE;
        Self::new_pid(page_count, pid)
    }
    pub fn new_size_offset(size: u64, offset: u64) -> Self {
        let page_count = size / PAGE_SIZE;
        Self::new_offset(page_count, offset)
    }
    pub fn unpack(self) -> (u64, u64) {
        (self.size(), self.pid())
    }
    pub fn unpack_offset(self) -> (u64, u64) {
        (self.size(), self.offset())
    }
}

impl std::fmt::Display for PageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:x}_{:x}", self.pid(), self.page_count())
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PageState(u64);

impl PageState {
    const UNLOCKED: PageState = PageState(0);
    const LOCK_MIN: PageState = PageState(1);
    const LOCKED_SHARED: PageState = PageState(124);
    const LOCKED: PageState = PageState(125);
    const MARKED: PageState = PageState(126);
    const EVICTED: PageState = PageState(127);

    fn get(x: u64) -> PageState {
        PageState(x >> 57)
    }
    fn set(x: u64, st: PageState) -> u64 {
        ((x << 7) >> 7) | st.0 << 57
    }
    fn inc(x: u64, st: PageState) -> u64 {
        (((x << 8) >> 8) + 1) | (x & DIRTY_BIT_MASK) | st.0 << 57
    }
    fn is_dirty(x: u64) -> bool {
        (x & DIRTY_BIT_MASK) != 0
    }
    fn set_dirty(x: u64) -> u64 {
        x | DIRTY_BIT_MASK
    }
    fn set_clean(x: u64) -> u64 {
        x | (!DIRTY_BIT_MASK)
    }
}

struct PageEntry(AtomicU64);
impl PageEntry {
    fn new() -> PageEntry {
        PageEntry(AtomicU64::new(PageState::EVICTED.0 << 56))
    }
    fn cas(&self, old: u64, new: u64) -> bool {
        self.0
            .compare_exchange(old, new, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }
    fn cas_weak(&self, old: u64, new: u64) -> bool {
        self.0
            .compare_exchange_weak(old, new, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }
    fn lock(&self, old: u64) -> bool {
        self.cas(old, PageState::set(old, PageState::LOCKED))
    }
    fn lock_weak(&self, old: u64) -> bool {
        self.cas_weak(old, PageState::set(old, PageState::LOCKED))
    }
    fn lock_shared(&self, old: u64) -> bool {
        let st = PageState::get(old);
        if st < PageState::LOCKED_SHARED {
            let new = PageState::set(old, PageState(st.0 + 1));
            return self.cas(old, new);
        }
        if st == PageState::MARKED {
            let new = PageState::set(old, PageState::LOCK_MIN);
            return self.cas(old, new);
        }
        false
    }
    fn unlock(&self) {
        let new = PageState::inc(self.0.load(Ordering::SeqCst), PageState::UNLOCKED);
        self.0.store(new, Ordering::Release);
    }
    fn unlock_shared(&self) {
        loop {
            let old = self.0.load(Ordering::SeqCst);
            let st = PageState::get(old);
            if self.cas_weak(old, PageState::set(old, PageState(st.0 - 1))) {
                return;
            }
        }
    }
    fn unlock_evicted(&self) {
        let new = PageState::inc(self.0.load(Ordering::SeqCst), PageState::EVICTED);
        self.0.store(new, Ordering::Release);
    }
    fn rollback_evicted(&self) {
        // assumes it was locked.
        let new = PageState::set(self.0.load(Ordering::SeqCst), PageState::EVICTED);
        self.0.store(new, Ordering::Release); // okay for this to be release?
    }
    fn mark(&self, old: u64) -> bool {
        self.cas(old, PageState::set(old, PageState::MARKED))
    }
    fn mark_clean(&self) {
        loop {
            let old = self.0.load(Ordering::SeqCst);
            if self.cas_weak(old, PageState::set_clean(old)) {
                return;
            }
        }
    }
    fn mark_dirty(&self) {
        loop {
            let old = self.0.load(Ordering::SeqCst);
            if self.cas_weak(old, PageState::set_dirty(old)) {
                return;
            }
        }
    }
    fn is_locked(&self) -> bool {
        PageState::get(self.0.load(Ordering::SeqCst)) == PageState::LOCKED
    }
}

impl Default for PageEntry {
    fn default() -> Self {
        Self::new()
    }
}

struct ResidentSet {
    hand: AtomicU64,
    entries: Box<[AtomicU64]>,
    rand_state: ahash::RandomState,
    mask: u64,
    used_space: AtomicU64,
    capacity: u64,
}

impl ResidentSet {
    fn new(capacity: u64) -> ResidentSet {
        let capacity = capacity.next_multiple_of(4096);
        let n_entries = capacity / 4096;
        let size = ((n_entries as f64 * 1.5) as u64).next_power_of_two(); // is this necessary?
        let mask = size - 1;
        let hand = AtomicU64::new(0);
        let entries = init_boxed_slice(size as usize, |_| AtomicU64::new(0));
        let rand_state = ahash::RandomState::new();
        let used_space = AtomicU64::new(0);
        ResidentSet {
            hand,
            entries,
            mask,
            rand_state,
            used_space,
            capacity,
        }
    }
    fn insert(&self, page_id: PageId) {
        let mut i = self.rand_state.hash_one(page_id.0) & self.mask;
        loop {
            let curr = PageId(self.entries[i as usize].load(Ordering::SeqCst));
            if curr.is_empty() || curr.is_tombstone() {
                if self.entries[i as usize]
                    .compare_exchange(curr.0, page_id.0, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    self.used_space.fetch_add(page_id.size(), Ordering::SeqCst);
                    return;
                }
            }
            i = (i + 1) & self.mask;
        }
    }
    fn remove(&self, page_id: PageId) -> bool {
        let mut i = self.rand_state.hash_one(page_id.0) & self.mask;
        const TOMBSTONE: u64 = 0xFFF;
        loop {
            let curr = PageId(self.entries[i as usize].load(Ordering::SeqCst));
            if curr.is_empty() {
                return false;
            }
            if curr == page_id {
                if self.entries[i as usize]
                    .compare_exchange(curr.0, TOMBSTONE, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    self.used_space.fetch_sub(page_id.size(), Ordering::SeqCst);
                    return true;
                }
            }
            i = (i + 1) & self.mask;
        }
    }
    fn next(&self) -> Option<PageId> {
        let (mut i_old, mut i_new);
        loop {
            i_old = self.hand.load(Ordering::SeqCst);
            i_new = (i_old + 1) % (self.entries.len() as u64); // vmcache uses %count instead of &mask here. idk why
            if self
                .hand
                .compare_exchange(i_old, i_new, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                // costlier because it's not a batch update?
                break;
            }
        }
        let curr = PageId(self.entries[i_old as usize].load(Ordering::SeqCst));
        if !curr.is_tombstone() && !curr.is_empty() {
            Some(curr)
        } else {
            None
        }
    }
}

pub struct PageCache {
    mem: Mmap,
    page_state: DashMap<PageId, PageEntry, ahash::RandomState>,
    f: File,
    residents: ResidentSet,
}

impl PageCache {
    pub fn new(f: File, vm_size_bytes: u64, cache_capacity_bytes: u64) -> Result<PageCache> {
        // hint that we're going to be doing random reads on this file
        #[cfg(all(
            target_family = "unix",
            not(any(target_os = "macos", target_os = "ios"))
        ))]
        os::advise_random_read(&f)?;

        let mem = Mmap::create_virt_mem(vm_size_bytes)?;
        let ahasher = ahash::RandomState::new();
        let page_state = DashMap::with_hasher(ahasher);
        let residents = ResidentSet::new(cache_capacity_bytes);
        Ok(PageCache {
            mem,
            page_state,
            f,
            residents,
        })
    }
    fn get_page_entry(&self, page_id: PageId) -> Ref<'_, PageId, PageEntry, ahash::RandomState> {
        // &self.page_state[page_id.pid() as usize]
        let entry = self.page_state.entry(page_id);
        let or_def = entry.or_default();
        or_def.downgrade()
    }
    unsafe fn disk_read_page(&self, page_id: PageId) -> Result<u64> {
        let buf = self.mem.to_ptr(page_id);
        let offset = page_id.offset();
        let size = page_id.size();
        self.f.pread(buf, offset, size)
    }
    unsafe fn disk_write_page(&self, page_id: PageId) -> Result<u64> {
        // let m = self.mem.to_slice_mut(page_id);
        let buf = self.mem.to_ptr(page_id);
        let offset = page_id.offset();
        let size = page_id.size();
        self.f.pwrite(buf, offset, size)
    }
    fn ensure_free_pages(&self, size: u64) {
        if (size + self.residents.used_space.load(Ordering::SeqCst))
            >= (((self.residents.capacity as f64) * 0.95) as u64)
        {
            self.evict(size);
        }
    }
    unsafe fn handle_fault(&self, page_id: PageId) -> Result<()> {
        self.ensure_free_pages(page_id.size());
        self.disk_read_page(page_id)?;
        self.residents.insert(page_id);
        Ok(())
    }
    pub fn write_page(&self, page_id: PageId) -> Result<PageMut> {
        let entry = self.get_page_entry(page_id);
        loop {
            let old = entry.0.load(Ordering::SeqCst);
            match PageState::get(old) {
                PageState::EVICTED => {
                    // cache miss
                    if entry.lock_weak(old) {
                        // ensure there's enough space to read this page
                        // then actually read it.
                        match unsafe { self.handle_fault(page_id) } {
                            Ok(_) => {
                                // return actual write pointer to page
                                let slice = unsafe { self.mem.to_slice_mut(page_id) };
                                let pg = PageMut {
                                    page_entry: entry,
                                    data: slice,
                                };
                                return Ok(pg);
                            }
                            Err(e) => {
                                entry.rollback_evicted();
                                return Err(e);
                            }
                        }
                    }
                }
                PageState::MARKED | PageState::UNLOCKED => {
                    // cache hit
                    if entry.lock_weak(old) {
                        let slice = unsafe { self.mem.to_slice_mut(page_id) };
                        let pg = PageMut {
                            page_entry: entry,
                            data: slice,
                        };
                        return Ok(pg);
                    }
                }
                _ => (),
            }
        }
    }
    fn unfix_write(&self, page_id: PageId) {
        self.get_page_entry(page_id).unlock()
    }
    fn unfix_read(&self, page_id: PageId) {
        self.get_page_entry(page_id).unlock_shared()
    }
    pub fn read_page(&self, page_id: PageId) -> Result<Page> {
        let entry = self.get_page_entry(page_id);
        loop {
            let old = entry.0.load(Ordering::SeqCst);
            match PageState::get(old) {
                PageState::LOCKED => (),
                PageState::EVICTED => {
                    if entry.lock(old) {
                        // handle fault
                        match unsafe { self.handle_fault(page_id) } {
                            Ok(_) => {
                                entry.unlock();
                                // return actual pointer to page
                                let slice = unsafe { self.mem.to_slice(page_id) };
                                let pg = Page {
                                    page_entry: entry,
                                    data: slice,
                                };
                                return Ok(pg);
                            }
                            Err(e) => {
                                entry.rollback_evicted();
                                return Err(e);
                            }
                        }
                    }
                }
                _ => {
                    if entry.lock_shared(old) {
                        // return shared ref to page
                        let slice = unsafe { self.mem.to_slice(page_id) };
                        let pg = Page {
                            page_entry: entry,
                            data: slice,
                        };
                        return Ok(pg);
                    }
                }
            }
        }
    }
    pub fn optimistic_read<F>(&self, page_id: PageId, f: F) -> Result<()>
    where
        F: Fn(&[u8]) -> (),
    {
        let mut restart_counter = 0;
        let entry = self.get_page_entry(page_id);
        loop {
            // if we've restarted too many times, just fall back to grabbing a lock
            if restart_counter == MAX_RESTARTS_OPTIMISTIC_READ {
                let pg = self.read_page(page_id)?;
                f(&pg);
            }
            let old = entry.0.load(Ordering::SeqCst); // does this need to be seqcst?
            match PageState::get(old) {
                PageState::UNLOCKED => {
                    // pass into function
                    let slice = unsafe { self.mem.to_slice(page_id) };
                    f(slice);
                    if entry.0.load(Ordering::SeqCst) == old {
                        return Ok(());
                    }
                    restart_counter += 1;
                }
                PageState::MARKED => {
                    let new = PageState::set(old, PageState::UNLOCKED);
                    entry.cas_weak(old, new);
                }
                PageState::EVICTED => {
                    self.write_page(page_id)?; // unpins in call to drop
                }
                _ => (),
            }
        }
    }
    // eviction based on CLOCK, but consider trying SIEVE
    // https://cachemon.github.io/SIEVE-website/blog/2023/12/17/sieve-is-simpler-than-lru/
    fn evict(&self, required_size: u64) {
        let mut to_evict: Vec<PageId> = Vec::with_capacity(BATCH_SIZE as usize);
        let mut to_write: Vec<PageId> = Vec::with_capacity(BATCH_SIZE as usize);
        let mut total_size: u64 = 0;
        let needed = required_size.max(BATCH_SPACE);
        while total_size < needed {
            if let Some(page_id) = self.residents.next() {
                let entry = self.get_page_entry(page_id);
                let old = entry.0.load(Ordering::SeqCst);
                match PageState::get(old) {
                    PageState::MARKED => {
                        // if it's marked, we evict if we can grab a lock on the page
                        if PageState::is_dirty(old) {
                            // if entry.lock_shared(old) {
                            if entry.lock(old) {
                                to_write.push(page_id);
                                total_size += page_id.size();
                                // this does mean that size checks will allow insertion to proceed
                                // but it means this can be turned into a pop()
                                self.residents.remove(page_id);
                            }
                        } else {
                            if entry.lock(old) {
                                to_evict.push(page_id);
                                total_size += page_id.size();
                                // this does mean that size checks will allow insertion to proceed
                                // but it means this can be turned into a pop()
                                self.residents.remove(page_id);
                            }
                        }
                    }
                    PageState::UNLOCKED => {
                        // if it's unmarked, we mark it and wait till we wrap around
                        entry.mark(old);
                    }
                    _ => (),
                }
            }
        }
        for page_id in &to_write {
            unsafe {
                if let Err(e) = self.disk_write_page(*page_id) {
                    error!("failed to write page {}: {}", page_id, e);
                }
            };
            let entry = self.get_page_entry(*page_id);
            entry.mark_clean();
            to_evict.push(*page_id);
        }
        for page_id in &to_evict {
            unsafe {
                if let Err(e) = self.mem.release(*page_id) {
                    error!("failed to release page {}: {}", page_id, e);
                }
            }
        }
        for page_id in &to_evict {
            let entry = self.get_page_entry(*page_id);
            entry.unlock_evicted();
        }
    }
}

#[derive(Debug)]
struct FsInfo {
    disk_size: u64,
    max_file_size: u64,
}

fn max_file_size_from_fs_name(fs_name: &str) -> u64 {
    match fs_name {
        "apfs" => u64::MAX, // 16 EIB (technically it's 1+MAX but this really doesn't matter)
        "ext4" => 16 * TB,
        _ => todo!(),
    }
}

// not sure what we return here. probably some db object.
pub fn open_db(path: &Path) -> Result<PageCache> {
    // create db if it does not exist.
    let f = File::create(path)?;

    // get info about its respective disk's size and maximum supported file size
    let inf = fsinfo::get_fs_info(&f)?;

    // the limit of our virtual memory mapping
    let vm_size_bytes = MAX_PAGES_LARGE * PAGE_SIZE.min(inf.disk_size).min(inf.max_file_size);
    PageCache::new(f, vm_size_bytes, DEFAULT_CACHE_CAPACITY_BYTES)
}

pub struct Page<'a, 'b> {
    page_entry: Ref<'a, PageId, PageEntry, ahash::RandomState>,
    data: &'b [u8],
}

impl Deref for Page<'_, '_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.data
    }
}

impl Drop for Page<'_, '_> {
    fn drop(&mut self) {
        self.page_entry.unlock_shared()
    }
}

pub struct PageMut<'a, 'b> {
    page_entry: Ref<'a, PageId, PageEntry, ahash::RandomState>,
    data: &'b mut [u8],
}

impl Deref for PageMut<'_, '_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.data
    }
}

impl DerefMut for PageMut<'_, '_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.page_entry.mark_dirty();
        &mut self.data
    }
}

impl Drop for PageMut<'_, '_> {
    fn drop(&mut self) {
        self.page_entry.unlock()
    }
}
