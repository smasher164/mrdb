#![feature(new_uninit)]

use dashmap::{mapref::one::RefMut, DashMap};
use init_array::init_boxed_slice;
use std::{
    cmp::min,
    fs::File,
    io,
    path::Path,
    str::Utf8Error,
    sync::atomic::{AtomicU64, Ordering},
};

use thiserror::Error;

#[cfg_attr(target_family = "unix", path = "os_unix.rs")]
#[cfg_attr(target_family = "windows", path = "os_windows.rs")]
mod os;
pub use os::*;

#[cfg_attr(target_os = "linux", path = "fsinfo_linux.rs")]
#[cfg_attr(
    all(target_family = "unix", not(target_os = "linux")),
    path = "fsinfo_unix.rs"
)]
#[cfg_attr(target_family = "windows", path = "fsinfo_windows.rs")]
mod fsinfo;
pub use fsinfo::*;

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

// TODO: use async i/o for disk reads and writes instead of pread/pwrite.

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
        release(self.to_ptr(pid), pid.size())
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
struct PageId(u64);
impl PageId {
    fn is_empty(self) -> bool {
        self.page_count() == 0
    }
    fn is_tombstone(self) -> bool {
        self.page_count() == 0xFFF
    }
    // should this return a u16?
    fn page_count(self) -> u64 {
        let n = (self.0 & PAGE_COUNT_MASK); // no longer +1, because when cleared it's empty.
        n // for now, keep it simple.
          // n*n*n // 1, 8, 27, ...
    }
    fn size(self) -> u64 {
        self.page_count() * PAGE_SIZE // 4K,32K,81K,256K,500K,864K,1372K,..256TB
    }
    fn pid(self) -> u64 {
        (self.0 & PID_MASK) >> 12
    }
    fn offset(self) -> u64 {
        self.pid() * PAGE_SIZE
    }
    fn new_pid(page_count: u64, pid: u64) -> Self {
        Self((pid << 12) | (page_count - 1))
    }
    fn new_size_pid(size: u64, pid: u64) -> Self {
        let page_count = size / PAGE_SIZE;
        Self::new_pid(page_count, pid)
    }
    fn new_offset(page_count: u64, offset: u64) -> Self {
        let pid = offset / PAGE_SIZE;
        Self::new_pid(page_count, offset)
    }
    fn new_size_offset(size: u64, offset: u64) -> Self {
        let page_count = size / PAGE_SIZE;
        Self::new_offset(page_count, offset)
    }
    fn unpack(self) -> (u64, u64) {
        (self.size(), self.pid())
    }
    fn unpack_offset(self) -> (u64, u64) {
        (self.size(), self.offset())
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

struct PageEntry(AtomicU64); // TODO: maybe we'll need a dirty bit for logging
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
}

impl Default for PageEntry {
    fn default() -> Self {
        Self::new()
    }
}

pub struct PageCache {
    mem: Mmap,
    page_state: DashMap<PageId, PageEntry, ahash::RandomState>, // Should the key be PageId or u64?
    f: File,
    residents: ResidentSet,
}

struct ResidentSet {
    hand: AtomicU64,
    entries: Box<[AtomicU64]>,
    rand_state: ahash::RandomState,
    mask: u64,
    used: AtomicU64,
    capacity: u64,
}

impl ResidentSet {
    fn new(size: u64) -> ResidentSet {
        let size = ((size as f64 * 1.5) as u64).next_power_of_two();
        let mask = size - 1;
        let hand = AtomicU64::new(0);
        let entries = init_boxed_slice(size as usize, |_| AtomicU64::new(0));
        let rand_state = ahash::RandomState::new();
        let used = AtomicU64::new(0);
        ResidentSet {
            hand,
            entries,
            mask,
            rand_state,
            used,
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
                    self.used.fetch_add(1, Ordering::SeqCst);
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
                    self.used.fetch_sub(1, Ordering::SeqCst);
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

impl PageCache {
    pub fn new(f: File, vm_size_bytes: u64, cache_elems: u64) -> Result<PageCache> {
        // hint that we're going to be doing random reads
        // maybe this should be done outside of new?
        #[cfg(all(
            target_family = "unix",
            not(any(target_os = "macos", target_os = "ios"))
        ))]
        advise_random_read(&f)?;
        /*
        vmSize = min(
            MAX_PAGES * PAGE_SIZE,
            MAX_PAGES_LARGE * PAGE_SIZE,
            disk_size,
            max_file_size,
        )
        count = vmSize / PAGE_SIZE
        */
        let mem = Mmap::create_virt_mem(vm_size_bytes)?;
        let ahasher = ahash::RandomState::new();
        let page_state = DashMap::with_hasher(ahasher);
        let residents = ResidentSet::new(cache_elems);
        Ok(PageCache {
            mem,
            page_state,
            f,
            residents,
        })
    }
    fn get_page_entry(&self, page_id: PageId) -> RefMut<'_, PageId, PageEntry, ahash::RandomState> {
        // &self.page_state[page_id.pid() as usize]
        let entry = self.page_state.entry(page_id);
        let or_def = entry.or_default();
        or_def
    }
    unsafe fn read_page(&self, page_id: PageId) -> Result<u64> {
        let buf = self.mem.to_ptr(page_id);
        let offset = page_id.offset();
        let size = page_id.size();
        self.f.pread(buf, offset, size)
    }
    unsafe fn write_page(&self, page_id: PageId) -> Result<u64> {
        // should unset dirty bit.
        let m = self.mem.to_slice_mut(page_id);
        m[0] = 0;
        let buf = self.mem.to_ptr(page_id);
        let offset = page_id.offset();
        let size = page_id.size();
        self.f.pwrite(buf, offset, size)
    }
    fn ensure_free_pages(&self) {
        if (1 + self.residents.used.load(Ordering::SeqCst))
            >= (((self.residents.entries.len() as f64) * 0.95) as u64)
        {
            self.evict();
        }
    }
    // fn readpg(self, pid: PageId) -> *mut u8 {
    //     pid.0
    // }
    // unsafe fn read_page(self, pid: PageId, offset: u64) -> Result<usize> {
    //     self.f.pread(self.mem.to_ptr(pid), offset)
    // }
    // unsafe fn write_page(self, pid: PageId, offset: u64) -> Result<usize> {
    //     self.f.pwrite(self.mem.to_ptr(pid), offset)
    // }
    unsafe fn fix(&self, page_id: PageId) -> Result<()> {
        let entry = self.get_page_entry(page_id);
        loop {
            let old = entry.0.load(Ordering::SeqCst);
            match PageState::get(old) {
                PageState::EVICTED => {
                    // cache miss
                    if entry.lock_weak(old) {
                        // ensure there's enough space to read this page
                        // then actually read it.
                        self.ensure_free_pages();
                        self.read_page(page_id)?;
                        self.residents.insert(page_id);
                        return Ok(());
                        // return self.mem.to_slice(page_id) // do we want read or write?
                    }
                }
                PageState::MARKED | PageState::UNLOCKED => {
                    // cache hit
                    if entry.lock_weak(old) {
                        return Ok(());
                        // return self.mem.to_slice(page_id) // do we want read or write?
                    }
                }
                _ => (),
            }
        }
    }
    fn unfix(&self, page_id: PageId) {
        let entry = self.get_page_entry(page_id);
        let old = entry.0.load(Ordering::SeqCst);
        let new = PageState::inc(old, PageState::UNLOCKED);
        entry.0.store(new, Ordering::Release);
    }
    unsafe fn optimistic_read<F>(&self, page_id: PageId, f: F)
    where
        F: Fn(&[u8]) -> (),
    {
        let entry = self.get_page_entry(page_id);
        loop {
            // TODO: consider falling back to exclusive locking to prevent repeated restarts.
            let old = entry.0.load(Ordering::SeqCst); // does this need to be seqcst?
            match PageState::get(old) {
                PageState::UNLOCKED => {
                    // pass into function
                    let slice = self.mem.to_slice(page_id);
                    f(slice);
                    if entry.0.load(Ordering::SeqCst) == old {
                        return;
                    }
                }
                PageState::MARKED => {
                    let new = PageState::set(old, PageState::UNLOCKED);
                    entry.cas_weak(old, new);
                }
                PageState::EVICTED => {
                    self.fix(page_id);
                    self.unfix(page_id);
                }
                _ => (),
            }
        }
    }
    // eviction based on CLOCK, but consider trying SIEVE
    // https://cachemon.github.io/SIEVE-website/blog/2023/12/17/sieve-is-simpler-than-lru/
    fn evict(&self) {
        let mut to_evict: Vec<PageId> = Vec::with_capacity(BATCH_SIZE as usize);
        let mut to_write: Vec<PageId> = Vec::with_capacity(BATCH_SIZE as usize);
        while ((to_evict.len() + to_write.len()) as u64) < BATCH_SIZE {
            if let Some(page_id) = self.residents.next() {
                let entry = self.get_page_entry(page_id);
                let old = entry.0.load(Ordering::SeqCst);
                match PageState::get(old) {
                    PageState::MARKED => {
                        // if it's marked, we evict
                        if PageState::is_dirty(old) {
                            if entry.lock_shared(old) {
                                to_write.push(page_id);
                            }
                        } else {
                            to_evict.push(page_id);
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
            unsafe { self.write_page(*page_id) }; // TODO: make this async
        }
        to_evict.retain(|page_id| {
            let entry = self.get_page_entry(*page_id);
            let old = entry.0.load(Ordering::SeqCst);
            PageState::get(old) == PageState::MARKED && entry.lock(old)
        });
        for page_id in &to_write {
            let entry = self.get_page_entry(*page_id);
            entry.mark_clean(); // unset dirty bit. we do it here since we already got the entry.
            let old = entry.0.load(Ordering::SeqCst);
            if PageState::get(old) == PageState::LOCK_MIN
                && entry.cas_weak(old, PageState::set(old, PageState::LOCKED))
            {
                to_evict.push(*page_id);
            } else {
                entry.unlock_shared();
            }
        }
        for page_id in &to_evict {
            unsafe {
                self.mem.release(*page_id);
            }
        }
        for page_id in &to_evict {
            self.residents.remove(*page_id);
            let entry = self.get_page_entry(*page_id);
            entry.unlock_evicted();
        }
    }
}

#[derive(Debug)]
pub struct FsInfo {
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
fn open_db(path: &Path) -> Result<()> {
    let f = File::create(path)?;
    let inf = get_fs_info(&f)?;
    // maybe advise random reads here?
    /*        vmSize = min(
        MAX_PAGES * PAGE_SIZE,
        MAX_PAGES_LARGE * PAGE_SIZE,
        disk_size,
        max_file_size,
    )
    count = vmSize / PAGE_SIZE */
    let vm_size_bytes = (MAX_PAGES * PAGE_SIZE)
        .min(MAX_PAGES_LARGE * PAGE_SIZE)
        .min(inf.disk_size)
        .min(inf.max_file_size);
    // cache elems 
    // variable size 63.984375 mb
    // 
    PageCache::new(vm_size_bytes, _);
    Ok(())
}
