use ahash::{AHasher, RandomState};
use std::{
    collections::HashMap,
    fs::File,
    io, mem,
    sync::{
        atomic::{AtomicU64, Ordering},
        RwLock, RwLockReadGuard, RwLockWriteGuard,
    },
};

use bitfield_struct::bitfield;
use bitflags::bitflags;
use dashmap::DashMap;
use init_array::init_boxed_slice;
use typed_arena::Arena;

use core::ffi::c_void;

use thiserror::Error;

#[derive(Error, Debug)]
enum Error {
    #[error("{0}")]
    Io(#[from] io::Error),
}

type Result<T> = core::result::Result<T, Error>;

const KB: u64 = 1024;
const MB: u64 = 1024 * KB;
const GB: u64 = 1024 * MB;
const PAGE_SIZE: u64 = 4 * KB;
const PAGE_COUNT_MASK: u64 = PAGE_SIZE - 1;
const PID_MASK: u64 = !PAGE_COUNT_MASK;
const OFFSET_MASK: u64 = PID_MASK;
const MAX_PAGES: u64 = 1 << 52;
const MAX_PAGES_LARGE: u64 = 1 << 40;
const DIRTY_BIT_MASK: u64 = 1 << (57 - 1);
// TODO: disable huge pages (MADV_NOHUGEPAGE)
// TODO: set random reads likely
// TODO: use async i/o for disk reads and writes instead of pread/pwrite.
//       We already don't flush
// TODO: use platform-native page size
// TODO: also tag pointer.

struct Mmap(*mut u8);

#[cfg(target_family = "unix")]
impl Mmap {
    fn create_virt_mem(vm_size: u64) -> Result<Mmap> {
        unsafe {
            use libc::{
                mmap, MAP_ANONYMOUS, MAP_FAILED, MAP_NORESERVE, MAP_PRIVATE, PROT_READ, PROT_WRITE,
            };
            use std::ptr::null_mut;
            match mmap(
                null_mut(),
                vm_size as usize,
                PROT_READ | PROT_WRITE,
                MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE,
                -1,
                0,
            ) {
                MAP_FAILED => Err(io::Error::last_os_error().into()),
                p => Ok(Mmap(p as *mut u8)),
            }
        }
    }
}

#[cfg(target_family = "windows")]
impl Mmap {
    fn create_virt_mem(vm_size: u64) -> Result<Mmap> {
        unsafe {
            use std::ptr::null;
            use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_RESERVE, PAGE_READWRITE};
            let p = VirtualAlloc(null(), vm_size as usize, MEM_RESERVE, PAGE_READWRITE);
            if p.is_null() {
                Err(io::Error::last_os_error().into())
            } else {
                Ok(Mmap(p as *mut u8))
            }
        }
    }
}

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
        // ptr should be tagged with the (page_count-1)
        // Since we're 4K aligned, we have 12 bits free in the LSB.
        // requires masking before use.
        let u_ptr = ptr as u64;
        let page_count = (u_ptr & PAGE_COUNT_MASK) + 1;
        // determine offset into vm
        let offset = ptr.offset_from(self.0) as u64;
        PageId::new_offset(page_count, offset & OFFSET_MASK)
    }
    unsafe fn release(self, pid: PageId) -> Result<()> {
        release(self.to_ptr(pid), pid.size())
    }
}

trait Pread {
    fn pread(&self, buf: *mut u8, offset: u64, size: u64) -> Result<u64>;
}

trait Pwrite {
    fn pwrite(&self, buf: *const u8, offset: u64, size: u64) -> Result<u64>;
}

#[cfg(target_family = "unix")]
impl Pread for File {
    fn pread(&self, buf: *mut u8, offset: u64, size: u64) -> Result<u64> {
        use std::os::unix::fs::FileExt;
        use std::slice::from_raw_parts_mut;
        let slice = unsafe { from_raw_parts_mut(buf, size as usize) };
        self.read_at(slice, offset)
            .map(|us| us as u64)
            .map_err(Into::into)
    }
}

#[cfg(target_family = "unix")]
impl Pwrite for File {
    fn pwrite(&self, buf: *const u8, offset: u64, size: u64) -> Result<u64> {
        use std::os::unix::fs::FileExt;
        use std::slice::from_raw_parts;
        let slice = unsafe { from_raw_parts(buf, size as usize) };
        self.write_at(slice, offset)
            .map(|us| us as u64)
            .map_err(Into::into)
    }
}

// Note: Windows will advance the file pointer.
// A correct implementation of pread/pwrite would reset it.
// But we always use pread so this isn't an issue.
#[cfg(target_family = "windows")]
impl Pread for File {
    fn pread(&self, buf: *mut u8, offset: u64, size: u64) -> Result<u64> {
        use std::os::windows::fs::FileExt;
        use std::slice::from_raw_parts_mut;
        let slice = unsafe { from_raw_parts_mut(buf, size as usize) };
        self.seek_read(slice, offset)
            .map(|us| us as u64)
            .map_err(Into::into)
    }
}

#[cfg(target_family = "windows")]
impl Pwrite for File {
    fn pwrite(&self, buf: *const u8, offset: u64, size: u64) -> Result<u64> {
        use std::os::windows::fs::FileExt;
        let slice = unsafe { from_raw_parts(buf, size as usize) };
        self.seek_write(slice, offset)
            .map(|us| us as u64)
            .map_err(Into::into)
    }
}

#[cfg(target_family = "unix")]
unsafe fn release(buf: *mut u8, size: u64) -> Result<()> {
    use libc::{madvise, MADV_DONTNEED};
    if madvise(buf as *mut c_void, size as usize, MADV_DONTNEED) != 0 {
        Err(io::Error::last_os_error().into())
    } else {
        Ok(())
    }
}

#[cfg(target_family = "windows")]
unsafe fn release(buf: *mut u8, size: u64) -> Result<()> {
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
    if VirtualFree(buf as *mut c_void, size as usize, MEM_RELEASE) == 0 {
        Err(io::Error::last_os_error().into())
    } else {
        Ok(())
    }
}

// actually, bottom 12 bits will store it.
#[derive(Hash, PartialEq, Eq, Copy, Clone)]
struct PageId(u64);
impl PageId {
    // should this return a u16?
    fn page_count(self) -> u64 {
        (self.0 & PAGE_COUNT_MASK) + 1
    }
    fn size(self) -> u64 {
        self.page_count() * PAGE_SIZE
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

// let's ignore synchronization for now
// lock: Mutex<Map<PageId, Arc<Frame>>>,
//     lock: RwLock<Vec<u8>>,

fn advise_random_read(f: &File) -> Result<()> {
    unsafe {
        use libc::{posix_fadvise, POSIX_FADV_RANDOM};
        use std::os::unix::io::AsRawFd;
        if posix_fadvise(f.as_raw_fd(), 0, 0, POSIX_FADV_RANDOM) != 0 {
            return Err(io::Error::last_os_error().into());
        }
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PageState(u64);

// TODO: Change top 7 bits of PageState to hold these tags. to be 7 bits
// Reserve one bit for dirty bit.
// Use rest of 56 bits for ABA counter.
impl PageState {
    const UNLOCKED: PageState = PageState(0);
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
    // fn load(&self) ->
    // fn get_state(self) -> PageState {
    //     PageState(self.0.load(Ordering::SeqCst) >> 56)
    // }
}

impl Default for PageEntry {
    fn default() -> Self {
        Self::new()
    }
}

struct PageCache {
    mem: Mmap,
    page_state: DashMap<PageId, PageEntry, ahash::RandomState>, // Should the key be PageId or u64?
    // page_state: Box<[PageEntry]>,
    // page_state:
    f: File,
    cache_elems: u64,
    // frames: Arena<Frame>,
    // page_table: HashMap<PageId, &'a Frame>,
    // free_list
    // lru
    // free_list
    // storage for frames
    // want to avoid returning index
}

impl PageCache {
    fn new(f: File, vm_size_bytes: u64, cache_elems: u64) -> Result<PageCache> {
        // hint that we're going to be doing random reads
        if cfg!(target_family = "unix") {
            advise_random_read(&f)?;
        }
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
        let page_state = DashMap::with_hasher(ahash::RandomState::new());

        // let page_state: Box<[PageEntry]> =
        //     init_boxed_slice(cache_elems as usize, |_| PageEntry::new());

        // TODO: maybe we'll need counts
        Ok(PageCache {
            mem,
            page_state,
            f,
            cache_elems,
        })
    }
    // fn to_ptr
    // should this be owned? or a reference? C++ does refs, but they're mutable.
    // maybe mut ref?
    fn get_page_entry(&self, page_id: PageId) -> &PageEntry {
        // &self.page_state[page_id.pid() as usize]
        self.page_state.entry(page_id).or_default().value()
    }
    unsafe fn read_page(&self, page_id: PageId) -> Result<u64> {
        let buf = self.mem.to_ptr(page_id);
        let offset = page_id.offset();
        let size = page_id.size();
        self.f.pread(buf, offset, size)
    }
    unsafe fn write_page(&self, page_id: PageId) -> Result<u64> {
        let m = self.mem.to_slice_mut(page_id);
        m[0] = 0;
        let buf = self.mem.to_ptr(page_id);
        let offset = page_id.offset();
        let size = page_id.size();
        self.f.pwrite(buf, offset, size)
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
                    let new = PageState::set(old, PageState::LOCKED);
                    if entry.cas_weak(old, new) {
                        // ensure there's enough space to read this page
                        // then actually read it.
                        self.read_page(page_id)?;
                        return Ok(());
                        // return self.mem.to_slice(page_id) // do we want read or write?
                    }
                }
                PageState::MARKED | PageState::UNLOCKED => {
                    // cache hit
                    let new = PageState::set(old, PageState::LOCKED);
                    if entry.cas_weak(old, new) {
                        return Ok(());
                        // return self.mem.to_slice(page_id) // do we want read or write?
                    }
                }
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
            }
        }
    }
    // fn is_dirty(&self, page_id: PageId) -> bool {
    //     0 != unsafe { self.mem.to_slice(page_id)[0] }
    //     // stores dirty/non-dirty in page itself.
    // }
    // eviction based on CLOCK, but consider trying SIEVE
    // https://cachemon.github.io/SIEVE-website/blog/2023/12/17/sieve-is-simpler-than-lru/
    fn evict(&self) {
        const BATCH_SIZE: usize = 64;
        let candidates: Vec<PageId> = self
            .page_state
            .iter()
            .filter_map(|pg| {
                let old = pg.value().0.load(Ordering::SeqCst);
                match PageState::get(old) {
                    PageState::MARKED => {
                        // Obtain exclusive lock to page if it was locked for eviction.
                        let new = PageState::set(old, PageState::LOCKED);
                        if pg.value().cas(old, new) {
                            return Some(*pg.key());
                        }
                    }
                    // PageState::UNLOCKED => todo!(),
                }
                None
                // pg.key()
                // pg.value()
                // we need to store clock pos
                // pg.0.l
                // let old = entry.0.load(Ordering::SeqCst);
            })
            .take(64)
            .collect();
        // Check if page is dirty. If so, write it.
        candidates.iter().for_each(|page_id| {
            let st = self.get_page_entry(*page_id);
            let old = st.0.load(Ordering::SeqCst);
            if PageState::is_dirty(old) {
                unsafe { self.write_page(*page_id) };
            }
        });
        candidates.iter().for_each(|page_id| unsafe {
            let m = self.mem.to_ptr(*page_id);
            release(m, page_id.size());
        });
        candidates.iter().for_each(|page_id| {
            let entry = self.get_page_entry(*page_id);
            let old = entry.0.load(Ordering::SeqCst);
            match PageState::get(old) {
                PageState::LOCKED => {
                    let new = PageState::inc(old, PageState::UNLOCKED);
                    entry.0.store(new, Ordering::Release);
                }
            }
        });
    }
}

pub struct Frame {
    // reference to lru
    lock: RwLock<Vec<u8>>,
}

impl Frame {
    fn read(&self) -> RwLockReadGuard<'_, Vec<u8>> {
        self.lock.read().unwrap()
    }
    fn write(&self) -> RwLockWriteGuard<'_, Vec<u8>> {
        self.lock.write().unwrap()
    }
}

fn read_disk() {}

pub fn foo() {
    // let buf = Page{
    //     lock: Arc::from(RwLock::from(vec![])),
    // };
    // let mut read_guard = buf.write();
    // read_guard.push(42);
    // let v: Vec<u8> = *read_guard;
    // v.push(42);
    // let buf = *read_guard;
    // let x : [u8] = [1, 2];
}
