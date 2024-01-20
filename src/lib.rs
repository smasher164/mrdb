#![feature(new_uninit)]

use dashmap::{mapref::one::RefMut, DashMap};
use init_array::init_boxed_slice;
use std::{
    fs::File,
    io,
    str::Utf8Error,
    sync::atomic::{AtomicU64, Ordering},
};

use core::ffi::c_void;

use thiserror::Error;

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
        use std::ptr::null;
        unsafe {
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
        // ptr should be tagged with the page_count.
        // Since we're 4K aligned, we have 12 bits free in the LSB.
        // requires masking before use.
        let page_count = (ptr as u64) & PAGE_COUNT_MASK;
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
        use std::slice::from_raw_parts;
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

#[cfg(target_family = "unix")]
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
}

impl Default for PageEntry {
    fn default() -> Self {
        Self::new()
    }
}

struct PageCache {
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
    fn new(f: File, vm_size_bytes: u64, cache_elems: u64) -> Result<PageCache> {
        // hint that we're going to be doing random reads
        #[cfg(target_family = "unix")]
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
                        if PageState::is_dirty(old) {
                            if entry.lock_shared(old) {
                                to_write.push(page_id);
                            }
                        } else {
                            to_evict.push(page_id);
                        }
                    }
                    PageState::UNLOCKED => {
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
                let buf = self.mem.to_ptr(*page_id);
                release(buf, page_id.size());
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

#[cfg(target_os = "linux")]
fn fs_name_from_f_type(f_type: i64) -> &'static str {
    match f_type {
        libc::EXT4_SUPER_MAGIC => "ext4",
        _ => todo!(),
    }
}

// Given a file, obtain the
// - The size of the disk it's contained on.
// - The maximum size of a file on the file system it's contained on.
#[cfg(all(target_family = "unix", not(target_os = "linux")))]
pub fn get_fs_info(file: &File) -> Result<FsInfo> {
    use libc::{fstatfs, statfs};
    use std::{ffi::CStr, mem::MaybeUninit, os::fd::AsRawFd};
    let mut buf: MaybeUninit<statfs> = MaybeUninit::uninit();
    unsafe {
        if fstatfs(file.as_raw_fd(), buf.as_mut_ptr()) != 0 {
            return Err(io::Error::last_os_error().into());
        }
        let st = buf.assume_init();
        let fs_name = CStr::from_ptr(st.f_fstypename.as_ptr()).to_str()?;
        let disk_size = st.f_blocks * (st.f_bsize as u64);
        let max_file_size = max_file_size_from_fs_name(fs_name);
        Ok(FsInfo {
            disk_size,
            max_file_size,
        })
    }
}

#[cfg(target_os = "linux")]
pub fn get_fs_info(file: &File) -> Result<FsInfo> {
    use libc::{fstatfs64, statfs64};
    use std::{mem::MaybeUninit, os::fd::AsRawFd};
    let mut buf: MaybeUninit<statfs64> = MaybeUninit::uninit();
    let st = unsafe {
        if fstatfs64(file.as_raw_fd(), buf.as_mut_ptr()) != 0 {
            return Err(io::Error::last_os_error().into());
        }
        buf.assume_init()
    };
    let disk_size = st.f_blocks * (st.f_bsize as u64);
    let fsname = fs_name_from_f_type(st.f_type);
    let max_file_size = max_file_size_from_fs_name(fsname);
    Ok(FsInfo {
        disk_size,
        max_file_size,
    })
}

#[cfg(target_family = "windows")]
pub fn get_fs_info(file: &File) -> Result<FsInfo> {
    use std::{mem::MaybeUninit, os::windows::io::AsRawHandle, ptr::null_mut};
    use widestring::U16CStr;
    use windows_sys::Win32::{
        Foundation::MAX_PATH,
        Storage::FileSystem::{
            GetDiskFreeSpaceExW, GetFinalPathNameByHandleW, GetVolumeInformationByHandleW,
            VOLUME_NAME_GUID,
        },
    };
    let raw_handle = file.as_raw_handle() as isize;
    let max_file_size = unsafe {
        let mut fs_name_buf =
            MaybeUninit::<[MaybeUninit<u16>; MAX_PATH as usize + 1]>::uninit().assume_init();
        if GetVolumeInformationByHandleW(
            raw_handle,
            null_mut(),
            0,
            null_mut(),
            null_mut(),
            null_mut(),
            fs_name_buf.as_mut_ptr() as *mut u16,
            fs_name_buf.len() as u32,
        ) == 0
        {
            return Err(io::Error::last_os_error().into());
        }
        let fs_name =
            U16CStr::from_ptr_truncate(fs_name_buf.as_ptr() as *const u16, fs_name_buf.len())?
                .to_string()?;
        max_file_size_from_fs_name(&fs_name)
    };
    let disk_size = unsafe {
        let path_len = GetFinalPathNameByHandleW(raw_handle, null_mut(), 0, VOLUME_NAME_GUID);
        if path_len == 0 {
            return Err(io::Error::last_os_error().into());
        }
        let mut path_buf = Box::<[u16]>::new_uninit_slice(path_len as usize);
        if GetFinalPathNameByHandleW(
            raw_handle,
            path_buf.as_mut_ptr() as *mut u16,
            path_len,
            VOLUME_NAME_GUID,
        ) != (path_len - 1)
        {
            return Err(io::Error::last_os_error().into());
        }
        let mut path_buf = path_buf.assume_init();
        path_buf[49] = 0; // this is the char after the volume path
        let mut disk_size = MaybeUninit::<u64>::uninit();
        if GetDiskFreeSpaceExW(
            path_buf.as_ptr(),
            null_mut(),
            disk_size.as_mut_ptr(),
            null_mut(),
        ) == 0
        {
            return Err(io::Error::last_os_error().into());
        }
        disk_size.assume_init()
    };
    Ok(FsInfo {
        disk_size,
        max_file_size,
    })
}
