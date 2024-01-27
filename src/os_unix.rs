use crate::{Result, Mmap, Pread, Pwrite};
use core::ffi::c_void;
use std::{io, fs::File};

pub unsafe fn release(buf: *mut u8, size: u64) -> Result<()> {
    use libc::{madvise, MADV_DONTNEED};
    if madvise(buf as *mut c_void, size as usize, MADV_DONTNEED) != 0 {
        Err(io::Error::last_os_error().into())
    } else {
        Ok(())
    }
}

impl Mmap {
    pub fn create_virt_mem(vm_size: u64) -> Result<Mmap> {
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

#[cfg(all(target_family = "unix", not(any(target_os = "macos", target_os = "ios"))))]
pub fn advise_random_read(f: &File) -> Result<()> {
    unsafe {
        use libc::{posix_fadvise, POSIX_FADV_RANDOM};
        use std::os::unix::io::AsRawFd;
        if posix_fadvise(f.as_raw_fd(), 0, 0, POSIX_FADV_RANDOM) != 0 {
            return Err(io::Error::last_os_error().into());
        }
    }
    Ok(())
}