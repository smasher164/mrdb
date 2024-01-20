use crate::{Result, Mmap, Pread, Pwrite, FsInfo, max_file_size_from_fs_name};
use core::ffi::c_void;
use std::{io, fs::File};

pub unsafe fn release(buf: *mut u8, size: u64) -> Result<()> {
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
    if VirtualFree(buf as *mut c_void, size as usize, MEM_RELEASE) == 0 {
        Err(io::Error::last_os_error().into())
    } else {
        Ok(())
    }
}

impl Mmap {
    pub fn create_virt_mem(vm_size: u64) -> Result<Mmap> {
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

// Note: Windows will advance the file pointer.
// A correct implementation of pread/pwrite would reset it.
// But we always use pread so this isn't an issue.
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