use crate::{Result, FsInfo, max_file_size_from_fs_name};
use std::{io, fs::File};

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