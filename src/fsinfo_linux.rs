use crate::{Result, FsInfo, max_file_size_from_fs_name};
use std::{io, fs::File};

fn fs_name_from_f_type(f_type: i64) -> &'static str {
    match f_type {
        libc::EXT4_SUPER_MAGIC => "ext4",
        _ => todo!(),
    }
}

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