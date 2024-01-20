use crate::{Result, FsInfo, max_file_size_from_fs_name};
use std::{io, fs::File};

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