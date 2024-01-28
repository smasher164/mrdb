use std::{path::Path};

use mrdb;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() {
    // let mut file = File::create("foo.txt").unwrap();
    // mrdb::get_fs_info(&file).unwrap();
    // yay! miri is happy
    // unsafe {
    //     let mut s = MaybeUninit::<[MaybeUninit<u16>; 4]>::uninit().assume_init();
    //     fill_half(s.as_mut_ptr() as *mut u16);
    //     let us = U16CStr::from_ptr_truncate(s.as_ptr() as *const u16, s.len()).unwrap();
    //     println!("{}", us.display());
    //     println!("{}", us.len());
    // };
    // unsafe {
    //     let mut s = MaybeUninit::<[MaybeUninit<i8>; 4]>::uninit().assume_init();
    //     fill_half8(s.as_mut_ptr() as *mut i8);
    //     let s = CStr::from_ptr(s.as_ptr() as _).to_str().unwrap();
    //     println!("{}", s);
    // };
    // let f = File::open("/home/akhil/Projects/mrdb/README.md").unwrap();
    // let fs_info = mrdb::get_fs_info(&f).unwrap();
    // println!("{:?}", fs_info);
    let stdout_log = tracing_subscriber::fmt::layer().compact();
    tracing_subscriber::registry().with(stdout_log).init();
    let db = mrdb::open_db(Path::new("hola")).unwrap();
    let pid = mrdb::PageId::new_pid(1, 0);
    db.dump_page(pid);
    {
    // println!("pid = {}", pid);
    let mut pg = db.write_page(pid).unwrap();
    pg[0] = 72;
    pg[1] = 69;
    pg[2] = 76;
    pg[3] = 76;
    }
    db.flush_pid(pid);
}
