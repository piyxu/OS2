#![no_std]
#![no_main]

use bootloader::{entry_point, BootInfo};
use core::fmt::Write;
use x86_64::instructions::hlt;

mod serial;

entry_point!(kernel_main);

fn kernel_main(_boot_info: &'static BootInfo) -> ! {
    serial::init();
    let mut writer = serial::writer();
    let _ = writeln!(writer, "PIYXU OS2 boot image ready - awaiting evolution directives...");
    loop {
        hlt();
    }
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    serial::init();
    let mut writer = serial::writer();
    let _ = writeln!(writer, "KERNEL PANIC: {info}");
    loop {
        hlt();
    }
}
