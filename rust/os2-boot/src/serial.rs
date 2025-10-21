use core::fmt::{self, Write};
use spin::Mutex;
use uart_16550::SerialPort;

static SERIAL: Mutex<Option<SerialPort>> = Mutex::new(None);

pub fn init() {
    let mut guard = SERIAL.lock();
    if guard.is_none() {
        let mut port = unsafe { SerialPort::new(0x3F8) };
        port.init();
        *guard = Some(port);
    }
}

pub struct SerialWriter;

impl Write for SerialWriter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let mut guard = SERIAL.lock();
        if let Some(port) = guard.as_mut() {
            for byte in s.bytes() {
                if byte == b'\n' {
                    port.send(b'\r');
                }
                port.send(byte);
            }
            Ok(())
        } else {
            Err(core::fmt::Error)
        }
    }
}

pub fn writer() -> SerialWriter {
    SerialWriter
}
