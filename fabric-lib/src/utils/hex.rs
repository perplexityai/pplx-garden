use std::num::ParseIntError;

use bytes::{BufMut, Bytes, BytesMut};

pub fn fmt_hex(f: &mut std::fmt::Formatter<'_>, bytes: &[u8]) -> std::fmt::Result {
    for x in bytes {
        write!(f, "{:02x}", x)?;
    }
    Ok(())
}

pub fn from_hex(s: &str) -> Result<Bytes, ParseIntError> {
    let mut bytes = BytesMut::with_capacity(s.len() / 2);
    for i in (0..s.len()).step_by(2) {
        bytes.put_u8(u8::from_str_radix(&s[i..i + 2], 16)?);
    }
    Ok(bytes.freeze())
}
