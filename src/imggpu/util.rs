pub fn padded_bytes_per_row(width: u32) -> usize {
    let bytes_per_row = width as usize * 4;
    let padding = (256 - bytes_per_row % 256) % 256;
    bytes_per_row + padding
}

pub fn int_div_round_up(divisor: u32, dividend: u32) -> u32 {
    (divisor / dividend)
        + match divisor % dividend {
            0 => 0,
            _ => 1,
        }
}
