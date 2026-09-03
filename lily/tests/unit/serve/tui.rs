use std::time::Duration;

use super::*;

#[test]
fn uptime_formats_as_hours_minutes_seconds() {
    assert_eq!(fmt_uptime(Duration::from_secs(0)), "0:00:00");
    assert_eq!(fmt_uptime(Duration::from_secs(59)), "0:00:59");
    assert_eq!(fmt_uptime(Duration::from_secs(3_600 + 120 + 5)), "1:02:05");
    assert_eq!(fmt_uptime(Duration::from_secs(100 * 3_600)), "100:00:00");
}

#[test]
fn gpu_line_says_whether_tensor_ops_are_emulated() {
    assert_eq!(gpu_line(10), "GPU family 10: native tensor units");
    assert_eq!(
        gpu_line(9),
        "GPU family 9: Metal-4 tensor ops emulated, performance not representative"
    );
}
