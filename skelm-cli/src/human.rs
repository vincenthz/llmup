use std::time::Duration;

const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];

/// print human size string
pub fn size_units(bytes: u64) -> String {
    let mut out = [0; 5];
    repeated_div(&mut out, bytes, 1024);

    for (i, u) in UNITS.iter().enumerate().skip(1).rev() {
        if out[i] > 0 {
            return format!("{}.{:03} {}", out[i], out[i - 1], u);
        }
    }
    format!("{} {}", out[0], UNITS[0])
}

pub fn bench_duration_units(dur: Duration) -> String {
    const TIME_UNITS: [&str; 4] = ["ns", "us", "ms", "s"];
    let mut out = [0; 3];
    let seconds = repeated_div_by(&mut out, dur.as_nanos(), &[1000, 1000, 1000]);
    let [ns, us, ms] = out;
    if seconds > 0 {
        format!("{}.{:3} {}", seconds, ms, TIME_UNITS[3])
    } else if ms > 0 {
        format!("{}.{:3} {}", ms, us, TIME_UNITS[2])
    } else if us > 0 {
        format!("{}.{:3} {}", us, ns, TIME_UNITS[1])
    } else {
        format!("{:3} {}", ns, TIME_UNITS[0])
    }
}

pub fn duration_units(dur: Duration) -> String {
    const TIME_UNITS: [&str; 7] = [
        "seconds", "minutes", "hours", "days", "weeks", "months", "years",
    ];
    // seconds, minutes, hours, days
    let mut out = [0; 3];
    let days = repeated_div_by(&mut out, dur.as_secs(), &[60, 60, 24]);
    let [seconds, minutes, hours] = out;
    if days == 0 {
        if hours > 0 {
            format!("{} {}", hours, TIME_UNITS[2])
        } else if minutes > 0 {
            format!("{} {}", minutes, TIME_UNITS[1])
        } else {
            format!("{} {}", seconds, TIME_UNITS[0])
        }
    } else {
        if days > 365 {
            format!("{} {}", days / 365, TIME_UNITS[6])
        } else if days >= 60 {
            format!("{} {}", days / 30, TIME_UNITS[5])
        } else if days >= 14 {
            format!("{} {}", days / 7, TIME_UNITS[4])
        } else {
            format!("{} {}", days, TIME_UNITS[3])
        }
    }
}

pub fn repeated_div<const N: usize, T>(out: &mut [T; N], value: T, base: T)
where
    T: Copy + core::ops::Div<Output = T> + core::ops::Rem<Output = T>,
{
    let mut rem = value;
    for o in out.iter_mut() {
        *o = rem % base;
        rem = rem / base;
    }
}

pub fn repeated_div_by<const N: usize, T>(out: &mut [T; N], value: T, divisors: &[T; N]) -> T
where
    T: Copy + core::ops::Div<Output = T> + core::ops::Rem<Output = T>,
{
    let mut rem = value;
    for (o, base) in out.iter_mut().zip(divisors.iter()) {
        *o = rem % *base;
        rem = rem / *base;
    }
    rem
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repeated_div() {
        let mut out = [0, 0, 0];
        let rem = repeated_div_by(&mut out, 1_234_456_789, &[1000, 1000, 1000]);

        assert_eq!(rem, 1);
        assert_eq!(out[2], 234);
        assert_eq!(out[1], 456);
        assert_eq!(out[0], 789);
    }
}
