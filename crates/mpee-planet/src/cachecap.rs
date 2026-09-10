//! A ceiling on the resident page cache, so a big dataset can be proven to run
//! on a small machine.
//!
//! Every array in this dataset is a file mapping, which means the memory a
//! query touches is a *cache*, not an allocation: the kernel is free to drop
//! those pages and re-read them from disk. That is the whole reason the planet
//! fits somewhere with 512 MB of RAM. But it is only an argument until the
//! pages are actually taken away, and on a 39 GB development machine they
//! never are — nothing ever applies the pressure.
//!
//! So the process applies it to itself. `CacheCap` watches its own resident
//! set and, when it crosses the budget, hands the mapped pages back. What
//! remains is the search state, which is anonymous memory and cannot be
//! reclaimed. Running a query under a cap therefore measures the two things
//! that decide whether a small machine can serve it: whether the answer is
//! still right (it must be — dropping a clean page changes nothing), and how
//! much slower it gets when the pages have to come back off the disk.

use memmap2::Mmap;

/// This process's resident set size, in bytes.
#[cfg(target_os = "macos")]
pub fn rss_bytes() -> usize {
    // MACH_TASK_BASIC_INFO. `resident_size` counts both anonymous pages and
    // the file-backed pages currently mapped in, which is exactly the figure a
    // machine with a fixed amount of RAM is constrained by.
    #[repr(C)]
    #[derive(Default)]
    struct TaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [i32; 2],
        system_time: [i32; 2],
        policy: i32,
        suspend_count: i32,
    }
    const MACH_TASK_BASIC_INFO: i32 = 20;
    let mut info = TaskBasicInfo::default();
    let mut count = (std::mem::size_of::<TaskBasicInfo>() / std::mem::size_of::<i32>()) as u32;
    extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(task: u32, flavor: i32, out: *mut TaskBasicInfo, count: *mut u32) -> i32;
    }
    unsafe {
        if task_info(mach_task_self(), MACH_TASK_BASIC_INFO, &mut info, &mut count) == 0 {
            return info.resident_size as usize;
        }
    }
    0
}

/// This process's resident set size, in bytes.
#[cfg(not(target_os = "macos"))]
pub fn rss_bytes() -> usize {
    // statm's second field is resident pages.
    let s = match std::fs::read_to_string("/proc/self/statm") {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let pages: usize = match s.split_whitespace().nth(1).and_then(|f| f.parse().ok()) {
        Some(p) => p,
        None => return 0,
    };
    pages * 4096
}

/// A budget on resident memory, enforced by releasing mapped pages.
pub struct CacheCap {
    budget: usize,
    /// Page-aligned spans of the mappings this cap is allowed to release.
    spans: Vec<(usize, usize)>,
    /// How many times the budget was hit. A query that never flushes fits in
    /// the budget outright; one that flushes often is genuinely re-reading.
    pub flushes: u64,
    /// Highest RSS seen, so a query under a generous cap still reports what it
    /// would have needed.
    pub peak: usize,
}

impl CacheCap {
    pub fn new(budget_bytes: usize) -> CacheCap {
        CacheCap { budget: budget_bytes, spans: Vec::new(), flushes: 0, peak: 0 }
    }

    /// Put a set of mappings under the cap.
    ///
    /// Only whole pages can be released, so a mapping shorter than a page — or
    /// the partial page at the end of one — is simply left alone. Those are
    /// bytes, not gigabytes.
    pub fn govern(&mut self, maps: &[Mmap]) {
        let page = page_size();
        for m in maps {
            let start = m.as_ptr() as usize;
            let end = start + m.len();
            let lo = start.next_multiple_of(page);
            let hi = end & !(page - 1);
            if hi > lo {
                self.spans.push((lo, hi - lo));
            }
        }
    }

    /// Release every governed page if the resident set is over budget.
    ///
    /// The pages are clean, file-backed and read-only, so this cannot lose
    /// data or change an answer; the next access faults them back in.
    pub fn enforce(&mut self) {
        let rss = rss_bytes();
        self.peak = self.peak.max(rss);
        if rss <= self.budget {
            return;
        }
        self.flushes += 1;
        for &(addr, len) in &self.spans {
            unsafe { release(addr, len) };
        }
    }

    pub fn budget(&self) -> usize {
        self.budget
    }
}

fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// Hand a span of clean file-backed pages back to the kernel.
unsafe fn release(addr: usize, len: usize) {
    let p = addr as *mut libc::c_void;
    // On Linux MADV_DONTNEED drops the pages outright. On macOS it is advisory
    // and often a no-op for file mappings, where msync(MS_INVALIDATE) is what
    // actually invalidates them — so ask both ways and let the platform take
    // whichever it honours.
    libc::madvise(p, len, libc::MADV_DONTNEED);
    #[cfg(target_os = "macos")]
    libc::msync(p, len, libc::MS_INVALIDATE);
}


/// Ask the scheduler which kind of core this thread belongs on.
///
/// Apple Silicon has no thread affinity — `THREAD_AFFINITY_POLICY` returns
/// `KERN_NOT_SUPPORTED` — so a program states an *intent* and the kernel picks
/// the cluster. The three that matter here behave very differently, and the
/// difference is not a gradient:
///
/// - `background` runs on the efficiency cores only, at their low background
///   clock, and can never be promoted to a performance core even when one is
///   idle. Measured on this warm-up: 3 rows/s against 114 on the performance
///   cores.
/// - `utility` prefers the performance cores and spills to efficiency ones
///   when they are full — and a spilled thread clocks the efficiency cluster
///   *up*, to something like two thirds of performance speed. This is what a
///   long batch job wants: all the machine, without taking the machine.
/// - `default` is what a foreground process already has.
///
/// It has to be set on each worker, because a thread inherits the QoS of
/// whoever spawned it, and it cannot be set from outside: `taskpolicy -b` can
/// only demote a running process, never promote it back.
#[cfg(target_os = "macos")]
pub fn set_thread_qos(name: &str) -> bool {
    // From `pthread/qos.h`. Declared here rather than taken from `libc`,
    // which does not name the constants in every version.
    const BACKGROUND: u32 = 0x09;
    const UTILITY: u32 = 0x11;
    const DEFAULT: u32 = 0x15;
    const USER_INITIATED: u32 = 0x19;
    extern "C" {
        fn pthread_set_qos_class_self_np(class: u32, priority: i32) -> i32;
    }
    let class = match name {
        "background" => BACKGROUND,
        "utility" => UTILITY,
        "default" => DEFAULT,
        "user_initiated" => USER_INITIATED,
        _ => return false,
    };
    unsafe { pthread_set_qos_class_self_np(class, 0) == 0 }
}

#[cfg(not(target_os = "macos"))]
pub fn set_thread_qos(_name: &str) -> bool {
    false
}
