//! Double-array trie with a per-node bitmap, packed one node per cache line.

use std::collections::VecDeque;

/// Sentinel in `check[]` for an unoccupied slot.
const FREE: i32 = -1;
/// Sentinel in `node_hot.token_id` for a non-terminal node.
const NON_TERMINAL: i32 = -1;
/// Sentinel in `node_to_slot[]` for a node that hasn't been placed yet.
const UNPLACED: i32 = -1;

/// Empirical multiplier — a double-array trie packs out to ~5-6x node count.
const FILL_FACTOR: usize = 6;
/// Room for the full byte alphabet at the root.
const ALPHABET_PAD: usize = 256;
/// Extra slack so tiny vocabs don't immediately resize.
const SMALL_VOCAB_SLACK: usize = 1024;

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct NodeHot {
    /// 256-bit child-presence bitmap.
    pub bitmap: [u64; 4],
    /// Darts `base[slot]`.
    pub base: i32,
    /// Terminal token id, or NON_TERMINAL (-1) for non-terminal slots.
    pub token_id: i32,
    /// Terminal log-score; meaningless for non-terminal slots.
    pub score: f64,
    /// Pad to 64 B.
    _pad: [u8; 16],
}

impl Default for NodeHot {
    #[inline(always)]
    fn default() -> Self {
        Self {
            bitmap: [0; 4],
            base: 0,
            token_id: NON_TERMINAL,
            score: 0.0,
            _pad: [0; 16],
        }
    }
}

impl NodeHot {
    // `byte >> 6` picks one of four 64-bit words; `byte & 63` is the bit inside it.
    #[inline(always)]
    fn bitmap_set(&mut self, byte: u8) {
        self.bitmap[(byte >> 6) as usize] |= 1u64 << (byte & 63);
    }
    #[inline(always)]
    pub fn bitmap_test(&self, byte: u8) -> bool {
        self.bitmap[(byte >> 6) as usize] & (1u64 << (byte & 63)) != 0
    }
}

pub struct DartsPackedTrie {
    pub node_hot: &'static [NodeHot],
}

#[derive(Default)]
struct TempNode {
    token_id: i32,
    children: Vec<(u8, u32)>,
}

/// Walk the free-list starting at `start` and return the first slot whose
/// `check` entry is FREE. Path-compresses visited links into the answer so a
/// later scan from any earlier position jumps straight past known-occupied
/// prefixes (amortized constant time over the construction).
fn next_free(start: usize, free_link: &mut [usize], check: &[i32]) -> usize {
    let mut slot = start;
    while slot < check.len() && check[slot] != FREE {
        slot = free_link[slot];
    }
    let mut cur = start;
    while cur < slot {
        let next = free_link[cur];
        free_link[cur] = slot;
        cur = next;
    }
    slot
}

impl DartsPackedTrie {
    pub fn from_vocab(vocab: &[Vec<u8>], vocab_scores: &[f64]) -> Self {
        assert_eq!(vocab.len(), vocab_scores.len(), "vocab and scores length mismatch");
        assert!(vocab.iter().all(|t| !t.is_empty()), "vocab contains an empty token");

        // 1. Build a temporary pointer-trie from the vocabulary.
        let mut nodes: Vec<TempNode> =
            vec![TempNode { token_id: NON_TERMINAL, children: Vec::new() }];
        for (tid, bytes) in vocab.iter().enumerate() {
            let mut cur: u32 = 0;
            for &b in bytes {
                let next = nodes[cur as usize]
                    .children
                    .iter()
                    .find(|(byte, _)| *byte == b)
                    .map(|(_, c)| *c);
                cur = match next {
                    Some(c) => c,
                    None => {
                        let new_id = nodes.len() as u32;
                        nodes.push(TempNode {
                            token_id: NON_TERMINAL,
                            children: Vec::new(),
                        });
                        nodes[cur as usize].children.push((b, new_id));
                        new_id
                    }
                };
            }
            nodes[cur as usize].token_id = tid as i32;
        }
        // Sort children by byte. The placement loop below reads `children[0]`
        // and `children[len-1]` as the min/max byte for collision bookkeeping
        // and to size each parent's slot window, so this ordering is
        // load-bearing, not cosmetic.
        for n in &mut nodes {
            n.children.sort_unstable_by_key(|(b, _)| *b);
        }

        // 2. Darts BFS placement. `check[]` is local to construction — used
        // only to detect collisions while choosing base offsets.
        let init_cap = (nodes.len() * FILL_FACTOR + ALPHABET_PAD + SMALL_VOCAB_SLACK)
            .next_power_of_two();
        let mut check: Vec<i32> = vec![FREE; init_cap];
        let mut base: Vec<i32> = vec![0; init_cap];
        let mut token_id: Vec<i32> = vec![NON_TERMINAL; init_cap];
        let mut nf: Vec<usize> = (0..init_cap).map(|i| i + 1).collect();

        let mut node_to_slot: Vec<i32> = vec![UNPLACED; nodes.len()];
        node_to_slot[0] = 0;
        token_id[0] = nodes[0].token_id;
        check[0] = 0;

        let mut queue: VecDeque<u32> = VecDeque::new();
        queue.push_back(0);

        while let Some(tmp_id) = queue.pop_front() {
            let parent_slot = node_to_slot[tmp_id as usize] as usize;
            let children = &nodes[tmp_id as usize].children;
            if children.is_empty() {
                continue;
            }
            let first_byte = children[0].0 as usize;
            let last_byte = children[children.len() - 1].0 as usize;

            let mut candidate = next_free(first_byte + 1, &mut nf, &check);

            loop {
                let max_slot = candidate + (last_byte - first_byte);
                if max_slot >= check.len() {
                    let old = check.len();
                    let new_len = (max_slot + 1).next_power_of_two();
                    base.resize(new_len, 0);
                    check.resize(new_len, FREE);
                    token_id.resize(new_len, NON_TERMINAL);
                    nf.resize(new_len, 0);
                    for j in old..new_len {
                        nf[j] = j + 1;
                    }
                }

                let try_base = candidate - first_byte;
                debug_assert!(try_base >= 1);

                let mut ok = true;
                for &(b, _) in children {
                    let slot = try_base + b as usize;
                    if check[slot] != FREE {
                        ok = false;
                        break;
                    }
                }

                if ok {
                    base[parent_slot] = try_base as i32;
                    for &(b, child_tmp) in children {
                        let slot = try_base + b as usize;
                        check[slot] = parent_slot as i32;
                        token_id[slot] = nodes[child_tmp as usize].token_id;
                        node_to_slot[child_tmp as usize] = slot as i32;
                        queue.push_back(child_tmp);
                    }
                    break;
                }

                candidate = next_free(candidate + 1, &mut nf, &check);
            }
        }

        // Trim trailing free slots.
        let mut last_used = check.len();
        while last_used > 0 && check[last_used - 1] == FREE {
            last_used -= 1;
        }

        // 3. Materialize `node_hot[]` from base/token_id/score + bitmap of
        // each parent's children. `check[]` is intentionally dropped.
        let mut node_hot: Vec<NodeHot> = vec![NodeHot::default(); last_used];
        for slot in 0..last_used {
            node_hot[slot].base = base[slot];
            node_hot[slot].token_id = token_id[slot];
            if token_id[slot] >= 0 {
                node_hot[slot].score = vocab_scores[token_id[slot] as usize];
            }
        }
        for (tmp_id, n) in nodes.iter().enumerate() {
            let parent_slot = node_to_slot[tmp_id] as usize;
            for &(b, _) in &n.children {
                node_hot[parent_slot].bitmap_set(b);
            }
        }

        // 4. Move the table into a 2 MB-huge-page-backed region on Linux when
        // the HugeTLB pool is available; otherwise leak the boxed slice. The
        // trie lives for the process so leaking is fine.
        let node_hot: &'static [NodeHot] = allocate_static(node_hot);

        Self { node_hot }
    }
}

#[cfg(all(feature = "huge-pages", target_os = "linux"))]
const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

#[cfg(all(feature = "huge-pages", target_os = "linux"))]
fn allocate_static(node_hot: Vec<NodeHot>) -> &'static [NodeHot] {
    let needed = node_hot.len() * std::mem::size_of::<NodeHot>();
    let alloc_len = (needed + (HUGE_PAGE_SIZE - 1)) & !(HUGE_PAGE_SIZE - 1);
    let raw = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            alloc_len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB,
            -1,
            0,
        )
    };
    if raw == libc::MAP_FAILED {
        // No HugeTLB pool available. Fall back to a plain boxed slice.
        return Box::leak(node_hot.into_boxed_slice());
    }
    let n = node_hot.len();
    unsafe {
        std::ptr::copy_nonoverlapping(node_hot.as_ptr(), raw as *mut NodeHot, n);
    }
    unsafe { std::slice::from_raw_parts(raw as *const NodeHot, n) }
}

#[cfg(not(all(feature = "huge-pages", target_os = "linux")))]
fn allocate_static(node_hot: Vec<NodeHot>) -> &'static [NodeHot] {
    Box::leak(node_hot.into_boxed_slice())
}
