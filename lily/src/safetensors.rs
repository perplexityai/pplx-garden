//! Local safetensors checkpoint reading: index and shard headers plus byte-range reads.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, bail, ensure};
use serde::Deserialize;

/// The index file name a checkpoint directory is expected to contain.
const INDEX_NAME: &str = "model.safetensors.index.json";
/// The single-file checkpoint name used when no index is present.
const SINGLE_FILE_NAME: &str = "model.safetensors";

/// Safetensors dtypes lily can load. Anything else stays `Other` so unrelated
/// tensors (e.g. vision tower) never fail `open`; reading one is an error.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SafetensorsDType {
    BF16,
    F32,
    /// Packed quantized weight codes in mlx checkpoints.
    U32,
    Other(String),
}

impl SafetensorsDType {
    fn from_str(s: &str) -> Self {
        match s {
            "BF16" => Self::BF16,
            "F32" => Self::F32,
            "U32" => Self::U32,
            other => Self::Other(other.to_string()),
        }
    }

    pub fn size(&self) -> Option<usize> {
        match self {
            Self::BF16 => Some(2),
            Self::F32 | Self::U32 => Some(4),
            Self::Other(_) => None,
        }
    }
}

/// Where one tensor lives: its shard file plus absolute byte offsets.
#[derive(Clone)]
pub struct TensorMeta {
    pub dtype: SafetensorsDType,
    pub shape: Vec<usize>,
    shard: PathBuf,
    start: u64,
    end: u64,
}

impl TensorMeta {
    pub fn byte_len(&self) -> usize {
        (self.end - self.start) as usize
    }
}

/// A single tensor's entry in a safetensors JSON header.
#[derive(Deserialize)]
struct HeaderEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Deserialize)]
struct IndexFile {
    #[serde(default)]
    weight_map: BTreeMap<String, String>,
}

/// A parsed checkpoint directory: every tensor name mapped to its location.
pub struct Checkpoint {
    tensors: HashMap<String, TensorMeta>,
}

impl Checkpoint {
    /// Opens the checkpoint under `dir`, via `model.safetensors.index.json`
    /// when present or a single `model.safetensors` otherwise.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let index_path = dir.join(INDEX_NAME);
        let shards: Vec<PathBuf> = if index_path.exists() {
            let bytes = std::fs::read(&index_path)
                .with_context(|| format!("reading {}", index_path.display()))?;
            let index: IndexFile =
                serde_json::from_slice(&bytes).context("parsing safetensors index")?;
            let mut names: Vec<&String> = index.weight_map.values().collect();
            names.sort();
            names.dedup();
            names.into_iter().map(|n| dir.join(n)).collect()
        } else {
            let single = dir.join(SINGLE_FILE_NAME);
            ensure!(
                single.exists(),
                "no {INDEX_NAME} or {SINGLE_FILE_NAME} under {}",
                dir.display()
            );
            vec![single]
        };

        let mut tensors = HashMap::new();
        for shard in shards {
            for (name, meta) in parse_shard_header(&shard)? {
                if tensors.insert(name.clone(), meta).is_some() {
                    bail!("tensor {name} appears in multiple shards");
                }
            }
        }
        Ok(Self { tensors })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn meta(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.get(name)
    }

    /// Reads one tensor's raw bytes (its native dtype layout).
    pub fn read(&self, name: &str) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.read_with(name, |meta| {
            bytes = vec![0u8; meta.byte_len()];
            Ok(&mut bytes)
        })?;
        Ok(bytes)
    }

    /// Reads one tensor's raw bytes into a caller-provided buffer, letting the
    /// caller size/place the destination (e.g. directly in an MTLBuffer).
    pub fn read_with<'a>(
        &self,
        name: &str,
        dest: impl FnOnce(&TensorMeta) -> Result<&'a mut [u8]>,
    ) -> Result<()> {
        let meta = self
            .tensors
            .get(name)
            .with_context(|| format!("tensor {name} not in checkpoint"))?;
        ensure!(
            meta.dtype.size().is_some(),
            "tensor {name} has unsupported dtype {:?}",
            meta.dtype
        );
        let buf = dest(meta)?;
        ensure!(
            buf.len() == meta.byte_len(),
            "destination for {name} is {} bytes, tensor is {}",
            buf.len(),
            meta.byte_len()
        );
        let mut file = File::open(&meta.shard)
            .with_context(|| format!("opening {}", meta.shard.display()))?;
        file.seek(SeekFrom::Start(meta.start))
            .with_context(|| format!("seeking to {name}"))?;
        file.read_exact(buf).with_context(|| format!("reading {name}"))?;
        Ok(())
    }
}

/// Parses one shard's header: 8-byte little-endian header length, then the
/// JSON header; `data_offsets` are relative to the data section that follows.
fn parse_shard_header(path: &Path) -> Result<HashMap<String, TensorMeta>> {
    let mut file =
        File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .with_context(|| format!("reading header length of {}", path.display()))?;
    let header_len = u64::from_le_bytes(len_bytes);
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .with_context(|| format!("reading header of {}", path.display()))?;
    let data_section = 8 + header_len;

    // `__metadata__` is an optional free-form object, not a tensor entry.
    let raw: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&header).context("parsing safetensors header")?;
    let mut out = HashMap::with_capacity(raw.len());
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let entry: HeaderEntry = serde_json::from_value(value)
            .with_context(|| format!("parsing header entry for {name}"))?;
        let [start, end] = entry.data_offsets;
        ensure!(end >= start, "tensor {name} has reversed data offsets");
        out.insert(
            name,
            TensorMeta {
                dtype: SafetensorsDType::from_str(&entry.dtype),
                shape: entry.shape,
                shard: path.to_path_buf(),
                start: data_section + start,
                end: data_section + end,
            },
        );
    }
    Ok(out)
}
