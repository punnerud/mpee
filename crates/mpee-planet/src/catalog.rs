//! Build catalogue and atomic publication.
//!
//! A dataset is ~45 loose files in a directory. That arrangement has two
//! failure modes, and both were hit while building this one:
//!
//!   * **A torn build looks like a working one.** The address pass was killed
//!     twice mid-run. Nothing on disk recorded that, so the directory held
//!     pass-4 output beside pass-5 output that did not exist yet, and a server
//!     started against it would have answered confidently from half a dataset.
//!   * **Provenance lived in file timestamps**, which is a poor database.
//!     Which planet file produced this? At what simplification? Which passes
//!     actually finished? The honest answer was "grep the build log".
//!
//! So a build now writes into its own directory under `builds/`, records every
//! pass and every artifact in an mpedb catalogue as it goes, and becomes live
//! only when the last pass has committed — by renaming a `HEAD` file, which is
//! atomic. A build that dies leaves its directory marked `running` and never
//! touches what readers are using.
//!
//! `HEAD` is a plain text file rather than a row in the catalogue because a
//! reader has to find the dataset *before* it can open any database, and
//! because `rename(2)` is the primitive that makes the switch indivisible.
//! The catalogue is the record of what happened; `HEAD` is the record of
//! what is live. `publish` writes the catalogue first and `HEAD` last, so a
//! crash between them leaves readers on the previous dataset — the safe
//! direction to fail.

use mpedb::{Config, Database, ExecResult, Value};
use std::io;
use std::path::{Path, PathBuf};

const SCHEMA: &str = r#"
[[table]]
name = "build"
primary_key = ["build_id"]
  [[table.column]]
  name = "build_id"
  type = "text"
  [[table.column]]
  name = "status"
  type = "text"
  nullable = false
  indexed = true
  [[table.column]]
  name = "started"
  type = "timestamp"
  nullable = false
  [[table.column]]
  name = "finished"
  type = "timestamp"
  [[table.column]]
  name = "source_path"
  type = "text"
  nullable = false
  [[table.column]]
  name = "source_bytes"
  type = "int64"
  nullable = false
  [[table.column]]
  name = "source_mtime"
  type = "timestamp"
  [[table.column]]
  name = "profile"
  type = "text"
  nullable = false
  [[table.column]]
  name = "simplify_m"
  type = "numeric"
  nullable = false
  [[table.column]]
  name = "tool_version"
  type = "text"
  nullable = false
  [[table.column]]
  name = "note"
  type = "text"

[[table]]
name = "build_pass"
primary_key = ["build_id", "pass"]
  [[table.column]]
  name = "build_id"
  type = "text"
  [[table.column]]
  name = "pass"
  type = "text"
  [[table.column]]
  name = "status"
  type = "text"
  nullable = false
  [[table.column]]
  name = "started"
  type = "timestamp"
  nullable = false
  [[table.column]]
  name = "finished"
  type = "timestamp"
  [[table.column]]
  name = "wall_s"
  type = "numeric"

[[table]]
name = "build_stat"
primary_key = ["build_id", "key"]
  [[table.column]]
  name = "build_id"
  type = "text"
  [[table.column]]
  name = "key"
  type = "text"
  [[table.column]]
  name = "value"
  type = "int64"
  nullable = false

[[table]]
name = "build_artifact"
primary_key = ["build_id", "name"]
  [[table.column]]
  name = "build_id"
  type = "text"
  [[table.column]]
  name = "name"
  type = "text"
  [[table.column]]
  name = "bytes"
  type = "int64"
  nullable = false
  [[table.column]]
  name = "role"
  type = "text"
  nullable = false
  [[table.column]]
  name = "digest"
  type = "text"
"#;

pub const STATUS_RUNNING: &str = "running";
pub const STATUS_COMPLETE: &str = "complete";
pub const STATUS_FAILED: &str = "failed";

/// Passes in the order the pipeline runs them. A build is only complete when
/// every one of these has committed.
pub const PASSES: &[&str] = &["scan", "contract", "graph", "addr", "toll", "tolldb", "overlay"];

fn now_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

fn toml_path(p: &Path) -> String {
    let s = p.display().to_string();
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn err<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::other(e.to_string())
}

// ---------------------------------------------------------------- layout

/// Where a dataset actually lives, given the root a user names.
///
/// Backward compatible on purpose: a root with no `HEAD` *is* the dataset, so
/// every directory built before the catalogue existed keeps working.
pub fn resolve(root: &Path) -> PathBuf {
    match std::fs::read_to_string(root.join("HEAD")) {
        Ok(id) => {
            let id = id.trim();
            let d = root.join("builds").join(id);
            if d.is_dir() {
                d
            } else {
                root.to_path_buf()
            }
        }
        Err(_) => root.to_path_buf(),
    }
}

pub fn build_dir(root: &Path, id: &str) -> PathBuf {
    root.join("builds").join(id)
}

/// A build id that sorts chronologically and is safe as a directory name.
pub fn new_build_id() -> String {
    // Local wall time is not available without a dependency; seconds since the
    // epoch sort identically and are unambiguous.
    format!("b{}", now_us() / 1_000_000)
}

// --------------------------------------------------------------- catalog

pub struct Catalog {
    db: Database,
    root: PathBuf,
}

impl Catalog {
    pub fn open(root: &Path) -> io::Result<Catalog> {
        std::fs::create_dir_all(root)?;
        let toml = format!(
            "[database]\npath = {}\nsize_mb = 32\nmax_readers = 8\ndurability = \"commit\"\n{SCHEMA}",
            toml_path(&root.join("catalog.mpedb"))
        );
        let db = Database::open_with_config(Config::from_toml_str(&toml).map_err(err)?)
            .map_err(err)?;
        Ok(Catalog { db, root: root.to_path_buf() })
    }

    fn exec(&self, sql: &str, params: &[Value]) -> io::Result<ExecResult> {
        let h = self.db.prepare(sql).map_err(err)?;
        let mut s = self.db.begin().map_err(err)?;
        let r = s.execute(&h, params).map_err(err)?;
        s.commit().map_err(err)?;
        Ok(r)
    }

    pub fn query(&self, sql: &str) -> io::Result<ExecResult> {
        self.db.query(sql, &[]).map_err(err)
    }

    /// Open a new build. Creates its directory; the build is not visible to
    /// readers until [`Catalog::publish`].
    pub fn begin_build(
        &self,
        id: &str,
        source: &Path,
        profile: &str,
        simplify_m: f64,
    ) -> io::Result<PathBuf> {
        let dir = build_dir(&self.root, id);
        std::fs::create_dir_all(&dir)?;
        let md = std::fs::metadata(source).ok();
        let mtime = md
            .as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| Value::Timestamp(d.as_micros() as i64))
            .unwrap_or(Value::Null);
        self.exec(
            "INSERT INTO build (build_id, status, started, finished, source_path, source_bytes, \
             source_mtime, profile, simplify_m, tool_version, note) \
             VALUES ($1,$2,$3,NULL,$4,$5,$6,$7,$8,$9,NULL)",
            &[
                Value::Text(id.into()),
                Value::Text(STATUS_RUNNING.into()),
                Value::Timestamp(now_us()),
                Value::Text(source.display().to_string()),
                Value::Int(md.as_ref().map(|m| m.len() as i64).unwrap_or(0)),
                mtime,
                Value::Text(profile.into()),
                Value::Numeric(format!("{simplify_m:.3}")),
                Value::Text(env!("CARGO_PKG_VERSION").into()),
            ],
        )?;
        Ok(dir)
    }

    pub fn pass_start(&self, id: &str, pass: &str) -> io::Result<()> {
        // A rerun of a pass replaces its previous record rather than failing on
        // the primary key — rebuilding one stage is a normal thing to do.
        let _ = self.exec(
            "DELETE FROM build_pass WHERE build_id = $1 AND pass = $2",
            &[Value::Text(id.into()), Value::Text(pass.into())],
        );
        self.exec(
            "INSERT INTO build_pass (build_id, pass, status, started, finished, wall_s) \
             VALUES ($1,$2,$3,$4,NULL,NULL)",
            &[
                Value::Text(id.into()),
                Value::Text(pass.into()),
                Value::Text(STATUS_RUNNING.into()),
                Value::Timestamp(now_us()),
            ],
        )?;
        Ok(())
    }

    pub fn pass_end(&self, id: &str, pass: &str, ok: bool, wall_s: f64) -> io::Result<()> {
        self.exec(
            "UPDATE build_pass SET status = $3, finished = $4, wall_s = $5 \
             WHERE build_id = $1 AND pass = $2",
            &[
                Value::Text(id.into()),
                Value::Text(pass.into()),
                Value::Text(if ok { STATUS_COMPLETE } else { STATUS_FAILED }.into()),
                Value::Timestamp(now_us()),
                Value::Numeric(format!("{wall_s:.3}")),
            ],
        )?;
        Ok(())
    }

    pub fn stat(&self, id: &str, key: &str, value: u64) -> io::Result<()> {
        let _ = self.exec(
            "DELETE FROM build_stat WHERE build_id = $1 AND key = $2",
            &[Value::Text(id.into()), Value::Text(key.into())],
        );
        self.exec(
            "INSERT INTO build_stat (build_id, key, value) VALUES ($1,$2,$3)",
            &[Value::Text(id.into()), Value::Text(key.into()), Value::Int(value as i64)],
        )?;
        Ok(())
    }

    /// Record every file the build produced, with its length and role.
    pub fn record_artifacts(&self, id: &str, dir: &Path, deep: bool) -> io::Result<u64> {
        let _ = self.exec(
            "DELETE FROM build_artifact WHERE build_id = $1",
            &[Value::Text(id.into())],
        );
        let h = self
            .db
            .prepare(
                "INSERT INTO build_artifact (build_id, name, bytes, role, digest) \
                 VALUES ($1,$2,$3,$4,$5)",
            )
            .map_err(err)?;
        let mut s = self.db.begin().map_err(err)?;
        let mut n = 0u64;
        let mut entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let name = e.file_name().to_string_lossy().to_string();
            let Ok(md) = e.metadata() else { continue };
            if !md.is_file() {
                continue;
            }
            let role = if crate::dataset::is_runtime_artifact(&name) {
                "runtime"
            } else {
                "scratch"
            };
            let digest = if deep && role == "runtime" {
                Value::Text(blake3_file(&e.path())?)
            } else {
                Value::Null
            };
            s.execute(
                &h,
                &[
                    Value::Text(id.into()),
                    Value::Text(name),
                    Value::Int(md.len() as i64),
                    Value::Text(role.into()),
                    digest,
                ],
            )
            .map_err(err)?;
            n += 1;
        }
        s.commit().map_err(err)?;
        Ok(n)
    }

    /// Mark the build complete and make it the live dataset.
    ///
    /// The catalogue is written first and `HEAD` last: a crash in between
    /// leaves a complete-but-unpublished build and readers on the old one,
    /// which is the failure that costs nothing.
    pub fn publish(&self, id: &str) -> io::Result<()> {
        let missing = self.missing_passes(id)?;
        if !missing.is_empty() {
            return Err(io::Error::other(format!(
                "refusing to publish {id}: passes not complete: {}",
                missing.join(", ")
            )));
        }
        self.exec(
            "UPDATE build SET status = $2, finished = $3 WHERE build_id = $1",
            &[
                Value::Text(id.into()),
                Value::Text(STATUS_COMPLETE.into()),
                Value::Timestamp(now_us()),
            ],
        )?;
        // rename(2) over an existing path is atomic: a reader sees either the
        // old id or the new one, never a partial write.
        let tmp = self.root.join("HEAD.new");
        std::fs::write(&tmp, format!("{id}\n"))?;
        std::fs::rename(&tmp, self.root.join("HEAD"))?;
        Ok(())
    }

    pub fn fail_build(&self, id: &str, note: &str) -> io::Result<()> {
        self.exec(
            "UPDATE build SET status = $2, finished = $3, note = $4 WHERE build_id = $1",
            &[
                Value::Text(id.into()),
                Value::Text(STATUS_FAILED.into()),
                Value::Timestamp(now_us()),
                Value::Text(note.into()),
            ],
        )?;
        Ok(())
    }

    /// Passes that have not committed for this build.
    pub fn missing_passes(&self, id: &str) -> io::Result<Vec<String>> {
        let h = self
            .db
            .prepare("SELECT pass FROM build_pass WHERE build_id = $1 AND status = $2")
            .map_err(err)?;
        let done: std::collections::HashSet<String> = match self
            .db
            .execute(&h, &[Value::Text(id.into()), Value::Text(STATUS_COMPLETE.into())])
            .map_err(err)?
        {
            ExecResult::Rows { rows, .. } => rows
                .into_iter()
                .filter_map(|r| match r.into_iter().next() {
                    Some(Value::Text(s)) => Some(s),
                    _ => None,
                })
                .collect(),
            _ => Default::default(),
        };
        Ok(PASSES
            .iter()
            .filter(|p| !done.contains(**p))
            .map(|p| p.to_string())
            .collect())
    }

    pub fn head(&self) -> Option<String> {
        std::fs::read_to_string(self.root.join("HEAD")).ok().map(|s| s.trim().to_string())
    }
}

/// BLAKE3 of a file, for `verify --deep`.
pub fn digest_of(p: &Path) -> io::Result<String> {
    blake3_file(p)
}

fn blake3_file(p: &Path) -> io::Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(p)?;
    let mut h = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_falls_back_to_the_root_when_there_is_no_head() {
        let d = std::env::temp_dir().join(format!("mpee-cat-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        assert_eq!(resolve(&d), d, "a directory built before the catalogue still opens");
        // A HEAD pointing at a directory that does not exist must not strand
        // the reader either.
        std::fs::write(d.join("HEAD"), "b999\n").unwrap();
        assert_eq!(resolve(&d), d);
        std::fs::create_dir_all(d.join("builds").join("b999")).unwrap();
        assert_eq!(resolve(&d), d.join("builds").join("b999"));
        let _ = std::fs::remove_dir_all(&d);
    }
}
