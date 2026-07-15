//! Engine trait + inventory-based registry + DAG resolver.
//!
//! An engine is a pluggable model that transforms one page. It declares the
//! artifacts it needs and produces; the DAG resolver derives execution order.
//!
//! **Engines emit ops, not mutations.** `run()` returns `Vec<Op>`; the driver
//! wraps them in `Op::Batch` and hands to `ProjectSession::apply`.
//!
//! ## Adding an engine
//!
//! 1. Define a struct holding your model.
//! 2. Implement `Engine` for it (returning `Vec<Op>`).
//! 3. Register via `inventory::submit! { EngineInfo { … } }` with a static
//!    async `load` function.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Result, bail};
use async_trait::async_trait;
use koharu_core::{NodeId, Op, PageId, ReadingOrder, Region, Scene};
use koharu_runtime::RuntimeManager;
use parking_lot::RwLock;
use petgraph::algo::toposort;
use petgraph::graph::DiGraph;
use tracing::Instrument;

use crate::blobs::BlobStore;
use crate::llm;
use crate::pipeline::artifacts::Artifact;
use crate::pipeline::{WarningSink, WarningTick};
use crate::renderer;

// ---------------------------------------------------------------------------
// EngineCtx — everything an engine needs to produce ops
// ---------------------------------------------------------------------------

pub struct EngineCtx<'a> {
    /// A cheap clone of the target page (read-only).
    pub scene: &'a Scene,
    pub page: PageId,
    pub blobs: &'a BlobStore,
    pub runtime: &'a RuntimeManager,
    pub cancel: &'a AtomicBool,
    pub options: &'a PipelineRunOptions,
    pub llm: &'a llm::Model,
    pub renderer: &'a renderer::Renderer,
    /// 0-based index of `page` within the current run's page list, and the
    /// total page count — needed to build a [`WarningTick`] for sub-step
    /// (partial/non-fatal) failures that don't warrant failing the whole
    /// engine run.
    pub page_index: usize,
    pub total_pages: usize,
    pub warnings: Option<&'a WarningSink>,
}

impl EngineCtx<'_> {
    /// Report a non-fatal, sub-step issue (e.g. one block out of many failed
    /// to render) without failing the engine's whole `run()`. Unlike an
    /// `Err` return, this doesn't skip the rest of the page's pipeline
    /// steps — it's purely a "here's something you should know" signal that
    /// reaches the same UI warning surface as a full step failure.
    pub fn warn(&self, step_id: &str, message: impl Into<String>) {
        emit_warning(
            self.warnings,
            self.page_index,
            self.total_pages,
            step_id,
            message,
        );
    }
}

/// Free-function core of [`EngineCtx::warn`], factored out so it's testable
/// without constructing a full `EngineCtx` (which needs real `Scene`,
/// `BlobStore`, `RuntimeManager`, `llm::Model`, and `renderer::Renderer`
/// instances that `warn` itself never touches).
fn emit_warning(
    sink: Option<&WarningSink>,
    page_index: usize,
    total_pages: usize,
    step_id: &str,
    message: impl Into<String>,
) {
    if let Some(sink) = sink {
        sink(WarningTick {
            step_id: step_id.to_string(),
            page_index,
            total_pages,
            message: message.into(),
        });
    }
}

/// Options threaded through a pipeline run.
#[derive(Debug, Clone, Default)]
pub struct PipelineRunOptions {
    pub target_language: Option<String>,
    pub system_prompt: Option<String>,
    pub default_font: Option<String>,
    /// Optional text-node scope for engines that can operate on individual
    /// text blocks. Engines that render full-page artifacts ignore it.
    pub text_node_ids: Option<Vec<NodeId>>,
    /// Optional bounding-box hint. Inpainter engines (lama/aot) honor it:
    /// composite onto the existing `Image { Inpainted }` (fallback Source)
    /// and process just that one block. Other engines ignore it.
    pub region: Option<Region>,
    pub reading_order: Option<ReadingOrder>,
}

// ---------------------------------------------------------------------------
// Engine trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait Engine: Send + Sync + 'static {
    /// Run the engine on one page. Return the ops to apply.
    /// Empty `Vec` = nothing changed (still a success).
    async fn run(&self, ctx: EngineCtx<'_>) -> Result<Vec<Op>>;
}

// ---------------------------------------------------------------------------
// EngineInfo — static descriptor + factory (registered via inventory)
// ---------------------------------------------------------------------------

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
pub type EngineLoadFn =
    for<'a> fn(&'a RuntimeManager, bool) -> BoxFuture<'a, Result<Box<dyn Engine>>>;

pub struct EngineInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub needs: &'static [Artifact],
    pub produces: &'static [Artifact],
    pub load: EngineLoadFn,
}

inventory::collect!(EngineInfo);

// ---------------------------------------------------------------------------
// Registry — lazy load + cache engine instances
// ---------------------------------------------------------------------------

pub struct Registry {
    engines: RwLock<HashMap<&'static str, Arc<dyn Engine>>>,
}

impl Default for Registry {
    fn default() -> Self {
        Self {
            engines: RwLock::new(HashMap::new()),
        }
    }
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or load an engine instance by id.
    pub async fn get(
        &self,
        id: &str,
        runtime: &RuntimeManager,
        cpu: bool,
    ) -> Result<Arc<dyn Engine>> {
        if let Some(engine) = self.engines.read().get(id).cloned() {
            return Ok(engine);
        }
        let info = Self::find(id)?;
        let loaded = async { (info.load)(runtime, cpu).await }
            .instrument(tracing::info_span!("engine_load", engine = id))
            .await?;
        let engine: Arc<dyn Engine> = Arc::from(loaded);
        self.engines.write().insert(info.id, engine.clone());
        Ok(engine)
    }

    /// Drop all cached engines (frees GPU memory).
    pub fn clear(&self) {
        self.engines.write().clear();
    }

    /// Find engine descriptor by id.
    pub fn find(id: &str) -> Result<&'static EngineInfo> {
        Self::catalog()
            .into_iter()
            .find(|e| e.id == id)
            .ok_or_else(|| anyhow::anyhow!("unknown engine: {id}"))
    }

    /// All registered engine descriptors.
    pub fn catalog() -> Vec<&'static EngineInfo> {
        inventory::iter::<EngineInfo>.into_iter().collect()
    }

    /// Engines that produce a given artifact.
    pub fn providers(artifact: Artifact) -> Vec<&'static EngineInfo> {
        Self::catalog()
            .into_iter()
            .filter(|e| e.produces.contains(&artifact))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// DAG — derive execution order from artifact dependencies
// ---------------------------------------------------------------------------

/// Build a topological execution order from a set of engine infos.
pub fn build_order(infos: &[&EngineInfo]) -> Result<Vec<usize>> {
    let mut g = DiGraph::<usize, ()>::new();
    let mut id_to_node: HashMap<&str, _> = HashMap::new();

    for (i, info) in infos.iter().enumerate() {
        let n = g.add_node(i);
        if id_to_node.insert(info.id, n).is_some() {
            bail!("duplicate engine: {}", info.id);
        }
    }

    let mut producers: HashMap<Artifact, usize> = HashMap::new();
    for (i, info) in infos.iter().enumerate() {
        for &artifact in info.produces {
            producers.insert(artifact, i);
        }
    }

    for info in infos.iter() {
        let to = id_to_node[info.id];
        for &artifact in info.needs {
            if let Some(&producer) = producers.get(&artifact) {
                g.add_edge(id_to_node[infos[producer].id], to, ());
            }
        }
    }

    let order = toposort(&g, None)
        .map_err(|c| anyhow::anyhow!("cycle at '{}'", infos[g[c.node_id()]].id))?;
    Ok(order.into_iter().map(|n| g[n]).collect())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[test]
    fn emit_warning_dispatches_to_sink_with_correct_fields() {
        let received: Arc<Mutex<Vec<WarningTick>>> = Arc::new(Mutex::new(Vec::new()));
        let received_c = received.clone();
        let sink: WarningSink = Arc::new(move |tick| received_c.lock().unwrap().push(tick));

        emit_warning(Some(&sink), 2, 5, "koharu-renderer", "block failed: oops");

        let ticks = received.lock().unwrap();
        assert_eq!(ticks.len(), 1);
        assert_eq!(ticks[0].step_id, "koharu-renderer");
        assert_eq!(ticks[0].page_index, 2);
        assert_eq!(ticks[0].total_pages, 5);
        assert_eq!(ticks[0].message, "block failed: oops");
    }

    #[test]
    fn emit_warning_without_a_sink_does_not_panic() {
        // `warnings: None` is the normal case for synchronous, non-batch
        // callers (e.g. the repair-brush endpoint) that don't wire up the
        // pipeline's warning stream. `warn()` must be a safe no-op there.
        emit_warning(None, 0, 1, "koharu-renderer", "unheard warning");
    }
}
