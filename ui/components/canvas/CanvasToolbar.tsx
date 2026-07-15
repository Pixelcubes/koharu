'use client'

import {
  LanguagesIcon,
  LoaderCircleIcon,
  ScanIcon,
  ScanTextIcon,
  SparklesIcon,
  SquareIcon,
  TypeIcon,
  Wand2Icon,
} from 'lucide-react'
import { motion } from 'motion/react'
import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { Button } from '@/components/ui/button'
import { LlmModelSelect, type LlmModelOption } from '@/components/ui/llm-model-select'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Textarea } from '@/components/ui/textarea'
import {
  cancelOperation,
  deleteCurrentLlm,
  getConfig,
  putCurrentLlm,
  startPipeline,
  useGetCatalog,
  useGetCurrentLlm,
} from '@/lib/api/default/default'
import type { LlmCatalog, LlmCatalogModel, LlmProviderCatalog, LlmTarget } from '@/lib/api/schemas'
import { useEditorUiStore } from '@/lib/stores/editorUiStore'
import { type JobEntry, useJobsStore } from '@/lib/stores/jobsStore'
import { usePreferencesStore } from '@/lib/stores/preferencesStore'
import { useSelectionStore } from '@/lib/stores/selectionStore'

// ---------------------------------------------------------------------------
// Helpers (inlined from former llmTargets util)
// ---------------------------------------------------------------------------

function llmTargetKey(t: LlmTarget): string {
  return `${t.kind}:${t.providerId ?? ''}:${t.modelId}`
}

function sameLlmTarget(a?: LlmTarget | null, b?: LlmTarget | null): boolean {
  if (!a || !b) return false
  return (
    a.kind === b.kind &&
    a.modelId === b.modelId &&
    (a.providerId ?? null) === (b.providerId ?? null)
  )
}

type SelectableLlmModel = { model: LlmCatalogModel; provider?: LlmProviderCatalog }

const flattenCatalogModels = (catalog?: LlmCatalog): SelectableLlmModel[] => [
  ...(catalog?.localModels ?? []).map((model) => ({ model })),
  ...(catalog?.providers ?? [])
    .filter((p) => p.status === 'ready')
    .flatMap((p) => p.models.map((model) => ({ model, provider: p }))),
]

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function CanvasToolbar() {
  return (
    <div className='flex items-center gap-2 border-b border-border/60 bg-card px-3 py-2 text-xs text-foreground'>
      <WorkflowButtons />
      <div className='flex-1' />
      <LlmStatusPopover />
    </div>
  )
}

/** Currently-busy step (derived from jobsStore). */
function useCurrentStep(): string | null {
  const jobs = useJobsStore((s) => s.jobs)
  for (const j of Object.values(jobs)) {
    if (j.status === 'running' && j.progress?.step) return String(j.progress.step)
  }
  return null
}

function useIsProcessing(): boolean {
  const jobs = useJobsStore((s) => s.jobs)
  return Object.values(jobs).some((j) => j.status === 'running')
}

/** Id of the currently-running job, if any — used to cancel it. */
function useRunningJobId(): string | null {
  const jobs = useJobsStore((s) => s.jobs)
  for (const j of Object.values(jobs)) {
    if (j.status === 'running') return j.id
  }
  return null
}

/**
 * Resolve once `jobId` reaches a non-`running` status (SSE-driven, via
 * `useJobsStore`). Checks the current store state first — the job may
 * already have finished by the time the caller starts waiting — then
 * subscribes for the transition. Used to sequence the auto-translate
 * button's stages: `POST /pipelines` returns as soon as the run is
 * accepted, not when it finishes, so awaiting the HTTP call alone doesn't
 * wait for the stage to actually complete.
 */
function waitForJobSettled(jobId: string): Promise<JobEntry | undefined> {
  return new Promise((resolve) => {
    const check = (): boolean => {
      const job = useJobsStore.getState().jobs[jobId]
      if (!job || job.status === 'running') return false
      resolve(job)
      return true
    }
    if (check()) return
    const unsubscribe = useJobsStore.subscribe(() => {
      if (check()) unsubscribe()
    })
  })
}

function WorkflowButtons() {
  const { t } = useTranslation()
  const { data: llmState } = useGetCurrentLlm()
  const llmReady = llmState?.status === 'ready'
  const pageId = useSelectionStore((s) => s.pageId)
  const hasPage = pageId !== null
  const isProcessing = useIsProcessing()
  const currentStep = useCurrentStep()
  const runningJobId = useRunningJobId()

  type PipelinePick = (
    p: NonNullable<Awaited<ReturnType<typeof getConfig>>['pipeline']>,
  ) => string[]

  /**
   * Start a pipeline step (or a small chain) and return its operation id,
   * or `null` if there was nothing to run. `GET /config` is the single
   * source of truth for engine ids — every field has a serde default in
   * the Rust `PipelineConfig`, so we trust what the server returns and
   * never hard-code fallbacks here.
   *
   * Detect is the only multi-engine button; it bundles detector +
   * segmenter + font-detector so the subsequent single-engine steps
   * (OCR / Inpaint / Render) find their inputs already on the page. The
   * backend driver skips any step whose artifact is already satisfied,
   * so re-running is idempotent.
   */
  const startStep = async (pick: PipelinePick): Promise<string | null> => {
    if (!pageId) return null
    const cfg = await getConfig()
    if (!cfg.pipeline) return null
    const steps = pick(cfg.pipeline).filter((s): s is string => !!s)
    if (steps.length === 0) return null
    const editor = useEditorUiStore.getState()
    const prefs = usePreferencesStore.getState()
    const { operationId } = await startPipeline({
      steps,
      pages: [pageId],
      targetLanguage: editor.selectedLanguage,
      systemPrompt: prefs.customSystemPrompt,
      defaultFont: prefs.defaultFont,
      readingOrder: editor.readingOrder === 'custom' ? undefined : editor.readingOrder,
    })
    return operationId
  }

  // Thin wrapper for the five individual buttons below — fire-and-forget,
  // exactly the previous behavior. Auto Translate uses `startStep`
  // directly so it can wait for each stage to actually finish.
  const runStep = async (pick: PipelinePick) => {
    await startStep(pick)
  }

  const detectChain: PipelinePick = (p) => [
    p.detector!,
    p.segmenter!,
    p.bubble_segmenter!,
    p.font_detector!,
  ]
  const ocrChain: PipelinePick = (p) => [p.ocr!]
  const translateChain: PipelinePick = (p) => [p.translator!]
  const inpaintChain: PipelinePick = (p) => [p.inpainter!]
  const renderChain: PipelinePick = (p) => [p.renderer!]

  /**
   * Run detect -> ocr -> translate -> inpaint -> render as five separate,
   * sequential pipeline runs — waiting for each stage's job to actually
   * finish (not just be accepted) before starting the next. A single
   * combined `startPipeline` call was tried first, but the backend
   * resolves execution order from a dependency DAG, not the literal step
   * order it's given (see `build_order` in `koharu-app/src/pipeline/
   * engine.rs`) — steps with no forced relationship (e.g. OCR/translate
   * vs. segment/font-detect, which only depend on detect, not on each
   * other) can interleave, so the toolbar's step indicator could flash
   * Detect -> OCR -> Translate -> Detect again -> Inpaint -> Render.
   * Five separate runs guarantee the clean, visible sequence the button
   * groups imply. Stops the whole chain if a stage fails or is cancelled,
   * rather than pressing on with steps whose inputs never got produced.
   */
  const runAutoTranslateSequential = async () => {
    for (const pick of [detectChain, ocrChain, translateChain, inpaintChain, renderChain]) {
      const operationId = await startStep(pick)
      if (!operationId) continue
      const job = await waitForJobSettled(operationId)
      if (job?.status === 'failed' || job?.status === 'cancelled') return
    }
  }

  const isDetecting = currentStep === 'detect'
  const isOcr = currentStep === 'ocr'
  const isInpainting = currentStep === 'inpaint'
  const isTranslating = currentStep === 'llmGenerate'
  const isRendering = currentStep === 'render'

  const handleStop = async () => {
    if (!runningJobId) return
    await cancelOperation(runningJobId)
  }

  return (
    <div className='flex items-center gap-0.5'>
      <Button
        variant='ghost'
        size='xs'
        onClick={() => void runStep(detectChain)}
        data-testid='toolbar-detect'
        disabled={!hasPage || isProcessing}
      >
        {isDetecting ? (
          <LoaderCircleIcon className='size-4 animate-spin' />
        ) : (
          <ScanIcon className='size-4' />
        )}
        {t('processing.detect')}
      </Button>
      <Separator orientation='vertical' className='mx-0.5 h-4' />
      <Button
        variant='ghost'
        size='xs'
        onClick={() => void runStep(ocrChain)}
        data-testid='toolbar-ocr'
        disabled={!hasPage || isProcessing}
      >
        {isOcr ? (
          <LoaderCircleIcon className='size-4 animate-spin' />
        ) : (
          <ScanTextIcon className='size-4' />
        )}
        {t('processing.ocr')}
      </Button>
      <Separator orientation='vertical' className='mx-0.5 h-4' />
      <Button
        variant='ghost'
        size='xs'
        onClick={() => void runStep(translateChain)}
        disabled={!hasPage || !llmReady || isProcessing}
        data-testid='toolbar-translate'
      >
        {isTranslating ? (
          <LoaderCircleIcon className='size-4 animate-spin' />
        ) : (
          <LanguagesIcon className='size-4' />
        )}
        {t('llm.generate')}
      </Button>
      <Separator orientation='vertical' className='mx-0.5 h-4' />
      <Button
        variant='ghost'
        size='xs'
        onClick={() => void runStep(inpaintChain)}
        data-testid='toolbar-inpaint'
        disabled={!hasPage || isProcessing}
      >
        {isInpainting ? (
          <LoaderCircleIcon className='size-4 animate-spin' />
        ) : (
          <Wand2Icon className='size-4' />
        )}
        {t('mask.inpaint')}
      </Button>
      <Separator orientation='vertical' className='mx-0.5 h-4' />
      <Button
        variant='ghost'
        size='xs'
        onClick={() => void runStep(renderChain)}
        data-testid='toolbar-render'
        disabled={!hasPage || isProcessing}
      >
        {isRendering ? (
          <LoaderCircleIcon className='size-4 animate-spin' />
        ) : (
          <TypeIcon className='size-4' />
        )}
        {t('llm.render')}
      </Button>
      <Separator orientation='vertical' className='mx-1.5 h-4' />
      <Button
        variant={isProcessing ? 'destructive' : 'ghost'}
        size='xs'
        onClick={() => (isProcessing ? void handleStop() : void runAutoTranslateSequential())}
        data-testid='toolbar-auto-translate'
        disabled={isProcessing ? !runningJobId : !hasPage || !llmReady}
      >
        {isProcessing ? (
          <SquareIcon className='size-4' />
        ) : (
          <SparklesIcon className='size-4' />
        )}
        {isProcessing ? t('processing.stop') : t('processing.autoTranslate')}
      </Button>
    </div>
  )
}

function LlmStatusPopover() {
  const { t } = useTranslation()
  const { data: llmCatalog } = useGetCatalog()
  const { data: llmState } = useGetCurrentLlm()
  const llmReady = llmState?.status === 'ready'
  const llmLoading = llmState?.status === 'loading'
  const [popoverOpen, setPopoverOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const llmModels: LlmModelOption[] = useMemo(() => flattenCatalogModels(llmCatalog), [llmCatalog])
  const selectedTarget = useEditorUiStore((s) => s.selectedTarget)
  const customSystemPrompt = usePreferencesStore((s) => s.customSystemPrompt)
  const setCustomSystemPrompt = usePreferencesStore((s) => s.setCustomSystemPrompt)
  const llmSelectedLanguage = useEditorUiStore((s) => s.selectedLanguage)

  const selectedModel = useMemo(
    () => llmModels.find(({ model }) => sameLlmTarget(model.target, selectedTarget)),
    [llmModels, selectedTarget],
  )
  const selectedTargetKey = selectedTarget ? llmTargetKey(selectedTarget) : undefined
  const selectedModelLanguages = selectedModel?.model.languages ?? []
  const selectedIsLoaded = llmReady && sameLlmTarget(llmState?.target, selectedTarget)

  const handleSetSelectedModel = (key: string) => {
    const next = llmModels.find(({ model }) => llmTargetKey(model.target) === key)
    if (!next) return
    const nextLanguages = next.model.languages
    const nextLanguage =
      llmSelectedLanguage && nextLanguages.includes(llmSelectedLanguage)
        ? llmSelectedLanguage
        : nextLanguages[0]
    useEditorUiStore.setState({ selectedTarget: next.model.target, selectedLanguage: nextLanguage })
  }

  const handleSetSelectedLanguage = (language: string) => {
    if (!selectedModelLanguages.includes(language)) return
    useEditorUiStore.setState({ selectedLanguage: language })
  }

  const handleToggleLoadUnload = async () => {
    const target = useEditorUiStore.getState().selectedTarget
    if (!target) return
    setBusy(true)
    try {
      if (selectedIsLoaded) {
        await deleteCurrentLlm()
      } else {
        await putCurrentLlm({ target })
      }
    } catch (e) {
      useEditorUiStore.getState().showError(String(e))
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    if (llmModels.length === 0) return
    const hasCurrent = llmModels.some(({ model }) => sameLlmTarget(model.target, selectedTarget))
    const nextModel = hasCurrent ? selectedModel?.model : llmModels[0]?.model
    if (!nextModel) return
    const nextLanguages = nextModel.languages
    const nextLanguage =
      llmSelectedLanguage && nextLanguages.includes(llmSelectedLanguage)
        ? llmSelectedLanguage
        : nextLanguages[0]
    const cur = useEditorUiStore.getState()
    if (
      sameLlmTarget(cur.selectedTarget, nextModel.target) &&
      cur.selectedLanguage === nextLanguage
    ) {
      return
    }
    useEditorUiStore.setState({
      selectedTarget: nextModel.target,
      selectedLanguage: nextLanguage,
    })
  }, [llmModels, llmSelectedLanguage, selectedModel?.model, selectedTarget])

  const indicatorBusy = busy || llmLoading

  return (
    <Popover open={popoverOpen} onOpenChange={setPopoverOpen}>
      <PopoverTrigger asChild>
        <button
          data-testid='llm-trigger'
          data-llm-ready={llmReady ? 'true' : 'false'}
          data-llm-loading={indicatorBusy ? 'true' : 'false'}
          className={`flex h-6 cursor-pointer items-center gap-1.5 rounded-full px-2.5 text-[11px] font-medium shadow-sm transition hover:opacity-80 ${
            llmReady
              ? 'bg-rose-400 text-white ring-1 ring-rose-400/30'
              : indicatorBusy
                ? 'bg-amber-400 text-white ring-1 ring-amber-400/30'
                : 'bg-muted text-muted-foreground ring-1 ring-border/50'
          }`}
        >
          <motion.span
            className={`size-1.5 rounded-full ${
              llmReady ? 'bg-white' : indicatorBusy ? 'bg-white' : 'bg-muted-foreground/40'
            }`}
            animate={
              llmReady
                ? { opacity: [1, 0.5, 1] }
                : indicatorBusy
                  ? { opacity: [1, 0.4, 1] }
                  : { opacity: 1 }
            }
            transition={
              llmReady || indicatorBusy
                ? { duration: indicatorBusy ? 1 : 2, repeat: Infinity, ease: 'easeInOut' }
                : {}
            }
          />
          LLM
        </button>
      </PopoverTrigger>
      <PopoverContent align='end' className='w-[280px] p-0' data-testid='llm-popover'>
        <div className='flex flex-col gap-1.5 px-3 pt-3 pb-2.5'>
          <span className='text-[10px] font-medium text-muted-foreground uppercase'>
            {t('llm.model')}
          </span>
          <div className='flex items-center gap-1.5'>
            <LlmModelSelect
              data-testid='llm-model-select'
              value={selectedTargetKey}
              options={llmModels}
              getKey={({ model }) => llmTargetKey(model.target)}
              placeholder={t('llm.selectPlaceholder')}
              onChange={handleSetSelectedModel}
              triggerClassName='min-w-0 flex-1'
            />
            <Button
              data-testid='llm-load-toggle'
              data-llm-ready={selectedIsLoaded ? 'true' : 'false'}
              data-llm-loading={indicatorBusy ? 'true' : 'false'}
              variant={selectedIsLoaded ? 'ghost' : 'default'}
              size='sm'
              onClick={() => void handleToggleLoadUnload()}
              disabled={!selectedTarget || indicatorBusy}
              className='h-6 shrink-0 gap-1 px-2 text-[11px]'
            >
              {indicatorBusy ? <LoaderCircleIcon className='size-3 animate-spin' /> : null}
              {selectedIsLoaded ? t('llm.unload') : t('llm.load')}
            </Button>
          </div>
        </div>
        <div className='px-3'>
          <Separator />
        </div>
        <div className='flex flex-col gap-1 px-3 pt-2.5 pb-3'>
          <span className='text-[10px] font-medium text-muted-foreground uppercase'>
            {t('llm.translationSettings')}
          </span>
          <div className='flex flex-col gap-1.5'>
            {selectedModelLanguages.length > 0 ? (
              <Select
                value={llmSelectedLanguage ?? selectedModelLanguages[0]}
                onValueChange={handleSetSelectedLanguage}
              >
                <SelectTrigger data-testid='llm-language-select' className='w-full'>
                  <SelectValue placeholder={t('llm.languagePlaceholder')} />
                </SelectTrigger>
                <SelectContent position='popper'>
                  {selectedModelLanguages.map((language, index) => (
                    <SelectItem
                      key={language}
                      value={language}
                      data-testid={`llm-language-option-${index}`}
                    >
                      {t(`llm.languages.${language}`, { defaultValue: language })}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : null}
            <Textarea
              data-testid='llm-system-prompt'
              value={customSystemPrompt ?? ''}
              onChange={(e) => setCustomSystemPrompt(e.target.value || undefined)}
              placeholder={t('llm.systemPromptPlaceholder')}
              rows={5}
              className='min-h-0 resize-y px-2 py-1.5 text-xs leading-snug md:text-xs'
            />
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}
