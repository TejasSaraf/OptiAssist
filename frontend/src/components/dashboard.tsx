import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
} from "react";
import {
  Activity,
  Brain,
  CheckCircle2,
  ChevronRight,
  Eye,
  FileText,
  GitBranch,
  Loader2,
  MessageSquare,
  ScanEye,
  Search,
  Sparkles,
  Stethoscope,
  Upload,
  X,
  Zap,
} from "lucide-react";

/* ── Brand tokens ─────────────────────────────────────────────── */
const G = "#7fee64";
const G_RGB = "127, 238, 100";

interface DashboardProps {
  onBackHome: () => void;
}

type StageStatus = "idle" | "running" | "complete" | "error" | "skipped";

interface StageState {
  status: StageStatus;
  message: string;
  thinking: string | null;
  output: unknown;
}

interface DiagnosisResult {
  condition?: string;
  severity?: string;
  severity_level?: number;
  confidence?: number;
  findings?: string[];
  recommendation?: string;
  disclaimer?: string;
}

interface SegmentationResult {
  summary?: string;
  raw_output?: string;
  detections?: unknown[];
  annotated_image_base64?: string;
}

interface AnalysisResults {
  prescan?: string | null;
  routing?: string | null;
  segmentation?: SegmentationResult | null;
  diagnosis?: DiagnosisResult | null;
  synthesis?: string | null;
}

const DEFAULT_QUESTION =
  "What is the cup to disc ratio in this image?";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const STAGE_ORDER = [
  "input",
  "gemma3",
  "routing",
  "paligemma",
  "medgemma",
  "synthesis",
] as const;

type StageId = (typeof STAGE_ORDER)[number];

const STAGE_LABELS: Record<StageId, string> = {
  input: "Input",
  gemma3: "Gemma 3 — Pre-scan",
  routing: "FunctionGemma — Routing",
  paligemma: "PaliGemma 2 — Analysis",
  medgemma: "MedGemma 4B — Diagnosis",
  synthesis: "Gemma 3 — Summary",
};

const STAGE_ICONS: Record<StageId, React.ReactNode> = {
  input: <Upload size={14} />,
  gemma3: <Eye size={14} />,
  routing: <GitBranch size={14} />,
  paligemma: <Search size={14} />,
  medgemma: <Stethoscope size={14} />,
  synthesis: <FileText size={14} />,
};

function createInitialStages(): Record<string, StageState> {
  return Object.fromEntries(
    STAGE_ORDER.map((id) => [
      id,
      {
        status: "idle" as StageStatus,
        message: "Pending",
        thinking: null,
        output: null,
      },
    ]),
  );
}

function isDone(status: StageStatus): boolean {
  return (
    status === "complete" ||
    status === "skipped" ||
    status === "error"
  );
}

export default function Dashboard({ onBackHome }: DashboardProps) {
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [imageDims, setImageDims] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stages, setStages] = useState<Record<string, StageState>>(
    createInitialStages,
  );
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setError(null);
    setImageDims(null);
    setPreviewUrl((currentUrl) => {
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      return file ? URL.createObjectURL(file) : null;
    });
  };

  const onPreviewLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const { naturalWidth, naturalHeight } = e.currentTarget;
    setImageDims(`${naturalWidth} × ${naturalHeight}px`);
  };

  const applyStageUpdate = (
    stageId: string,
    status: StageStatus,
    message: string,
    data?: unknown,
  ) => {
    setStages((current) => {
      const prev = current[stageId] ?? {
        status: "idle" as StageStatus,
        message: "",
        thinking: null,
        output: null,
      };
      let thinking: string | null = prev.thinking;
      let output: unknown = prev.output;
      if (status === "running") {
        thinking = message;
      } else {
        thinking = null;
        if (data !== undefined) {
          output = data;
        }
      }
      return {
        ...current,
        [stageId]: { status, message, thinking, output },
      };
    });
  };

  const processSseEvent = (block: string) => {
    let eventName = "message";
    let dataPayload = "";

    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataPayload += line.slice(5).trim();
      }
    }

    if (!dataPayload) return;

    const payload = JSON.parse(dataPayload) as {
      message?: string;
      results?: AnalysisResults;
      id?: string;
      status?: StageStatus;
      data?: unknown;
    };

    if (eventName === "stage" && payload.id && payload.status) {
      applyStageUpdate(
        payload.id,
        payload.status,
        payload.message ?? "Updated",
        payload.data,
      );
      return;
    }

    if (eventName === "complete") {
      setResults(payload.results ?? null);
      return;
    }

    if (eventName === "error") {
      setError(payload.message ?? "Analysis failed.");
    }
  };

  const handleCancel = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsAnalyzing(false);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError("Upload a retinal image before starting analysis.");
      return;
    }

    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const { signal } = abortRef.current;

    setIsAnalyzing(true);
    setError(null);
    setResults(null);
    setStages(createInitialStages());

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append(
      "question",
      question.trim() ||
        "Analyze this retinal fundus image. What condition is present and what is the severity?",
    );

    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        body: formData,
        signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`Backend returned ${response.status}.`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let sep = buffer.indexOf("\n\n");
        while (sep !== -1) {
          const block = buffer.slice(0, sep).trim();
          buffer = buffer.slice(sep + 2);
          if (block) processSseEvent(block);
          sep = buffer.indexOf("\n\n");
        }
      }

      const tail = buffer.trim();
      if (tail) processSseEvent(tail);
    } catch (caught) {
      if (caught instanceof Error && caught.name === "AbortError") {
        setError("Analysis cancelled.");
      } else {
        const message =
          caught instanceof Error
            ? caught.message
            : "Unable to complete analysis.";
        setError(message);
      }
    } finally {
      setIsAnalyzing(false);
      abortRef.current = null;
    }
  };

  const completedCount = STAGE_ORDER.filter((id) =>
    isDone(stages[id]?.status ?? "idle"),
  ).length;
  const totalStages = STAGE_ORDER.length;
  const progressPct = (completedCount / totalStages) * 100;

  const hasStaleTrace = STAGE_ORDER.some(
    (id) => stages[id] != null && stages[id].status !== "idle",
  );

  /* ── Stage body (trace panel) ────────────────────────────── */
  const renderPipelineStageBody = (id: StageId, stage: StageState) => {
    if (stage.status === "running") {
      return (
        <div
          className="rounded-lg px-3 py-3"
          style={{
            background: `rgba(${G_RGB}, 0.06)`,
          }}
        >
          <div className="mb-2 flex items-center gap-2">
            <Brain className="h-4 w-4 shrink-0" style={{ color: G }} />
            <span className="text-[11px] font-bold uppercase tracking-wider" style={{ color: G }}>
              Model thinking
            </span>
            <Loader2 className="h-3.5 w-3.5 animate-spin" style={{ color: G }} />
          </div>
          <p className="thinking-stream text-sm leading-relaxed text-neutral-300">
            {stage.thinking ?? stage.message}
          </p>
        </div>
      );
    }

    if (stage.status === "error") {
      return (
        <p className="text-sm text-red-400">{stage.message}</p>
      );
    }

    if (stage.status === "skipped") {
      return (
        <p className="text-sm italic text-neutral-600">{stage.message}</p>
      );
    }

    if (stage.status === "idle") {
      if (isAnalyzing) {
        return (
          <p className="text-xs text-neutral-700">Waiting for pipeline…</p>
        );
      }
      if (hasStaleTrace && results == null) {
        return (
          <p className="text-xs italic text-neutral-700">Not reached.</p>
        );
      }
      return null;
    }

    if (id === "input") {
      return (
        <div>
          <p className="leading-relaxed text-neutral-300">{question}</p>
          <p className="mt-2 text-xs text-neutral-600">{stage.message}</p>
        </div>
      );
    }

    if (id === "gemma3") {
      const text =
        (results?.prescan as string | undefined) ??
        (typeof stage.output === "string" ? stage.output : null);
      if (!text)
        return <p className="text-xs text-neutral-600">No pre-scan text.</p>;
      return <p className="leading-relaxed text-neutral-300">{text}</p>;
    }

    if (id === "routing") {
      const extra =
        stage.output &&
        typeof stage.output === "object" &&
        stage.output !== null &&
        "run_segmentation" in stage.output
          ? (stage.output as Record<string, unknown>)
          : null;
      return (
        <div>
          {extra ? (
            <pre className="max-h-32 overflow-auto rounded-lg bg-black p-3 font-mono text-[11px] text-neutral-500">
              {JSON.stringify(
                {
                  run_segmentation: extra.run_segmentation,
                  run_diagnosis: extra.run_diagnosis,
                  advisory_segmentation: extra.advisory_segmentation,
                  advisory_diagnosis: extra.advisory_diagnosis,
                },
                null,
                2,
              )}
            </pre>
          ) : (
            <p className="text-sm text-neutral-400">
              {stage.message || "Routing complete."}
            </p>
          )}
        </div>
      );
    }

    if (id === "paligemma") {
      const seg = results?.segmentation;
      const o = stage.output as
        | {
            summary?: string;
            raw_output?: string;
            detections_count?: number;
          }
        | undefined;
      const summary = seg?.summary ?? o?.summary;
      const raw = seg?.raw_output ?? o?.raw_output;
      const n =
        (seg?.detections?.length ?? o?.detections_count ?? null) as
          | number
          | null;
      if (!summary && !raw && n == null) {
        return (
          <p className="text-xs text-neutral-600">No PaliGemma output yet.</p>
        );
      }
      return (
        <div>
          {summary && (
            <p className="mb-2 leading-relaxed text-neutral-300">{summary}</p>
          )}
          {raw && (
            <details className="mt-1">
              <summary className="cursor-pointer text-xs font-medium" style={{ color: G }}>
                Raw model output
              </summary>
              <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap rounded-lg bg-black p-3 text-[11px] text-neutral-500">
                {raw}
              </pre>
            </details>
          )}
          {n != null && (
            <p className="mt-2 text-xs text-neutral-600">
              Detections: {n} region(s)
            </p>
          )}
        </div>
      );
    }

    if (id === "medgemma") {
      const d = results?.diagnosis ?? (stage.output as DiagnosisResult | null);
      if (!d || typeof d !== "object") {
        return (
          <p className="text-xs text-neutral-600">No diagnosis payload.</p>
        );
      }
      return (
        <div>
          {d.findings && d.findings.length > 0 ? (
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-neutral-500">
                Findings
              </p>
              <ul className="list-inside list-disc space-y-1 text-neutral-300">
                {d.findings.map((f) => (
                  <li key={f}>{f}</li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="text-xs text-neutral-600">No findings reported.</p>
          )}
        </div>
      );
    }

    if (id === "synthesis") {
      const text =
        results?.synthesis?.trim() ??
        (typeof stage.output === "string" ? stage.output : null);
      if (!text) {
        return (
          <p className="text-xs text-neutral-600">No summary generated.</p>
        );
      }
      return (
        <p className="text-base leading-relaxed text-neutral-100">{text}</p>
      );
    }

    return (
      <p className="text-xs text-neutral-600">{stage.message}</p>
    );
  };

  /* ── Render ──────────────────────────────────────────────── */
  return (
    <div
      className="min-h-screen text-white"
      style={{ backgroundColor: "#000", fontFamily: "Inter, system-ui, sans-serif" }}
    >
      <div className="flex min-h-screen flex-col md:flex-row">
        {/* ── Left sidebar ───────────────────────────────────── */}
        <aside
          className="flex w-full shrink-0 flex-col border-b px-5 py-6 md:w-[min(100%,380px)] md:border-b-0 md:border-r"
          style={{ backgroundColor: "#050505", borderColor: `rgba(${G_RGB}, 0.1)` }}
        >
          {/* Brand header */}
          <header className="mb-8 flex items-start gap-3">
            <div
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl"
              style={{
                background: `rgba(${G_RGB}, 0.1)`,
                border: `1px solid rgba(${G_RGB}, 0.3)`,
                boxShadow: `0 0 24px rgba(${G_RGB}, 0.15)`,
              }}
            >
              <Activity style={{ color: G }} size={22} strokeWidth={2.25} />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white">
                OpusAI
              </h1>
              <p
                className="text-[10px] font-semibold uppercase tracking-[0.2em]"
                style={{ color: `rgba(${G_RGB}, 0.5)` }}
              >
                Agentic diagnostic AI
              </p>
            </div>
          </header>

          {/* Retinal image upload */}
          <section className="mb-6">
            <div className="mb-2 flex items-center gap-2" style={{ color: `rgba(${G_RGB}, 0.7)` }}>
              <ScanEye size={15} strokeWidth={2} />
              <span className="text-xs font-medium">Retinal Image</span>
            </div>
            <label className="block cursor-pointer">
              <div
                className="relative overflow-hidden rounded-2xl border-2 border-dashed p-1 transition-colors hover:border-opacity-80"
                style={{ borderColor: `rgba(${G_RGB}, 0.35)` }}
              >
                <div className="relative mx-auto aspect-square w-full max-w-[220px] overflow-hidden rounded-full bg-black">
                  {previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Fundus preview"
                      className="h-full w-full object-cover"
                      onLoad={onPreviewLoad}
                    />
                  ) : (
                    <div className="flex h-full flex-col items-center justify-center gap-3 px-4 text-center">
                      <div
                        className="flex h-12 w-12 items-center justify-center rounded-full"
                        style={{ background: `rgba(${G_RGB}, 0.08)`, border: `1px solid rgba(${G_RGB}, 0.2)` }}
                      >
                        <Upload size={20} style={{ color: G, opacity: 0.6 }} />
                      </div>
                      <span className="text-xs text-neutral-600">Click to upload</span>
                    </div>
                  )}
                  {imageDims && previewUrl && (
                    <span
                      className="absolute bottom-2 left-1/2 -translate-x-1/2 rounded-md px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-black"
                      style={{ background: G }}
                    >
                      {imageDims}
                    </span>
                  )}
                </div>
              </div>
              <input
                type="file"
                accept="image/*"
                className="sr-only"
                onChange={handleFileChange}
              />
            </label>
          </section>

          {/* Clinical inquiry */}
          <section className="mb-6">
            <div className="mb-2 flex items-center gap-2" style={{ color: `rgba(${G_RGB}, 0.7)` }}>
              <MessageSquare size={15} strokeWidth={2} />
              <span className="text-xs font-medium">Clinical Inquiry</span>
            </div>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={4}
              disabled={isAnalyzing}
              className="w-full resize-none rounded-xl border bg-black px-3 py-2.5 text-sm leading-relaxed text-neutral-200 outline-none transition-colors placeholder:text-neutral-700"
              style={{
                borderColor: `rgba(${G_RGB}, 0.12)`,
                minHeight: "100px",
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = `rgba(${G_RGB}, 0.4)`;
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = `rgba(${G_RGB}, 0.12)`;
              }}
            />
          </section>

          {/* Pipeline sidebar */}
          <section className="mb-6 flex-1">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Zap size={16} style={{ color: G }} />
                <span className="text-sm font-semibold text-neutral-200">Pipeline</span>
              </div>
              <span className="font-mono text-xs" style={{ color: `rgba(${G_RGB}, 0.5)` }}>
                {completedCount}/{totalStages}
              </span>
            </div>
            <div className="mb-4 h-1 overflow-hidden rounded-full bg-neutral-900">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${progressPct}%`,
                  background: `linear-gradient(90deg, #22c55e, ${G})`,
                  boxShadow: `0 0 12px rgba(${G_RGB}, 0.5)`,
                }}
              />
            </div>
            <ul className="space-y-0.5">
              {STAGE_ORDER.map((id) => {
                const st = stages[id]?.status ?? "idle";
                const isActive = st === "running";
                const done = isDone(st);
                return (
                  <li
                    key={id}
                    className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-all"
                    style={
                      isActive
                        ? {
                            color: "#fff",
                            background: `rgba(${G_RGB}, 0.08)`,
                            boxShadow: `0 0 24px rgba(${G_RGB}, 0.1), inset 0 0 0 1px rgba(${G_RGB}, 0.2)`,
                          }
                        : { color: done ? G : "rgb(115, 115, 115)" }
                    }
                  >
                    {done ? (
                      <CheckCircle2
                        className="shrink-0"
                        size={18}
                        strokeWidth={2}
                        style={{ color: G }}
                      />
                    ) : isActive ? (
                      <Loader2
                        className="shrink-0 animate-spin"
                        size={18}
                        style={{ color: G }}
                      />
                    ) : (
                      <span
                        className="inline-block h-[18px] w-[18px] shrink-0 rounded-full border"
                        style={{ borderColor: "rgb(64, 64, 64)" }}
                      />
                    )}
                    <span className="flex items-center gap-1.5 font-medium">
                      {STAGE_ICONS[id]}
                      {STAGE_LABELS[id]}
                    </span>
                    {isActive && (
                      <ChevronRight size={14} className="ml-auto opacity-60" style={{ color: G }} />
                    )}
                  </li>
                );
              })}
            </ul>
          </section>

          {error && (
            <div className="mb-3 rounded-xl border border-red-500/30 bg-red-950/30 px-3 py-2 text-xs text-red-300">
              {error}
            </div>
          )}

          <button
            type="button"
            onClick={handleAnalyze}
            disabled={isAnalyzing || !selectedFile}
            className="mb-3 flex w-full items-center justify-center gap-2 rounded-xl py-3 text-sm font-bold text-black transition enabled:hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
            style={{ background: G }}
          >
            {isAnalyzing ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Analyzing…
              </>
            ) : (
              <>
                <Sparkles size={16} />
                Run analysis
              </>
            )}
          </button>

          <button
            type="button"
            onClick={isAnalyzing ? handleCancel : onBackHome}
            className="flex w-full items-center justify-center gap-2 rounded-xl border py-3 text-sm font-semibold text-neutral-300 transition hover:text-white"
            style={{
              background: "rgba(255, 255, 255, 0.03)",
              borderColor: "rgba(255, 255, 255, 0.08)",
            }}
          >
            <X size={16} strokeWidth={2.5} />
            {isAnalyzing ? "Cancel" : "Exit"}
          </button>
        </aside>

        {/* ── Main: pipeline only ───────────────────────────── */}
        <main className="relative flex min-h-0 min-w-0 flex-1 flex-col bg-black">
          <div
            className="pointer-events-none absolute inset-0 opacity-[0.45]"
            style={{
              backgroundImage: `
                linear-gradient(rgba(115, 115, 115, 0.12) 1px, transparent 1px),
                linear-gradient(90deg, rgba(115, 115, 115, 0.12) 1px, transparent 1px)
              `,
              backgroundSize: "24px 24px",
            }}
          />
          <div className="relative z-[1] flex min-h-0 flex-1 flex-col px-6 py-5 md:px-10 md:py-6">
            <div className="custom-scrollbar flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl bg-black/40 backdrop-blur-[2px]">
              <div className="flex shrink-0 items-center justify-between gap-3 px-5 py-4">
                <div className="flex items-center gap-2">
                  <Zap size={18} style={{ color: G }} />
                  <h2 className="text-sm font-bold uppercase tracking-widest md:text-base" style={{ color: G }}>
                    Pipeline
                  </h2>
                </div>
                {isAnalyzing ? (
                  <span
                    className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-semibold"
                    style={{
                      background: `rgba(${G_RGB}, 0.12)`,
                      color: G,
                    }}
                  >
                    <Loader2 className="h-3 w-3 animate-spin" />
                    Running
                  </span>
                ) : results ? (
                  <span
                    className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-semibold"
                    style={{
                      background: `rgba(${G_RGB}, 0.14)`,
                      color: G,
                    }}
                  >
                    <CheckCircle2 size={12} />
                    Done
                  </span>
                ) : (
                  <span className="text-[11px] font-medium text-neutral-600">
                    {completedCount}/{totalStages} steps
                  </span>
                )}
              </div>

              <div className="custom-scrollbar min-h-0 flex-1 overflow-y-auto p-5 text-sm">
                {STAGE_ORDER.map((id, idx) => {
                  const stage = stages[id] ?? {
                    status: "idle" as StageStatus,
                    message: "",
                    thinking: null,
                    output: null,
                  };
                  const highlightSynth =
                    id === "synthesis" &&
                    stage.status === "complete" &&
                    Boolean(results);

                  const done = isDone(stage.status);
                  const isRunning = stage.status === "running";
                  const isLast = idx === STAGE_ORDER.length - 1;

                  return (
                    <div
                      key={id}
                      className="stage-reveal flex gap-4"
                      style={{ animationDelay: `${idx * 120}ms` }}
                    >
                      <div className="flex flex-col items-center pt-1">
                        <div
                          className="relative flex h-3 w-3 shrink-0 items-center justify-center rounded-full transition-all duration-300"
                          style={
                            done
                              ? {
                                  background: G,
                                  boxShadow: `0 0 8px rgba(${G_RGB}, 0.5)`,
                                }
                              : isRunning
                                ? {
                                    background: "transparent",
                                    border: `2px solid ${G}`,
                                    boxShadow: `0 0 10px rgba(${G_RGB}, 0.4)`,
                                  }
                                : {
                                    background: "transparent",
                                    border: "2px solid rgb(64, 64, 64)",
                                  }
                          }
                        >
                          {isRunning && (
                            <span
                              className="absolute inset-[-4px] rounded-full"
                              style={{
                                border: `1.5px solid rgba(${G_RGB}, 0.3)`,
                                animation: "dot-ping 1.5s ease-in-out infinite",
                              }}
                            />
                          )}
                        </div>
                        {!isLast && (
                          <div
                            className="mt-1 w-px flex-1 transition-colors duration-500"
                            style={{
                              background: done
                                ? `rgba(${G_RGB}, 0.3)`
                                : "rgba(255, 255, 255, 0.06)",
                              minHeight: "24px",
                            }}
                          />
                        )}
                      </div>

                      <div
                        className="mb-4 min-w-0 flex-1 rounded-xl p-4"
                        style={
                          highlightSynth
                            ? { background: `rgba(${G_RGB}, 0.06)` }
                            : { background: "rgba(0, 0, 0, 0.25)" }
                        }
                      >
                        <h4
                          className="mb-3 text-[11px] font-bold uppercase tracking-wider"
                          style={{
                            color: highlightSynth ? G : done ? G : "rgb(115, 115, 115)",
                          }}
                        >
                          {STAGE_LABELS[id]}
                        </h4>
                        {renderPipelineStageBody(id, stage)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </main>
      </div>

      <style>{`
        @keyframes thinking-shimmer {
          0%, 100% { opacity: 0.85; }
          50% { opacity: 1; }
        }
        .thinking-stream {
          animation: thinking-shimmer 2s ease-in-out infinite;
        }
        @keyframes stage-fade-in {
          from {
            opacity: 0;
            transform: translateY(12px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .stage-reveal {
          animation: stage-fade-in 0.4s ease-out both;
        }
        @keyframes dot-ping {
          0%, 100% {
            transform: scale(1);
            opacity: 0.6;
          }
          50% {
            transform: scale(1.6);
            opacity: 0;
          }
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
          background: rgba(${G_RGB}, 0.15);
          border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
          background: rgba(${G_RGB}, 0.3);
        }
      `}</style>
    </div>
  );
}
