import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
} from "react";
import {
  Activity,
  CheckCircle2,
  Image as ImageIcon,
  Loader2,
  MessageSquare,
  Microscope,
  X,
  Zap,
} from "lucide-react";

interface DashboardProps {
  onBackHome: () => void;
}

type StageStatus = "idle" | "running" | "complete" | "error" | "skipped";

interface StageState {
  status: StageStatus;
  message: string;
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

/** Matches backend `complete` payload `results` from /api/analyze */
interface AnalysisResults {
  prescan?: string | null;
  routing?: string | null;
  segmentation?: SegmentationResult | null;
  diagnosis?: DiagnosisResult | null;
  synthesis?: string | null;
}

const DEFAULT_QUESTION =
  "What is the cup ti disc ration in this image?";

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

function createInitialStages(): Record<string, StageState> {
  return Object.fromEntries(
    STAGE_ORDER.map((id) => [
      id,
      { status: "idle" as StageStatus, message: "Pending" },
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

function getActiveStageId(
  stages: Record<string, StageState>,
  analyzing: boolean,
): StageId | null {
  const running = STAGE_ORDER.find(
    (id) => stages[id]?.status === "running",
  );
  if (running) return running;
  if (!analyzing) return null;
  const firstPending = STAGE_ORDER.find(
    (id) => stages[id]?.status === "idle",
  );
  return firstPending ?? null;
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

  const updateStage = (
    stageId: string,
    status: StageStatus,
    message: string,
  ) => {
    setStages((current) => ({
      ...current,
      [stageId]: { status, message },
    }));
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
    };

    if (eventName === "stage" && payload.id && payload.status) {
      updateStage(
        payload.id,
        payload.status,
        payload.message ?? "Updated",
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

  const activeId = getActiveStageId(stages, isAnalyzing);
  const activeLabel = activeId
    ? STAGE_LABELS[activeId]
    : results
      ? "Analysis complete"
      : "Ready";
  const activeSubtext = activeId
    ? stages[activeId]?.message ?? "Working…"
    : results
      ? "Full pipeline output is on the right."
      : "Upload a fundus image and run analysis.";

  const showProcessingBadge = isAnalyzing;

  const renderStagePill = (id: StageId) => {
    const st = stages[id]?.status ?? "idle";
    const isRun = st === "running";
    const done = isDone(st);
    return (
      <span
        key={id}
        className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[11px] font-semibold md:text-xs ${
          isRun
            ? "border-[#22c55e] bg-[#22c55e] text-black"
            : done
              ? "border-zinc-700 bg-zinc-900/80 text-[#22c55e]"
              : "border-zinc-800 bg-zinc-900/40 text-zinc-600"
        }`}
        style={
          isRun
            ? { boxShadow: "0 0 20px rgba(34, 197, 94, 0.45)" }
            : undefined
        }
      >
        {done && !isRun && (
          <CheckCircle2 size={12} className="text-[#22c55e]" />
        )}
        {isRun && (
          <Loader2 size={12} className="animate-spin text-black" />
        )}
        {STAGE_LABELS[id]}
      </span>
    );
  };

  return (
    <div
      className="min-h-screen text-white"
      style={{ backgroundColor: "#0d0d0d", fontFamily: "Inter, system-ui, sans-serif" }}
    >
      <div className="flex min-h-screen flex-col md:flex-row">
        {/* ── Left sidebar ───────────────────────────────────── */}
        <aside
          className="flex w-full shrink-0 flex-col border-b border-white/[0.06] px-5 py-6 md:w-[min(100%,380px)] md:border-b-0 md:border-r"
          style={{ backgroundColor: "#121212" }}
        >
          <header className="mb-8 flex items-start gap-3">
            <div
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl"
              style={{
                background: "rgba(34, 197, 94, 0.15)",
                border: "1px solid rgba(34, 197, 94, 0.35)",
                boxShadow: "0 0 20px rgba(34, 197, 94, 0.2)",
              }}
            >
              <Activity
                className="text-[#22c55e]"
                size={22}
                strokeWidth={2.25}
              />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white">
                OpusAI
              </h1>
              <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-zinc-500">
                Agentic diagnostic AI
              </p>
            </div>
          </header>

          {/* Retinal image */}
          <section className="mb-6">
            <div className="mb-2 flex items-center gap-2 text-zinc-400">
              <ImageIcon size={15} strokeWidth={2} />
              <span className="text-xs font-medium">Retinal Image</span>
            </div>
            <label className="block cursor-pointer">
              <div
                className="relative overflow-hidden rounded-2xl border-2 border-dashed p-1"
                style={{ borderColor: "rgba(34, 197, 94, 0.45)" }}
              >
                <div className="relative mx-auto aspect-square w-full max-w-[220px] overflow-hidden rounded-full bg-black/50">
                  {previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Fundus preview"
                      className="h-full w-full object-cover"
                      onLoad={onPreviewLoad}
                    />
                  ) : (
                    <div className="flex h-full flex-col items-center justify-center gap-2 px-4 text-center text-xs text-zinc-600">
                      <ImageIcon className="opacity-40" size={28} />
                      Click to upload
                    </div>
                  )}
                  {imageDims && previewUrl && (
                    <span
                      className="absolute bottom-2 left-1/2 -translate-x-1/2 rounded-md px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-black"
                      style={{ background: "#22c55e" }}
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
            <div className="mb-2 flex items-center gap-2 text-zinc-400">
              <MessageSquare size={15} strokeWidth={2} />
              <span className="text-xs font-medium">Clinical Inquiry</span>
            </div>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              rows={4}
              disabled={isAnalyzing}
              className="w-full resize-none rounded-xl border border-white/[0.08] bg-black/40 px-3 py-2.5 text-sm leading-relaxed text-zinc-200 outline-none placeholder:text-zinc-600 focus:border-[#22c55e]/40"
              style={{ minHeight: "100px" }}
            />
          </section>

          {/* Pipeline */}
          <section className="mb-6 flex-1">
            <div className="mb-3 flex items-center justify-between">
              <div className="flex items-center gap-2 text-zinc-300">
                <Zap className="text-[#22c55e]" size={16} />
                <span className="text-sm font-semibold">Pipeline</span>
              </div>
              <span className="text-xs font-mono text-zinc-500">
                {completedCount}/{totalStages}
              </span>
            </div>
            <div className="mb-4 h-1 overflow-hidden rounded-full bg-zinc-800">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${progressPct}%`,
                  background: "linear-gradient(90deg, #16a34a, #22c55e)",
                  boxShadow: "0 0 12px rgba(34, 197, 94, 0.5)",
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
                    className={`flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-all ${
                      isActive
                        ? "text-white"
                        : done
                          ? "text-[#22c55e]"
                          : "text-zinc-600"
                    }`}
                    style={
                      isActive
                        ? {
                            background: "rgba(34, 197, 94, 0.12)",
                            boxShadow:
                              "0 0 24px rgba(34, 197, 94, 0.15), inset 0 0 0 1px rgba(34, 197, 94, 0.25)",
                          }
                        : undefined
                    }
                  >
                    {done ? (
                      <CheckCircle2
                        className="shrink-0 text-[#22c55e]"
                        size={18}
                        strokeWidth={2}
                      />
                    ) : isActive ? (
                      <Loader2
                        className="shrink-0 animate-spin text-[#22c55e]"
                        size={18}
                      />
                    ) : (
                      <span className="inline-block h-[18px] w-[18px] shrink-0 rounded-full border border-zinc-700" />
                    )}
                    <span
                      className={`font-medium ${
                        isActive ? "" : done ? "" : "text-zinc-600"
                      }`}
                    >
                      {STAGE_LABELS[id]}
                    </span>
                  </li>
                );
              })}
            </ul>
          </section>

          {error && (
            <div className="mb-3 rounded-xl border border-red-500/30 bg-red-950/40 px-3 py-2 text-xs text-red-200">
              {error}
            </div>
          )}

          <button
            type="button"
            onClick={handleAnalyze}
            disabled={isAnalyzing || !selectedFile}
            className="mb-3 w-full rounded-xl py-3 text-sm font-bold text-black transition enabled:hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
            style={{ background: "#22c55e" }}
          >
            {isAnalyzing ? "Analyzing…" : "Run analysis"}
          </button>

          <button
            type="button"
            onClick={isAnalyzing ? handleCancel : onBackHome}
            className="flex w-full items-center justify-center gap-2 rounded-xl py-3 text-sm font-semibold text-white transition hover:opacity-90"
            style={{ background: "#7f1d1d" }}
          >
            <X size={18} strokeWidth={2.5} />
            {isAnalyzing ? "Cancel" : "Exit"}
          </button>
        </aside>

        {/* ── Main: Active Context ───────────────────────────── */}
        <main className="relative flex min-h-0 min-w-0 flex-1 flex-col">
          <div
            className="pointer-events-none absolute inset-0 opacity-[0.35]"
            style={{
              backgroundImage: `
                linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)
              `,
              backgroundSize: "32px 32px",
            }}
          />

          <div className="relative z-[1] flex min-h-0 flex-1 flex-col px-8 py-6 md:px-12 md:py-8">
            <div className="mb-10 flex items-start justify-between gap-4">
              <div className="flex items-center gap-3">
                <Microscope
                  className="text-[#22c55e]"
                  size={26}
                  strokeWidth={2}
                />
                <h2 className="text-xl font-bold tracking-tight md:text-2xl">
                  Active Context
                </h2>
              </div>
              {showProcessingBadge ? (
                <span
                  className="inline-flex items-center gap-2 rounded-full border px-4 py-1.5 text-xs font-semibold"
                  style={{
                    borderColor: "rgba(96, 165, 250, 0.35)",
                    background: "rgba(59, 130, 246, 0.12)",
                    color: "#93c5fd",
                  }}
                >
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Processing
                </span>
              ) : results ? (
                <span
                  className="inline-flex items-center gap-2 rounded-full border border-[#22c55e]/40 bg-[#22c55e]/10 px-4 py-1.5 text-xs font-semibold text-[#86efac]"
                >
                  <CheckCircle2 size={14} />
                  Complete
                </span>
              ) : null}
            </div>

            <div
              className={`flex flex-col items-center justify-center pb-6 ${
                results ? "py-4" : "flex-1 pb-8"
              }`}
            >
              <div
                className={`relative flex items-center justify-center ${
                  results
                    ? "mb-4 h-28 w-28 md:h-32 md:w-32"
                    : "mb-8 h-48 w-48 md:h-56 md:w-56"
                }`}
              >
                <div
                  className="absolute inset-0 rounded-full"
                  style={{
                    border: "3px solid rgba(34, 197, 94, 0.25)",
                    boxShadow:
                      "0 0 60px rgba(34, 197, 94, 0.25), inset 0 0 40px rgba(34, 197, 94, 0.06)",
                  }}
                />
                <div
                  className="absolute inset-2 rounded-full"
                  style={{
                    border: "2px solid rgba(34, 197, 94, 0.5)",
                    animation: "dashboard-pulse 2.2s ease-in-out infinite",
                  }}
                />
                <div
                  className={`relative flex items-center justify-center rounded-full ${
                    results ? "h-16 w-16 md:h-20 md:w-20" : "h-28 w-28 md:h-32 md:w-32"
                  }`}
                  style={{
                    background:
                      "radial-gradient(circle at 30% 30%, rgba(34,197,94,0.25), rgba(0,0,0,0.9))",
                    border: "1px solid rgba(34, 197, 94, 0.35)",
                    boxShadow: "0 0 32px rgba(34, 197, 94, 0.35)",
                  }}
                >
                  <Microscope
                    className="text-[#22c55e]"
                    size={results ? 28 : 44}
                    strokeWidth={1.75}
                  />
                </div>
              </div>

              <h3
                className={`mb-2 text-center font-bold tracking-tight text-white ${
                  results ? "text-lg md:text-xl" : "text-2xl md:text-3xl"
                }`}
              >
                {activeLabel}
              </h3>
              <p className="max-w-md text-center text-sm text-zinc-500 md:text-base">
                {activeSubtext}
              </p>
            </div>

            {/* Horizontal pipeline pills (summary centered on second row, like reference) */}
            <div className="relative z-[1] flex flex-col items-center gap-3 pb-6">
              <div className="flex flex-wrap items-center justify-center gap-2 md:gap-3">
                {STAGE_ORDER.filter((id) => id !== "synthesis").map((id) =>
                  renderStagePill(id),
                )}
              </div>
              <div className="flex justify-center">
                {renderStagePill("synthesis")}
              </div>
            </div>

            {results && (
              <div className="custom-scrollbar relative z-[1] mx-auto mt-2 flex min-h-[200px] w-full max-w-3xl flex-1 flex-col overflow-hidden rounded-2xl border border-white/[0.08] bg-black/40 backdrop-blur-sm md:max-w-4xl md:min-h-[280px]">
                <div className="overflow-y-auto p-5 text-sm">
                  <p className="mb-4 text-xs font-semibold uppercase tracking-widest text-[#22c55e]">
                    Pipeline results
                  </p>

                  <section className="mb-5 rounded-xl border border-[#22c55e]/25 bg-[#22c55e]/[0.06] p-4">
                    <h4 className="mb-1 text-[11px] font-bold uppercase tracking-wider text-[#86efac]">
                      6 · Gemma 3 — Clinical summary
                    </h4>
                    <p className="text-base leading-relaxed text-zinc-100">
                      {results.synthesis?.trim() ||
                        "No summary was returned. Check Ollama (gemma3) and merger logs."}
                    </p>
                  </section>

                  <section className="mb-4 rounded-xl border border-white/[0.08] bg-black/30 p-4">
                    <h4 className="mb-2 text-[11px] font-bold uppercase tracking-wider text-zinc-500">
                      1 · Input
                    </h4>
                    <p className="text-zinc-300">{question}</p>
                  </section>

                  {results.prescan != null && results.prescan !== "" && (
                    <section className="mb-4 rounded-xl border border-white/[0.08] bg-black/30 p-4">
                      <h4 className="mb-2 text-[11px] font-bold uppercase tracking-wider text-zinc-500">
                        2 · Gemma 3 — Pre-scan
                      </h4>
                      <p className="leading-relaxed text-zinc-300">
                        {results.prescan}
                      </p>
                    </section>
                  )}

                  {results.routing != null && results.routing !== "" && (
                    <section className="mb-4 rounded-xl border border-white/[0.08] bg-black/30 p-4">
                      <h4 className="mb-2 text-[11px] font-bold uppercase tracking-wider text-zinc-500">
                        3 · FunctionGemma — Routing
                      </h4>
                      <p className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-zinc-400">
                        {results.routing}
                      </p>
                      <p className="mt-2 text-[11px] text-zinc-600">
                        Full PaliGemma + MedGemma stages always run so MedGemma
                        can use PaliGemma output.
                      </p>
                    </section>
                  )}

                  {results.segmentation && (
                    <section className="mb-4 rounded-xl border border-white/[0.08] bg-black/30 p-4">
                      <h4 className="mb-2 text-[11px] font-bold uppercase tracking-wider text-zinc-500">
                        4 · PaliGemma 2 — What&apos;s in the image
                      </h4>
                      {results.segmentation.summary && (
                        <p className="mb-2 leading-relaxed text-zinc-300">
                          {results.segmentation.summary}
                        </p>
                      )}
                      {results.segmentation.raw_output && (
                        <details className="mt-2">
                          <summary className="cursor-pointer text-xs text-[#22c55e]">
                            Raw model output
                          </summary>
                          <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap rounded-lg bg-black/50 p-3 text-[11px] text-zinc-500">
                            {results.segmentation.raw_output}
                          </pre>
                        </details>
                      )}
                      <p className="mt-2 text-xs text-zinc-600">
                        Detections:{" "}
                        {(results.segmentation.detections ?? []).length} region(s)
                      </p>
                    </section>
                  )}

                  {results.diagnosis && (
                    <section className="mb-2 rounded-xl border border-white/[0.08] bg-black/30 p-4">
                      <h4 className="mb-3 text-[11px] font-bold uppercase tracking-wider text-zinc-500">
                        5 · MedGemma 4B — Diagnosis (uses PaliGemma context)
                      </h4>
                      <div className="grid gap-3 sm:grid-cols-2">
                        <div>
                          <p className="text-xs text-zinc-500">Condition</p>
                          <p className="font-semibold text-white">
                            {results.diagnosis.condition ?? "—"}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs text-zinc-500">Severity</p>
                          <p className="font-semibold text-white">
                            {results.diagnosis.severity ?? "—"}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs text-zinc-500">Confidence</p>
                          <p className="font-semibold text-white">
                            {typeof results.diagnosis.confidence === "number"
                              ? `${Math.round(results.diagnosis.confidence * 100)}%`
                              : "—"}
                          </p>
                        </div>
                      </div>
                      {results.diagnosis.findings &&
                        results.diagnosis.findings.length > 0 && (
                          <div className="mt-3">
                            <p className="mb-1 text-xs text-zinc-500">Findings</p>
                            <ul className="list-inside list-disc text-zinc-400">
                              {results.diagnosis.findings.map((f) => (
                                <li key={f}>{f}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      {results.diagnosis.recommendation && (
                        <div className="mt-3 border-t border-white/[0.06] pt-3">
                          <p className="mb-1 text-xs text-zinc-500">
                            Recommendation
                          </p>
                          <p className="text-zinc-300">
                            {results.diagnosis.recommendation}
                          </p>
                        </div>
                      )}
                    </section>
                  )}
                </div>
              </div>
            )}
          </div>
        </main>
      </div>

      <style>{`
        @keyframes dashboard-pulse {
          0%, 100% { opacity: 0.5; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.02); }
        }
      `}</style>
    </div>
  );
}
