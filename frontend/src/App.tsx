import { useState, useRef, useCallback } from 'react';
import {
  UploadCloud, MessageSquare, Activity, Image as ImageIcon, Sparkles,
  Microscope, CheckCircle2, Loader2, FileText, AlertCircle, Eye, Brain,
  Zap, ShieldCheck, ChevronRight, X, GitBranch, Send,
} from 'lucide-react';

/* ------------------------------------------------------------------ */
/*  6-Stage Pipeline Definition                                        */
/* ------------------------------------------------------------------ */

const STAGES = [
  { id: 'input',     label: 'Input',                   icon: Send       },
  { id: 'gemma3',    label: 'Gemma 3 — Pre-scan',     icon: Eye        },
  { id: 'routing',   label: 'FunctionGemma — Routing', icon: GitBranch  },
  { id: 'paligemma', label: 'PaliGemma 2 — Analysis',  icon: Microscope },
  { id: 'medgemma',  label: 'MedGemma 4B — Diagnosis', icon: Brain      },
  { id: 'synthesis', label: 'Gemma 3 — Summary',       icon: Zap        },
] as const;

type StageId = (typeof STAGES)[number]['id'];
type Status  = 'idle' | 'running' | 'complete' | 'error';

interface StageState {
  status: Status;
  message: string;
  data?: any;
}

interface Results {
  prescan: string | null;
  segmentation: any | null;
  diagnosis: any | null;
  synthesis: string | null;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

function Dashboard() {
  const [file, setFile]               = useState<File | null>(null);
  const [preview, setPreview]         = useState<string | null>(null);
  const [res, setRes]                 = useState<string | null>(null);
  const [question, setQuestion]       = useState('');
  const [analyzing, setAnalyzing]     = useState(false);
  const [results, setResults]         = useState<Results | null>(null);
  const [error, setError]             = useState<string | null>(null);
  const [stages, setStages]           = useState<Record<StageId, StageState>>({} as any);
  const abortRef                      = useRef<AbortController | null>(null);

  /* ── Upload ── */
  const onUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(url);
    setResults(null); setError(null); setStages({} as any);
    const img = new Image();
    img.onload = () => setRes(`${img.naturalWidth} × ${img.naturalHeight}px`);
    img.src = url;
  };

  /* ── Stage updater ── */
  const setStage = useCallback((id: StageId, s: StageState) => {
    setStages(prev => ({ ...prev, [id]: s }));
  }, []);

  /* ── Run pipeline ── */
  const runAnalysis = async () => {
    if (!file) return;
    setAnalyzing(true); setResults(null); setError(null);
    // Initialize all stages to idle
    const init: Record<string, StageState> = {};
    STAGES.forEach(s => init[s.id] = { status: 'idle', message: '' });
    setStages(init as any);

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const fd = new FormData();
    fd.append('file', file);
    fd.append('question', question.trim() || 'Analyze this retinal fundus image. What condition is present and what is the severity?');

    try {
      const resp = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST', body: fd, signal: ctrl.signal,
      });
      if (!resp.ok) throw new Error(`Server ${resp.status}`);
      const reader = resp.body?.getReader();
      if (!reader) throw new Error('No stream');

      const dec = new TextDecoder();
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\n'); buf = '';
        let ev = '', dt = '';
        for (const ln of lines) {
          if (ln.startsWith('event: ')) ev = ln.slice(7).trim();
          else if (ln.startsWith('data: ')) dt = ln.slice(6);
          else if (ln === '' && ev && dt) {
            try { handleEvent(ev, JSON.parse(dt)); } catch {}
            ev = ''; dt = '';
          } else if (ln !== '') buf += ln + '\n';
        }
      }
    } catch (e: any) {
      if (e.name !== 'AbortError')
        setError('Failed to connect to backend. Ensure `python backend/main.py` is running on port 8000.');
    } finally { setAnalyzing(false); }
  };

  /* ── SSE handler ── */
  const handleEvent = (ev: string, d: any) => {
    if (ev === 'stage') {
      setStage(d.id as StageId, {
        status: d.status,
        message: d.message || '',
        data: d.data,
      });
    } else if (ev === 'complete') {
      setResults(d.results);
      setAnalyzing(false);
    } else if (ev === 'error') {
      setError(d.message);
      setAnalyzing(false);
    }
  };

  const cancel = () => { abortRef.current?.abort(); setAnalyzing(false); };
  const reset  = () => { setFile(null); setPreview(null); setRes(null); setResults(null); setError(null); setStages({} as any); setQuestion(''); };

  /* ── Computed ── */
  const stageList = STAGES.map(s => ({ ...s, ...(stages[s.id] || { status: 'idle' as Status, message: '' }) }));
  const done  = stageList.filter(s => s.status === 'complete').length;
  const active = stageList.find(s => s.status === 'running');

  /* ================================================================ */
  /*  Render                                                           */
  /* ================================================================ */
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 flex font-sans selection:bg-green-500/30">

      {/* ── LEFT SIDEBAR ── */}
      <aside className="w-full md:w-[380px] xl:w-[420px] flex-shrink-0 bg-zinc-900/40 backdrop-blur-3xl border-r border-zinc-800/60 p-6 flex flex-col space-y-5 relative overflow-y-auto">

        {/* Brand */}
        <div className="flex items-center space-x-3 pb-5 border-b border-zinc-800/50">
          <div className="bg-green-600/20 p-2.5 rounded-xl border border-green-500/30 shadow-[0_0_20px_rgba(74,222,128,0.2)]">
            <Activity className="w-6 h-6 text-green-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-white to-zinc-400 bg-clip-text text-transparent">OptiAssist</h1>
            <p className="text-[10px] text-zinc-500 font-bold tracking-widest uppercase mt-0.5">Agentic Diagnostic AI</p>
          </div>
        </div>

        <div className="flex-1 space-y-5">

          {/* Upload */}
          <div className="space-y-2">
            <label className="text-sm font-semibold text-zinc-300 flex items-center space-x-2">
              <ImageIcon className="w-4 h-4 text-green-400" /><span>Retinal Image</span>
            </label>
            <div className="relative group cursor-pointer">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-green-500 to-emerald-500 rounded-2xl blur opacity-10 group-hover:opacity-30 transition duration-500" />
              <label htmlFor="upload" className="relative flex flex-col items-center justify-center w-full h-40 border-2 border-dashed border-zinc-700/50 rounded-2xl hover:border-green-500/50 hover:bg-green-500/5 transition-all duration-300 cursor-pointer overflow-hidden bg-zinc-900/50">
                {preview ? (
                  <>
                    <img src={preview} className="absolute inset-0 w-full h-full object-contain p-2 opacity-90 group-hover:opacity-40 transition-opacity" alt="Preview" />
                    {res && <div className="absolute bottom-2 left-1/2 -translate-x-1/2 bg-black/70 backdrop-blur-md text-[10px] uppercase font-mono tracking-wider text-green-400 px-2.5 py-1 rounded-full border border-green-500/20 shadow-lg pointer-events-none group-hover:opacity-0 transition-opacity">{res}</div>}
                  </>
                ) : (
                  <div className="flex flex-col items-center pt-4 pb-5">
                    <UploadCloud className="w-9 h-9 text-zinc-500 mb-2 group-hover:text-green-400 group-hover:-translate-y-1 transition-all duration-300" />
                    <p className="mb-1 text-sm text-zinc-400"><span className="font-semibold text-green-400">Click to upload</span> or drag and drop</p>
                    <p className="text-xs text-zinc-600">JPEG, PNG or TIFF</p>
                  </div>
                )}
                <input id="upload" type="file" className="hidden" accept="image/*" onChange={onUpload} disabled={analyzing} />
              </label>
            </div>
            {preview && !analyzing && (
              <button onClick={reset} className="text-xs text-red-400 hover:text-red-300 font-medium flex items-center space-x-1">
                <X className="w-3 h-3" /><span>Remove</span>
              </button>
            )}
          </div>

          {/* Question */}
          <div className="space-y-2">
            <label className="text-sm font-semibold text-zinc-300 flex items-center space-x-2">
              <MessageSquare className="w-4 h-4 text-emerald-400" /><span>Clinical Inquiry</span>
            </label>
            <textarea
              className="w-full h-24 bg-zinc-900/60 border border-zinc-800 rounded-xl p-3 text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-green-500/50 focus:border-green-500/50 transition-all resize-none shadow-inner disabled:opacity-50"
              placeholder="What is the cup-to-disc ratio in this image?"
              value={question} onChange={e => setQuestion(e.target.value)} disabled={analyzing}
            />
          </div>

          {/* ── Pipeline tracker ── */}
          {Object.keys(stages).length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-semibold text-zinc-300 flex items-center space-x-2">
                  <Zap className="w-4 h-4 text-amber-400" /><span>Pipeline</span>
                </label>
                <span className="text-[10px] text-zinc-500 font-mono">{done}/{STAGES.length}</span>
              </div>
              <div className="w-full h-1 bg-zinc-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-green-500 to-emerald-400 rounded-full transition-all duration-500 ease-out" style={{ width: `${(done / STAGES.length) * 100}%` }} />
              </div>
              <div className="space-y-1">
                {stageList.map(s => {
                  const Icon = s.icon;
                  return (
                    <div key={s.id} className={`flex items-center space-x-2.5 px-2.5 py-1.5 rounded-lg transition-all duration-300 ${
                      s.status === 'running' ? 'bg-green-500/10 border border-green-500/20' :
                      s.status === 'complete' ? 'bg-zinc-800/20 border border-zinc-800/30' :
                      s.status === 'error' ? 'bg-red-500/10 border border-red-500/20' :
                      'bg-zinc-900/20 border border-zinc-800/10'
                    }`}>
                      <div className="w-5 h-5 flex items-center justify-center flex-shrink-0">
                        {s.status === 'running' ? <Loader2 className="w-3.5 h-3.5 text-green-400 animate-spin" /> :
                         s.status === 'complete' ? <CheckCircle2 className="w-3.5 h-3.5 text-green-500" /> :
                         s.status === 'error' ? <AlertCircle className="w-3.5 h-3.5 text-red-400" /> :
                         <Icon className="w-3.5 h-3.5 text-zinc-600" />}
                      </div>
                      <span className={`text-[11px] font-medium truncate ${
                        s.status === 'running' ? 'text-green-300' :
                        s.status === 'complete' ? 'text-zinc-500' :
                        s.status === 'error' ? 'text-red-300' :
                        'text-zinc-600'
                      }`}>{s.label}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-xs p-3 rounded-xl flex items-start space-x-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" /><span>{error}</span>
          </div>
        )}

        {/* CTA */}
        <div className="pt-1">
          {analyzing ? (
            <button onClick={cancel} className="w-full flex items-center justify-center space-x-2 py-3.5 rounded-xl font-bold text-sm bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 transition-all">
              <X className="w-4 h-4" /><span>Cancel</span>
            </button>
          ) : (
            <button onClick={runAnalysis} disabled={!preview} className={`w-full group relative flex items-center justify-center space-x-2 py-3.5 rounded-xl font-bold text-sm transition-all duration-300 overflow-hidden ${
              preview ? 'bg-zinc-100 text-zinc-900 hover:scale-[1.02] active:scale-95 shadow-[0_0_30px_rgba(255,255,255,0.1)]' :
              'bg-zinc-800/50 text-zinc-600 cursor-not-allowed border border-zinc-800'
            }`}>
              {preview && <div className="absolute inset-0 bg-[linear-gradient(to_right,transparent,rgba(0,0,0,0.1),transparent)] -translate-x-[150%] group-hover:translate-x-[150%] transition-transform duration-1000" />}
              <Sparkles className={`w-4 h-4 ${preview ? 'text-green-600' : 'text-zinc-600'}`} />
              <span>Run Agentic Analysis</span>
            </button>
          )}
        </div>

        <div className="flex items-center space-x-2 text-[10px] text-zinc-600">
          <ShieldCheck className="w-3 h-3" /><span>For research use only. Not for clinical diagnosis.</span>
        </div>
      </aside>

      {/* ── RIGHT PANEL ── */}
      <main className="flex-1 flex flex-col relative bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-zinc-900 via-zinc-950 to-black overflow-hidden">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#8080800a_1px,transparent_1px),linear-gradient(to_bottom,#8080800a_1px,transparent_1px)] bg-[size:24px_24px]" />

        <div className="flex-1 flex items-center justify-center p-8 sm:p-12 z-10 relative">

          {/* ── Analyzing ── */}
          {analyzing && preview && (
            <div className="w-full h-full max-w-4xl bg-zinc-900/30 backdrop-blur-xl border border-zinc-800/60 rounded-3xl p-8 flex flex-col shadow-2xl">
              <div className="flex justify-between items-center mb-6 pb-6 border-b border-zinc-800/50">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20">
                    <Microscope className="w-4 h-4 text-emerald-400" />
                  </div>
                  <h2 className="text-xl font-bold tracking-tight">Active Context</h2>
                </div>
                <span className="flex items-center space-x-2 bg-blue-500/10 text-blue-400 py-1.5 px-3 rounded-full text-xs font-semibold border border-blue-500/20">
                  <Loader2 className="w-3 h-3 animate-spin" /><span>Processing</span>
                </span>
              </div>
              <div className="flex-1 flex flex-col items-center justify-center bg-black/40 rounded-2xl border border-zinc-800/80 p-8">
                <div className="w-28 h-28 bg-zinc-900 rounded-full border-2 border-zinc-800 flex items-center justify-center shadow-[0_0_60px_rgba(74,222,128,0.1)] relative mb-8">
                  <div className="absolute inset-0 border-t-2 border-green-400 rounded-full animate-spin" />
                  <div className="absolute inset-1 border-b-2 border-emerald-300/30 rounded-full animate-[spin_2s_linear_infinite_reverse]" />
                  {active ? (() => { const I = active.icon; return <I className="w-8 h-8 text-green-400" />; })() : <Sparkles className="w-8 h-8 text-green-400" />}
                </div>
                <h3 className="text-2xl font-bold text-zinc-100 mb-2">{active?.label || 'Initializing…'}</h3>
                <p className="text-zinc-400 text-sm text-center max-w-md">{active?.message || 'Starting multi-model pipeline…'}</p>
                <div className="flex flex-wrap justify-center gap-1.5 mt-6">
                  {stageList.map(s => (
                    <div key={s.id} className={`flex items-center space-x-1 px-2.5 py-1 rounded-full text-[10px] font-semibold transition-all duration-500 ${
                      s.status === 'running' ? 'bg-green-500/15 text-green-400 border border-green-500/30 shadow-[0_0_10px_rgba(74,222,128,0.1)]' :
                      s.status === 'complete' ? 'bg-zinc-800/50 text-zinc-500 border border-zinc-700/30' :
                      s.status === 'error' ? 'bg-red-500/10 text-red-400 border border-red-500/30' :
                      'bg-zinc-900/50 text-zinc-600 border border-zinc-800/30'
                    }`}>
                      {s.status === 'running' ? <Loader2 className="w-2.5 h-2.5 animate-spin" /> :
                       s.status === 'complete' ? <CheckCircle2 className="w-2.5 h-2.5" /> :
                       s.status === 'error' ? <AlertCircle className="w-2.5 h-2.5" /> :
                       <span className="w-1.5 h-1.5 rounded-full bg-zinc-700" />}
                      <span>{s.label}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ── Results ── */}
          {!analyzing && results && preview && (
            <div className="w-full h-full max-w-5xl bg-zinc-900/30 backdrop-blur-xl border border-zinc-800/60 rounded-3xl p-6 md:p-8 flex flex-col shadow-2xl">
              <div className="flex justify-between items-center mb-6 pb-6 border-b border-zinc-800/50">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20">
                    <FileText className="w-4 h-4 text-emerald-400" />
                  </div>
                  <h2 className="text-xl font-bold tracking-tight">Diagnostic Report</h2>
                </div>
                <span className="flex items-center space-x-2 bg-green-500/10 text-green-400 py-1.5 px-3 rounded-full text-xs font-semibold border border-green-500/20">
                  <CheckCircle2 className="w-3 h-3" /><span>Complete</span>
                </span>
              </div>

              <div className="flex-1 overflow-y-auto space-y-5 pr-2 custom-scrollbar">

                {/* Synthesis */}
                <div className="bg-black/40 rounded-2xl border border-zinc-800/80 p-6">
                  <div className="flex items-center mb-4">
                    <Zap className="w-5 h-5 text-green-400 mr-2" />
                    <h3 className="text-lg font-bold text-zinc-100">Gemma 3 Clinical Synthesis</h3>
                  </div>
                  <div className="text-zinc-300 leading-relaxed whitespace-pre-wrap text-sm font-medium">
                    {results.synthesis || 'No synthesis available.'}
                  </div>
                </div>

                {/* MedGemma */}
                {results.diagnosis && (
                  <div className="bg-zinc-900/80 border border-zinc-800 rounded-2xl p-5">
                    <div className="flex items-center mb-4">
                      <Brain className="w-5 h-5 text-purple-400 mr-2" />
                      <h3 className="text-sm font-bold text-zinc-300 uppercase tracking-widest">MedGemma Assessment</h3>
                    </div>
                    <div className="grid grid-cols-3 gap-3 mb-4">
                      <div className="bg-zinc-800/50 rounded-xl p-3 border border-zinc-700/30">
                        <p className="text-[10px] text-zinc-500 uppercase font-bold tracking-widest mb-1">Condition</p>
                        <p className="text-green-400 font-bold">{results.diagnosis.condition || '—'}</p>
                      </div>
                      <div className="bg-zinc-800/50 rounded-xl p-3 border border-zinc-700/30">
                        <p className="text-[10px] text-zinc-500 uppercase font-bold tracking-widest mb-1">Severity</p>
                        <p className="text-amber-400 font-bold">{results.diagnosis.severity || '—'}</p>
                      </div>
                      <div className="bg-zinc-800/50 rounded-xl p-3 border border-zinc-700/30">
                        <p className="text-[10px] text-zinc-500 uppercase font-bold tracking-widest mb-1">Confidence</p>
                        <p className="text-blue-400 font-bold">{typeof results.diagnosis.confidence === 'number' ? `${Math.round(results.diagnosis.confidence * 100)}%` : '—'}</p>
                      </div>
                    </div>
                    {results.diagnosis.findings?.length > 0 && (
                      <div className="mb-4">
                        <p className="text-[10px] text-zinc-500 uppercase font-bold tracking-widest mb-2">Findings</p>
                        <div className="flex flex-wrap gap-1.5">
                          {results.diagnosis.findings.map((f: string, i: number) => (
                            <span key={i} className="bg-zinc-800/80 text-zinc-300 text-[11px] px-2.5 py-1 rounded-full border border-zinc-700/40 flex items-center space-x-1">
                              <ChevronRight className="w-3 h-3 text-green-500" /><span>{f}</span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    {results.diagnosis.recommendation && (
                      <div className="bg-emerald-500/5 border border-emerald-500/10 rounded-xl p-3">
                        <p className="text-[10px] text-emerald-500 uppercase font-bold tracking-widest mb-1">Recommendation</p>
                        <p className="text-zinc-300 text-sm">{results.diagnosis.recommendation}</p>
                      </div>
                    )}
                  </div>
                )}

                {/* PaliGemma */}
                {results.segmentation && (
                  <div className="bg-zinc-900/80 border border-zinc-800 rounded-2xl p-5">
                    <div className="flex items-center mb-3">
                      <Microscope className="w-5 h-5 text-cyan-400 mr-2" />
                      <h3 className="text-sm font-bold text-zinc-300 uppercase tracking-widest">PaliGemma 2 Analysis</h3>
                    </div>
                    {results.segmentation.summary && <p className="text-zinc-400 text-sm mb-3">{results.segmentation.summary}</p>}
                    {results.segmentation.raw_output && (
                      <div className="bg-zinc-800/40 rounded-lg p-3 border border-zinc-700/20">
                        <p className="text-[10px] text-zinc-500 uppercase font-bold tracking-widest mb-1">Raw Model Output</p>
                        <p className="text-zinc-400 text-xs font-mono whitespace-pre-wrap">{results.segmentation.raw_output}</p>
                      </div>
                    )}
                  </div>
                )}

                {/* Prescan */}
                {results.prescan && (
                  <div className="bg-zinc-900/80 border border-zinc-800 rounded-2xl p-5">
                    <div className="flex items-center mb-3">
                      <Eye className="w-5 h-5 text-yellow-400 mr-2" />
                      <h3 className="text-sm font-bold text-zinc-300 uppercase tracking-widest">Gemma 3 Pre-Scan</h3>
                    </div>
                    <p className="text-zinc-400 text-sm leading-relaxed">{results.prescan}</p>
                  </div>
                )}

                <div className="flex items-center space-x-2 text-[10px] text-zinc-600 pb-2">
                  <ShieldCheck className="w-3 h-3" /><span>For research use only. Not intended for clinical diagnosis.</span>
                </div>
              </div>
            </div>
          )}

          {/* ── Ready ── */}
          {!analyzing && !results && preview && (
            <div className="w-full h-full max-w-4xl bg-zinc-900/30 backdrop-blur-xl border border-zinc-800/60 rounded-3xl p-8 flex flex-col shadow-2xl">
              <div className="flex justify-between items-center mb-6 pb-6 border-b border-zinc-800/50">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20"><Microscope className="w-4 h-4 text-emerald-400" /></div>
                  <h2 className="text-xl font-bold tracking-tight">Active Context</h2>
                </div>
                <span className="flex items-center space-x-2 bg-emerald-500/10 text-emerald-400 py-1.5 px-3 rounded-full text-xs font-semibold border border-emerald-500/20">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" /><span>Image Attached</span>
                </span>
              </div>
              <div className="flex-1 flex flex-col items-center justify-center bg-black/40 rounded-2xl border border-zinc-800/80 p-8 relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-t from-green-500/5 to-transparent pointer-events-none" />
                <Sparkles className="w-12 h-12 text-green-400/50 mb-6 drop-shadow-[0_0_15px_rgba(74,222,128,0.3)] animate-pulse" />
                <h3 className="text-2xl font-bold text-zinc-100 mb-3">Pipeline Ready</h3>
                <p className="text-zinc-400 max-w-md text-center text-sm">
                  Click <strong className="text-green-400">Run Agentic Analysis</strong> to start the 6-stage pipeline:
                </p>
                <div className="flex flex-wrap justify-center gap-1.5 mt-5">
                  {STAGES.map(({ id, label, icon: I }) => (
                    <div key={id} className="flex items-center space-x-1 px-2 py-1 rounded-full text-[10px] font-semibold bg-zinc-900/50 text-zinc-500 border border-zinc-800/30">
                      <I className="w-3 h-3" /><span>{label}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ── Empty ── */}
          {!preview && (
            <div className="text-center max-w-lg bg-zinc-900/20 backdrop-blur-xl p-12 rounded-3xl border border-zinc-800/30 shadow-2xl">
              <div className="w-24 h-24 mx-auto bg-gradient-to-br from-green-500/10 to-emerald-500/10 rounded-full flex items-center justify-center mb-6 shadow-[0_0_60px_rgba(74,222,128,0.05)] border border-green-500/20">
                <ImageIcon className="w-10 h-10 text-green-400" />
              </div>
              <h2 className="text-2xl font-bold mb-4 text-zinc-100">Awaiting Submissions</h2>
              <p className="text-zinc-500 text-sm leading-relaxed">Upload a retinal fundus image to start the pipeline.</p>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}

export default function App() {
  
  return <Dashboard />;
}
