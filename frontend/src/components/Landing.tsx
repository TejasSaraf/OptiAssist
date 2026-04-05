import { Activity, Microscope, Brain, GitBranch, ArrowRight, Eye, ShieldCheck, Sparkles, Network } from 'lucide-react';

interface LandingProps {
  onStart: () => void;
}

export default function Landing({ onStart }: LandingProps) {
  return (
    <div className="min-h-screen bg-black text-zinc-100 flex flex-col font-sans selection:bg-green-500/40 selection:text-white overflow-hidden relative">
      {/* Subtle green grid on black */}
      <div
        className="absolute inset-0 opacity-[0.12] pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(to right, rgb(34 197 94 / 0.35) 1px, transparent 1px),
            linear-gradient(to bottom, rgb(34 197 94 / 0.35) 1px, transparent 1px)
          `,
          backgroundSize: '32px 32px',
        }}
      />
      <div className="absolute top-0 right-0 w-[min(100%,42rem)] h-[min(100vh,36rem)] bg-green-500/[0.07] rounded-full blur-[140px] pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-[min(100%,36rem)] h-[min(100vh,28rem)] bg-emerald-600/[0.06] rounded-full blur-[120px] pointer-events-none" />

      <header className="relative z-10 px-6 py-5 flex items-center justify-between border-b border-green-500/20 bg-black/80 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="bg-green-500/15 p-2.5 rounded-xl border border-green-500/40 shadow-[0_0_24px_rgba(34,197,94,0.25)]">
            <Activity className="w-6 h-6 text-green-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white tracking-tight">OpusAI</h1>
            <p className="text-[10px] text-green-500/80 font-semibold tracking-[0.2em] uppercase mt-0.5">
              Agentic Diagnostic AI
            </p>
          </div>
        </div>
        <div className="hidden sm:flex items-center gap-2 text-[10px] text-green-400/90 font-medium px-3 py-1.5 rounded-full border border-green-500/30 bg-green-500/5">
          <ShieldCheck className="w-3.5 h-3.5 text-green-400" />
          <span>Research Access Only</span>
        </div>
      </header>

      <main className="flex-1 relative z-10 flex flex-col items-center justify-center p-6 text-center max-w-5xl mx-auto w-full">
        <div className="inline-flex items-center justify-center gap-2 px-4 py-2 bg-green-500/10 border border-green-500/35 rounded-full mb-8 shadow-[0_0_28px_rgba(34,197,94,0.2)]">
          <Sparkles className="w-4 h-4 text-green-400 animate-pulse" />
          <span className="text-xs font-semibold text-green-300 tracking-wide">Retinal Intelligence</span>
        </div>

        <h2 className="text-5xl sm:text-7xl font-extrabold tracking-tight mb-4 leading-[1.08] text-white">
          OpusAI <span className="bg-gradient-to-r from-green-300 via-green-400 to-emerald-500 bg-clip-text text-transparent">Retinal Intelligence</span>
          <br />
          <span className="text-3xl sm:text-5xl font-bold text-zinc-300 mt-2 block">
            for Diabetic Retinopathy
          </span>
        </h2>

        <p className="text-lg sm:text-xl text-zinc-400 max-w-2xl mx-auto mb-12 leading-relaxed">
          Upload a fundus image. OpusAI detects diabetic retinopathy, streams live clinical
          reasoning, and lets you interrogate every finding in seconds.
        </p>

        <button
          type="button"
          onClick={onStart}
          className="group relative inline-flex items-center justify-center gap-2 px-8 py-4 bg-green-500 text-black font-bold rounded-2xl hover:bg-green-400 active:scale-[0.98] transition-all duration-300 shadow-[0_0_40px_rgba(34,197,94,0.35)] border border-green-400/50 overflow-hidden"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/25 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700" />
          <span className="relative">Launch Diagnostic Pipeline</span>
          <ArrowRight className="w-5 h-5 relative group-hover:translate-x-0.5 transition-transform" />
        </button>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-20 w-full">
          {[
            {
              icon: Eye,
              title: '1. Initial Pre-Scan',
              body: 'Gemma 3 performs a preliminary visual assessment to identify obvious abnormalities and set clinical context.',
              accent: 'from-green-500/20 to-green-600/5',
            },
            {
              icon: GitBranch,
              title: '2. Autonomous Routing',
              body: 'FunctionGemma analyzes the pre-scan and dynamically routes to specialized models for deeper insight.',
              accent: 'from-emerald-500/20 to-emerald-600/5',
            },
            {
              icon: Microscope,
              title: '3. Precise Segmentation',
              body: 'PaliGemma 2 detects and segments key features (optic disc/cup, lesions) to compute structural metrics.',
              accent: 'from-lime-500/15 to-green-600/5',
            },
            {
              icon: Brain,
              title: '4. Medical Diagnosis',
              body: 'MedGemma 4B processes all collected data to provide the final structured condition assessment.',
              accent: 'from-green-400/15 to-emerald-700/10',
            },
          ].map(({ icon: Icon, title, body, accent }) => (
            <div
              key={title}
              className="group bg-zinc-950/80 backdrop-blur-md border border-green-500/20 p-6 rounded-3xl text-left hover:border-green-400/40 hover:bg-black/90 transition-all duration-300"
            >
              <div
                className={`w-10 h-10 rounded-xl bg-gradient-to-br ${accent} flex items-center justify-center border border-green-500/30 mb-4 shadow-[0_0_16px_rgba(34,197,94,0.12)] group-hover:shadow-[0_0_20px_rgba(34,197,94,0.22)]`}
              >
                <Icon className="w-5 h-5 text-green-400" />
              </div>
              <h3 className="text-sm font-bold text-white mb-2">{title}</h3>
              <p className="text-xs text-zinc-500 leading-relaxed group-hover:text-zinc-400 transition-colors">{body}</p>
            </div>
          ))}
        </div>
      </main>

      <footer className="relative z-10 border-t border-green-500/20 px-6 py-5 flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-zinc-500 bg-black/90 backdrop-blur-md">
        <div className="flex items-center gap-2 text-green-500/70">
          <Network className="w-4 h-4 text-green-500/80" />
          <span>Local execution — QLoRA-capable pipeline</span>
        </div>
        <p className="text-zinc-600">© 2026 OpusAI Diagnostics. Local inference mode.</p>
      </footer>
    </div>
  );
}
