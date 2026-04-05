import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
    Stethoscope,
    ArrowRight,
    Eye,
    Activity,
    Sparkles,
    GitBranch,
    Search,
    ScanEye,
} from "lucide-react";

// ─────────────────────────────────────────────
//  Three.js Background — fast moving points only
// ─────────────────────────────────────────────
function ThreeBackground() {
    const mountRef = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        let animId: number | undefined;

        const init = async () => {
            // @ts-expect-error Runtime CDN import is resolved by the browser.
            const THREE = await import("https://esm.sh/three@0.160.0");

            const mount = mountRef.current;
            if (!mount) return;

            const scene = new THREE.Scene();
            const W = mount.clientWidth;
            const H = mount.clientHeight;

            const camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 1000);
            camera.position.z = 40;

            const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
            renderer.setSize(W, H);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.setClearColor(0x000000, 0);
            mount.appendChild(renderer.domElement);

            const COUNT = 350;
            const pointData: Array<{ vx: number; vy: number; vz: number }> = [];
            const positions = new Float32Array(COUNT * 3);
            const colors = new Float32Array(COUNT * 3);

            const greenShades = [
                [0.29, 0.87, 0.50],
                [0.13, 0.77, 0.37],
                [0.53, 0.93, 0.69],
                [0.08, 0.64, 0.34],
            ];
            for (let i = 0; i < COUNT; i++) {
                positions[i * 3 + 0] = (Math.random() - 0.5) * 90;
                positions[i * 3 + 1] = (Math.random() - 0.5) * 70;
                positions[i * 3 + 2] = (Math.random() - 0.5) * 50;

                pointData.push({
                    vx: (Math.random() - 0.5) * 0.028,
                    vy: (Math.random() - 0.5) * 0.028,
                    vz: (Math.random() - 0.5) * 0.018,
                });

                const shade = greenShades[Math.floor(Math.random() * greenShades.length)];
                colors[i * 3 + 0] = shade[0];
                colors[i * 3 + 1] = shade[1];
                colors[i * 3 + 2] = shade[2];
            }

            const geo = new THREE.BufferGeometry();
            geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
            geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

            const mat = new THREE.PointsMaterial({
                size: 0.18,
                vertexColors: true,
                transparent: true,
                opacity: 0.85,
                sizeAttenuation: true,
            });

            const points = new THREE.Points(geo, mat);
            scene.add(points);

            const onResize = () => {
                const W2 = mount.clientWidth;
                const H2 = mount.clientHeight;
                camera.aspect = W2 / H2;
                camera.updateProjectionMatrix();
                renderer.setSize(W2, H2);
            };
            window.addEventListener("resize", onResize);

            const BOUND_X = 45, BOUND_Y = 35, BOUND_Z = 25;

            const animate = () => {
                animId = requestAnimationFrame(animate);
                const pos = geo.attributes.position.array;

                for (let i = 0; i < COUNT; i++) {
                    pos[i * 3 + 0] += pointData[i].vx;
                    pos[i * 3 + 1] += pointData[i].vy;
                    pos[i * 3 + 2] += pointData[i].vz;

                    if (pos[i * 3 + 0] > BOUND_X) pos[i * 3 + 0] = -BOUND_X;
                    if (pos[i * 3 + 0] < -BOUND_X) pos[i * 3 + 0] = BOUND_X;
                    if (pos[i * 3 + 1] > BOUND_Y) pos[i * 3 + 1] = -BOUND_Y;
                    if (pos[i * 3 + 1] < -BOUND_Y) pos[i * 3 + 1] = BOUND_Y;
                    if (pos[i * 3 + 2] > BOUND_Z) pos[i * 3 + 2] = -BOUND_Z;
                    if (pos[i * 3 + 2] < -BOUND_Z) pos[i * 3 + 2] = BOUND_Z;
                }

                geo.attributes.position.needsUpdate = true;
                renderer.render(scene, camera);
            };
            animate();

            return () => {
                window.removeEventListener("resize", onResize);
                if (animId !== undefined) {
                    cancelAnimationFrame(animId);
                }
                renderer.dispose();
                if (mount.contains(renderer.domElement)) {
                    mount.removeChild(renderer.domElement);
                }
            };
        };

        let cleanup: (() => void) | undefined;
        void init().then((fn) => {
            cleanup = fn;
        });
        return () => {
            cleanup?.();
        };
    }, []);

    return (
        <div
            ref={mountRef}
            style={{ position: "fixed", inset: 0, zIndex: 0, pointerEvents: "none" }}
        />
    );
}

// ─────────────────────────────────────────────
//  Data
// ─────────────────────────────────────────────
/** Core capabilities — model lineup shown as simple cards */
const modelCards = [
    { id: 1, name: "Gemma 3 4b", Icon: Sparkles, accentColor: "#4ade80" },
    { id: 2, name: "PaliGemma 3b + LoRA", Icon: Search, accentColor: "#7fee64" },
    { id: 3, name: "MedGemma 4b", Icon: Stethoscope, accentColor: "#86efac" },
    { id: 4, name: "FunctionGemma 270m", Icon: GitBranch, accentColor: "#22c55e" },
];

/** Shown in the How it works terminal — matches backend /api/analyze order */
const onDeviceValueProps = [
    {
        title: "Patient Data Privacy",
        body: "Patient retinal images are protected under HIPAA and GDPR. Uploading to cloud AI creates serious compliance risk.",
    },
    {
        title: "Real-time Speed",
        body: "Cloud round-trips add 500ms+ of delay. On-device inference responds in under 50ms — fast enough for live consultation.",
    },
    {
        title: "No Cloud Upload Required",
        body: "Existing tools send scans to external servers. OpusAI runs entirely on-device, built for regulated clinical environments.",
    },
    {
        title: "Smarter Clinical Tools",
        body: "Manual retinal review is slow and error-prone. AI detects patterns and anomalies instantly, augmenting clinical decision-making.",
    },
];

const analyzePipelineSteps = [
    "Input — receive fundus image and optional clinical question",
    "Gemma 3 — pre-scan for anatomy and diabetic retinopathy-related signs",
    "FunctionGemma — route the workflow (segmentation vs. diagnosis tools)",
    "PaliGemma 2 — detect and localize fundus structures and lesions",
    "MedGemma — structured reasoning (condition, severity, findings, recommendation)",
    "Gemma 3 — merge stage outputs into one clinical summary",
];

// ─────────────────────────────────────────────
//  Main Component
// ─────────────────────────────────────────────
export default function LandingPage() {
    const [scrolled, setScrolled] = useState(false);
    const [activeFeature, setActiveFeature] = useState<number | null>(null);
    const [pipelineLinesVisible, setPipelineLinesVisible] = useState(0);
    const pipelineTerminalRef = useRef<HTMLDivElement | null>(null);
    const pipelineAnimStartedRef = useRef(false);
    const pipelineIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const navigate = useNavigate();

    useEffect(() => {
        const onScroll = () => setScrolled(window.scrollY > 20);
        window.addEventListener("scroll", onScroll);
        return () => window.removeEventListener("scroll", onScroll);
    }, []);

    useEffect(() => {
        const el = pipelineTerminalRef.current;
        if (!el) return;

        const obs = new IntersectionObserver(
            ([entry]) => {
                if (!entry.isIntersecting || pipelineAnimStartedRef.current) return;
                pipelineAnimStartedRef.current = true;

                setPipelineLinesVisible(1);
                let shown = 1;
                pipelineIntervalRef.current = window.setInterval(() => {
                    if (shown >= analyzePipelineSteps.length) {
                        if (pipelineIntervalRef.current !== null) {
                            window.clearInterval(pipelineIntervalRef.current);
                            pipelineIntervalRef.current = null;
                        }
                        return;
                    }
                    shown += 1;
                    setPipelineLinesVisible(shown);
                }, 520);
            },
            { threshold: 0.2, rootMargin: "0px 0px -5% 0px" },
        );

        obs.observe(el);
        return () => {
            obs.disconnect();
            if (pipelineIntervalRef.current !== null) {
                window.clearInterval(pipelineIntervalRef.current);
                pipelineIntervalRef.current = null;
            }
        };
    }, []);

    const handleGetStarted = () => {
        navigate("/app");
    };

    return (
        <div style={{
            minHeight: "100vh",
            backgroundColor: "#000000",
            color: "#fff",
            fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
            overflowX: "hidden",
            position: "relative",
        }}>
            <ThreeBackground />

            {/* Green-on-black vignette */}
            <div style={{
                position: "fixed", inset: 0, zIndex: 1, pointerEvents: "none",
                background: `
                    radial-gradient(ellipse 90% 70% at 50% 40%, rgba(22, 101, 52, 0.22) 0%, transparent 55%),
                    radial-gradient(ellipse 60% 50% at 50% 100%, rgba(34, 197, 94, 0.08) 0%, transparent 50%),
                    linear-gradient(180deg, #000000 0%, #020804 45%, #000000 100%)
                `,
            }} />

            <div style={{ position: "relative", zIndex: 2 }}>

                {/* ═══════════ NAVBAR ═══════════ */}
                <nav style={{
                    position: "fixed", top: 0, left: 0, right: 0, zIndex: 50,
                    transition: "all 0.3s ease",
                    background: scrolled ? "rgba(0,0,0,0.92)" : "rgba(0,0,0,0.35)",
                    backdropFilter: scrolled ? "blur(14px)" : "blur(8px)",
                    borderBottom: scrolled ? "1px solid rgba(34,197,94,0.25)" : "1px solid rgba(34,197,94,0.08)",
                }}>
                    <div style={{ maxWidth: 1400, margin: "0 auto", padding: "0 2rem", height: 56, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                            <Eye size={22} color="#4ade80" strokeWidth={2.25} aria-hidden />
                            <span style={{ fontSize: 17, fontWeight: 700, color: "#fff", letterSpacing: "-0.02em" }}>OpusAI</span>
                        </div>

                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <button
                                type="button"
                                onClick={handleGetStarted}
                                style={{
                                    padding: "6px 14px",
                                    borderRadius: 8,
                                    background: "#4ade80",
                                    color: "#030706",
                                    border: "none",
                                    fontSize: 12,
                                    fontWeight: 700,
                                    cursor: "pointer",
                                    transition: "all 0.2s",
                                }}
                                onMouseEnter={(e) => { e.currentTarget.style.background = "#22c55e"; e.currentTarget.style.transform = "scale(1.02)"; }}
                                onMouseLeave={(e) => { e.currentTarget.style.background = "#4ade80"; e.currentTarget.style.transform = "scale(1)"; }}
                            >
                                Get Started
                            </button>
                        </div>
                    </div>
                </nav>

                {/* ═══════════ SECTION 1 — FULL-SCREEN HERO ═══════════ */}
                <section style={{ position: "relative", minHeight: "100dvh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", paddingTop: 56, paddingBottom: 48, paddingLeft: "1.5rem", paddingRight: "1.5rem", overflow: "hidden", background: "transparent", boxSizing: "border-box" }}>

                    <div
                        style={{
                            position: "relative",
                            zIndex: 10,
                            width: "100%",
                            maxWidth: 920,
                            margin: "0 auto",
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            textAlign: "center",
                        }}
                    >
                        {/* Badge */}
                        <div
                            style={{
                                display: "inline-flex",
                                alignItems: "center",
                                justifyContent: "center",
                                gap: 8,
                                padding: "7px 18px",
                                borderRadius: 999,
                                border: "1px solid rgba(34, 197, 94, 0.45)",
                                background: "transparent",
                                marginBottom: 28,
                            }}
                        >
                            <ScanEye size={15} color="#4ade80" strokeWidth={2.25} aria-hidden style={{ flexShrink: 0 }} />
                            <span style={{ color: "#86efac", fontSize: 11, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase" }}>
                                CLINICAL VISION AI
                            </span>
                        </div>

                        <h1
                            style={{
                                fontSize: "clamp(40px, 7vw, 86px)",
                                fontWeight: 800,
                                lineHeight: 1.12,
                                letterSpacing: "-0.03em",
                                margin: "0 0 18px",
                                width: "100%",
                                textAlign: "center",
                            }}
                        >
                            <span className="hero-headline-line1">
                                <span style={{ color: "#fff" }}>OpusAI </span>
                                <span
                                    style={{
                                        backgroundImage: "linear-gradient(135deg, rgb(74, 222, 128) 0%, rgb(34, 211, 238) 100%)",
                                        WebkitBackgroundClip: "text",
                                        backgroundClip: "text",
                                        color: "transparent",
                                        WebkitTextFillColor: "transparent",
                                    }}
                                >
                                    Retinal intelligence
                                </span>
                            </span>
                            <br />
                            <span style={{ color: "rgba(255,255,255,0.92)", display: "block", fontSize: "clamp(30px, 5.2vw, 48px)", fontWeight: 800, letterSpacing: "-0.02em", marginTop: "0.12em" }}>
                                for Diabetic Retinopathy
                            </span>
                        </h1>

                        <p style={{ color: "rgba(255,255,255,0.85)", fontSize: "clamp(15px, 2.2vw, 18px)", lineHeight: 1.75, maxWidth: 560, margin: "0 0 40px", fontWeight: 400, textAlign: "center", padding: "0 0.25rem" }}>
                            Upload a fundus image. OpusAI detects diabetic retinopathy, streams live clinical reasoning, and lets you interrogate every findingf in seconds.
                        </p>

                        {/* CTA Buttons */}
                        <div style={{ display: "flex", gap: 14, justifyContent: "center", flexWrap: "wrap", width: "100%" }}>
                            <button
                                type="button"
                                onClick={handleGetStarted}
                                style={{
                                    padding: "14px 28px",
                                    borderRadius: 12,
                                    background: "#4ade80",
                                    color: "#030706",
                                    border: "none",
                                    fontSize: 15,
                                    fontWeight: 700,
                                    cursor: "pointer",
                                    transition: "all 0.3s",
                                    display: "inline-flex",
                                    alignItems: "center",
                                    gap: 10,
                                }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.background = "#34d399";
                                    e.currentTarget.style.transform = "translateY(-2px)";
                                    e.currentTarget.style.boxShadow = "0 14px 32px rgba(74, 222, 128, 0.35)";
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.background = "#4ade80";
                                    e.currentTarget.style.transform = "none";
                                    e.currentTarget.style.boxShadow = "none";
                                }}
                            >
                                Analyze an Image <ArrowRight size={18} strokeWidth={2.5} />
                            </button>
                            <button
                                type="button"
                                style={{
                                    padding: "14px 28px",
                                    borderRadius: 12,
                                    border: "1px solid rgba(55, 65, 60, 0.95)",
                                    boxShadow: "inset 0 0 0 1px rgba(74, 222, 128, 0.12)",
                                    background: "transparent",
                                    color: "#fff",
                                    fontSize: 15,
                                    fontWeight: 600,
                                    cursor: "pointer",
                                    transition: "all 0.3s",
                                    display: "inline-flex",
                                    alignItems: "center",
                                    gap: 10,
                                }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.borderColor = "rgba(74, 222, 128, 0.45)";
                                    e.currentTarget.style.background = "rgba(74, 222, 128, 0.06)";
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.borderColor = "rgba(55, 65, 60, 0.95)";
                                    e.currentTarget.style.background = "transparent";
                                }}
                            >
                                Watch Demo
                            </button>
                        </div>
                    </div>

                    {/* Scroll indicator */}
                    <div style={{ position: "absolute", bottom: 32, left: "50%", transform: "translateX(-50%)", display: "flex", flexDirection: "column", alignItems: "center", gap: 8, opacity: 0.4 }}>
                        <div style={{ width: 1, height: 40, background: "linear-gradient(to bottom, transparent, #22c55e)" }} />
                        <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22c55e" }} />
                    </div>
                </section>

                {/* ═══════════ SECTION 2 — HOW IT WORKS ═══════════ */}
                <section id="how-it-works" style={{ position: "relative", padding: "96px 0", background: "linear-gradient(180deg, #000000 0%, #031505 50%, #000000 100%)" }}>
                    {/* Grid bg */}
                    <div style={{ position: "absolute", inset: 0, pointerEvents: "none", opacity: 0.9, backgroundImage: "linear-gradient(rgba(34,197,94,0.07) 1px, transparent 1px), linear-gradient(90deg, rgba(34,197,94,0.07) 1px, transparent 1px)", backgroundSize: "48px 48px" }} />
                    {/* Divider */}
                    <div style={{ position: "absolute", top: 0, left: "50%", transform: "translateX(-50%)", width: 600, height: 1, background: "linear-gradient(90deg, transparent, rgba(127,238,100,0.3), transparent)" }} />
                    <div style={{ position: "absolute", top: 0, left: "50%", transform: "translateX(-50%)", width: 700, height: 400, pointerEvents: "none", background: "transparent" }} />

                    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 1.5rem", position: "relative" }}>
                        <div className="landing-how-columns" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 64, alignItems: "center" }}>

                            {/* LEFT */}
                            <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
                                <div style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "6px 14px", borderRadius: 999, border: "1px solid rgba(127,238,100,0.35)", background: "rgba(127,238,100,0.08)", width: "fit-content" }}>
                                    <Activity size={12} color="#7fee64" />
                                    <span style={{ color: "#7fee64", fontSize: 11, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase" }}>How It Works</span>
                                </div>

                                <h2 style={{ fontSize: "clamp(28px, 4vw, 42px)", fontWeight: 800, color: "#fff", letterSpacing: "-0.02em", lineHeight: 1.15, margin: 0 }}>
                                    Three Steps. Zero Cloud
                                </h2>

                                {/* Steps */}
                                <div style={{ display: "flex", flexDirection: "column", gap: 20, paddingTop: 20, borderTop: "1px solid rgba(34,197,94,0.12)" }}>
                                    {[
                                        { num: "01", title: "Upload + question", desc: "Fundus file plus optional clinical inquiry (default example: DR severity)" },
                                        { num: "02", title: "Stream the pipeline", desc: "SSE events for input, Gemma 3, FunctionGemma, PaliGemma 2, MedGemma 4B, synthesis" },
                                        { num: "03", title: "Read outputs", desc: "Structured diagnosis JSON, segmentation payload, prescan text, and merged narrative" },
                                    ].map(({ num, title, desc }, i) => (
                                        <div key={i} style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
                                            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", width: 44, height: 44, borderRadius: 12, background: "rgba(34,197,94,0.14)", border: "1px solid rgba(74,222,128,0.3)", flexShrink: 0 }}>
                                                <span style={{ fontSize: 14, fontWeight: 700, color: "#4ade80" }}>{num}</span>
                                            </div>
                                            <div>
                                                <h4 style={{ fontSize: 15, fontWeight: 700, margin: "0 0 4px", color: "#fff" }}>{title}</h4>
                                                <p style={{ fontSize: 13, color: "rgba(255,255,255,0.45)", margin: 0 }}>{desc}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* RIGHT — Terminal */}
                            <div style={{ position: "relative" }}>
                                <div style={{ position: "absolute", inset: -40, background: " transparent", pointerEvents: "none" }} />
                                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

                                    {/* Terminal */}
                                    <div
                                        ref={pipelineTerminalRef}
                                        style={{ borderRadius: 16, border: "1px solid rgba(34,197,94,0.28)", background: "rgba(0,0,0,0.85)", backdropFilter: "blur(16px)", padding: 24 }}
                                    >
                                        <div style={{ display: "flex", gap: 7, marginBottom: 20, alignItems: "center" }}>
                                            {["#ef4444", "#eab308", "#22c55e"].map((c) => (
                                                <div key={c} style={{ width: 11, height: 11, borderRadius: "50%", background: c }} />
                                            ))}
                                            <span style={{ fontSize: 11, color: "rgba(127,238,100,0.55)", marginLeft: 8, fontFamily: "ui-monospace, monospace" }}>POST /api/analyze</span>
                                        </div>
                                        <div
                                            style={{
                                                display: "flex",
                                                flexDirection: "column",
                                                gap: 10,
                                                fontFamily: "ui-monospace, monospace",
                                                fontSize: 13,
                                                lineHeight: 1.55,
                                                minHeight: 200,
                                            }}
                                        >
                                            {analyzePipelineSteps.slice(0, pipelineLinesVisible).map((line, i) => (
                                                <div key={i} className="pipeline-line-animate" style={{ display: "flex", gap: 12, alignItems: "baseline" }}>
                                                    <span style={{ color: "#4ade80", fontWeight: 700, flexShrink: 0, lineHeight: 1.55 }} aria-hidden>•</span>
                                                    <span style={{ color: "rgba(255,255,255,0.82)" }}>{line}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* ═══════════ SECTION 3 — FEATURES ═══════════ */}
                <section id="features" style={{ position: "relative", padding: "96px 0", background: "#000000" }}>
                    <div style={{ position: "absolute", top: 0, left: "50%", transform: "translateX(-50%)", width: 600, height: 1, background: "linear-gradient(90deg, transparent, rgba(34,197,94,0.35), transparent)" }} />

                    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 1.5rem" }}>
                        <div style={{ textAlign: "center", marginBottom: 48 }}>
                            <p style={{ color: "#4ade80", fontSize: 11, fontWeight: 700, letterSpacing: "0.18em", textTransform: "uppercase", margin: 0 }}>Core capabilities</p>
                        </div>

                        <div className="landing-features-grid" style={{ display: "grid", gap: 24 }}>
                            {modelCards.map(({ id, name, Icon, accentColor }) => (
                                <div key={id}
                                    onMouseEnter={() => setActiveFeature(id)}
                                    onMouseLeave={() => setActiveFeature(null)}
                                    style={{
                                        border: `1px solid ${activeFeature === id ? `${accentColor}55` : "rgba(255,255,255,0.08)"}`,
                                        borderRadius: 18,
                                        padding: "28px 20px",
                                        background: activeFeature === id ? "rgba(127,238,100,0.04)" : "rgba(255,255,255,0.02)",
                                        backdropFilter: "blur(8px)", transition: "all 0.3s ease", cursor: "default",
                                        transform: activeFeature === id ? "translateY(-4px)" : "none",
                                        boxShadow: activeFeature === id ? "0 20px 56px rgba(127,238,100,0.08)" : "none",
                                        position: "relative", overflow: "hidden",
                                        display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center",
                                    }}
                                >
                                    <div style={{ position: "absolute", top: 0, left: 24, right: 24, height: 1, background: activeFeature === id ? `linear-gradient(90deg, transparent, ${accentColor}, transparent)` : "transparent", transition: "all 0.3s" }} />

                                    <div style={{ width: 52, height: 52, borderRadius: 14, background: `${accentColor}15`, border: `1px solid ${accentColor}28`, display: "flex", alignItems: "center", justifyContent: "center" }}>
                                        <Icon size={26} color={accentColor} />
                                    </div>

                                    <h3 style={{ fontSize: 17, fontWeight: 700, marginTop: 18, marginBottom: 0, color: activeFeature === id ? accentColor : "#fff", transition: "color 0.2s", letterSpacing: "-0.02em", lineHeight: 1.3 }}>{name}</h3>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* ═══════════ SECTION 4 — CTA BANNER ═══════════ */}
                <section style={{ position: "relative", padding: "80px 0", background: "linear-gradient(180deg, #000000 0%, #021208 100%)" }}>
                    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 1.5rem" }}>
                        <div style={{ position: "relative", borderRadius: 28, border: "1px solid rgba(34,197,94,0.3)", background: "linear-gradient(145deg, rgba(34,197,94,0.08) 0%, rgba(0,0,0,0.92) 45%, #000000 100%)", padding: "clamp(40px, 6vw, 72px) clamp(24px, 4vw, 48px)", textAlign: "center", overflow: "hidden" }}>
                            <div style={{ position: "absolute", top: -100, right: -100, width: 350, height: 350, borderRadius: "50%", background: "radial-gradient(circle, rgba(127,238,100,0.12) 0%, transparent 70%)", pointerEvents: "none", filter: "blur(40px)" }} />
                            <div style={{ position: "absolute", bottom: -80, left: -80, width: 250, height: 250, borderRadius: "50%", background: "radial-gradient(circle, rgba(34,197,94,0.08) 0%, transparent 70%)", pointerEvents: "none", filter: "blur(40px)" }} />

                            <div style={{ position: "relative", zIndex: 2 }}>
                                <h2 style={{ fontSize: "clamp(28px, 4.5vw, 44px)", fontWeight: 800, letterSpacing: "-0.03em", margin: "0 auto 12px", lineHeight: 1.15, color: "#fff", maxWidth: 800 }}>
                                    Private, fast, and local—fundus intelligence without the cloud
                                </h2>
                                <p style={{ color: "rgba(255,255,255,0.45)", maxWidth: 640, margin: "0 auto 40px", fontSize: 16, lineHeight: 1.65 }}>
                                    Keep PHI on hardware you control, cut latency for live workflows, and augment review—without routing retinal scans through third-party servers.
                                </p>

                                <div className="landing-value-grid">
                                    {onDeviceValueProps.map(({ title, body }) => (
                                        <div
                                            key={title}
                                            style={{
                                                textAlign: "left",
                                                padding: "22px 22px 24px",
                                                borderRadius: 16,
                                                border: "1px solid rgba(255,255,255,0.08)",
                                                background: "rgba(0,0,0,0.35)",
                                            }}
                                        >
                                            <h3 style={{ fontSize: 17, fontWeight: 700, color: "#7fee64", margin: "0 0 10px", letterSpacing: "-0.02em" }}>{title}</h3>
                                            <p style={{ fontSize: 14, color: "rgba(255,255,255,0.58)", lineHeight: 1.7, margin: 0 }}>{body}</p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* ═══════════ FOOTER ═══════════ */}
                <footer style={{ borderTop: "1px solid rgba(34,197,94,0.15)", padding: "32px 0", background: "#000000" }}>
                    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 1.5rem", display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: 16 }}>
                        <p style={{ color: "rgba(255,255,255,0.25)", fontSize: 12, margin: 0 }}>
                            Built for ScarletHacks
                        </p>

                        <p style={{ color: "rgba(255,255,255,0.25)", fontSize: 12, margin: 0, textAlign: "center", flex: "1 1 220px" }}>
                            © 2026 OpusAI
                        </p>

                        <div style={{ display: "flex", gap: 24 }}>
                            {["Privacy", "Terms", "Contact"].map((l) => (
                                <a key={l} href="#"
                                    style={{ fontSize: 12, color: "rgba(255,255,255,0.25)", textDecoration: "none", transition: "color 0.2s" }}
                                    onMouseEnter={(e) => (e.currentTarget.style.color = "#7fee64")}
                                    onMouseLeave={(e) => (e.currentTarget.style.color = "rgba(255,255,255,0.25)")}
                                >{l}</a>
                            ))}
                        </div>
                    </div>
                </footer>
            </div>

            <style>{`
                @keyframes pipelineLineIn {
                    from {
                        opacity: 0;
                        transform: translateY(8px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
                .pipeline-line-animate {
                    animation: pipelineLineIn 0.42s ease-out both;
                }
                .hero-headline-line1 {
                    display: inline-block;
                    white-space: nowrap;
                }
                @media (max-width: 540px) {
                    .hero-headline-line1 { white-space: normal; }
                }
                .landing-features-grid {
                    grid-template-columns: repeat(2, 1fr);
                }
                @media (min-width: 1100px) {
                    .landing-features-grid { grid-template-columns: repeat(4, 1fr); }
                }
                @media (max-width: 640px) {
                    .landing-features-grid { grid-template-columns: 1fr; }
                }
                .landing-value-grid {
                    display: grid;
                    grid-template-columns: 1fr;
                    gap: 20px;
                }
                @media (min-width: 768px) {
                    .landing-value-grid { grid-template-columns: 1fr 1fr; gap: 24px; }
                }
                #how-it-works .landing-how-columns {
                    grid-template-columns: 1fr 1fr;
                }
                @media (max-width: 900px) {
                    #how-it-works .landing-how-columns { grid-template-columns: 1fr !important; gap: 48px !important; }
                }
            `}</style>
        </div>
    );
}
