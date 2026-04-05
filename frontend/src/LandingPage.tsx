import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import logo from "./logo.png";
import {
    Zap,
    Wrench,
    Stethoscope,
    ArrowRight,
    Eye,
    Activity,
    Cpu,
    Layers,
    LogIn,
    CheckCircle,
    TrendingUp,
    Users,
    Clock,
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
                [0.13, 0.77, 0.37],
                [0.30, 0.86, 0.50],
                [0.08, 0.64, 0.34],
                [0.53, 0.93, 0.69],
            ];

            for (let i = 0; i < COUNT; i++) {
                positions[i * 3 + 0] = (Math.random() - 0.5) * 90;
                positions[i * 3 + 1] = (Math.random() - 0.5) * 70;
                positions[i * 3 + 2] = (Math.random() - 0.5) * 50;

                pointData.push({
                    vx: (Math.random() - 0.5) * 0.055,
                    vy: (Math.random() - 0.5) * 0.055,
                    vz: (Math.random() - 0.5) * 0.030,
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
                size: 0.22,
                vertexColors: true,
                transparent: true,
                opacity: 0.75,
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
const features = [
    {
        id: 1,
        name: "Paligema",
        Icon: Zap,
        description: "High-performance vision encoder that streamlines fundus image analysis with intelligent automation and real-time lesion detection.",
        tag: "Core Engine",
        accentColor: "#22c55e",
    },
    {
        id: 2,
        name: "FunctionGemma",
        Icon: Wrench,
        description: "Advanced agentic routing layer that executes complex multi-step operations seamlessly, bridging raw data with actionable clinical insights.",
        tag: "Functional Layer",
        accentColor: "#7fee64",
    },
    {
        id: 3,
        name: "MedGemma",
        Icon: Stethoscope,
        description: "Specialized medical-grade language module delivering precision diagnostics, evidence-based recommendations, and live clinical decision support.",
        tag: "Medical Module",
        accentColor: "#86efac",
    },
];



const infoPoints = [
    "AI-powered optimization for every workflow",
    "Real-time insights and recommendations",
    "Seamlessly integrates with your existing tools",
];

const typewriterPoints = [
    "Real-time DR detection — 5-class severity grading with confidence scores in under 2 seconds",
    "Explainable AI reasoning — watch the clinical logic stream step-by-step, not just a result",
    "Doctor-first design — chat with MedGemma about findings; AI assists, clinicians decide",
];

// ─────────────────────────────────────────────
//  Main Component
// ─────────────────────────────────────────────
export default function LandingPage() {
    const [scrolled, setScrolled] = useState(false);
    const [activeFeature, setActiveFeature] = useState<number | null>(null);
    const [displayedText, setDisplayedText] = useState(["", "", ""]);
    const [currentPoint, setCurrentPoint] = useState(0);
    const navigate = useNavigate();

    useEffect(() => {
        const onScroll = () => setScrolled(window.scrollY > 20);
        window.addEventListener("scroll", onScroll);
        return () => window.removeEventListener("scroll", onScroll);
    }, []);

    useEffect(() => {
        if (currentPoint >= typewriterPoints.length) return;
        let charIndex = 0;
        const interval = setInterval(() => {
            setDisplayedText((prev) => {
                const updated = [...prev];
                updated[currentPoint] = typewriterPoints[currentPoint].slice(0, charIndex + 1);
                return updated;
            });
            charIndex++;
            if (charIndex === typewriterPoints[currentPoint].length) {
                clearInterval(interval);
                setTimeout(() => setCurrentPoint((p) => p + 1), 900);
            }
        }, 18);
        return () => clearInterval(interval);
    }, [currentPoint]);

    const handleGetStarted = () => {
        navigate("/app");
    };

    // shared inline style helpers
    const btn = {
        primary: {
            display: "flex", alignItems: "center", gap: 8,
            padding: "12px 28px", borderRadius: 10,
            background: "#22c55e", color: "#000",
            border: "none", fontSize: 14, fontWeight: 700, cursor: "pointer",
            transition: "all 0.2s",
        },
        ghost: {
            display: "flex", alignItems: "center", gap: 8,
            padding: "12px 28px", borderRadius: 10,
            border: "1px solid rgba(255,255,255,0.15)", background: "transparent",
            color: "#fff", fontSize: 14, fontWeight: 500, cursor: "pointer",
            transition: "all 0.2s",
        },
    };

    return (
        <div style={{
            minHeight: "100vh",
            backgroundColor: "#000",
            color: "#fff",
            fontFamily: "'DM Sans', 'Inter', sans-serif",
            overflowX: "hidden",
            position: "relative",
        }}>
            <ThreeBackground />

            {/* Ambient radial glow */}
            <div style={{
                position: "fixed", inset: 0, zIndex: 1, pointerEvents: "none",
                background: "transparent",
            }} />

            <div style={{ position: "relative", zIndex: 2 }}>

                {/* ═══════════ NAVBAR ═══════════ */}
                <nav style={{
                    position: "fixed", top: 0, left: 0, right: 0, zIndex: 50,
                    transition: "all 0.3s ease",
                    background: scrolled ? "rgba(0,0,0,0.88)" : "transparent",
                    backdropFilter: scrolled ? "blur(14px)" : "none",
                    borderBottom: scrolled ? "1px solid rgba(127,238,100,0.18)" : "1px solid transparent",
                }}>
                    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 1.5rem", height: 64, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                        <div style={{ display: "flex", alignItems: "center" }}>
                            <img
                                src={logo}
                                alt="OpusAI Logo"
                                style={{
                                    height: 90,
                                    width: "auto",
                                    objectFit: "contain",
                                }}
                            />
                        </div>

                        

                        <button onClick={handleGetStarted}
                            style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 20px", borderRadius: 8, border: "1px solid rgba(127,238,100,0.5)", background: "transparent", color: "#7fee64", fontSize: 14, fontWeight: 500, cursor: "pointer", transition: "all 0.2s" }}
                            onMouseEnter={(e) => { e.currentTarget.style.background = "#22c55e"; e.currentTarget.style.color = "#000"; }}
                            onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "#7fee64"; }}
                        >
                            <LogIn size={15} /> Get Started
                        </button>
                    </div>
                </nav>

                {/* ═══════════ SECTION 1 — FULL-SCREEN HERO ═══════════ */}
                <section style={{ position: "relative", minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", paddingTop: 80, overflow: "hidden" }}>
                    <div style={{ position: "absolute", inset: 0, pointerEvents: "none", background: "transparent 65%)" }} />

                    <div style={{ position: "relative", zIndex: 10, textAlign: "center", maxWidth: 860, margin: "0 auto", padding: "0 1.5rem" }}>
                        {/* Badge */}
                        <div style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "6px 16px", borderRadius: 999, border: "1px solid rgba(127,238,100,0.4)", marginBottom: 36 }}>
                            <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#7fee64" }} />
                            <span style={{ color: "#7fee64", fontSize: 11, fontWeight: 600, letterSpacing: "0.14em", textTransform: "uppercase" }}>
                                AI-Powered Ophthalmology Optimization
                            </span>
                        </div>

                        <h1
                            style={{
                                fontSize: "clamp(40px, 7vw, 72px)", // slightly smaller
                                fontWeight: 800, // fatter text
                                lineHeight: 1.0, // tighter height
                                letterSpacing: "-1.5px", // slightly reduced spacing
                                margin: "0 0 16px", // less bottom space
                            }}
                        >
                            <span style={{ color: "#fff" }}>Train Smarter. </span>
                            <span style={{ color: "#22c55e" }}>Diagnose Faster. </span>
                            <br />
                            <span style={{ color: "#fff" }}>Perform at Your Peak.</span>
                        </h1>



                        <p style={{ color: "rgba(255,255,255,0.45)", fontSize: 17, lineHeight: 1.75, maxWidth: 580, margin: "0 auto 40px" }}>
                            OpusAI uses deep learning to analyze fundus images, identify diabetic
                            retinopathy, and deliver explainable clinical insights — adapting in
                            real time to keep clinicians ahead of every diagnosis.
                        </p>



                        {/* Stats row */}

                    </div>

                    {/* Scroll indicator */}
                    <div style={{ position: "absolute", bottom: 32, left: "50%", transform: "translateX(-50%)", display: "flex", flexDirection: "column", alignItems: "center", gap: 8, opacity: 0.4 }}>
                        <div style={{ width: 1, height: 40, background: "linear-gradient(to bottom, transparent, #22c55e)" }} />
                        <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22c55e" }} />
                    </div>
                </section>

                {/* ═══════════ SECTION 2 — HOW IT WORKS ═══════════ */}
                <section id="how-it-works" style={{ position: "relative", padding: "96px 0" }}>
                    {/* Grid bg */}
                    <div style={{ position: "absolute", inset: 0, pointerEvents: "none", backgroundImage: "linear-gradient(rgba(127,238,100,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(127,238,100,0.04) 1px, transparent 1px)", backgroundSize: "48px 48px" }} />
                    {/* Divider */}
                    <div style={{ position: "absolute", top: 0, left: "50%", transform: "translateX(-50%)", width: 600, height: 1, background: "linear-gradient(90deg, transparent, rgba(127,238,100,0.3), transparent)" }} />
                    <div style={{ position: "absolute", top: 0, left: "50%", transform: "translateX(-50%)", width: 700, height: 400, pointerEvents: "none", background: "transparent" }} />

                    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 1.5rem", position: "relative" }}>
                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 64, alignItems: "center" }}>

                            {/* LEFT */}
                            <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
                                <div style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "6px 14px", borderRadius: 999, border: "1px solid rgba(127,238,100,0.35)", background: "rgba(127,238,100,0.08)", width: "fit-content" }}>
                                    <Activity size={12} color="#7fee64" />
                                    <span style={{ color: "#7fee64", fontSize: 11, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase" }}>How It Works</span>
                                </div>

                                <div>
                                    <h2 style={{ fontSize: 64, fontWeight: 800, color: "#22c55e", letterSpacing: "-2px", lineHeight: 1, margin: 0 }}>OpusAI</h2>
                                    <div style={{ marginTop: 8, height: 3, width: 120, background: "linear-gradient(to right, #22c55e, transparent)", borderRadius: 99 }} />
                                </div>

                                <p style={{ fontSize: 24, color: "rgba(255,255,255,0.7)", fontWeight: 300, lineHeight: 1.5, margin: 0, maxWidth: 440 }}>
                                    Smarter diagnostics —{" "}
                                    <span style={{ color: "#7fee64", fontWeight: 500 }}>better decisions</span>, faster results.
                                </p>

                                <p style={{ color: "rgba(255,255,255,0.38)", fontSize: 15, lineHeight: 1.7, maxWidth: 420, margin: 0 }}>
                                    OpusAI brings together powerful AI modules to help clinicians
                                    optimize, analyze, and act — all from one unified platform.
                                </p>



                                {/* Stats */}
                                <div style={{ display: "flex", alignItems: "center", gap: 24, paddingTop: 16, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                                    {[
                                        { value: "10k+", label: "Active users", Icon: Users },
                                        { value: "99.9%", label: "Uptime", Icon: TrendingUp },
                                        { value: "3x", label: "Faster workflows", Icon: Clock },
                                    ].map(({ value, label, Icon: StatIcon }, i) => (
                                        <div key={i} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                                            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                                <StatIcon size={12} color="#22c55e" />
                                                <span style={{ fontSize: 20, fontWeight: 700 }}>{value}</span>
                                            </div>
                                            <span style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", textTransform: "uppercase", letterSpacing: "0.08em" }}>{label}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* RIGHT — terminal + module cards */}
                            <div style={{ position: "relative" }}>
                                <div style={{ position: "absolute", inset: -40, background: " transparent", pointerEvents: "none" }} />
                                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

                                    {/* Terminal */}
                                    <div style={{ borderRadius: 16, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(0,0,0,0.7)", backdropFilter: "blur(16px)", padding: 24 }}>
                                        <div style={{ display: "flex", gap: 7, marginBottom: 20, alignItems: "center" }}>
                                            {["#ef4444", "#eab308", "#22c55e"].map((c) => (
                                                <div key={c} style={{ width: 11, height: 11, borderRadius: "50%", background: c }} />
                                            ))}
                                            <span style={{ fontSize: 11, color: "rgba(255,255,255,0.2)", marginLeft: 8, fontFamily: "monospace" }}>OpusAI — clinical analysis</span>
                                        </div>
                                        <div style={{ display: "flex", flexDirection: "column", gap: 18, minHeight: 120 }}>
                                            {displayedText.map((text, i) => (
                                                <div key={i} style={{ display: "flex", gap: 12, opacity: i <= currentPoint ? 1 : 0, transform: i <= currentPoint ? "translateY(0)" : "translateY(8px)", transition: "all 0.4s ease" }}>
                                                    <div style={{ marginTop: 6, width: 7, height: 7, borderRadius: "50%", background: "#22c55e", flexShrink: 0, boxShadow: "0 0 6px #22c55e" }} />
                                                    <p style={{ color: "rgba(255,255,255,0.7)", fontSize: 13.5, lineHeight: 1.6, margin: 0 }}>
                                                        <span style={{ color: "#fff", fontWeight: 600 }}>{text.split(" — ")[0]}</span>
                                                        {text.includes(" — ") && <span style={{ color: "rgba(255,255,255,0.45)" }}>{" — "}{text.split(" — ")[1]}</span>}
                                                        {currentPoint === i && <span style={{ color: "#22c55e", marginLeft: 2, animation: "blink 1s step-end infinite" }}>▍</span>}
                                                    </p>
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Module cards */}
                                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
                                        {[
                                            { title: "PaliGemma", subtitle: "Vision encoder", Icon: Cpu },
                                            { title: "FunctionGemma", subtitle: "Agentic routing", Icon: Layers },
                                            { title: "MedGemma", subtitle: "Clinical reasoning", Icon: Stethoscope },
                                        ].map(({ title, subtitle, Icon: CardIcon }, i) => (
                                            <div key={i}
                                                style={{ borderRadius: 14, border: "1px solid rgba(127,238,100,0.18)", background: "rgba(3,19,10,0.7)", padding: "18px 14px", textAlign: "center", cursor: "default", transition: "all 0.25s ease" }}
                                                onMouseEnter={(e) => { e.currentTarget.style.borderColor = "rgba(127,238,100,0.5)"; e.currentTarget.style.transform = "translateY(-3px)"; e.currentTarget.style.boxShadow = "0 0 30px rgba(127,238,100,0.15)"; }}
                                                onMouseLeave={(e) => { e.currentTarget.style.borderColor = "rgba(127,238,100,0.18)"; e.currentTarget.style.transform = "none"; e.currentTarget.style.boxShadow = "none"; }}
                                            >
                                                <div style={{ width: 44, height: 44, margin: "0 auto 12px", borderRadius: 10, background: "rgba(127,238,100,0.1)", border: "1px solid rgba(127,238,100,0.2)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                                                    <CardIcon size={18} color="#22c55e" />
                                                </div>
                                                <p style={{ color: "#7fee64", fontWeight: 600, fontSize: 12, margin: 0 }}>{title}</p>
                                                <p style={{ color: "rgba(255,255,255,0.3)", fontSize: 11, margin: "4px 0 0" }}>{subtitle}</p>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* ═══════════ SECTION 3 — FEATURES ═══════════ */}
                <section id="features" style={{ position: "relative", padding: "96px 0" }}>
                    <div style={{ position: "absolute", top: 0, left: "50%", transform: "translateX(-50%)", width: 600, height: 1, background: "linear-gradient(90deg, transparent, rgba(127,238,100,0.3), transparent)" }} />

                    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 1.5rem" }}>
                        <div style={{ textAlign: "center", marginBottom: 56, display: "flex", flexDirection: "column", gap: 12 }}>
                            <p style={{ color: "#22c55e", fontSize: 11, fontWeight: 700, letterSpacing: "0.18em", textTransform: "uppercase", margin: 0 }}>Modules</p>
                            <h2 style={{ fontSize: 42, fontWeight: 800, margin: 0, letterSpacing: "-1px" }}>Three Powerful Engines</h2>
                            <p style={{ color: "rgba(255,255,255,0.4)", maxWidth: 480, margin: "0 auto", fontSize: 15, lineHeight: 1.7 }}>
                                Each module handles a distinct layer — from vision to agentic routing to clinical intelligence.
                            </p>
                        </div>

                        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 20 }}>
                            {features.map(({ id, name, Icon, description, tag, accentColor }) => (
                                <div key={id}
                                    onMouseEnter={() => setActiveFeature(id)}
                                    onMouseLeave={() => setActiveFeature(null)}
                                    style={{
                                        border: `1px solid ${activeFeature === id ? `${accentColor}55` : "rgba(255,255,255,0.08)"}`,
                                        borderRadius: 16, padding: "28px 24px",
                                        background: activeFeature === id ? "rgba(127,238,100,0.04)" : "rgba(255,255,255,0.02)",
                                        backdropFilter: "blur(8px)", transition: "all 0.3s ease", cursor: "default",
                                        transform: activeFeature === id ? "translateY(-4px)" : "none",
                                        boxShadow: activeFeature === id ? "0 20px 60px rgba(127,238,100,0.08)" : "none",
                                        position: "relative", overflow: "hidden",
                                    }}
                                >
                                    <div style={{ position: "absolute", top: 0, left: 32, right: 32, height: 1, background: activeFeature === id ? `linear-gradient(90deg, transparent, ${accentColor}, transparent)` : "transparent", transition: "all 0.3s" }} />

                                    <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: accentColor, background: `${accentColor}18`, border: `1px solid ${accentColor}30`, padding: "3px 10px", borderRadius: 99 }}>{tag}</span>

                                    <div style={{ marginTop: 20, width: 48, height: 48, borderRadius: 12, background: `${accentColor}15`, border: `1px solid ${accentColor}25`, display: "flex", alignItems: "center", justifyContent: "center" }}>
                                        <Icon size={22} color={accentColor} />
                                    </div>

                                    <h3 style={{ fontSize: 18, fontWeight: 700, marginTop: 16, marginBottom: 10, color: activeFeature === id ? accentColor : "#fff", transition: "color 0.2s", letterSpacing: "-0.3px" }}>{name}</h3>
                                    <p style={{ color: "rgba(255,255,255,0.4)", fontSize: 14, lineHeight: 1.7, margin: 0 }}>{description}</p>


                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* ═══════════ SECTION 4 — CTA BANNER ═══════════ */}
                <section style={{ position: "relative", padding: "80px 0" }}>
                    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 1.5rem" }}>
                        <div style={{ position: "relative", borderRadius: 28, border: "1px solid rgba(127,238,100,0.25)", background: "transparent", padding: "72px 48px", textAlign: "center", overflow: "hidden" }}>
                            <div style={{ position: "absolute", top: -80, right: -80, width: 280, height: 280, borderRadius: "50%", background: "radial-gradient(circle, rgba(127,238,100,0.15) 0%, transparent 70%)", pointerEvents: "none" }} />
                            <div style={{ position: "absolute", bottom: -60, left: -60, width: 200, height: 200, borderRadius: "50%", background: "radial-gradient(circle, rgba(127,238,100,0.08) 0%, transparent 70%)", pointerEvents: "none" }} />

                            <p style={{ color: "#22c55e", fontSize: 11, fontWeight: 700, letterSpacing: "0.18em", textTransform: "uppercase", margin: "0 0 16px" }}>Ready to optimize?</p>
                            <h2 style={{ fontSize: "clamp(32px,5vw,50px)", fontWeight: 800, letterSpacing: "-1.5px", margin: "0 0 20px", lineHeight: 1.15 }}>
                                Start using OpusAI<br />
                                <span style={{ color: "#22c55e" }}>today for free</span>
                            </h2>
                            <p style={{ color: "rgba(255,255,255,0.4)", maxWidth: 440, margin: "0 auto 36px", fontSize: 15, lineHeight: 1.7 }}>
                                No credit card required. Get access to all three modules and start
                                optimizing your workflow in minutes.
                            </p>

                            {/* Checkpoints */}
                            <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 10, marginBottom: 36 }}>
                                {infoPoints.map((pt, i) => (
                                    <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 16px", borderRadius: 99, border: "1px solid rgba(127,238,100,0.2)", background: "rgba(127,238,100,0.06)" }}>
                                        <CheckCircle size={13} color="#22c55e" />
                                        <span style={{ fontSize: 13, color: "rgba(255,255,255,0.6)" }}>{pt}</span>
                                    </div>
                                ))}
                            </div>

                            {/* Quick-start card */}
                            <div style={{ maxWidth: 340, margin: "0 auto 28px", borderRadius: 16, border: "1px solid rgba(127,238,100,0.2)", background: "rgba(255,255,255,0.03)", backdropFilter: "blur(8px)", padding: 20 }}>
                                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                                    {infoPoints.map((point, i) => (
                                        <div key={i} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                            <div style={{ width: 20, height: 20, borderRadius: "50%", background: "rgba(127,238,100,0.2)", border: "1px solid rgba(127,238,100,0.4)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                                                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#7fee64" }} />
                                            </div>
                                            <p style={{ color: "rgba(255,255,255,0.6)", fontSize: 13, margin: 0 }}>{point}</p>
                                        </div>
                                    ))}
                                </div>
                                <div style={{ marginTop: 16, paddingTop: 16, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                                    <button onClick={handleGetStarted}
                                        style={{ width: "100%", padding: "11px 0", borderRadius: 10, background: "rgba(127,238,100,0.1)", border: "1px solid rgba(127,238,100,0.3)", color: "#7fee64", fontSize: 14, fontWeight: 500, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 10, transition: "all 0.2s" }}
                                        onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(127,238,100,0.2)"; }}
                                        onMouseLeave={(e) => { e.currentTarget.style.background = "rgba(127,238,100,0.1)"; }}
                                    >
                                        <ArrowRight size={16} />
                                        Open Dashboard
                                    </button>
                                </div>
                            </div>

                            <div style={{ display: "flex", flexWrap: "wrap", gap: 14, justifyContent: "center" }}>
                                <button onClick={handleGetStarted}
                                    style={{ ...btn.primary, padding: "13px 30px", fontSize: 15 }}
                                    onMouseEnter={(e) => { e.currentTarget.style.background = "#16a34a"; e.currentTarget.style.transform = "scale(1.03)"; }}
                                    onMouseLeave={(e) => { e.currentTarget.style.background = "#22c55e"; e.currentTarget.style.transform = "none"; }}
                                >
                                    Get Started <ArrowRight size={16} />
                                </button>
                                <button
                                    style={{ ...btn.ghost, padding: "13px 30px", fontSize: 15 }}
                                    onMouseEnter={(e) => { e.currentTarget.style.borderColor = "rgba(127,238,100,0.5)"; e.currentTarget.style.color = "#7fee64"; }}
                                    onMouseLeave={(e) => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.15)"; e.currentTarget.style.color = "#fff"; }}
                                >
                                    Talk to Sales
                                </button>
                            </div>
                        </div>
                    </div>
                </section>

                {/* ═══════════ FOOTER ═══════════ */}
                <footer style={{ borderTop: "1px solid rgba(255,255,255,0.06)", padding: "32px 0" }}>
                    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 1.5rem", display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: 16 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <div style={{ width: 26, height: 26, borderRadius: 6, background: "linear-gradient(135deg,#22c55e,#16a34a)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                                <Eye size={13} color="#000" strokeWidth={2.5} />
                            </div>
                            <span style={{ fontSize: 14, fontWeight: 700 }}>
                                <span style={{ color: "#7fee64" }}>Opus</span>
                                <span style={{ color: "#fff" }}>AI</span>
                            </span>
                        </div>

                        <p style={{ color: "rgba(255,255,255,0.2)", fontSize: 12, margin: 0 }}>© 2026 OpusAI. All rights reserved.</p>

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
                @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
                @media(max-width:768px){ .nav-links{ display:none !important; } }
            `}</style>
        </div>
    );
}
