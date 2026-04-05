
// code with full working nd also with the new three.js background and updated styling


// import { useState, useEffect, useRef } from "react";
// import { supabase } from "./supabaseClient";
// import {
//     Zap,
//     Wrench,
//     Stethoscope,
//     ArrowRight,
//     Eye,
//     Brain,
//     Activity,
//     ChevronRight,
//     Cpu,
//     Shield,
//     Layers,
//     LogIn,
// } from "lucide-react";

// // ─────────────────────────────────────────────
// //  Three.js Background
// // ─────────────────────────────────────────────
// function ThreeBackground() {
//     const mountRef = useRef(null);

//     useEffect(() => {
//         let animId;
//         let THREE;

//         const init = async () => {
//             THREE = await import("https://esm.sh/three@0.160.0");

//             const mount = mountRef.current;
//             if (!mount) return;

//             const scene = new THREE.Scene();
//             const W = mount.clientWidth;
//             const H = mount.clientHeight;

//             const camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 1000);
//             camera.position.z = 40;

//             const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
//             renderer.setSize(W, H);
//             renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
//             renderer.setClearColor(0x000000, 0);
//             mount.appendChild(renderer.domElement);

//             // ── 350 individual points with random velocities
//             const COUNT = 350;
//             const pointData = [];

//             const positions = new Float32Array(COUNT * 3);
//             const colors = new Float32Array(COUNT * 3);

//             const greenShades = [
//                 [0.13, 0.77, 0.37],  // #22c55e
//                 [0.30, 0.86, 0.50],  // #4ade80
//                 [0.08, 0.64, 0.34],  // #16a34a
//                 [0.53, 0.93, 0.69],  // #86efac
//             ];

//             for (let i = 0; i < COUNT; i++) {
//                 const x = (Math.random() - 0.5) * 90;
//                 const y = (Math.random() - 0.5) * 70;
//                 const z = (Math.random() - 0.5) * 50;

//                 positions[i * 3 + 0] = x;
//                 positions[i * 3 + 1] = y;
//                 positions[i * 3 + 2] = z;

//                 // individual velocity per point
//                 pointData.push({
//                     vx: (Math.random() - 0.5) * 0.055,
//                     vy: (Math.random() - 0.5) * 0.055,
//                     vz: (Math.random() - 0.5) * 0.030,
//                 });

//                 const shade = greenShades[Math.floor(Math.random() * greenShades.length)];
//                 colors[i * 3 + 0] = shade[0];
//                 colors[i * 3 + 1] = shade[1];
//                 colors[i * 3 + 2] = shade[2];
//             }

//             const geo = new THREE.BufferGeometry();
//             geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
//             geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

//             const mat = new THREE.PointsMaterial({
//                 size: 0.22,
//                 vertexColors: true,
//                 transparent: true,
//                 opacity: 0.75,
//             });

//             const points = new THREE.Points(geo, mat);
//             scene.add(points);

//             // Resize handler
//             const onResize = () => {
//                 const W2 = mount.clientWidth;
//                 const H2 = mount.clientHeight;
//                 camera.aspect = W2 / H2;
//                 camera.updateProjectionMatrix();
//                 renderer.setSize(W2, H2);
//             };
//             window.addEventListener("resize", onResize);

//             // Animate — move each point individually, wrap at bounds
//             const BOUND_X = 45, BOUND_Y = 35, BOUND_Z = 25;

//             const animate = () => {
//                 animId = requestAnimationFrame(animate);

//                 const pos = geo.attributes.position.array;

//                 for (let i = 0; i < COUNT; i++) {
//                     pos[i * 3 + 0] += pointData[i].vx;
//                     pos[i * 3 + 1] += pointData[i].vy;
//                     pos[i * 3 + 2] += pointData[i].vz;

//                     // wrap around bounds
//                     if (pos[i * 3 + 0] >  BOUND_X) pos[i * 3 + 0] = -BOUND_X;
//                     if (pos[i * 3 + 0] < -BOUND_X) pos[i * 3 + 0] =  BOUND_X;
//                     if (pos[i * 3 + 1] >  BOUND_Y) pos[i * 3 + 1] = -BOUND_Y;
//                     if (pos[i * 3 + 1] < -BOUND_Y) pos[i * 3 + 1] =  BOUND_Y;
//                     if (pos[i * 3 + 2] >  BOUND_Z) pos[i * 3 + 2] = -BOUND_Z;
//                     if (pos[i * 3 + 2] < -BOUND_Z) pos[i * 3 + 2] =  BOUND_Z;
//                 }

//                 geo.attributes.position.needsUpdate = true;
//                 renderer.render(scene, camera);
//             };
//             animate();

//             return () => {
//                 window.removeEventListener("resize", onResize);
//                 cancelAnimationFrame(animId);
//                 renderer.dispose();
//                 if (mount.contains(renderer.domElement)) {
//                     mount.removeChild(renderer.domElement);
//                 }
//             };
//         };

//         let cleanup;
//         init().then((fn) => { cleanup = fn; });
//         return () => { cleanup && cleanup(); };
//     }, []);

//     return (
//         <div
//             ref={mountRef}
//             style={{
//                 position: "fixed",
//                 inset: 0,
//                 zIndex: 0,
//                 pointerEvents: "none",
//             }}
//         />
//     );
// }

// // ─────────────────────────────────────────────
// //  Feature data
// // ─────────────────────────────────────────────
// const features = [
//     {
//         id: 1,
//         name: "Paligema",
//         Icon: Zap,
//         description:
//             "High-performance vision encoder that streamlines fundus image analysis with intelligent automation and real-time lesion detection.",
//         tag: "Core Engine",
//         accentColor: "#22c55e",
//     },
//     {
//         id: 2,
//         name: "FunctionGemma",
//         Icon: Wrench,
//         description:
//             "Advanced agentic routing layer that executes complex multi-step operations seamlessly, bridging raw data with actionable clinical insights.",
//         tag: "Functional Layer",
//         accentColor: "#4ade80",
//     },
//     {
//         id: 3,
//         name: "MedGemma",
//         Icon: Stethoscope,
//         description:
//             "Specialized medical-grade language module delivering precision diagnostics, evidence-based recommendations, and live clinical decision support.",
//         tag: "Medical Module",
//         accentColor: "#86efac",
//     },
// ];

// const stats = [
//     { value: "< 2s", label: "Inference time", Icon: Activity },
//     { value: "5-class", label: "DR grading", Icon: Eye },
//     { value: "XAI", label: "Explainability", Icon: Brain },
//     { value: "HIPAA", label: "Compliant", Icon: Shield },
// ];

// // ─────────────────────────────────────────────
// //  Main component
// // ─────────────────────────────────────────────
// export default function LandingPage() {
//     const [scrolled, setScrolled] = useState(false);

//     const points = [
//         "Real-time DR detection — 5-class severity grading with confidence scores in under 2 seconds",
//         "Explainable AI reasoning — watch the clinical logic stream step-by-step, not just a result",
//         "Doctor-first design — chat with MedGemma about findings; AI assists, clinicians decide",
//     ];

//     const [displayedText, setDisplayedText] = useState(["", "", ""]);
//     const [currentPoint, setCurrentPoint] = useState(0);

//     useEffect(() => {
//         const onScroll = () => setScrolled(window.scrollY > 20);
//         window.addEventListener("scroll", onScroll);
//         return () => window.removeEventListener("scroll", onScroll);
//     }, []);

//     useEffect(() => {
//         if (currentPoint >= points.length) return;
//         let charIndex = 0;
//         const interval = setInterval(() => {
//             setDisplayedText((prev) => {
//                 const updated = [...prev];
//                 updated[currentPoint] = points[currentPoint].slice(0, charIndex + 1);
//                 return updated;
//             });
//             charIndex++;
//             if (charIndex === points[currentPoint].length) {
//                 clearInterval(interval);
//                 setTimeout(() => setCurrentPoint((p) => p + 1), 900);
//             }
//         }, 18);
//         return () => clearInterval(interval);
//     }, [currentPoint]);

//     const handleLogin = async () => {
//         await supabase.auth.signOut({ scope: "global" });
//         Object.keys(localStorage).forEach((key) => {
//             if (key.startsWith("sb-")) localStorage.removeItem(key);
//         });
//         await supabase.auth.signInWithOAuth({
//             provider: "google",
//             options: {
//                 redirectTo: "http://localhost:5173/auth/callback",
//                 queryParams: { prompt: "select_account" },
//             },
//         });
//     };

//     return (
//         <div
//             style={{
//                 minHeight: "100vh",
//                 backgroundColor: "#000",
//                 color: "#fff",
//                 fontFamily: "'DM Sans', 'Inter', sans-serif",
//                 overflowX: "hidden",
//                 position: "relative",
//             }}
//         >
//             {/* Three.js canvas */}
//             <ThreeBackground />

//             {/* Radial gradient overlay */}
//             <div
//                 style={{
//                     position: "fixed",
//                     inset: 0,
//                     zIndex: 1,
//                     pointerEvents: "none",
//                     background:
//                         "radial-gradient(ellipse 80% 60% at 60% 20%, rgba(20,83,45,0.22) 0%, transparent 70%)",
//                 }}
//             />

//             {/* All page content above z-index 1 */}
//             <div style={{ position: "relative", zIndex: 2 }}>
//                 {/* ── NAVBAR ── */}
//                 <nav
//                     style={{
//                         position: "fixed",
//                         top: 0,
//                         left: 0,
//                         right: 0,
//                         zIndex: 50,
//                         transition: "all 0.3s ease",
//                         background: scrolled
//                             ? "rgba(0,0,0,0.88)"
//                             : "transparent",
//                         backdropFilter: scrolled ? "blur(14px)" : "none",
//                         borderBottom: scrolled
//                             ? "1px solid rgba(34,197,94,0.18)"
//                             : "1px solid transparent",
//                     }}
//                 >
//                     <div
//                         style={{
//                             maxWidth: 1200,
//                             margin: "0 auto",
//                             padding: "0 1.5rem",
//                             height: 64,
//                             display: "flex",
//                             alignItems: "center",
//                             justifyContent: "space-between",
//                         }}
//                     >
//                         {/* Logo */}
//                         <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
//                             <div
//                                 style={{
//                                     width: 34,
//                                     height: 34,
//                                     borderRadius: 8,
//                                     background: "linear-gradient(135deg,#22c55e,#16a34a)",
//                                     display: "flex",
//                                     alignItems: "center",
//                                     justifyContent: "center",
//                                 }}
//                             >
//                                 <Eye size={18} color="#000" strokeWidth={2.5} />
//                             </div>
//                             <span style={{ fontSize: 20, fontWeight: 700, letterSpacing: "-0.5px" }}>
//                                 <span style={{ color: "#4ade80" }}>Opti</span>
//                                 <span style={{ color: "#fff" }}>Assist</span>
//                             </span>
//                         </div>

//                         {/* Nav links */}
//                         <div
//                             style={{
//                                 display: "flex",
//                                 alignItems: "center",
//                                 gap: 32,
//                             }}
//                             className="nav-links"
//                         >
//                             {["Home", "App", "Features", "Docs"].map((link) => (
//                                 <a
//                                     key={link}
//                                     href="#"
//                                     style={{
//                                         fontSize: 14,
//                                         color: "rgba(255,255,255,0.5)",
//                                         textDecoration: "none",
//                                         transition: "color 0.2s",
//                                         fontWeight: 400,
//                                     }}
//                                     onMouseEnter={(e) => (e.target.style.color = "#4ade80")}
//                                     onMouseLeave={(e) => (e.target.style.color = "rgba(255,255,255,0.5)")}
//                                 >
//                                     {link}
//                                 </a>
//                             ))}
//                         </div>

//                         {/* Login button */}
//                         <button
//                             onClick={handleLogin}
//                             style={{
//                                 display: "flex",
//                                 alignItems: "center",
//                                 gap: 8,
//                                 padding: "8px 20px",
//                                 borderRadius: 8,
//                                 border: "1px solid rgba(34,197,94,0.5)",
//                                 background: "transparent",
//                                 color: "#4ade80",
//                                 fontSize: 14,
//                                 fontWeight: 500,
//                                 cursor: "pointer",
//                                 transition: "all 0.2s",
//                             }}
//                             onMouseEnter={(e) => {
//                                 e.currentTarget.style.background = "#22c55e";
//                                 e.currentTarget.style.color = "#000";
//                                 e.currentTarget.style.borderColor = "#22c55e";
//                             }}
//                             onMouseLeave={(e) => {
//                                 e.currentTarget.style.background = "transparent";
//                                 e.currentTarget.style.color = "#4ade80";
//                                 e.currentTarget.style.borderColor = "rgba(34,197,94,0.5)";
//                             }}
//                         >
//                             <LogIn size={15} />
//                             Login
//                         </button>
//                     </div>
//                 </nav>

//                 {/* ── HERO ── */}
//                 <section
//                     style={{
//                         minHeight: "100vh",
//                         display: "flex",
//                         alignItems: "center",
//                         paddingTop: 80,
//                     }}
//                 >
//                     <div
//                         style={{
//                             maxWidth: 1200,
//                             margin: "0 auto",
//                             padding: "0 1.5rem",
//                             width: "100%",
//                         }}
//                     >
//                         <div
//                             style={{
//                                 display: "grid",
//                                 gridTemplateColumns: "1fr 1fr",
//                                 gap: 64,
//                                 alignItems: "center",
//                             }}
//                         >
//                             {/* LEFT */}
//                             <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
//                                 {/* Badge */}
//                                 <div
//                                     style={{
//                                         display: "inline-flex",
//                                         alignItems: "center",
//                                         gap: 8,
//                                         padding: "6px 14px",
//                                         borderRadius: 999,
//                                         border: "1px solid rgba(34,197,94,0.35)",
//                                         background: "rgba(34,197,94,0.08)",
//                                         width: "fit-content",
//                                     }}
//                                 >
//                                     <Activity size={13} color="#4ade80" />
//                                     <span
//                                         style={{
//                                             color: "#4ade80",
//                                             fontSize: 11,
//                                             fontWeight: 600,
//                                             letterSpacing: "0.12em",
//                                             textTransform: "uppercase",
//                                         }}
//                                     >
//                                         AI-Ophthalmology System
//                                     </span>
//                                 </div>

//                                 {/* Headline */}
//                                 <div>
//                                     <h1
//                                         style={{
//                                             fontSize: 68,
//                                             fontWeight: 800,
//                                             color: "#22c55e",
//                                             letterSpacing: "-2px",
//                                             lineHeight: 1,
//                                             margin: 0,
//                                         }}
//                                     >
//                                         OptiAssist
//                                     </h1>
//                                     <h2
//                                         style={{
//                                             fontSize: 38,
//                                             fontWeight: 700,
//                                             lineHeight: 1.2,
//                                             margin: "16px 0 0",
//                                             letterSpacing: "-1px",
//                                         }}
//                                     >
//                                         See smarter.
//                                         <br />
//                                         <span style={{ color: "rgba(255,255,255,0.45)" }}>
//                                             Diagnose faster.
//                                         </span>
//                                     </h2>
//                                 </div>

//                                 <p
//                                     style={{
//                                         color: "rgba(255,255,255,0.4)",
//                                         fontSize: 16,
//                                         lineHeight: 1.7,
//                                         maxWidth: 420,
//                                         margin: 0,
//                                     }}
//                                 >
//                                     AI-powered fundus analysis that detects diabetic retinopathy,
//                                     explains its reasoning in real-time, and enables doctors to
//                                     interact naturally with clinical findings.
//                                 </p>

//                                 {/* CTA row */}
//                                 <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
//                                     <button
//                                         onClick={handleLogin}
//                                         style={{
//                                             display: "flex",
//                                             alignItems: "center",
//                                             gap: 8,
//                                             padding: "12px 28px",
//                                             background: "#22c55e",
//                                             color: "#000",
//                                             borderRadius: 10,
//                                             border: "none",
//                                             fontSize: 15,
//                                             fontWeight: 700,
//                                             cursor: "pointer",
//                                             transition: "all 0.2s",
//                                         }}
//                                         onMouseEnter={(e) => {
//                                             e.currentTarget.style.background = "#16a34a";
//                                             e.currentTarget.style.transform = "translateY(-1px)";
//                                         }}
//                                         onMouseLeave={(e) => {
//                                             e.currentTarget.style.background = "#22c55e";
//                                             e.currentTarget.style.transform = "translateY(0)";
//                                         }}
//                                     >
//                                         Get Started Free
//                                         <ArrowRight size={16} />
//                                     </button>
//                                     <a
//                                         href="#features"
//                                         style={{
//                                             display: "flex",
//                                             alignItems: "center",
//                                             gap: 6,
//                                             color: "rgba(255,255,255,0.45)",
//                                             fontSize: 14,
//                                             textDecoration: "none",
//                                             transition: "color 0.2s",
//                                         }}
//                                         onMouseEnter={(e) => (e.currentTarget.style.color = "#4ade80")}
//                                         onMouseLeave={(e) => (e.currentTarget.style.color = "rgba(255,255,255,0.45)")}
//                                     >
//                                         Learn more <ChevronRight size={14} />
//                                     </a>
//                                 </div>

//                                 {/* Stats row */}
//                                 <div style={{ display: "flex", gap: 24, marginTop: 8 }}>
//                                     {stats.map(({ value, label, Icon }) => (
//                                         <div
//                                             key={label}
//                                             style={{ display: "flex", flexDirection: "column", gap: 4 }}
//                                         >
//                                             <div
//                                                 style={{
//                                                     display: "flex",
//                                                     alignItems: "center",
//                                                     gap: 6,
//                                                 }}
//                                             >
//                                                 <Icon size={13} color="#22c55e" />
//                                                 <span
//                                                     style={{
//                                                         fontSize: 18,
//                                                         fontWeight: 700,
//                                                         color: "#fff",
//                                                         letterSpacing: "-0.5px",
//                                                     }}
//                                                 >
//                                                     {value}
//                                                 </span>
//                                             </div>
//                                             <span
//                                                 style={{
//                                                     fontSize: 11,
//                                                     color: "rgba(255,255,255,0.35)",
//                                                     textTransform: "uppercase",
//                                                     letterSpacing: "0.08em",
//                                                 }}
//                                             >
//                                                 {label}
//                                             </span>
//                                         </div>
//                                     ))}
//                                 </div>
//                             </div>

//                             {/* RIGHT */}
//                             <div style={{ position: "relative" }}>
//                                 {/* Glow behind panel */}
//                                 <div
//                                     style={{
//                                         position: "absolute",
//                                         inset: -40,
//                                         background:
//                                             "radial-gradient(ellipse at center, rgba(34,197,94,0.1) 0%, transparent 65%)",
//                                         pointerEvents: "none",
//                                     }}
//                                 />

//                                 <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
//                                     {/* Terminal-style panel */}
//                                     <div
//                                         style={{
//                                             borderRadius: 16,
//                                             border: "1px solid rgba(255,255,255,0.12)",
//                                             background: "rgba(0,0,0,0.7)",
//                                             backdropFilter: "blur(16px)",
//                                             padding: 24,
//                                             boxShadow: "0 0 60px rgba(34,197,94,0.06)",
//                                         }}
//                                     >
//                                         {/* Window dots */}
//                                         <div style={{ display: "flex", gap: 7, marginBottom: 20 }}>
//                                             {["#ef4444", "#eab308", "#22c55e"].map((c) => (
//                                                 <div
//                                                     key={c}
//                                                     style={{
//                                                         width: 11,
//                                                         height: 11,
//                                                         borderRadius: "50%",
//                                                         background: c,
//                                                     }}
//                                                 />
//                                             ))}
//                                             <span
//                                                 style={{
//                                                     fontSize: 11,
//                                                     color: "rgba(255,255,255,0.2)",
//                                                     marginLeft: 8,
//                                                     fontFamily: "monospace",
//                                                 }}
//                                             >
//                                                 optiassist — clinical analysis
//                                             </span>
//                                         </div>

//                                         {/* Typewriter lines */}
//                                         <div
//                                             style={{
//                                                 display: "flex",
//                                                 flexDirection: "column",
//                                                 gap: 18,
//                                                 minHeight: 120,
//                                             }}
//                                         >
//                                             {displayedText.map((text, i) => (
//                                                 <div
//                                                     key={i}
//                                                     style={{
//                                                         display: "flex",
//                                                         gap: 12,
//                                                         opacity: i <= currentPoint ? 1 : 0,
//                                                         transform:
//                                                             i <= currentPoint
//                                                                 ? "translateY(0)"
//                                                                 : "translateY(8px)",
//                                                         transition: "all 0.4s ease",
//                                                     }}
//                                                 >
//                                                     <div
//                                                         style={{
//                                                             marginTop: 6,
//                                                             width: 7,
//                                                             height: 7,
//                                                             borderRadius: "50%",
//                                                             background: "#22c55e",
//                                                             flexShrink: 0,
//                                                             boxShadow: "0 0 6px #22c55e",
//                                                         }}
//                                                     />
//                                                     <p
//                                                         style={{
//                                                             color: "rgba(255,255,255,0.7)",
//                                                             fontSize: 13.5,
//                                                             lineHeight: 1.6,
//                                                             margin: 0,
//                                                         }}
//                                                     >
//                                                         <span
//                                                             style={{
//                                                                 color: "#fff",
//                                                                 fontWeight: 600,
//                                                             }}
//                                                         >
//                                                             {text.split(" — ")[0]}
//                                                         </span>
//                                                         {text.includes(" — ") && (
//                                                             <span style={{ color: "rgba(255,255,255,0.45)" }}>
//                                                                 {" — "}
//                                                                 {text.split(" — ")[1]}
//                                                             </span>
//                                                         )}
//                                                         {currentPoint === i && (
//                                                             <span
//                                                                 style={{
//                                                                     color: "#22c55e",
//                                                                     marginLeft: 2,
//                                                                     animation: "blink 1s step-end infinite",
//                                                                 }}
//                                                             >
//                                                                 ▍
//                                                             </span>
//                                                         )}
//                                                     </p>
//                                                 </div>
//                                             ))}
//                                         </div>
//                                     </div>

//                                     {/* 3 module cards */}
//                                     <div
//                                         style={{
//                                             display: "grid",
//                                             gridTemplateColumns: "repeat(3, 1fr)",
//                                             gap: 12,
//                                         }}
//                                     >
//                                         {[
//                                             { title: "PaliGemma", subtitle: "Vision encoder", Icon: Cpu },
//                                             { title: "FunctionGemma", subtitle: "Agentic routing", Icon: Layers },
//                                             { title: "MedGemma", subtitle: "Clinical reasoning", Icon: Stethoscope },
//                                         ].map(({ title, subtitle, Icon: CardIcon }, i) => (
//                                             <div
//                                                 key={i}
//                                                 style={{
//                                                     borderRadius: 14,
//                                                     border: "1px solid rgba(34,197,94,0.18)",
//                                                     background: "rgba(3,19,10,0.7)",
//                                                     padding: "18px 14px",
//                                                     textAlign: "center",
//                                                     cursor: "default",
//                                                     transition: "all 0.25s ease",
//                                                 }}
//                                                 onMouseEnter={(e) => {
//                                                     e.currentTarget.style.borderColor = "rgba(34,197,94,0.5)";
//                                                     e.currentTarget.style.transform = "translateY(-3px)";
//                                                     e.currentTarget.style.boxShadow =
//                                                         "0 0 30px rgba(34,197,94,0.15)";
//                                                 }}
//                                                 onMouseLeave={(e) => {
//                                                     e.currentTarget.style.borderColor = "rgba(34,197,94,0.18)";
//                                                     e.currentTarget.style.transform = "translateY(0)";
//                                                     e.currentTarget.style.boxShadow = "none";
//                                                 }}
//                                             >
//                                                 <div
//                                                     style={{
//                                                         width: 44,
//                                                         height: 44,
//                                                         margin: "0 auto 12px",
//                                                         borderRadius: 10,
//                                                         background: "rgba(34,197,94,0.1)",
//                                                         border: "1px solid rgba(34,197,94,0.2)",
//                                                         display: "flex",
//                                                         alignItems: "center",
//                                                         justifyContent: "center",
//                                                     }}
//                                                 >
//                                                     <CardIcon size={18} color="#22c55e" />
//                                                 </div>
//                                                 <p
//                                                     style={{
//                                                         color: "#4ade80",
//                                                         fontWeight: 600,
//                                                         fontSize: 12,
//                                                         margin: 0,
//                                                     }}
//                                                 >
//                                                     {title}
//                                                 </p>
//                                                 <p
//                                                     style={{
//                                                         color: "rgba(255,255,255,0.3)",
//                                                         fontSize: 11,
//                                                         margin: "4px 0 0",
//                                                     }}
//                                                 >
//                                                     {subtitle}
//                                                 </p>
//                                             </div>
//                                         ))}
//                                     </div>
//                                 </div>
//                             </div>
//                         </div>
//                     </div>
//                 </section>

//                 {/* ── FEATURES ── */}
//                 <section
//                     id="features"
//                     style={{ padding: "96px 0", position: "relative" }}
//                 >
//                     {/* Subtle divider glow */}
//                     <div
//                         style={{
//                             position: "absolute",
//                             top: 0,
//                             left: "50%",
//                             transform: "translateX(-50%)",
//                             width: 600,
//                             height: 1,
//                             background:
//                                 "linear-gradient(90deg, transparent, rgba(34,197,94,0.3), transparent)",
//                         }}
//                     />

//                     <div
//                         style={{ maxWidth: 1200, margin: "0 auto", padding: "0 1.5rem" }}
//                     >
//                         {/* Section header */}
//                         <div
//                             style={{
//                                 textAlign: "center",
//                                 marginBottom: 56,
//                                 display: "flex",
//                                 flexDirection: "column",
//                                 gap: 12,
//                             }}
//                         >
//                             <p
//                                 style={{
//                                     color: "#22c55e",
//                                     fontSize: 11,
//                                     fontWeight: 700,
//                                     letterSpacing: "0.18em",
//                                     textTransform: "uppercase",
//                                     margin: 0,
//                                 }}
//                             >
//                                 Modules
//                             </p>
//                             <h2
//                                 style={{
//                                     fontSize: 40,
//                                     fontWeight: 800,
//                                     margin: 0,
//                                     letterSpacing: "-1px",
//                                 }}
//                             >
//                                 Three Powerful Engines
//                             </h2>
//                             <p
//                                 style={{
//                                     color: "rgba(255,255,255,0.4)",
//                                     maxWidth: 480,
//                                     margin: "0 auto",
//                                     fontSize: 15,
//                                     lineHeight: 1.7,
//                                 }}
//                             >
//                                 Each module handles a distinct layer — from vision to agentic
//                                 routing to clinical intelligence.
//                             </p>
//                         </div>

//                         {/* Feature cards */}
//                         <div
//                             style={{
//                                 display: "grid",
//                                 gridTemplateColumns: "repeat(3, 1fr)",
//                                 gap: 20,
//                             }}
//                         >
//                             {features.map(({ id, name, Icon, description, tag, accentColor }) => (
//                                 <div
//                                     key={id}
//                                     style={{
//                                         border: "1px solid rgba(255,255,255,0.08)",
//                                         borderRadius: 16,
//                                         padding: "28px 24px",
//                                         background: "rgba(255,255,255,0.02)",
//                                         backdropFilter: "blur(8px)",
//                                         transition: "all 0.3s ease",
//                                         cursor: "default",
//                                     }}
//                                     onMouseEnter={(e) => {
//                                         e.currentTarget.style.borderColor = `${accentColor}40`;
//                                         e.currentTarget.style.background = "rgba(34,197,94,0.04)";
//                                         e.currentTarget.style.transform = "translateY(-4px)";
//                                         e.currentTarget.style.boxShadow = `0 20px 60px rgba(34,197,94,0.08)`;
//                                     }}
//                                     onMouseLeave={(e) => {
//                                         e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)";
//                                         e.currentTarget.style.background = "rgba(255,255,255,0.02)";
//                                         e.currentTarget.style.transform = "translateY(0)";
//                                         e.currentTarget.style.boxShadow = "none";
//                                     }}
//                                 >
//                                     {/* Tag */}
//                                     <span
//                                         style={{
//                                             fontSize: 10,
//                                             fontWeight: 600,
//                                             letterSpacing: "0.1em",
//                                             textTransform: "uppercase",
//                                             color: accentColor,
//                                             background: `${accentColor}18`,
//                                             border: `1px solid ${accentColor}30`,
//                                             padding: "3px 10px",
//                                             borderRadius: 99,
//                                         }}
//                                     >
//                                         {tag}
//                                     </span>

//                                     {/* Icon */}
//                                     <div
//                                         style={{
//                                             marginTop: 20,
//                                             width: 48,
//                                             height: 48,
//                                             borderRadius: 12,
//                                             background: `${accentColor}15`,
//                                             border: `1px solid ${accentColor}25`,
//                                             display: "flex",
//                                             alignItems: "center",
//                                             justifyContent: "center",
//                                         }}
//                                     >
//                                         <Icon size={22} color={accentColor} />
//                                     </div>

//                                     <h3
//                                         style={{
//                                             fontSize: 18,
//                                             fontWeight: 700,
//                                             marginTop: 16,
//                                             marginBottom: 10,
//                                             letterSpacing: "-0.3px",
//                                         }}
//                                     >
//                                         {name}
//                                     </h3>
//                                     <p
//                                         style={{
//                                             color: "rgba(255,255,255,0.4)",
//                                             fontSize: 14,
//                                             lineHeight: 1.7,
//                                             margin: 0,
//                                         }}
//                                     >
//                                         {description}
//                                     </p>

//                                     {/* Arrow link */}
//                                     <div
//                                         style={{
//                                             marginTop: 20,
//                                             display: "flex",
//                                             alignItems: "center",
//                                             gap: 6,
//                                             color: accentColor,
//                                             fontSize: 13,
//                                             fontWeight: 500,
//                                         }}
//                                     >
//                                         Learn more <ArrowRight size={13} />
//                                     </div>
//                                 </div>
//                             ))}
//                         </div>
//                     </div>
//                 </section>

//                 {/* ── FOOTER ── */}
//                 <footer
//                     style={{
//                         borderTop: "1px solid rgba(255,255,255,0.06)",
//                         padding: "28px 0",
//                         textAlign: "center",
//                         color: "rgba(255,255,255,0.25)",
//                         fontSize: 13,
//                     }}
//                 >
//                     © 2026 OptiAssist — All rights reserved
//                 </footer>
//             </div>

//             {/* Cursor blink keyframe */}
//             <style>{`
//                 @keyframes blink {
//                     0%, 100% { opacity: 1; }
//                     50% { opacity: 0; }
//                 }
//                 @media (max-width: 768px) {
//                     .nav-links { display: none !important; }
//                 }
//             `}</style>
//         </div>
//     );
// }



//////////////////////////////////////////////////////////////////////

// trying to add more of one new page in it 



////////////////////////////////////////////////////////////////////////




import { useState, useEffect, useRef } from "react";
import { supabase } from "./supabaseClient";
import {
    Zap,
    Wrench,
    Stethoscope,
    ArrowRight,
    Eye,
    Brain,
    Activity,
    ChevronRight,
    Cpu,
    Shield,
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
    const mountRef = useRef(null);

    useEffect(() => {
        let animId;
        let THREE;

        const init = async () => {
            THREE = await import("https://esm.sh/three@0.160.0");

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
            const pointData = [];
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

                    if (pos[i * 3 + 0] >  BOUND_X) pos[i * 3 + 0] = -BOUND_X;
                    if (pos[i * 3 + 0] < -BOUND_X) pos[i * 3 + 0] =  BOUND_X;
                    if (pos[i * 3 + 1] >  BOUND_Y) pos[i * 3 + 1] = -BOUND_Y;
                    if (pos[i * 3 + 1] < -BOUND_Y) pos[i * 3 + 1] =  BOUND_Y;
                    if (pos[i * 3 + 2] >  BOUND_Z) pos[i * 3 + 2] = -BOUND_Z;
                    if (pos[i * 3 + 2] < -BOUND_Z) pos[i * 3 + 2] =  BOUND_Z;
                }

                geo.attributes.position.needsUpdate = true;
                renderer.render(scene, camera);
            };
            animate();

            return () => {
                window.removeEventListener("resize", onResize);
                cancelAnimationFrame(animId);
                renderer.dispose();
                if (mount.contains(renderer.domElement)) {
                    mount.removeChild(renderer.domElement);
                }
            };
        };

        let cleanup;
        init().then((fn) => { cleanup = fn; });
        return () => { cleanup && cleanup(); };
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
        accentColor: "#4ade80",
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

const heroStats = [
    { value: "< 2s", label: "Inference time", Icon: Activity },
    { value: "5-class", label: "DR grading", Icon: Eye },
    { value: "XAI", label: "Explainability", Icon: Brain },
    { value: "HIPAA", label: "Compliant", Icon: Shield },
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
    const [activeFeature, setActiveFeature] = useState(null);
    const [displayedText, setDisplayedText] = useState(["", "", ""]);
    const [currentPoint, setCurrentPoint] = useState(0);

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

    const handleLogin = async () => {
        await supabase.auth.signOut({ scope: "global" });
        Object.keys(localStorage).forEach((key) => {
            if (key.startsWith("sb-")) localStorage.removeItem(key);
        });
        await supabase.auth.signInWithOAuth({
            provider: "google",
            options: {
                redirectTo: "http://localhost:5173/auth/callback",
                queryParams: { prompt: "select_account" },
            },
        });
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
                    borderBottom: scrolled ? "1px solid rgba(34,197,94,0.18)" : "1px solid transparent",
                }}>
                    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 1.5rem", height: 64, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                            <div style={{ width: 34, height: 34, borderRadius: 8, background: "linear-gradient(135deg,#22c55e,#16a34a)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                                <Eye size={18} color="#000" strokeWidth={2.5} />
                            </div>
                            <span style={{ fontSize: 20, fontWeight: 700, letterSpacing: "-0.5px" }}>
                                <span style={{ color: "#4ade80" }}>Opti</span>
                                <span style={{ color: "#fff" }}>Assist</span>
                            </span>
                        </div>

                        <div style={{ display: "flex", alignItems: "center", gap: 32 }} className="nav-links">
                            {["Home", "App", "Features", "Docs"].map((link) => (
                                <a key={link} href="#"
                                    style={{ fontSize: 14, color: "rgba(255,255,255,0.5)", textDecoration: "none", transition: "color 0.2s" }}
                                    onMouseEnter={(e) => (e.target.style.color = "#4ade80")}
                                    onMouseLeave={(e) => (e.target.style.color = "rgba(255,255,255,0.5)")}
                                >{link}</a>
                            ))}
                        </div>

                        <button onClick={handleLogin}
                            style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 20px", borderRadius: 8, border: "1px solid rgba(34,197,94,0.5)", background: "transparent", color: "#4ade80", fontSize: 14, fontWeight: 500, cursor: "pointer", transition: "all 0.2s" }}
                            onMouseEnter={(e) => { e.currentTarget.style.background = "#22c55e"; e.currentTarget.style.color = "#000"; }}
                            onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "#4ade80"; }}
                        >
                            <LogIn size={15} /> Login
                        </button>
                    </div>
                </nav>

                {/* ═══════════ SECTION 1 — FULL-SCREEN HERO ═══════════ */}
                <section style={{ position: "relative", minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", paddingTop: 80, overflow: "hidden" }}>
                    <div style={{ position: "absolute", inset: 0, pointerEvents: "none", background: "linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)" }} />

                    <div style={{ position: "relative", zIndex: 10, textAlign: "center", maxWidth: 860, margin: "0 auto", padding: "0 1.5rem" }}>
                        {/* Badge */}
                        <div style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "6px 16px", borderRadius: 999, border: "1px solid rgba(34,197,94,0.4)", marginBottom: 36 }}>
                            <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#4ade80" }} />
                            <span style={{ color: "#4ade80", fontSize: 11, fontWeight: 600, letterSpacing: "0.14em", textTransform: "uppercase" }}>
                                AI-Powered Ophthalmology Optimization
                            </span>
                        </div>

                        {/* Headline */}
                        <h1 style={{ fontSize: "clamp(44px,8vw,80px)", fontWeight: 600, lineHeight: 1.1, letterSpacing: "-2px", margin: "0 0 28px" }}>
                            <span style={{ color: "#fff" }}>Train Smarter. </span><br/>
                            <h1 style={{ fontSize: "clamp(44px,8vw,80px)", fontWeight: 600, lineHeight: 1.1, letterSpacing: "-2px", margin: "0 0 28px" }}>
                                <span style={{ color: "#22c55e" }}>Diagnose </span>
                                <span style={{ color: "#22c55e" }}>Faster.</span><br />
                            </h1>
                            <span style={{ color: "#fff" }}>Perform at Your Peak</span>
                        </h1>

                        <p style={{ color: "rgba(255,255,255,0.45)", fontSize: 17, lineHeight: 1.75, maxWidth: 580, margin: "0 auto 40px" }}>
                            OptiAssist uses deep learning to analyze fundus images, identify diabetic
                            retinopathy, and deliver explainable clinical insights — adapting in real
                            time to keep clinicians ahead of every diagnosis.
                        </p>

                        {/* CTAs */}
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 14, justifyContent: "center", marginBottom: 48 }}>
                            
                            
                        </div>

                        {/* Stats row */}
                        <div style={{ display: "flex", justifyContent: "center", gap: 32, flexWrap: "wrap" }}>
                            {heroStats.map(({ value, label, Icon }) => (
                                <div key={label} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                        <Icon size={13} color="#22c55e" />
                                        <span style={{ fontSize: 20, fontWeight: 700, letterSpacing: "-0.5px" }}>{value}</span>
                                    </div>
                                    <span style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", textTransform: "uppercase", letterSpacing: "0.1em" }}>{label}</span>
                                </div>
                            ))}
                        </div>
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
                    <div style={{ position: "absolute", inset: 0, pointerEvents: "none", backgroundImage: "linear-gradient(rgba(34,197,94,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(34,197,94,0.04) 1px, transparent 1px)", backgroundSize: "48px 48px" }} />
                    {/* Divider */}
                    <div style={{ position: "absolute", top: 0, left: "50%", transform: "translateX(-50%)", width: 600, height: 1, background: "linear-gradient(90deg, transparent, rgba(34,197,94,0.3), transparent)" }} />
                    <div style={{ position: "absolute", top: 0, left: "50%", transform: "translateX(-50%)", width: 700, height: 400, pointerEvents: "none", background: "transparent" }} />

                    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 1.5rem", position: "relative" }}>
                        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 64, alignItems: "center" }}>

                            {/* LEFT */}
                            <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
                                <div style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "6px 14px", borderRadius: 999, border: "1px solid rgba(34,197,94,0.35)", background: "rgba(34,197,94,0.08)", width: "fit-content" }}>
                                    <Activity size={12} color="#4ade80" />
                                    <span style={{ color: "#4ade80", fontSize: 11, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase" }}>How It Works</span>
                                </div>

                                <div>
                                    <h2 style={{ fontSize: 64, fontWeight: 800, color: "#22c55e", letterSpacing: "-2px", lineHeight: 1, margin: 0 }}>OptiAssist</h2>
                                    <div style={{ marginTop: 8, height: 3, width: 120, background: "linear-gradient(to right, #22c55e, transparent)", borderRadius: 99 }} />
                                </div>

                                <p style={{ fontSize: 24, color: "rgba(255,255,255,0.7)", fontWeight: 300, lineHeight: 1.5, margin: 0, maxWidth: 440 }}>
                                    Smarter diagnostics —{" "}
                                    <span style={{ color: "#4ade80", fontWeight: 500 }}>better decisions</span>, faster results.
                                </p>

                                <p style={{ color: "rgba(255,255,255,0.38)", fontSize: 15, lineHeight: 1.7, maxWidth: 420, margin: 0 }}>
                                    OptiAssist brings together powerful AI modules to help clinicians
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
                                <div style={{ position: "absolute", inset: -40, background: "transparent", pointerEvents: "none" }} />
                                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

                                    {/* Terminal */}
                                    <div style={{ borderRadius: 16, border: "1px solid rgba(255,255,255,0.12)", background: "rgba(0,0,0,0.7)", backdropFilter: "blur(16px)", padding: 24 }}>
                                        <div style={{ display: "flex", gap: 7, marginBottom: 20, alignItems: "center" }}>
                                            {["#ef4444", "#eab308", "#22c55e"].map((c) => (
                                                <div key={c} style={{ width: 11, height: 11, borderRadius: "50%", background: c }} />
                                            ))}
                                            <span style={{ fontSize: 11, color: "rgba(255,255,255,0.2)", marginLeft: 8, fontFamily: "monospace" }}>optiassist — clinical analysis</span>
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
                                                style={{ borderRadius: 14, border: "1px solid rgba(34,197,94,0.18)", background: "rgba(3,19,10,0.7)", padding: "18px 14px", textAlign: "center", cursor: "default", transition: "all 0.25s ease" }}
                                                onMouseEnter={(e) => { e.currentTarget.style.borderColor = "rgba(34,197,94,0.5)"; e.currentTarget.style.transform = "translateY(-3px)"; e.currentTarget.style.boxShadow = "0 0 30px rgba(34,197,94,0.15)"; }}
                                                onMouseLeave={(e) => { e.currentTarget.style.borderColor = "rgba(34,197,94,0.18)"; e.currentTarget.style.transform = "none"; e.currentTarget.style.boxShadow = "none"; }}
                                            >
                                                <div style={{ width: 44, height: 44, margin: "0 auto 12px", borderRadius: 10, background: "rgba(34,197,94,0.1)", border: "1px solid rgba(34,197,94,0.2)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                                                    <CardIcon size={18} color="#22c55e" />
                                                </div>
                                                <p style={{ color: "#4ade80", fontWeight: 600, fontSize: 12, margin: 0 }}>{title}</p>
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
                    <div style={{ position: "absolute", top: 0, left: "50%", transform: "translateX(-50%)", width: 600, height: 1, background: "linear-gradient(90deg, transparent, rgba(34,197,94,0.3), transparent)" }} />

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
                                        background: activeFeature === id ? "rgba(34,197,94,0.04)" : "rgba(255,255,255,0.02)",
                                        backdropFilter: "blur(8px)", transition: "all 0.3s ease", cursor: "default",
                                        transform: activeFeature === id ? "translateY(-4px)" : "none",
                                        boxShadow: activeFeature === id ? "0 20px 60px rgba(34,197,94,0.08)" : "none",
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

                

                {/* ═══════════ FOOTER ═══════════ */}
                <footer style={{ borderTop: "1px solid rgba(255,255,255,0.06)", padding: "32px 0" }}>
                    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 1.5rem", display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: 16 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <div style={{ width: 26, height: 26, borderRadius: 6, background: "linear-gradient(135deg,#22c55e,#16a34a)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                                <Eye size={13} color="#000" strokeWidth={2.5} />
                            </div>
                            <span style={{ fontSize: 14, fontWeight: 700 }}>
                                <span style={{ color: "#4ade80" }}>Opti</span>
                                <span style={{ color: "#fff" }}>Assist</span>
                            </span>
                        </div>

                        <p style={{ color: "rgba(255,255,255,0.2)", fontSize: 12, margin: 0 }}>© 2026 OptiAssist. All rights reserved.</p>

                        <div style={{ display: "flex", gap: 24 }}>
                            {["Privacy", "Terms", "Contact"].map((l) => (
                                <a key={l} href="#"
                                    style={{ fontSize: 12, color: "rgba(255,255,255,0.25)", textDecoration: "none", transition: "color 0.2s" }}
                                    onMouseEnter={(e) => (e.target.style.color = "#4ade80")}
                                    onMouseLeave={(e) => (e.target.style.color = "rgba(255,255,255,0.25)")}
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
