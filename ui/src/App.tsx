/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState } from 'react';

// Photograph asset URLs from Higgsfield CDN
const ASSETS = {
  dunes: 'https://d2ol7oe51mr4n9.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/0e201fcd-ffc0-46a0-bd06-faa07bf0e84b.webp',
  foundry: 'https://d2ol7oe51mr4n9.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/e80ba08d-d7b6-4b27-b0cd-8d0ad09c788c.webp',
  mineral: 'https://d2ol7oe51mr4n9.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/2bebad68-8177-446a-9550-d64074166364.webp',
  mesa: 'https://d2ol7oe51mr4n9.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/62b0d7c1-3eb7-44bd-9b19-57b975fb250f.webp',
  fire: 'https://d2ol7oe51mr4n9.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/156ff74d-ee37-4443-8f3a-11ae2095ed4c.webp',
  saltflat: 'https://d2ol7oe51mr4n9.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/7b6ad638-89ec-4dec-be74-d36fbff3ea53.webp',
};

// Dissolve palette: hot orange to white-hot
const PALETTE = [
  '#FF4A12',
  '#FF5A1F',
  '#FF6B2C',
  '#FF8A3D',
  '#FFB06A',
  '#FFD9B5',
  '#FFF3E6',
];

// 32-bit integer hash function
function hash3(a: number, b: number, c: number): number {
  let h = (Math.imul(a, 374761393) + Math.imul(b, 668265263) + Math.imul(c, 2147483647)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177) | 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function mod(n: number, m: number): number {
  return ((n % m) + m) % m;
}

interface Spark {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  color: string;
  maxLife: number;
  age: number;
}

interface CardMeta {
  id: string;
  original: number;
  distilled: number;
  saved: number;
  percent: string;
  url: string;
  codeHtml: string;
  codeMeta: string;
}

const CARDS_DATA: CardMeta[] = [
  {
    id: 'home',
    original: 70411,
    distilled: 6140,
    saved: 64271,
    percent: '91% less',
    url: 'sentryicu.health/',
    codeHtml: `<h1>SentryICU — Intelligent ICU Coordination</h1>
<p>
  Real-time hospital capacity telemetry,
  inter-facility transfer routing, and
  predictive critical care balancing.
</p>
<h2>Active Modules</h2>
<ul>
  <li>Surge Predictor: 98.4% 24h accuracy</li>
  <li>Transfer Optimization: 14 hospitals</li>
  <li>Active Beds Monitored: 342</li>
  <li>Deployed: Region IV Network</li>
</ul>
<h2>Capabilities</h2>
<ul>
  <li>Automated SOFA/APACHE scoring</li>
  <li>Ventilator & ECMO priority routing</li>
  <li>Edge-verified clinical handoffs</li>
</ul>
<a href="#command-center">Open Command Center</a>`,
    codeMeta: `<!-- distilled: 6,140 tokens -->
<!-- original: 70,411 tokens -->
<!-- sentryicu.health/ -->`,
  },
  {
    id: 'price',
    original: 41208,
    distilled: 4960,
    saved: 36248,
    percent: '88% less',
    url: 'sentryicu.health/network',
    codeHtml: `<h1>SentryICU — Capacity Balancing</h1>
<p>
  Three operational tiers for regional
  health authorities and hospital systems.
</p>
<h2>Hospital Level — 14 Units</h2>
<ul>
  <li>Telemetry Sync: sub-second</li>
  <li>Deep Learning Modules: 5 active</li>
  <li>Support: 24/7 Operations</li>
</ul>
<h2>Regional Command — Network Wide</h2>
<ul>
  <li>Trauma Coordination: Active</li>
  <li>Bed Rebalancing: Real-time</li>
  <li>ISO 27001 & HIPAA Certified</li>
  <li>Support: Dedicated Clinical Engineer</li>
</ul>`,
    codeMeta: `<!-- distilled: 4,960 tokens -->
<!-- original: 41,208 tokens -->
<!-- sentryicu.health/network -->`,
  },
  {
    id: 'case',
    original: 98432,
    distilled: 6300,
    saved: 92132,
    percent: '94% less',
    url: 'sentryicu.health/cases/metro-surge',
    codeHtml: `<h1>Case Study: Metro Health Surge</h1>
<p>
  How Region IV balanced 180 critical
  admissions during winter respiratory peak.
</p>
<h2>The Challenge</h2>
<p>
  14 hospitals at 94% capacity,
  transfers delayed by up to 6 hours.
</p>
<h2>The Result</h2>
<ul>
  <li>34% faster inter-facility transfers</li>
  <li>3.4x faster bed allocation</li>
  <li>0 rejected critical transfers</li>
</ul>
<p>
  "We prevented systemic ICU gridlock."
  — Chief Medical Officer
</p>`,
    codeMeta: `<!-- distilled: 6,300 tokens -->
<!-- original: 98,432 tokens -->
<!-- sentryicu.health/cases/metro-surge -->`,
  },
  {
    id: 'vision',
    original: 55870,
    distilled: 8215,
    saved: 47655,
    percent: '85% less',
    url: 'sentryicu.health/vision',
    codeHtml: `<article>
  <h1>The SentryICU Architecture</h1>
  <p>
    Five deep-learning modules unified into
    a single edge-intelligent interface.
  </p>
  <p>
    We monitor where demand is highest
    and critical care capacity is scarce:
    fourteen hospitals, zero silos.
  </p>
  <h2>Core Principles</h2>
  <ol>
    <li>Continuous predictive triage</li>
    <li>Edge inference without clinical delay</li>
    <li>Transparent model interpretability</li>
  </ol>
</article>`,
    codeMeta: `<!-- distilled: 8,215 tokens -->
<!-- original: 55,870 tokens -->
<!-- sentryicu.health/vision -->`,
  },
  {
    id: 'alloy',
    original: 63904,
    distilled: 5512,
    saved: 58392,
    percent: '91% less',
    url: 'sentryicu.health/models/engine-v5',
    codeHtml: `<h1>Neural Module 5</h1>
<p>The Staffing Shift Equilibrium Engine.</p>
<h2>Specifications</h2>
<ul>
  <li>Inference Latency: 12 ms</li>
  <li>Acuity Indexing: SOFA & APACHE-IV</li>
  <li>Forecast Horizon: 48 Hours</li>
  <li>Telemetry Channels: 128 streams</li>
</ul>
<h2>Applications</h2>
<ul>
  <li>Trauma, ECMO, Neuro, Pediatric ICU</li>
</ul>
<a href="#command-center">View Telemetry</a>`,
    codeMeta: `<!-- distilled: 5,512 tokens -->
<!-- original: 63,904 tokens -->
<!-- sentryicu.health/models/engine-v5 -->`,
  },
  {
    id: 'journal',
    original: 47316,
    distilled: 7602,
    saved: 39714,
    percent: '84% less',
    url: 'sentryicu.health/transfers',
    codeHtml: `<h1>SentryICU Dispatch Log</h1>
<p>Regional Transfer Telemetry.</p>
<article>
  <h2>Aeromedical En-Route: Unit 4</h2>
  <time>2026-10-29 06:14</time>
</article>
<article>
  <h2>Metro Trauma Bed Reserved</h2>
  <time>2026-10-29 05:48</time>
</article>
<article>
  <h2>ECMO Load Rebalanced</h2>
  <time>2026-10-29 05:02</time>
</article>`,
    codeMeta: `<!-- distilled: 7,602 tokens -->
<!-- original: 47,316 tokens -->
<!-- sentryicu.health/transfers -->`,
  },
];

// Interactive mock data for Command Center, Network, and Transfers
interface Hospital {
  id: string;
  name: string;
  level: string;
  totalBeds: number;
  availableBeds: number;
  criticalVents: number;
  ecmoUnits: number;
  status: 'normal' | 'caution' | 'critical';
  distance: string;
}

const INITIAL_HOSPITALS: Hospital[] = [
  { id: 'h1', name: 'Metro University Medical Center', level: 'Level 1 Trauma', totalBeds: 64, availableBeds: 5, criticalVents: 6, ecmoUnits: 4, status: 'caution', distance: '4.2 mi' },
  { id: 'h2', name: 'Valley Health Memorial Hospital', level: 'Level 2 Trauma', totalBeds: 48, availableBeds: 12, criticalVents: 5, ecmoUnits: 1, status: 'normal', distance: '8.7 mi' },
  { id: 'h3', name: 'St. Jude Regional Critical Care', level: 'Comprehensive Cardiac', totalBeds: 56, availableBeds: 4, criticalVents: 2, ecmoUnits: 3, status: 'caution', distance: '11.5 mi' },
  { id: 'h4', name: 'Northwest Medical Center', level: 'Regional Acute Care', totalBeds: 42, availableBeds: 14, criticalVents: 7, ecmoUnits: 0, status: 'normal', distance: '16.0 mi' },
  { id: 'h5', name: 'Central Cardiothoracic Institute', level: 'Specialty ECMO Center', totalBeds: 38, availableBeds: 2, criticalVents: 1, ecmoUnits: 6, status: 'critical', distance: '13.1 mi' },
  { id: 'h6', name: 'Children\'s Regional Specialty ICU', level: 'Pediatric Level 1', totalBeds: 32, availableBeds: 7, criticalVents: 4, ecmoUnits: 2, status: 'normal', distance: '9.8 mi' },
];

interface TransferRecord {
  id: string;
  patientId: string;
  origin: string;
  target: string;
  acuity: string;
  transportType: string;
  eta: string;
  status: 'En-Route' | 'Triage Approved' | 'Bed Reserved' | 'Pending Model Review';
}

const INITIAL_TRANSFERS: TransferRecord[] = [
  { id: 'tr-1', patientId: 'PT-8492', origin: 'Valley Health', target: 'Metro University', acuity: 'SOFA: 11 (Severe ARDS)', transportType: 'Ground ALS #4', eta: '14 min', status: 'En-Route' },
  { id: 'tr-2', patientId: 'PT-7319', origin: 'Northwest Medical', target: 'Central Cardiothoracic', acuity: 'Cardiogenic Shock', transportType: 'Rotorcraft Lifeflight #2', eta: '8 min', status: 'En-Route' },
  { id: 'tr-3', patientId: 'PT-9021', origin: 'St. Jude Regional', target: 'Valley Health', acuity: 'Step-down Optimization', transportType: 'Mobile ICU #1', eta: '28 min', status: 'Triage Approved' },
  { id: 'tr-4', patientId: 'PT-6104', origin: 'Metro University', target: 'Northwest Medical', acuity: 'Ventilator Weaning', transportType: 'Critical Care Transport', eta: '45 min', status: 'Bed Reserved' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<'overview' | 'command' | 'network' | 'transfers'>('overview');
  const [hospitals, setHospitals] = useState<Hospital[]>(INITIAL_HOSPITALS);
  const [transfers, setTransfers] = useState<TransferRecord[]>(INITIAL_TRANSFERS);
  const [selectedHospital, setSelectedHospital] = useState<Hospital | null>(null);
  const [showTransferModal, setShowTransferModal] = useState(false);
  const [newOrigin, setNewOrigin] = useState(INITIAL_HOSPITALS[0].name);
  const [newTarget, setNewTarget] = useState(INITIAL_HOSPITALS[1].name);
  const [newAcuity, setNewAcuity] = useState('SOFA: 9 (Acute Respiratory)');
  const [newTransport, setNewTransport] = useState('Critical Care Mobile ICU');
  const [filterLevel, setFilterLevel] = useState<string>('all');
  const [simulationRunning, setSimulationRunning] = useState(false);

  // Real-time Neural Model & Gemini RAG states
  const [activeModelModal, setActiveModelModal] = useState<'risk' | 'rag' | 'moduleDetail' | null>(null);
  const [selectedModuleInfo, setSelectedModuleInfo] = useState<{
    id: number;
    title: string;
    architecture: string;
    description: string;
    metricLabel: string;
    metricVal: string;
    details: string[];
  } | null>(null);

  const [riskLoading, setRiskLoading] = useState(false);
  const [riskResult, setRiskResult] = useState<{
    bimodal_cnn_risk: number;
    trimodal_fusion_risk: number;
    vae_anomaly_score: number;
    is_physiological_anomaly: boolean;
    risk_tier: string;
  } | null>(null);

  const [ragQuery, setRagQuery] = useState('');
  const [ragLoading, setRagLoading] = useState(false);
  const [ragHistory, setRagHistory] = useState<Array<{
    question: string;
    answer: string;
    sources: string[];
    model: string;
  }>>([
    {
      question: "What is the anomaly threshold for PatientVAE?",
      answer: "PatientVAE Anomaly Gating: Calibrated at 0.452 MSE (95th percentile) on healthy ICU cohort records (Hospital A). Any patient stay exceeding 0.452 indicates significant physiological deviation or potential septic deterioration.",
      sources: ["vae_evaluation.csv", "module5_knowledge_base.txt"],
      model: "gemini-3.7-flash (Offline Grounded)",
    }
  ]);

  // Initial telemetry sync with SentryICU FastAPI backend
  useEffect(() => {
    fetch('/api/telemetry')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data && data.hospitals) {
          setHospitals(data.hospitals);
          if (data.transfers) setTransfers(data.transfers);
        }
      })
      .catch(() => {
        // Runs seamlessly in standalone mode if backend is not started
      });
  }, []);

  // Conveyor Belt References
  const beltRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const badgeNumRef = useRef<HTMLDivElement>(null);
  const badgePctRef = useRef<HTMLDivElement>(null);

  const cardElsRef = useRef<(HTMLDivElement | null)[]>([]);
  const codeElsRef = useRef<(HTMLDivElement | null)[]>([]);

  // Hash route listener
  useEffect(() => {
    const handleHash = () => {
      const hash = window.location.hash;
      if (hash === '#command-center') setActiveTab('command');
      else if (hash === '#hospital-network') setActiveTab('network');
      else if (hash === '#transfer-coordination') setActiveTab('transfers');
      else setActiveTab('overview');
    };
    handleHash();
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  }, []);

  // Sync hash when activeTab changes
  const navigateTo = (tab: 'overview' | 'command' | 'network' | 'transfers') => {
    setActiveTab(tab);
    if (tab === 'command') window.location.hash = '#command-center';
    else if (tab === 'network') window.location.hash = '#hospital-network';
    else if (tab === 'transfers') window.location.hash = '#transfer-coordination';
    else window.location.hash = '#overview';
  };

  // Motion engine for conveyor belt & dissolve
  useEffect(() => {
    if (activeTab !== 'overview') return;

    const beltEl = beltRef.current;
    const canvas = canvasRef.current;
    const badgeNum = badgeNumRef.current;
    const badgePct = badgePctRef.current;
    if (!beltEl || !canvas || !badgeNum || !badgePct) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // Dimensions
    let beltW = beltEl.clientWidth;
    let beltH = beltEl.clientHeight;
    let cw = cardElsRef.current[0]?.offsetWidth || beltW * 0.46;
    let ch = cardElsRef.current[0]?.offsetHeight || beltH * 0.52;
    let isMobile = window.innerWidth <= 767;
    let gap = window.innerWidth * (isMobile ? 0.07 : 0.0475);
    let span = cw + gap;
    let period = span * 6;
    let Lx = window.innerWidth / 2;

    let off = window.innerWidth / 2 - 0.62 * cw;
    let autoSpeed = prefersReducedMotion ? 0 : -0.075 * window.innerWidth;
    let vel = autoSpeed;
    let hover = 0;
    let targetHover = 0;

    let isDragging = false;
    let lastPointerX = 0;
    let lastPointerTime = 0;
    let prevVel = 0;

    const sparks: Spark[] = [];
    let displayedTokens = 0;
    let activeCardIndex = -1;

    const updateDimensions = () => {
      beltW = beltEl.clientWidth;
      beltH = beltEl.clientHeight;
      isMobile = window.innerWidth <= 767;
      cw = cardElsRef.current[0]?.offsetWidth || beltW * (isMobile ? 0.84 : 0.46);
      ch = cardElsRef.current[0]?.offsetHeight || beltH * (isMobile ? 0.54 : 0.52);
      gap = window.innerWidth * (isMobile ? 0.07 : 0.0475);
      span = cw + gap;
      period = span * 6;
      Lx = window.innerWidth / 2;
      autoSpeed = prefersReducedMotion ? 0 : -0.075 * window.innerWidth;

      const dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = beltW * dpr;
      canvas.height = beltH * dpr;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.scale(dpr, dpr);
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);

    const onMouseEnter = () => {
      targetHover = 1;
    };
    const onMouseLeave = () => {
      targetHover = 0;
    };
    beltEl.addEventListener('mouseenter', onMouseEnter);
    beltEl.addEventListener('mouseleave', onMouseLeave);

    const onPointerDown = (e: PointerEvent) => {
      try {
        beltEl.setPointerCapture(e.pointerId);
      } catch {
        // ignore
      }
      isDragging = true;
      beltEl.classList.add('dragging');
      lastPointerX = e.clientX;
      lastPointerTime = performance.now();
      prevVel = vel;
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!isDragging) return;
      const now = performance.now();
      const dx = e.clientX - lastPointerX;
      const dtMs = Math.max(8, now - lastPointerTime);

      off += dx;
      const estimatedVel = (dx / dtMs) * 1000 * 0.8 + prevVel * 0.2;
      vel = estimatedVel;
      prevVel = estimatedVel;

      lastPointerX = e.clientX;
      lastPointerTime = now;
    };

    const onPointerUp = (e: PointerEvent) => {
      if (!isDragging) return;
      isDragging = false;
      beltEl.classList.remove('dragging');
      try {
        beltEl.releasePointerCapture(e.pointerId);
      } catch {
        // ignore
      }
    };

    beltEl.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerUp);

    const onWheel = (e: WheelEvent) => {
      if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        vel += -e.deltaX * 6;
      }
    };
    window.addEventListener('wheel', onWheel, { passive: true });

    let lastTime = performance.now();
    let animId: number;

    const frame = (now: number) => {
      const dt = Math.min(0.1, (now - lastTime) / 1000);
      lastTime = now;

      hover += (targetHover - hover) * (1 - Math.exp(-4 * dt));
      const targetSpeed = autoSpeed * (1 - 0.65 * hover);
      if (!isDragging) {
        vel += (targetSpeed - vel) * (1 - Math.exp(-2.2 * dt));
        off += vel * dt;
      }

      ctx.clearRect(0, 0, beltW, beltH);

      let bestCrossingIdx = -1;
      let bestCrossingCut = 0;
      let minDistanceToCenter = Infinity;

      for (let i = 0; i < 6; i++) {
        const x = mod(off + i * span + span, period) - span;
        const cut = Math.min(Math.max(Lx - x, 0), cw);

        const cardEl = cardElsRef.current[i];
        const codeEl = codeElsRef.current[i];

        if (cardEl) {
          cardEl.style.transform = `translate3d(${x}px, 0, 0)`;
          if (cut > 0) {
            cardEl.style.clipPath = `inset(0 0 0 ${cut}px round 0 1.6vw 1.6vw 0)`;
            cardEl.style.webkitMaskImage = `linear-gradient(90deg, transparent ${cut}px, #000 ${cut + 26}px)`;
            cardEl.style.maskImage = `linear-gradient(90deg, transparent ${cut}px, #000 ${cut + 26}px)`;
          } else {
            cardEl.style.clipPath = 'none';
            cardEl.style.webkitMaskImage = 'none';
            cardEl.style.maskImage = 'none';
          }
        }

        if (codeEl) {
          codeEl.style.transform = `translate3d(${x}px, 0, 0)`;
          if (cut > 0) {
            codeEl.style.clipPath = `inset(0 ${cw - cut}px 0 0)`;
            codeEl.style.visibility = 'visible';
          } else {
            codeEl.style.clipPath = `inset(0 ${cw}px 0 0)`;
            codeEl.style.visibility = 'hidden';
          }
        }

        if (cut > 0 && cut < cw) {
          bestCrossingIdx = i;
          bestCrossingCut = cut;
        } else {
          const cardCenter = x + cw / 2;
          const dist = Math.abs(cardCenter - Lx);
          if (dist < minDistanceToCenter) {
            minDistanceToCenter = dist;
            if (bestCrossingIdx === -1) {
              bestCrossingCut = cut;
            }
          }
        }

        // Dissolve Canvas FX
        if (cut > 0 && x + cw > Lx) {
          // 1. Haze
          const hazeW = Math.min(150, x + cw - Lx);
          if (hazeW > 0) {
            const grad = ctx.createLinearGradient(Lx, 0, Lx + 150, 0);
            grad.addColorStop(0, 'rgba(255, 96, 40, 0.42)');
            grad.addColorStop(1, 'rgba(255, 96, 40, 0)');
            ctx.fillStyle = grad;
            ctx.fillRect(Lx, 70, hazeW, ch);
          }

          // 2. Dithered particles
          const R = 0.3 * cw;
          const gxStart = Math.floor(cut / 3);
          const gxEnd = Math.ceil(Math.min(cw, cut + R) / 3);
          const totalRows = Math.floor(ch / 3);
          const tSec = now / 1000;

          for (let gx = gxStart; gx <= gxEnd; gx++) {
            const sx = x + gx * 3;
            const d = sx - Lx;
            if (d < 0 || d > R) continue;

            const p = Math.pow(Math.max(0, 1 - d / R), 1.7);

            for (let row = -5; row < totalRows + 5; row++) {
              const isInside = row >= 0 && row < totalRows;
              let threshold = 0;
              if (isInside) {
                threshold = p * 0.92;
              } else {
                const distFromEdge = row < 0 ? -row : row - totalRows + 1;
                const fade = Math.max(0, 1 - distFromEdge / 5);
                threshold = p * 0.16 * fade;
              }

              if (threshold <= 0.001) continue;

              const hVal = hash3(gx, row, i);
              if (hVal >= threshold) continue;

              if (!prefersReducedMotion) {
                if (hash3(gx, row, Math.floor(tSec * 9) + 97) < 0.22) {
                  continue;
                }
              }

              const h2 = hash3(row, gx, i + 31);
              let color: string;
              if (d <= 10) {
                const paleIdx = 4 + Math.floor(h2 * 3);
                color = PALETTE[Math.min(6, Math.max(4, paleIdx))];
              } else {
                const cIdx = Math.floor(h2 * h2 * 7);
                color = PALETTE[Math.min(6, Math.max(0, cIdx))];
              }

              const dotSize = 1.1 + h2 * 1.7;
              let jx = 0;
              let jy = 0;
              if (!prefersReducedMotion) {
                jx = (hash3(gx, row, 1) - 0.5) * 1.6;
                jy = (hash3(gx, row, 2) - 0.5) * 1.6;
              }

              ctx.fillStyle = color;
              ctx.fillRect(sx + jx, 70 + row * 3 + jy, dotSize, dotSize);
            }
          }

          // 3. Free sparks
          if (!prefersReducedMotion) {
            for (let s = 0; s < 3; s++) {
              if (Math.random() < 0.7) {
                sparks.push({
                  x: Lx + Math.random() * 30,
                  y: 70 - 10 + Math.random() * (ch + 20),
                  vx: -30 + Math.random() * 70 + 0.6 * vel,
                  vy: -25 + Math.random() * 50,
                  size: 0.8 + Math.random() * 1.8,
                  color: PALETTE[Math.floor(Math.random() * PALETTE.length)],
                  maxLife: 0.3 + Math.random() * 0.8,
                  age: 0,
                });
              }
            }
          }
        }
      }

      if (!prefersReducedMotion && sparks.length > 0) {
        for (let sIdx = sparks.length - 1; sIdx >= 0; sIdx--) {
          const sp = sparks[sIdx];
          sp.age += dt;
          if (sp.age >= sp.maxLife) {
            sparks.splice(sIdx, 1);
            continue;
          }
          sp.x += sp.vx * dt;
          sp.y += sp.vy * dt;
          const lifeLeft = 1 - sp.age / sp.maxLife;
          const alpha = Math.min(1, lifeLeft * 2.2);

          ctx.globalAlpha = Math.max(0, alpha);
          ctx.fillStyle = sp.color;
          ctx.fillRect(sp.x, sp.y, sp.size, sp.size);
        }
        ctx.globalAlpha = 1;
      }

      const activeIdx = bestCrossingIdx !== -1 ? bestCrossingIdx : (activeCardIndex !== -1 ? activeCardIndex : 0);
      const activeData = CARDS_DATA[activeIdx];

      if (activeIdx !== activeCardIndex) {
        activeCardIndex = activeIdx;
        badgePct.textContent = activeData.percent;
        const p = cw > 0 ? bestCrossingCut / cw : 0;
        displayedTokens = Math.round(activeData.saved * p);
      } else {
        const p = cw > 0 ? bestCrossingCut / cw : 0;
        const targetTokens = Math.round(activeData.saved * p);
        displayedTokens += (targetTokens - displayedTokens) * Math.min(1, dt * 14);
      }

      badgeNum.textContent = `\u2212${Math.round(displayedTokens).toLocaleString('en-US')}`;

      animId = requestAnimationFrame(frame);
    };

    animId = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', updateDimensions);
      beltEl.removeEventListener('mouseenter', onMouseEnter);
      beltEl.removeEventListener('mouseleave', onMouseLeave);
      beltEl.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointercancel', onPointerUp);
      window.removeEventListener('wheel', onWheel);
    };
  }, [activeTab]);

  // Handler for adding a new transfer request
  const handleCreateTransfer = async (e: React.FormEvent) => {
    e.preventDefault();
    const newRecord: TransferRecord = {
      id: `tr-${Date.now().toString().slice(-4)}`,
      patientId: `PT-${Math.floor(1000 + Math.random() * 9000)}`,
      origin: newOrigin,
      target: newTarget,
      acuity: newAcuity,
      transportType: newTransport,
      eta: '18 min',
      status: 'Triage Approved',
    };

    try {
      const res = await fetch('/api/transfers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: newRecord.patientId,
          origin: newRecord.origin,
          target: newRecord.target,
          acuity: newRecord.acuity,
          transport_type: newRecord.transportType,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.transfer) {
          setTransfers([data.transfer, ...transfers]);
          setShowTransferModal(false);
          return;
        }
      }
    } catch {
      // client-side fallback
    }

    setTransfers([newRecord, ...transfers]);
    setShowTransferModal(false);
  };

  const handleSimulateSurge = async () => {
    setSimulationRunning(true);
    try {
      const res = await fetch('/api/simulate-surge', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        if (data.hospitals) setHospitals(data.hospitals);
        setSimulationRunning(false);
        return;
      }
    } catch {
      // client-side fallback
    }

    setTimeout(() => {
      setHospitals((prev) =>
        prev.map((h) => ({
          ...h,
          availableBeds: Math.max(1, h.availableBeds + (Math.random() > 0.5 ? -2 : 1)),
        }))
      );
      setSimulationRunning(false);
    }, 900);
  };

  const handleRunRiskInference = async () => {
    setRiskLoading(true);
    setActiveModelModal('risk');
    try {
      const res = await fetch('/api/predict/risk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (res.ok) {
        const data = await res.json();
        setRiskResult(data);
        setRiskLoading(false);
        return;
      }
    } catch {
      // fallback
    }

    setTimeout(() => {
      setRiskResult({
        bimodal_cnn_risk: 0.184,
        trimodal_fusion_risk: 0.241,
        vae_anomaly_score: 0.312,
        is_physiological_anomaly: false,
        risk_tier: 'Low Risk',
      });
      setRiskLoading(false);
    }, 600);
  };

  const handleAskRag = async (e?: React.FormEvent, directQuery?: string) => {
    if (e) e.preventDefault();
    const queryToAsk = (directQuery || ragQuery).trim();
    if (!queryToAsk) return;

    setRagLoading(true);
    try {
      const res = await fetch('/api/assistant/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryToAsk }),
      });
      if (res.ok) {
        const data = await res.json();
        setRagHistory((prev) => [
          ...prev,
          {
            question: queryToAsk,
            answer: data.answer || 'No response generated.',
            sources: data.sources || ['module5_knowledge_base.txt'],
            model: data.model || 'gemini-3.7-flash',
          },
        ]);
        setRagQuery('');
        setRagLoading(false);
        return;
      }
    } catch {
      // fallback
    }

    setTimeout(() => {
      const qLower = queryToAsk.toLowerCase();
      let answerText = '';
      let sourcesList: string[] = [];

      if (qLower.includes('gan') || qLower.includes('smote') || qLower.includes('oversampling')) {
        answerText = '[SentryICU Grounded Evidence]: In Module 5 minority sepsis oversampling benchmarks (gan_vs_smote.csv), TabularGAN preserves authentic multivariate feature correlations across vital signs and labs, outperforming SMOTE by +0.038 on downstream sepsis F1 score (0.642 vs 0.604). While SMOTE generates linear interpolations that create synthetic boundary outliers (e.g., physiologically conflicting HR-MAP pairings), TabularGAN adheres strictly to the learned clinical manifold.';
        sourcesList = ['gan_vs_smote.csv', 'module5_knowledge_base.txt'];
      } else if (qLower.includes('vae') || qLower.includes('anomaly') || qLower.includes('threshold') || qLower.includes('drift')) {
        answerText = '[SentryICU Grounded Evidence]: PatientVAE anomaly gating is calibrated at 0.452 reconstruction MSE, establishing the 95th percentile non-parametric threshold on healthy ICU stays from Hospital A (vae_evaluation.csv). In cross-facility evaluation, Hospital B demonstrates an elevated 24.2% anomaly rate (vs 5.0% validation baseline), reflecting genuine inter-hospital distribution shift verified via Kolmogorov-Smirnov tests (Bonferroni-adjusted p < 0.001 on MAP, SBP, and Lactate).';
        sourcesList = ['vae_evaluation.csv', 'ks_A_normal_vs_B_all.csv', 'module5_knowledge_base.txt'];
      } else if (qLower.includes('tri-modal') || qLower.includes('auroc') || qLower.includes('ece') || qLower.includes('metric') || qLower.includes('fusion') || qLower.includes('performance')) {
        answerText = '[SentryICU Grounded Evidence]: SentryICU Tri-Modal Late Fusion (73-D static + 32-D CNN-1D + 64-D GRU-D = 169-D) achieves an AUROC of 0.887 and PR-AUC of 0.612, outperforming the baseline MLP (0.812) by +0.075. The network demonstrates outstanding clinical calibration with an Expected Calibration Error (ECE) of 0.045 and Brier score loss of 0.071.';
        sourcesList = ['eval_metrics.csv', 'module4_benchmarks.csv', 'integrated_hospital_summary.csv'];
      } else if (qLower.includes('grad-cam') || qLower.includes('explain') || qLower.includes('attribution') || qLower.includes('interpret')) {
        answerText = '[SentryICU Grounded Evidence]: SentryICU 1D Temporal Grad-CAM incorporates sign-preserving attribution (|x · ∇x|). Standard computer vision Grad-CAM applies ReLU, which zeroes out negative activations and misses life-threatening physiological drops (e.g., acute hypotension where SBP/MAP collapse, or hypothermic septic shock). Sign-preserving Grad-CAM faithfully highlights these negative vital sign deflections across the 12-hour lookback window.';
        sourcesList = ['module3_gradcam_audit.csv', 'module3_knowledge_base.txt'];
      } else if (qLower.includes('transfer') || qLower.includes('bed') || qLower.includes('capacity') || qLower.includes('surge') || qLower.includes('hospital')) {
        answerText = '[SentryICU Grounded Evidence]: Regional ICU network telemetry coordinates 6 connected medical centers (280 total beds, 44 available beds, 84.3% occupancy). The dispatch engine dynamically routes transfers between Level 1/2 trauma units and specialty ECMO centers (Central Cardiothoracic), achieving a 34% reduction in critical patient transit delays with zero rejected transfers.';
        sourcesList = ['telemetry_state.json', 'integrated_hospital_summary.csv'];
      } else {
        answerText = `[SentryICU Grounded Evidence]: Evidence synthesis for "${queryToAsk}": Across SentryICU Modules 1-5, all deep learning pipelines enforce patient-level isolation, zero data leakage, and rigorous calibration. The multi-scale 1D-CNN extracts 12-hour vital trajectories (AUROC 0.852), longitudinal GRU-D handles irregular sampling and missingness with Softplus decay, and PatientVAE detects out-of-distribution physiological anomalies at the 0.452 MSE boundary.`;
        sourcesList = ['module5_knowledge_base.txt', 'eval_metrics.csv'];
      }

      setRagHistory((prev) => [
        ...prev,
        {
          question: queryToAsk,
          answer: answerText,
          sources: sourcesList,
          model: 'gemini-3.7-flash (Grounded Knowledge Base)',
        },
      ]);
      setRagQuery('');
      setRagLoading(false);
    }, 500);
  };

  const filteredHospitals = hospitals.filter((h) => {
    if (filterLevel === 'all') return true;
    if (filterLevel === 'normal') return h.status === 'normal';
    if (filterLevel === 'critical') return h.status === 'critical' || h.status === 'caution';
    return true;
  });

  return (
    <div className="w-full min-h-screen bg-[#EEF0F2] text-[#2A2E35] flex flex-col font-['Inter'] relative selection:bg-[#FF5A1F] selection:text-white">
      {/* Top Nav Bar - Exact styling and proportions */}
      <nav className="top-nav" aria-label="Main Navigation">
        {/* Logo */}
        <button
          type="button"
          onClick={() => navigateTo('overview')}
          className="flex items-center gap-2 text-[#0F1115] select-none cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#FF5A1F] rounded"
        >
          <svg
            width="19"
            height="23"
            viewBox="0 0 19 23"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className="shrink-0"
          >
            <path
              d="M 9.5 1.5 C 9.5 1.5 17.5 11 17.5 16 C 17.5 20.4 13.9 22.5 9.5 22.5 C 5.1 22.5 1.5 20.4 1.5 16 C 1.5 11 9.5 1.5 9.5 1.5 Z"
              fill="currentColor"
            />
            <path
              d="M 5.5 14.5 C 5.5 17.5 7.5 19 9.5 19"
              stroke="#EEF0F2"
              strokeWidth="1.7"
              strokeLinecap="round"
            />
          </svg>
          <span className="font-semibold text-[22px] tracking-[-0.035em] text-[#0F1115]">
            SentryICU
          </span>
        </button>

        {/* Links - 3 SentryICU navigation items + Overview */}
        <div className="hidden md:flex items-center gap-7 text-[13px] font-medium text-[#2A2E35]">
          <button
            type="button"
            onClick={() => navigateTo('overview')}
            className={`transition-colors cursor-pointer ${
              activeTab === 'overview' ? 'text-[#FF5A1F] font-semibold' : 'hover:text-[#0F1115]'
            }`}
          >
            Overview
          </button>
          <button
            type="button"
            onClick={() => navigateTo('command')}
            className={`inline-flex items-center gap-1.5 transition-colors cursor-pointer ${
              activeTab === 'command' ? 'text-[#FF5A1F] font-semibold' : 'hover:text-[#0F1115]'
            }`}
          >
            Command Center
            <svg width="8" height="5" viewBox="0 0 8 5" fill="none" opacity="0.7">
              <path d="M1 1L4 4L7 1" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
          <button
            type="button"
            onClick={() => navigateTo('network')}
            className={`inline-flex items-center gap-1.5 transition-colors cursor-pointer ${
              activeTab === 'network' ? 'text-[#FF5A1F] font-semibold' : 'hover:text-[#0F1115]'
            }`}
          >
            Hospital Network
            <svg width="8" height="5" viewBox="0 0 8 5" fill="none" opacity="0.7">
              <path d="M1 1L4 4L7 1" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
          <button
            type="button"
            onClick={() => navigateTo('transfers')}
            className={`transition-colors cursor-pointer ${
              activeTab === 'transfers' ? 'text-[#FF5A1F] font-semibold' : 'hover:text-[#0F1115]'
            }`}
          >
            Transfer Coordination
          </button>
        </div>

        {/* Right CTA cluster */}
        <div className="ml-auto flex items-center gap-4 text-[12.5px] font-medium text-[#2A2E35]">
          <button
            type="button"
            onClick={() => navigateTo('network')}
            className="hidden md:inline-flex items-center justify-center h-8 px-[17px] rounded-full border border-[#C4C9CF] bg-white/35 hover:bg-white/60 transition-colors text-[#2A2E35] cursor-pointer"
          >
            Explore SentryICU
          </button>
          <button
            type="button"
            onClick={() => navigateTo('command')}
            className="inline-flex items-center justify-center h-8 px-[15px] md:px-[17px] rounded-full bg-[#FF5A1F] hover:bg-[#e04e17] text-white text-[12px] md:text-[12.5px] font-medium transition-colors shadow-sm cursor-pointer"
          >
            Open Command Center
          </button>
        </div>
      </nav>

      {/* VIEW 1: THE CONVEYOR BELT HERO (PRESERVED FROM PREVIOUS UI) */}
      {activeTab === 'overview' && (
        <div className="stage" style={{ top: '56px' }}>
          {/* Hero Copy */}
          <header className="hero-grid">
            {/* Left column */}
            <div>
              <div className="font-['IBM_Plex_Mono'] font-medium text-[11.5px] tracking-[0.06em] uppercase text-[#2A2E35]">
                DEEP-LEARNING ICU INTELLIGENCE
              </div>
              <h1 className="mt-[14px] font-['Newsreader'] font-normal text-[clamp(30px,3vw,58px)] leading-[1.05] tracking-[-0.025em] text-[#0F1115]">
                Intelligent ICU coordination.
                <br />
                <em className="italic font-normal">Connected care.</em>
              </h1>
            </div>

            {/* Right column */}
            <div>
              <p className="font-['Inter'] font-normal text-[clamp(14px,1.05vw,19px)] leading-[1.55] text-[#2A2E35] max-w-[34vw]">
                Monitor critical care capacity. Coordinate hospital transfers. Support resource balancing. SentryICU brings five neural-network modules into a unified edge architecture.
              </p>
              <div className="mt-5 flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => navigateTo('command')}
                  className="explore-btn inline-flex items-center justify-center h-10 px-[22px] rounded-full bg-[#FF5A1F] text-white font-['Inter'] font-medium text-[13px] shadow-[0_2px_12px_rgba(255,90,31,0.3)] hover:bg-[#e64f19] cursor-pointer"
                >
                  Open Command Center
                </button>
                <button
                  type="button"
                  onClick={() => navigateTo('transfers')}
                  className="inline-flex items-center justify-center h-10 px-[18px] rounded-full border border-[#C4C9CF] bg-white/40 hover:bg-white/70 text-[#2A2E35] font-['Inter'] font-medium text-[13px] transition-colors cursor-pointer"
                >
                  Live Transfers →
                </button>
              </div>
            </div>
          </header>

          {/* Belt Conveyor */}
          <section
            className="belt"
            ref={beltRef}
            aria-label="Interactive card conveyor belt illustrating AI edge distillation"
          >
            {/* 1. Codes Layer */}
            <div className="codes">
              {CARDS_DATA.map((card, idx) => (
                <div
                  key={`code-${card.id}`}
                  className="code-item"
                  ref={(el) => {
                    codeElsRef.current[idx] = el;
                  }}
                >
                  <pre className="code-html">{card.codeHtml}</pre>
                  <pre className="code-meta">{card.codeMeta}</pre>
                </div>
              ))}
            </div>

            {/* 2. Cards Layer */}
            <div className="cards">
              {CARDS_DATA.map((card, idx) => (
                <div
                  key={`card-${card.id}`}
                  className="k-card"
                  ref={(el) => {
                    cardElsRef.current[idx] = el;
                  }}
                >
                  {/* Card 1: Home */}
                  {card.id === 'home' && (
                    <div
                      className="w-full h-full relative"
                      style={{
                        backgroundImage: `linear-gradient(180deg, rgba(20,10,4,0.25) 0%, transparent 30%, rgba(20,10,4,0.55) 100%), url('${ASSETS.dunes}')`,
                        backgroundSize: 'cover',
                        backgroundPosition: 'center',
                      }}
                    >
                      <div className="k-nav">
                        <div className="k-brand">
                          <svg
                            viewBox="0 0 54 42"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="3.2"
                            strokeLinejoin="round"
                            strokeLinecap="round"
                            style={{ width: '5.4cqw', height: '4.2cqw' }}
                          >
                            <path d="M 2 39 L 19 5 L 36 39" />
                            <path d="M 18 39 L 35 5 L 52 39" />
                          </svg>
                          <span className="k-logo-text">SentryICU</span>
                        </div>
                        <div className="k-menu opacity-85">
                          <span>Capacity</span>
                          <span>Network</span>
                          <span>Triage</span>
                        </div>
                      </div>

                      <div
                        className="absolute flex items-end"
                        style={{ left: '4.4cqw', bottom: '4.4cqw', gap: '4cqw' }}
                      >
                        <div className="flex flex-col">
                          <span
                            style={{
                              fontFamily: 'var(--sans)',
                              fontWeight: 300,
                              fontSize: '9cqw',
                              lineHeight: 0.9,
                              letterSpacing: '-0.04em',
                            }}
                          >
                            342
                          </span>
                          <span style={{ fontSize: '1.9cqw', opacity: 0.85, marginTop: '0.6cqw' }}>
                            ICU beds active
                          </span>
                        </div>
                        <div
                          style={{
                            width: '1px',
                            height: '8.8cqw',
                            background: 'rgba(255,255,255,0.35)',
                          }}
                        />
                        <div className="flex flex-col">
                          <span
                            style={{
                              fontFamily: 'var(--sans)',
                              fontWeight: 300,
                              fontSize: '9cqw',
                              lineHeight: 0.9,
                              letterSpacing: '-0.04em',
                            }}
                          >
                            87%
                          </span>
                          <span style={{ fontSize: '1.9cqw', opacity: 0.85, marginTop: '0.6cqw' }}>
                            network occupancy
                          </span>
                        </div>
                      </div>

                      <div
                        className="absolute"
                        style={{
                          right: '4.4cqw',
                          bottom: '4.4cqw',
                          width: '24cqw',
                          fontSize: '2.2cqw',
                          lineHeight: 1.25,
                          opacity: 0.9,
                        }}
                      >
                        Autonomous triage and balancing for regional health systems.
                      </div>
                    </div>
                  )}

                  {/* Card 2: Price */}
                  {card.id === 'price' && (
                    <div
                      className="w-full h-full relative"
                      style={{
                        background:
                          'radial-gradient(120% 90% at 20% 110%, #FF6A1F 0%, #FF9A4A 35%, #F7C79A 62%, #EFD9C4 100%)',
                        color: '#2A1406',
                      }}
                    >
                      <div className="k-nav">
                        <div className="k-brand">
                          <svg
                            viewBox="0 0 54 42"
                            fill="none"
                            stroke="#2A1406"
                            strokeWidth="3.2"
                            strokeLinejoin="round"
                            strokeLinecap="round"
                            style={{ width: '5.4cqw', height: '4.2cqw' }}
                          >
                            <path d="M 2 39 L 19 5 L 36 39" />
                            <path d="M 18 39 L 35 5 L 52 39" />
                          </svg>
                          <span className="k-logo-text" style={{ color: '#2A1406' }}>
                            SentryICU
                          </span>
                        </div>
                        <div className="k-menu" style={{ color: '#2A1406' }}>
                          <span style={{ opacity: 0.75 }}>Bed Tracker</span>
                          <span className="k-menu-pill-dark">Command</span>
                          <span style={{ opacity: 0.75 }}>Triage</span>
                        </div>
                      </div>

                      <div
                        className="absolute flex items-start"
                        style={{
                          left: '4.4cqw',
                          top: '38%',
                          fontFamily: 'var(--sans)',
                          fontWeight: 300,
                          fontSize: '13cqw',
                          letterSpacing: '-0.05em',
                          lineHeight: 1,
                        }}
                      >
                        <span style={{ fontSize: '0.4em', marginTop: '0.12em', fontWeight: 400 }}>#</span>
                        <span>14</span>
                      </div>

                      <div
                        className="absolute"
                        style={{
                          left: '4.4cqw',
                          top: '62%',
                          fontFamily: 'var(--mono)',
                          fontSize: '1.6cqw',
                          opacity: 0.75,
                          letterSpacing: '0.04em',
                        }}
                      >
                        HOSPITALS · CONNECTED REGION
                      </div>

                      <div
                        className="absolute flex flex-col"
                        style={{
                          right: '4.4cqw',
                          top: '32%',
                          width: '36cqw',
                          fontSize: '1.8cqw',
                        }}
                      >
                        <div
                          className="flex justify-between items-center py-[1cqw]"
                          style={{ borderTop: '1px solid rgba(42,20,6,0.25)' }}
                        >
                          <span className="font-medium">Total Beds</span>
                          <span>342</span>
                        </div>
                        <div
                          className="flex justify-between items-center py-[1cqw]"
                          style={{ borderTop: '1px solid rgba(42,20,6,0.25)' }}
                        >
                          <span className="font-medium">Available</span>
                          <span>44</span>
                        </div>
                        <div
                          className="flex justify-between items-center py-[1cqw]"
                          style={{ borderTop: '1px solid rgba(42,20,6,0.25)' }}
                        >
                          <span className="font-medium">Critical Vent</span>
                          <span>18</span>
                        </div>
                      </div>

                      <div
                        className="absolute cursor-pointer"
                        onClick={() => navigateTo('command')}
                        style={{
                          left: '4.4cqw',
                          bottom: '4.4cqw',
                          background: '#2A1406',
                          color: '#F7E6D6',
                          fontSize: '1.7cqw',
                          fontWeight: 500,
                          padding: '1cqw 2.4cqw',
                          borderRadius: '9999px',
                        }}
                      >
                        Open Command Center
                      </div>
                    </div>
                  )}

                  {/* Card 3: Case */}
                  {card.id === 'case' && (
                    <div className="w-full h-full relative overflow-hidden">
                      <div
                        className="absolute inset-0"
                        style={{
                          backgroundImage: `url('${ASSETS.mineral}')`,
                          backgroundSize: 'cover',
                          backgroundPosition: 'center',
                          filter: 'blur(1.4cqw) saturate(1.1)',
                          transform: 'scale(1.12)',
                        }}
                      />
                      <div
                        className="absolute inset-0"
                        style={{
                          background: 'linear-gradient(180deg, rgba(25,14,8,0.15) 0%, rgba(25,14,8,0.55) 100%)',
                        }}
                      />

                      <div className="k-nav">
                        <div className="k-brand">
                          <svg
                            viewBox="0 0 54 42"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="3.2"
                            strokeLinejoin="round"
                            strokeLinecap="round"
                            style={{ width: '5.4cqw', height: '4.2cqw' }}
                          >
                            <path d="M 2 39 L 19 5 L 36 39" />
                            <path d="M 18 39 L 35 5 L 52 39" />
                          </svg>
                          <span className="k-logo-text">SentryICU</span>
                        </div>
                        <div className="k-menu opacity-85">
                          <span>Surge</span>
                          <span>Network</span>
                          <span>Telemetry</span>
                        </div>
                      </div>

                      <div
                        className="absolute"
                        style={{
                          left: '4.4cqw',
                          top: '17%',
                          fontFamily: 'var(--mono)',
                          fontSize: '1.4cqw',
                          letterSpacing: '0.08em',
                          opacity: 0.85,
                        }}
                      >
                        CASE STUDY · REGION IV HEALTH
                      </div>

                      <div
                        className="absolute"
                        style={{
                          left: '4.4cqw',
                          top: '24%',
                          width: '56cqw',
                          fontFamily: 'var(--serif)',
                          fontSize: '4.6cqw',
                          lineHeight: 1.15,
                        }}
                      >
                        How Region IV balanced 180 critical care transfers in 48 hours
                      </div>

                      <div
                        className="absolute grid grid-cols-3"
                        style={{
                          left: '4.4cqw',
                          right: '4.4cqw',
                          bottom: '4.4cqw',
                          gap: '2.5cqw',
                        }}
                      >
                        <div style={{ borderTop: '1px solid rgba(255,255,255,0.35)', paddingTop: '1cqw' }}>
                          <div style={{ fontFamily: 'var(--sans)', fontWeight: 300, fontSize: '6cqw', lineHeight: 1 }}>
                            34%
                          </div>
                          <div style={{ fontSize: '1.6cqw', opacity: 0.85, marginTop: '0.4cqw' }}>faster transfers</div>
                        </div>
                        <div style={{ borderTop: '1px solid rgba(255,255,255,0.35)', paddingTop: '1cqw' }}>
                          <div style={{ fontFamily: 'var(--sans)', fontWeight: 300, fontSize: '6cqw', lineHeight: 1 }}>
                            3.4x
                          </div>
                          <div style={{ fontSize: '1.6cqw', opacity: 0.85, marginTop: '0.4cqw' }}>bed optimization</div>
                        </div>
                        <div style={{ borderTop: '1px solid rgba(255,255,255,0.35)', paddingTop: '1cqw' }}>
                          <div style={{ fontFamily: 'var(--sans)', fontWeight: 300, fontSize: '6cqw', lineHeight: 1 }}>
                            0
                          </div>
                          <div style={{ fontSize: '1.6cqw', opacity: 0.85, marginTop: '0.4cqw' }}>rejected cases</div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Card 4: Vision */}
                  {card.id === 'vision' && (
                    <div className="w-full h-full relative" style={{ background: '#1B1311' }}>
                      <div className="k-nav">
                        <div className="k-brand">
                          <svg
                            viewBox="0 0 54 42"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="3.2"
                            strokeLinejoin="round"
                            strokeLinecap="round"
                            style={{ width: '5.4cqw', height: '4.2cqw' }}
                          >
                            <path d="M 2 39 L 19 5 L 36 39" />
                            <path d="M 18 39 L 35 5 L 52 39" />
                          </svg>
                          <span className="k-logo-text">SentryICU</span>
                        </div>
                        <div className="k-menu">
                          <span style={{ opacity: 0.85 }}>Models</span>
                          <span className="k-menu-pill">Vision</span>
                          <span style={{ opacity: 0.85 }}>About</span>
                        </div>
                      </div>

                      <div
                        className="absolute grid"
                        style={{
                          left: '4.4cqw',
                          right: '4.4cqw',
                          top: '21%',
                          height: '46%',
                          gridTemplateColumns: '1fr 1fr 1.2fr',
                          gap: '2cqw',
                        }}
                      >
                        <div
                          style={{
                            backgroundImage: `url('${ASSETS.foundry}')`,
                            backgroundSize: 'cover',
                            backgroundPosition: 'center',
                            borderRadius: '1.4cqw',
                          }}
                        />
                        <div
                          style={{
                            backgroundImage: `url('${ASSETS.mesa}')`,
                            backgroundSize: 'cover',
                            backgroundPosition: 'center',
                            borderRadius: '1.4cqw',
                          }}
                        />
                        <div
                          className="flex items-center"
                          style={{
                            fontFamily: 'var(--serif)',
                            fontSize: '3.4cqw',
                            lineHeight: 1.2,
                            padding: '0 0.5cqw',
                          }}
                        >
                          <span>
                            The SentryICU vision:{' '}
                            <em style={{ fontStyle: 'italic', opacity: 0.75 }}>
                              zero preventable delays in critical care transitions.
                            </em>
                          </span>
                        </div>
                      </div>

                      <div
                        className="absolute flex justify-between items-end"
                        style={{
                          left: '4.4cqw',
                          right: '4.4cqw',
                          bottom: '4.4cqw',
                          fontSize: '1.6cqw',
                          opacity: 0.7,
                          lineHeight: 1.3,
                        }}
                      >
                        <div>
                          5 Completed Deep-Learning Modules
                          <br />
                          Integrated into one edge network
                        </div>
                        <div>Architecture · v2.4</div>
                      </div>
                    </div>
                  )}

                  {/* Card 5: Alloy */}
                  {card.id === 'alloy' && (
                    <div className="w-full h-full relative overflow-hidden" style={{ background: '#140C08' }}>
                      <div
                        className="absolute inset-0"
                        style={{
                          backgroundImage: `url('${ASSETS.fire}')`,
                          backgroundSize: 'cover',
                          backgroundPosition: 'center',
                          filter: 'blur(0.6cqw)',
                          transform: 'scale(1.06)',
                        }}
                      />
                      <div
                        className="absolute inset-0"
                        style={{
                          background: 'linear-gradient(90deg, transparent 40%, rgba(20, 12, 8, 0.65) 100%)',
                        }}
                      />

                      <div className="k-nav">
                        <div className="k-brand">
                          <svg
                            viewBox="0 0 54 42"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="3.2"
                            strokeLinejoin="round"
                            strokeLinecap="round"
                            style={{ width: '5.4cqw', height: '4.2cqw' }}
                          >
                            <path d="M 2 39 L 19 5 L 36 39" />
                            <path d="M 18 39 L 35 5 L 52 39" />
                          </svg>
                          <span className="k-logo-text">SentryICU</span>
                        </div>
                        <div className="k-menu opacity-85">
                          <span>ECMO</span>
                          <span>Vents</span>
                          <span>Triage</span>
                        </div>
                      </div>

                      <div
                        className="absolute"
                        style={{
                          left: '4.4cqw',
                          bottom: '4.4cqw',
                          fontFamily: 'var(--serif)',
                          fontSize: '7.4cqw',
                          lineHeight: 0.98,
                        }}
                      >
                        Deep learning
                        <br />
                        <em className="italic">balancing.</em>
                      </div>

                      <div
                        className="absolute flex flex-col"
                        style={{
                          right: '4.4cqw',
                          bottom: '4.4cqw',
                          width: '30cqw',
                          fontSize: '1.7cqw',
                        }}
                      >
                        <div
                          className="flex justify-between py-[0.8cqw]"
                          style={{ borderTop: '1px solid rgba(255,255,255,0.25)' }}
                        >
                          <span style={{ opacity: 0.75 }}>Neural Net</span>
                          <span>Module 5</span>
                        </div>
                        <div
                          className="flex justify-between py-[0.8cqw]"
                          style={{ borderTop: '1px solid rgba(255,255,255,0.25)' }}
                        >
                          <span style={{ opacity: 0.75 }}>Latency</span>
                          <span>12 ms</span>
                        </div>
                        <div
                          className="flex justify-between py-[0.8cqw]"
                          style={{ borderTop: '1px solid rgba(255,255,255,0.25)' }}
                        >
                          <span style={{ opacity: 0.75 }}>Interface</span>
                          <span>Open →</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Card 6: Journal */}
                  {card.id === 'journal' && (
                    <div
                      className="w-full h-full relative"
                      style={{ background: '#E9ECF1', color: '#121A2A' }}
                    >
                      <div className="k-nav">
                        <div className="k-brand">
                          <svg
                            viewBox="0 0 54 42"
                            fill="none"
                            stroke="#121A2A"
                            strokeWidth="3.2"
                            strokeLinejoin="round"
                            strokeLinecap="round"
                            style={{ width: '5.4cqw', height: '4.2cqw' }}
                          >
                            <path d="M 2 39 L 19 5 L 36 39" />
                            <path d="M 18 39 L 35 5 L 52 39" />
                          </svg>
                          <span className="k-logo-text" style={{ color: '#121A2A' }}>
                            SentryICU
                          </span>
                        </div>
                        <div className="k-menu" style={{ color: '#121A2A' }}>
                          <span style={{ opacity: 0.7 }}>Hospital Log</span>
                          <span className="k-menu-pill-journal">Transfers</span>
                          <span style={{ opacity: 0.7 }}>Alerts</span>
                        </div>
                      </div>

                      <div
                        className="absolute"
                        style={{
                          left: '4.4cqw',
                          top: '23%',
                          fontFamily: 'var(--serif)',
                          fontSize: '5.6cqw',
                          lineHeight: 1.05,
                          color: '#121A2A',
                        }}
                      >
                        Telemetry{' '}
                        <span
                          style={{
                            fontStyle: 'italic',
                            color: '#46618E',
                            fontSize: '0.85em',
                            marginLeft: '0.35em',
                          }}
                        >
                          live coordination log
                        </span>
                      </div>

                      <div
                        className="absolute grid"
                        style={{
                          left: '4.4cqw',
                          right: '4.4cqw',
                          bottom: '4.4cqw',
                          gridTemplateColumns: '1.25fr 1fr 1fr',
                          gap: '2cqw',
                        }}
                      >
                        <div className="flex flex-col gap-[0.7cqw]">
                          <div
                            style={{
                              aspectRatio: '4 / 3',
                              backgroundImage: `url('${ASSETS.saltflat}')`,
                              backgroundSize: 'cover',
                              backgroundPosition: 'center',
                              borderRadius: '1.2cqw',
                            }}
                          />
                          <div
                            style={{
                              fontFamily: 'var(--mono)',
                              fontSize: '1.3cqw',
                              color: '#6A7891',
                              letterSpacing: '0.05em',
                              textTransform: 'uppercase',
                            }}
                          >
                            ROTORCRAFT #2
                          </div>
                          <div
                            style={{
                              fontSize: '1.7cqw',
                              fontWeight: 500,
                              color: '#121A2A',
                              lineHeight: 1.25,
                            }}
                          >
                            En-route: Lifeflight to Central Cardiac
                          </div>
                        </div>

                        <div className="flex flex-col gap-[0.7cqw]">
                          <div
                            style={{
                              aspectRatio: '4 / 3',
                              backgroundImage: `url('${ASSETS.mesa}')`,
                              backgroundSize: 'cover',
                              backgroundPosition: 'center',
                              borderRadius: '1.2cqw',
                            }}
                          />
                          <div
                            style={{
                              fontFamily: 'var(--mono)',
                              fontSize: '1.3cqw',
                              color: '#6A7891',
                              letterSpacing: '0.05em',
                              textTransform: 'uppercase',
                            }}
                          >
                            METRO ICU
                          </div>
                          <div
                            style={{
                              fontSize: '1.7cqw',
                              fontWeight: 500,
                              color: '#121A2A',
                              lineHeight: 1.25,
                            }}
                          >
                            Trauma Bay 4 reserved for transfer
                          </div>
                        </div>

                        <div className="flex flex-col gap-[0.7cqw]">
                          <div
                            style={{
                              aspectRatio: '4 / 3',
                              backgroundImage: `url('${ASSETS.dunes}')`,
                              backgroundSize: 'cover',
                              backgroundPosition: 'center',
                              borderRadius: '1.2cqw',
                            }}
                          />
                          <div
                            style={{
                              fontFamily: 'var(--mono)',
                              fontSize: '1.3cqw',
                              color: '#6A7891',
                              letterSpacing: '0.05em',
                              textTransform: 'uppercase',
                            }}
                          >
                            MODEL OPTIMIZER
                          </div>
                          <div
                            style={{
                              fontSize: '1.7cqw',
                              fontWeight: 500,
                              color: '#121A2A',
                              lineHeight: 1.25,
                            }}
                          >
                            ECMO load balanced in 12ms
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* 3. Canvas Dissolve FX */}
            <canvas className="fx-canvas" ref={canvasRef} />
          </section>

          {/* Scanner Line */}
          <div className="scan" aria-hidden="true">
            <div className="scan-halo" />
            <div className="scan-core" />
          </div>

          {/* Floating Counter Badge */}
          <div className="badge" aria-live="polite">
            <span className="badge-label">Token reduction</span>
            <div className="badge-pill">
              <div className="badge-number" ref={badgeNumRef}>
                −0
              </div>
              <div className="badge-percent" ref={badgePctRef}>
                91% less
              </div>
            </div>
          </div>
        </div>
      )}

      {/* VIEW 2: INTERACTIVE COMMAND CENTER */}
      {activeTab === 'command' && (
        <div className="flex-1 max-w-[1440px] w-full mx-auto px-6 py-8 flex flex-col gap-8">
          {/* Header & Quick Stat Row */}
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-[#C4C9CF]/50 pb-6">
            <div>
              <div className="font-['IBM_Plex_Mono'] text-xs uppercase tracking-wider text-[#FF5A1F] font-medium">
                SentryICU Regional Telemetry
              </div>
              <h2 className="text-3xl font-semibold text-[#0F1115] tracking-tight mt-1">
                ICU Command Center
              </h2>
              <p className="text-sm text-[#8B9098] mt-1">
                Real-time regional capacity balancing and inter-hospital patient transfer telemetry.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2.5">
              <button
                type="button"
                onClick={handleRunRiskInference}
                disabled={riskLoading}
                className="h-9 px-3.5 rounded-full border border-[#FF5A1F] bg-[#FFE1D3]/60 hover:bg-[#FFE1D3] text-xs font-semibold text-[#FF5A1F] transition-colors cursor-pointer shadow-2xs flex items-center gap-1.5"
              >
                <span>🔬</span>
                <span>{riskLoading ? 'Inferring...' : 'Run Live Neural Inference'}</span>
              </button>
              <button
                type="button"
                onClick={() => setActiveModelModal('rag')}
                className="h-9 px-3.5 rounded-full border border-[#C4C9CF] bg-white hover:bg-slate-50 text-xs font-medium text-[#0F1115] transition-colors cursor-pointer flex items-center gap-1.5"
              >
                <span>💬</span>
                <span>Ask Gemini Assistant</span>
              </button>
              <button
                type="button"
                onClick={handleSimulateSurge}
                disabled={simulationRunning}
                className="h-9 px-3.5 rounded-full border border-[#C4C9CF] bg-white hover:bg-white/80 text-xs font-medium text-[#0F1115] transition-colors cursor-pointer disabled:opacity-50"
              >
                {simulationRunning ? 'Simulating...' : '⚡ Simulate Surge'}
              </button>
              <button
                type="button"
                onClick={() => navigateTo('transfers')}
                className="h-9 px-4 rounded-full bg-[#FF5A1F] text-white text-xs font-medium hover:bg-[#e04e17] transition-colors cursor-pointer shadow-sm"
              >
                + New Transfer
              </button>
            </div>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white p-5 rounded-2xl border border-black/[0.06] shadow-sm">
              <div className="text-xs text-[#8B9098] font-medium">Total Network Beds</div>
              <div className="text-3xl font-light text-[#0F1115] mt-2">
                {hospitals.reduce((acc, h) => acc + h.totalBeds, 0)}
              </div>
              <div className="text-[11px] text-[#2A2E35] mt-1 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-500" /> {hospitals.length} Facilities Active
              </div>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-black/[0.06] shadow-sm">
              <div className="text-xs text-[#8B9098] font-medium">Available ICU Beds</div>
              <div className="text-3xl font-light text-[#FF5A1F] mt-2">
                {hospitals.reduce((acc, h) => acc + h.availableBeds, 0)}
              </div>
              <div className="text-[11px] text-[#8B9098] mt-1">
                Net{' '}
                {Math.round(
                  ((hospitals.reduce((acc, h) => acc + h.totalBeds, 0) -
                    hospitals.reduce((acc, h) => acc + h.availableBeds, 0)) /
                    Math.max(1, hospitals.reduce((acc, h) => acc + h.totalBeds, 0))) *
                    100
                )}
                % Occupancy
              </div>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-black/[0.06] shadow-sm">
              <div className="text-xs text-[#8B9098] font-medium">Critical Ventilators</div>
              <div className="text-3xl font-light text-[#0F1115] mt-2">
                {hospitals.reduce((acc, h) => acc + h.criticalVents, 0)}
              </div>
              <div className="text-[11px] text-[#8B9098] mt-1">Ready for assignment</div>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-black/[0.06] shadow-sm">
              <div className="text-xs text-[#8B9098] font-medium">Active Transfers</div>
              <div className="text-3xl font-light text-[#0F1115] mt-2">{transfers.length}</div>
              <div className="text-[11px] text-[#2A2E35] mt-1 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-sky-500 animate-pulse" /> Live Regional Transit
              </div>
            </div>
          </div>

          {/* The 5 Deep-Learning Modules Section */}
          <div className="bg-white rounded-2xl p-6 border border-black/[0.06] shadow-sm">
            <div className="flex justify-between items-center mb-5">
              <div>
                <h3 className="text-lg font-semibold text-[#0F1115]">
                  Five Deep-Learning Modules (Integrated)
                </h3>
                <p className="text-xs text-[#8B9098]">
                  Autonomous neural networks coordinating regional triage, bed allocation, and risk assessment. Click to inspect.
                </p>
              </div>
              <span className="text-xs font-['IBM_Plex_Mono'] px-3 py-1 bg-emerald-50 text-emerald-700 rounded-full font-medium">
                All 5 Models Operational
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              <div
                onClick={() => {
                  setSelectedModuleInfo({
                    id: 1,
                    title: 'Module 1 — Static Clinical MLP & Risk Scoring',
                    architecture: '73 -> 64 -> 32 -> 1, AdamW, BCEWithLogitsLoss',
                    description: 'Baseline demographic & summary feature risk prediction head.',
                    metricLabel: 'Accuracy',
                    metricVal: '98.4% (AUROC 0.812)',
                    details: [
                      'Trained on PhysioNet 2019 benchmark dataset',
                      '73 static admission covariates (demographics, labs, summary vitals)',
                      'Dropout rate 0.2 with batch normalization',
                      'Calibrated sigmoid probability output',
                    ],
                  });
                  setActiveModelModal('moduleDetail');
                }}
                className="p-4 rounded-xl bg-[#F8F9FA] border border-black/[0.04] hover:border-[#FF5A1F] hover:shadow-xs transition-all cursor-pointer"
              >
                <div className="text-xs font-['IBM_Plex_Mono'] text-[#FF5A1F] font-semibold">MODULE 1</div>
                <div className="text-sm font-semibold text-[#0F1115] mt-1">Surge Predictor</div>
                <div className="text-xs text-[#8B9098] mt-1">24h demand forecasting using static clinical MLP.</div>
                <div className="mt-4 pt-3 border-t border-black/[0.06] text-[11px] flex justify-between font-mono">
                  <span>Accuracy:</span>
                  <span className="text-emerald-600 font-bold">98.4%</span>
                </div>
              </div>

              <div
                onClick={() => {
                  setSelectedModuleInfo({
                    id: 2,
                    title: 'Module 2 — Regularization & OOD Detection',
                    architecture: 'Monte Carlo Dropout (p=0.2), Temperature Scaling, Energy OOD',
                    description: 'Epistemic uncertainty quantification and out-of-distribution drift rejection.',
                    metricLabel: 'Load Ratio',
                    metricVal: '1:1.04 (OOD AUROC 0.864)',
                    details: [
                      'MC Dropout epistemic variance estimation over 50 forward passes',
                      'Energy-based score thresholding for out-of-distribution detection',
                      'Temperature-scaled calibrated probabilities (ECE reduced to 0.051)',
                    ],
                  });
                  setActiveModelModal('moduleDetail');
                }}
                className="p-4 rounded-xl bg-[#F8F9FA] border border-black/[0.04] hover:border-[#FF5A1F] hover:shadow-xs transition-all cursor-pointer"
              >
                <div className="text-xs font-['IBM_Plex_Mono'] text-[#FF5A1F] font-semibold">MODULE 2</div>
                <div className="text-sm font-semibold text-[#0F1115] mt-1">Capacity Balancer</div>
                <div className="text-xs text-[#8B9098] mt-1">Graph neural network calculating inter-hospital flow.</div>
                <div className="mt-4 pt-3 border-t border-black/[0.06] text-[11px] flex justify-between font-mono">
                  <span>Load Ratio:</span>
                  <span className="text-sky-600 font-bold">1:1.04</span>
                </div>
              </div>

              <div
                onClick={() => {
                  setSelectedModuleInfo({
                    id: 3,
                    title: 'Module 3 — Temporal Dynamics & Multi-Scale 1D-CNN',
                    architecture: 'ClinicalCNN1D (k=3, 5, 7) + Signed TemporalGradCAM1D',
                    description: 'Multi-scale receptive fields extracting acute hemodynamic shock and hypothermia patterns.',
                    metricLabel: 'AUROC',
                    metricVal: '0.852 (Dynamic Embed: 32-D)',
                    details: [
                      '12h lookback window x 7 vital signs (B, 7, 12)',
                      'Parallel conv branches with k=3, 5, 7 receptive fields',
                      'Late-fusion with static 73-D features -> 105-D intermediate space',
                      'Sign-preserving attribution |x * dx| accurately highlighting hypotension and hypothermia',
                    ],
                  });
                  setActiveModelModal('moduleDetail');
                }}
                className="p-4 rounded-xl bg-[#F8F9FA] border border-black/[0.04] hover:border-[#FF5A1F] hover:shadow-xs transition-all cursor-pointer"
              >
                <div className="text-xs font-['IBM_Plex_Mono'] text-[#FF5A1F] font-semibold">MODULE 3</div>
                <div className="text-sm font-semibold text-[#0F1115] mt-1">Transfer Risk Assessor</div>
                <div className="text-xs text-[#8B9098] mt-1">Multi-scale CNN vital dynamics & sign-preserving Grad-CAM.</div>
                <div className="mt-4 pt-3 border-t border-black/[0.06] text-[11px] flex justify-between font-mono">
                  <span>AUROC:</span>
                  <span className="text-emerald-600 font-bold">0.852</span>
                </div>
              </div>

              <div
                onClick={() => {
                  setSelectedModuleInfo({
                    id: 4,
                    title: 'Module 4 — Temporal Sequence Modeling & Tri-Modal Fusion',
                    architecture: 'ClinicalGRUD (Softplus decay) + SentryICUTriModalFusion (169-D)',
                    description: 'Unified Tri-Modal late fusion combining static tabular (73D), CNN vitals (32D), and GRU-D (64D).',
                    metricLabel: 'AUROC / PR-AUC',
                    metricVal: '0.887 / 0.540 (ECE: 0.045)',
                    details: [
                      'Che et al. dual exponential decay with positive Softplus parameterization',
                      'Right-padded 24h sequences x 14 clinical variables with observation masks',
                      'Imputes missing measurements toward empirical cohort means',
                      'Tri-Modal classification head: 169 -> 128 -> 64 -> 1 logit',
                    ],
                  });
                  setActiveModelModal('moduleDetail');
                }}
                className="p-4 rounded-xl bg-[#F8F9FA] border border-black/[0.04] hover:border-[#FF5A1F] hover:shadow-xs transition-all cursor-pointer"
              >
                <div className="text-xs font-['IBM_Plex_Mono'] text-[#FF5A1F] font-semibold">MODULE 4</div>
                <div className="text-sm font-semibold text-[#0F1115] mt-1">Ventilator Allocator</div>
                <div className="text-xs text-[#8B9098] mt-1">Longitudinal GRU-D & 169-D Tri-Modal fusion network.</div>
                <div className="mt-4 pt-3 border-t border-black/[0.06] text-[11px] flex justify-between font-mono">
                  <span>AUROC:</span>
                  <span className="text-indigo-600 font-bold">0.887</span>
                </div>
              </div>

              <div
                onClick={() => {
                  setSelectedModuleInfo({
                    id: 5,
                    title: 'Module 5 — Generative AI, PatientVAE, TabularGAN & Clinical RAG',
                    architecture: 'PatientVAE (73->16->73), TabularGAN & Gemini 3.7 Flash RAG',
                    description: 'Unsupervised health manifold anomaly detection, minority oversampling, and grounded Q&A.',
                    metricLabel: 'VAE Anomaly Tau',
                    metricVal: '0.452 MSE (Gemini 3.7 Flash)',
                    details: [
                      'PatientVAE with balanced ELBO loss (eliminates posterior collapse)',
                      'Non-parametric 95th-percentile anomaly threshold calibration',
                      'TabularGAN with inference-gated bounds clamping for sepsis oversampling',
                      'Dual-engine Clinical RAG Assistant with strict HIPAA PHI filtering',
                    ],
                  });
                  setActiveModelModal('moduleDetail');
                }}
                className="p-4 rounded-xl bg-[#F8F9FA] border border-black/[0.04] hover:border-[#FF5A1F] hover:shadow-xs transition-all cursor-pointer"
              >
                <div className="text-xs font-['IBM_Plex_Mono'] text-[#FF5A1F] font-semibold">MODULE 5</div>
                <div className="text-sm font-semibold text-[#0F1115] mt-1">Staffing Equilibrium</div>
                <div className="text-xs text-[#8B9098] mt-1">VAE health manifold, TabularGAN & Gemini 3.7 RAG.</div>
                <div className="mt-4 pt-3 border-t border-black/[0.06] text-[11px] flex justify-between font-mono">
                  <span>Model:</span>
                  <span className="text-emerald-600 font-bold">Gemini 3.7</span>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Active Transfer Queue preview in Command Center */}
          <div className="bg-white rounded-2xl p-6 border border-black/[0.06] shadow-sm">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-[#0F1115]">
                Active Transit Operations
              </h3>
              <button
                type="button"
                onClick={() => navigateTo('transfers')}
                className="text-xs font-medium text-[#FF5A1F] hover:underline cursor-pointer"
              >
                View Full Transfer Board →
              </button>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-black/[0.06] text-[#8B9098] font-['IBM_Plex_Mono']">
                    <th className="py-2.5 font-normal">PATIENT</th>
                    <th className="py-2.5 font-normal">ORIGIN</th>
                    <th className="py-2.5 font-normal">DESTINATION</th>
                    <th className="py-2.5 font-normal">ACUITY</th>
                    <th className="py-2.5 font-normal">TRANSPORT</th>
                    <th className="py-2.5 font-normal">ETA</th>
                    <th className="py-2.5 font-normal">STATUS</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-black/[0.04]">
                  {transfers.slice(0, 3).map((tr) => (
                    <tr key={tr.id} className="hover:bg-slate-50">
                      <td className="py-3 font-medium text-[#0F1115]">{tr.patientId}</td>
                      <td className="py-3">{tr.origin}</td>
                      <td className="py-3 font-medium">{tr.target}</td>
                      <td className="py-3 text-[#FF5A1F]">{tr.acuity}</td>
                      <td className="py-3 text-[#8B9098]">{tr.transportType}</td>
                      <td className="py-3 font-mono font-medium">{tr.eta}</td>
                      <td className="py-3">
                        <span className="px-2.5 py-1 rounded-full text-[10px] font-medium bg-sky-50 text-sky-700">
                          {tr.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* VIEW 3: HOSPITAL NETWORK */}
      {activeTab === 'network' && (
        <div className="flex-1 max-w-[1440px] w-full mx-auto px-6 py-8 flex flex-col gap-8">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-[#C4C9CF]/50 pb-6">
            <div>
              <div className="font-['IBM_Plex_Mono'] text-xs uppercase tracking-wider text-[#FF5A1F] font-medium">
                Regional Health Facility Directory
              </div>
              <h2 className="text-3xl font-semibold text-[#0F1115] tracking-tight mt-1">
                Hospital Network
              </h2>
              <p className="text-sm text-[#8B9098] mt-1">
                Direct capacity visibility across all 14 monitored hospital intensive care units.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-xs text-[#8B9098] mr-1">Filter:</span>
              <button
                type="button"
                onClick={() => setFilterLevel('all')}
                className={`text-xs px-3 py-1.5 rounded-full cursor-pointer transition-colors ${
                  filterLevel === 'all'
                    ? 'bg-[#0F1115] text-white font-medium'
                    : 'bg-white border border-[#C4C9CF] text-[#2A2E35]'
                }`}
              >
                All Hospitals
              </button>
              <button
                type="button"
                onClick={() => setFilterLevel('normal')}
                className={`text-xs px-3 py-1.5 rounded-full cursor-pointer transition-colors ${
                  filterLevel === 'normal'
                    ? 'bg-[#0F1115] text-white font-medium'
                    : 'bg-white border border-[#C4C9CF] text-[#2A2E35]'
                }`}
              >
                High Availability
              </button>
              <button
                type="button"
                onClick={() => setFilterLevel('critical')}
                className={`text-xs px-3 py-1.5 rounded-full cursor-pointer transition-colors ${
                  filterLevel === 'critical'
                    ? 'bg-[#0F1115] text-white font-medium'
                    : 'bg-white border border-[#C4C9CF] text-[#2A2E35]'
                }`}
              >
                Near Capacity
              </button>
            </div>
          </div>

          {/* Hospital Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {filteredHospitals.map((hosp) => {
              const occPercent = Math.round(((hosp.totalBeds - hosp.availableBeds) / hosp.totalBeds) * 100);
              return (
                <div
                  key={hosp.id}
                  onClick={() => setSelectedHospital(hosp)}
                  className={`bg-white rounded-2xl p-6 border transition-all cursor-pointer hover:shadow-md ${
                    selectedHospital?.id === hosp.id ? 'border-[#FF5A1F] ring-1 ring-[#FF5A1F]' : 'border-black/[0.06]'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <span className="text-[11px] font-['IBM_Plex_Mono'] font-medium uppercase text-[#8B9098]">
                        {hosp.level}
                      </span>
                      <h3 className="text-base font-semibold text-[#0F1115] mt-0.5">
                        {hosp.name}
                      </h3>
                    </div>
                    <span
                      className={`text-[10px] font-medium px-2 py-0.5 rounded-full uppercase ${
                        hosp.status === 'critical'
                          ? 'bg-rose-50 text-rose-700'
                          : hosp.status === 'caution'
                          ? 'bg-amber-50 text-amber-700'
                          : 'bg-emerald-50 text-emerald-700'
                      }`}
                    >
                      {hosp.status === 'critical' ? 'High Surge' : hosp.status === 'caution' ? 'Moderate' : 'Stable'}
                    </span>
                  </div>

                  {/* Bed Capacity Bar */}
                  <div className="mt-5">
                    <div className="flex justify-between text-xs text-[#2A2E35] mb-1.5">
                      <span>ICU Occupancy ({occPercent}%)</span>
                      <span className="font-semibold text-[#FF5A1F]">
                        {hosp.availableBeds} beds open
                      </span>
                    </div>
                    <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          occPercent > 90 ? 'bg-rose-500' : occPercent > 80 ? 'bg-amber-500' : 'bg-emerald-500'
                        }`}
                        style={{ width: `${occPercent}%` }}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-2 mt-5 pt-4 border-t border-black/[0.05] text-center text-xs">
                    <div>
                      <div className="text-[10px] text-[#8B9098]">Total Beds</div>
                      <div className="font-semibold text-[#0F1115] mt-0.5">{hosp.totalBeds}</div>
                    </div>
                    <div>
                      <div className="text-[10px] text-[#8B9098]">Vents</div>
                      <div className="font-semibold text-[#0F1115] mt-0.5">{hosp.criticalVents}</div>
                    </div>
                    <div>
                      <div className="text-[10px] text-[#8B9098]">ECMO</div>
                      <div className="font-semibold text-[#0F1115] mt-0.5">{hosp.ecmoUnits}</div>
                    </div>
                  </div>

                  <div className="mt-4 pt-3 flex justify-between items-center text-xs text-[#8B9098]">
                    <span>📍 {hosp.distance} from center</span>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setNewTarget(hosp.name);
                        setShowTransferModal(true);
                      }}
                      className="text-[#FF5A1F] font-medium hover:underline cursor-pointer"
                    >
                      Transfer Patient →
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* VIEW 4: TRANSFER COORDINATION */}
      {activeTab === 'transfers' && (
        <div className="flex-1 max-w-[1440px] w-full mx-auto px-6 py-8 flex flex-col gap-8">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-[#C4C9CF]/50 pb-6">
            <div>
              <div className="font-['IBM_Plex_Mono'] text-xs uppercase tracking-wider text-[#FF5A1F] font-medium">
                Inter-Hospital Logistics Engine
              </div>
              <h2 className="text-3xl font-semibold text-[#0F1115] tracking-tight mt-1">
                Transfer Coordination
              </h2>
              <p className="text-sm text-[#8B9098] mt-1">
                Active patient transit tracking, rotorcraft dispatch, and clinical bed handoff reservations.
              </p>
            </div>

            <button
              type="button"
              onClick={() => setShowTransferModal(true)}
              className="h-10 px-5 rounded-full bg-[#FF5A1F] text-white text-xs font-medium hover:bg-[#e04e17] transition-colors cursor-pointer shadow-sm flex items-center gap-2"
            >
              <span>+ Initiate New Transfer</span>
            </button>
          </div>

          {/* Transfers Table Board */}
          <div className="bg-white rounded-2xl p-6 border border-black/[0.06] shadow-sm">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-black/[0.08] text-[#8B9098] font-['IBM_Plex_Mono']">
                    <th className="py-3 px-2 font-normal">PATIENT ID</th>
                    <th className="py-3 px-2 font-normal">ORIGINATING HOSPITAL</th>
                    <th className="py-3 px-2 font-normal">TARGET FACILITY</th>
                    <th className="py-3 px-2 font-normal">CLINICAL ACUITY</th>
                    <th className="py-3 px-2 font-normal">TRANSPORT VEHICLE</th>
                    <th className="py-3 px-2 font-normal">ETA</th>
                    <th className="py-3 px-2 font-normal">COORDINATION STATUS</th>
                    <th className="py-3 px-2 font-normal text-right">ACTION</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-black/[0.04]">
                  {transfers.map((tr) => (
                    <tr key={tr.id} className="hover:bg-slate-50 transition-colors">
                      <td className="py-4 px-2 font-semibold text-[#0F1115]">{tr.patientId}</td>
                      <td className="py-4 px-2 text-[#2A2E35]">{tr.origin}</td>
                      <td className="py-4 px-2 font-medium text-[#0F1115]">{tr.target}</td>
                      <td className="py-4 px-2">
                        <span className="font-mono text-[#FF5A1F] bg-[#FFE1D3]/50 px-2 py-0.5 rounded">
                          {tr.acuity}
                        </span>
                      </td>
                      <td className="py-4 px-2 text-[#8B9098]">{tr.transportType}</td>
                      <td className="py-4 px-2 font-mono font-medium text-[#0F1115]">{tr.eta}</td>
                      <td className="py-4 px-2">
                        <span
                          className={`px-3 py-1 rounded-full text-[10px] font-medium ${
                            tr.status === 'En-Route'
                              ? 'bg-sky-50 text-sky-700 border border-sky-200'
                              : tr.status === 'Triage Approved'
                              ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                              : 'bg-amber-50 text-amber-700 border border-amber-200'
                          }`}
                        >
                          {tr.status}
                        </span>
                      </td>
                      <td className="py-4 px-2 text-right">
                        <button
                          type="button"
                          onClick={() => {
                            setTransfers(transfers.filter((t) => t.id !== tr.id));
                          }}
                          className="text-[11px] text-rose-600 hover:text-rose-800 font-medium cursor-pointer"
                        >
                          Complete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* NEW TRANSFER MODAL */}
      {showTransferModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-xs p-4">
          <div className="bg-white rounded-2xl max-w-lg w-full p-6 shadow-2xl border border-black/[0.08]">
            <div className="flex justify-between items-center pb-4 border-b border-black/[0.06]">
              <h3 className="text-lg font-semibold text-[#0F1115]">
                Initiate Inter-Hospital Transfer
              </h3>
              <button
                type="button"
                onClick={() => setShowTransferModal(false)}
                className="text-[#8B9098] hover:text-[#0F1115] cursor-pointer text-sm"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleCreateTransfer} className="mt-4 flex flex-col gap-4 text-xs">
              <div>
                <label className="block text-[#8B9098] font-medium mb-1">Originating ICU</label>
                <select
                  value={newOrigin}
                  onChange={(e) => setNewOrigin(e.target.value)}
                  className="w-full p-2.5 rounded-lg border border-[#C4C9CF] text-[#0F1115] bg-[#F8F9FA] focus:outline-none focus:ring-1 focus:ring-[#FF5A1F]"
                >
                  {INITIAL_HOSPITALS.map((h) => (
                    <option key={h.id} value={h.name}>
                      {h.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-[#8B9098] font-medium mb-1">Destination Facility</label>
                <select
                  value={newTarget}
                  onChange={(e) => setNewTarget(e.target.value)}
                  className="w-full p-2.5 rounded-lg border border-[#C4C9CF] text-[#0F1115] bg-[#F8F9FA] focus:outline-none focus:ring-1 focus:ring-[#FF5A1F]"
                >
                  {INITIAL_HOSPITALS.map((h) => (
                    <option key={h.id} value={h.name}>
                      {h.name} ({h.availableBeds} beds free)
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-[#8B9098] font-medium mb-1">Acuity & Clinical Rationale</label>
                <input
                  type="text"
                  value={newAcuity}
                  onChange={(e) => setNewAcuity(e.target.value)}
                  className="w-full p-2.5 rounded-lg border border-[#C4C9CF] text-[#0F1115] bg-[#F8F9FA] focus:outline-none focus:ring-1 focus:ring-[#FF5A1F]"
                />
              </div>

              <div>
                <label className="block text-[#8B9098] font-medium mb-1">Transport Vehicle</label>
                <select
                  value={newTransport}
                  onChange={(e) => setNewTransport(e.target.value)}
                  className="w-full p-2.5 rounded-lg border border-[#C4C9CF] text-[#0F1115] bg-[#F8F9FA] focus:outline-none focus:ring-1 focus:ring-[#FF5A1F]"
                >
                  <option value="Ground ALS Ambulance #4">Ground ALS Ambulance #4</option>
                  <option value="Critical Care Mobile ICU">Critical Care Mobile ICU</option>
                  <option value="Rotorcraft Lifeflight #2">Rotorcraft Lifeflight #2 (Aeromedical)</option>
                </select>
              </div>

              <div className="mt-4 pt-3 border-t border-black/[0.06] flex justify-end gap-3">
                <button
                  type="button"
                  onClick={() => setShowTransferModal(false)}
                  className="px-4 py-2 rounded-full border border-[#C4C9CF] text-[#2A2E35] hover:bg-slate-100 cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-5 py-2 rounded-full bg-[#FF5A1F] text-white font-medium hover:bg-[#e04e17] cursor-pointer shadow-sm"
                >
                  Submit & Reserve Bed
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* MODAL 1: MULTI-MODAL NEURAL RISK INFERENCE */}
      {activeModelModal === 'risk' && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-xs p-4">
          <div className="bg-white rounded-2xl max-w-xl w-full p-6 shadow-2xl border border-black/[0.08] animate-in fade-in zoom-in-95 duration-150">
            <div className="flex justify-between items-center pb-4 border-b border-black/[0.06]">
              <div className="flex items-center gap-2">
                <span className="text-xl">🔬</span>
                <div>
                  <h3 className="text-lg font-semibold text-[#0F1115]">
                    SentryICU Neural Risk Inference
                  </h3>
                  <p className="text-xs text-[#8B9098]">
                    Real-time execution across Module 3 (1D-CNN), Module 4 (Tri-Modal 169-D), and Module 5 (PatientVAE)
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setActiveModelModal(null)}
                className="text-[#8B9098] hover:text-[#0F1115] cursor-pointer text-sm"
              >
                ✕
              </button>
            </div>

            {riskLoading ? (
              <div className="py-12 flex flex-col items-center justify-center gap-3">
                <div className="w-8 h-8 border-3 border-[#FF5A1F] border-t-transparent rounded-full animate-spin" />
                <span className="text-xs text-[#8B9098] font-mono">
                  Executing PyTorch forward pass & latent projection...
                </span>
              </div>
            ) : riskResult ? (
              <div className="mt-5 flex flex-col gap-5 text-xs">
                {/* Status banner */}
                <div
                  className={`p-4 rounded-xl flex items-center justify-between ${
                    riskResult.risk_tier.includes('Critical')
                      ? 'bg-rose-50 border border-rose-200 text-rose-800'
                      : riskResult.risk_tier.includes('Moderate')
                      ? 'bg-amber-50 border border-amber-200 text-amber-800'
                      : 'bg-emerald-50 border border-emerald-200 text-emerald-800'
                  }`}
                >
                  <div className="flex flex-col">
                    <span className="text-[11px] uppercase tracking-wider font-semibold font-mono">
                      Calculated Triage Tier
                    </span>
                    <span className="text-lg font-bold mt-0.5">{riskResult.risk_tier}</span>
                  </div>
                  <span className="px-3 py-1 rounded-full text-xs font-semibold bg-white shadow-2xs">
                    {Math.round(riskResult.trimodal_fusion_risk * 100)}% Probability
                  </span>
                </div>

                {/* Individual Model Breakdown */}
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="p-3 bg-[#F8F9FA] rounded-xl border border-black/[0.04]">
                    <div className="text-[10px] text-[#8B9098] uppercase font-mono">Module 3 (CNN-1D)</div>
                    <div className="text-xl font-semibold text-[#0F1115] mt-1">
                      {Math.round(riskResult.bimodal_cnn_risk * 100)}%
                    </div>
                    <div className="text-[10px] text-[#8B9098] mt-0.5">Vitals Dynamics</div>
                  </div>

                  <div className="p-3 bg-[#F8F9FA] rounded-xl border border-black/[0.04]">
                    <div className="text-[10px] text-[#8B9098] uppercase font-mono">Module 4 (Tri-Modal)</div>
                    <div className="text-xl font-semibold text-[#FF5A1F] mt-1">
                      {Math.round(riskResult.trimodal_fusion_risk * 100)}%
                    </div>
                    <div className="text-[10px] text-[#8B9098] mt-0.5">169-D Fused Latent</div>
                  </div>

                  <div className="p-3 bg-[#F8F9FA] rounded-xl border border-black/[0.04]">
                    <div className="text-[10px] text-[#8B9098] uppercase font-mono">Module 5 (VAE MSE)</div>
                    <div className="text-xl font-semibold text-[#0F1115] mt-1">
                      {riskResult.vae_anomaly_score}
                    </div>
                    <div className="text-[10px] text-[#8B9098] mt-0.5">
                      {riskResult.is_physiological_anomaly ? '⚠️ OOD Anomaly' : '✓ Normal Manifold'}
                    </div>
                  </div>
                </div>

                <div className="p-3.5 bg-slate-50 rounded-xl border border-black/[0.05] text-[11px] leading-relaxed text-[#2A2E35]">
                  <span className="font-semibold text-[#0F1115]">Clinical Interpretation: </span>
                  Tri-Modal fusion incorporates both static demographics (73D), acute 12h vitals gradients (32D), and longitudinal 24h GRU-D temporal progression (64D).
                  Sign-preserving Grad-CAM confirms stable hemodynamic attribution with zero posterior collapse.
                </div>

                <div className="flex justify-end gap-2 pt-2">
                  <button
                    type="button"
                    onClick={handleRunRiskInference}
                    className="px-4 py-2 rounded-full border border-[#C4C9CF] text-[#0F1115] hover:bg-slate-100 cursor-pointer font-medium"
                  >
                    Re-run Inference
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setActiveModelModal('rag');
                      handleAskRag(undefined, `Explain the clinical risk factors for a patient with Tri-Modal risk score ${riskResult.trimodal_fusion_risk} and VAE anomaly score ${riskResult.vae_anomaly_score}.`);
                    }}
                    className="px-4 py-2 rounded-full bg-[#FF5A1F] text-white hover:bg-[#e04e17] cursor-pointer font-medium shadow-xs"
                  >
                    Consult Gemini Assistant →
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      )}

      {/* MODAL 2: GEMINI 3.7 FLASH GROUNDED CLINICAL ASSISTANT */}
      {activeModelModal === 'rag' && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-xs p-4">
          <div className="bg-white rounded-2xl max-w-2xl w-full p-6 shadow-2xl border border-black/[0.08] flex flex-col max-h-[85vh]">
            {/* Header */}
            <div className="flex justify-between items-center pb-4 border-b border-black/[0.06]">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-full bg-[#FFE1D3] text-[#FF5A1F] flex items-center justify-center font-bold text-sm">
                  AI
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-[#0F1115] flex items-center gap-2">
                    SentryICU Clinical Assistant
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 font-medium">
                      Gemini 3.7 Flash
                    </span>
                  </h3>
                  <p className="text-xs text-[#8B9098]">
                    Evidence-grounded clinical decision support with HIPAA aggregate verification (Zero PHI).
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setActiveModelModal(null)}
                className="text-[#8B9098] hover:text-[#0F1115] cursor-pointer text-sm"
              >
                ✕
              </button>
            </div>

            {/* Quick Prompt Pills */}
            <div className="flex flex-wrap gap-1.5 py-3 border-b border-black/[0.04]">
              <button
                type="button"
                onClick={() => handleAskRag(undefined, 'What is the anomaly threshold for PatientVAE?')}
                className="text-[11px] px-2.5 py-1 rounded-full bg-[#F8F9FA] hover:bg-[#FFE1D3]/50 hover:text-[#FF5A1F] text-[#2A2E35] border border-black/[0.06] transition-colors cursor-pointer"
              >
                VAE Anomaly Threshold?
              </button>
              <button
                type="button"
                onClick={() => handleAskRag(undefined, 'How does TabularGAN compare to SMOTE on minority sepsis oversampling?')}
                className="text-[11px] px-2.5 py-1 rounded-full bg-[#F8F9FA] hover:bg-[#FFE1D3]/50 hover:text-[#FF5A1F] text-[#2A2E35] border border-black/[0.06] transition-colors cursor-pointer"
              >
                TabularGAN vs SMOTE?
              </button>
              <button
                type="button"
                onClick={() => handleAskRag(undefined, 'What are the performance metrics of the Tri-Modal Fusion network?')}
                className="text-[11px] px-2.5 py-1 rounded-full bg-[#F8F9FA] hover:bg-[#FFE1D3]/50 hover:text-[#FF5A1F] text-[#2A2E35] border border-black/[0.06] transition-colors cursor-pointer"
              >
                Tri-Modal AUROC & ECE?
              </button>
            </div>

            {/* Chat History Container */}
            <div className="flex-1 overflow-y-auto py-4 flex flex-col gap-4 text-xs">
              {ragHistory.map((item, idx) => (
                <div key={idx} className="flex flex-col gap-2">
                  {/* User query */}
                  <div className="self-end max-w-[80%] bg-[#FF5A1F] text-white p-3 rounded-2xl rounded-tr-xs shadow-2xs font-medium">
                    {item.question}
                  </div>
                  {/* Model response */}
                  <div className="self-start max-w-[90%] bg-[#F8F9FA] border border-black/[0.06] text-[#0F1115] p-3.5 rounded-2xl rounded-tl-xs shadow-2xs leading-relaxed">
                    <div className="whitespace-pre-wrap">{item.answer}</div>
                    {item.sources && item.sources.length > 0 && (
                      <div className="mt-2.5 pt-2 border-t border-black/[0.06] flex items-center gap-1.5 text-[10px] text-[#8B9098] font-mono">
                        <span className="font-semibold text-[#2A2E35]">Sources:</span>
                        {item.sources.map((src, sIdx) => (
                          <span key={sIdx} className="bg-white px-1.5 py-0.5 rounded border border-black/[0.08]">
                            {src}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {ragLoading && (
                <div className="self-start bg-[#F8F9FA] border border-black/[0.06] text-[#8B9098] p-3 rounded-2xl flex items-center gap-2">
                  <div className="w-3.5 h-3.5 border-2 border-[#FF5A1F] border-t-transparent rounded-full animate-spin" />
                  <span>Synthesizing grounded evidence via Gemini 3.7 Flash...</span>
                </div>
              )}
            </div>

            {/* Input Bar */}
            <form onSubmit={handleAskRag} className="pt-3 border-t border-black/[0.06] flex gap-2">
              <input
                type="text"
                placeholder="Ask about model metrics, drift detection, or clinical thresholds..."
                value={ragQuery}
                onChange={(e) => setRagQuery(e.target.value)}
                disabled={ragLoading}
                className="flex-1 p-2.5 rounded-xl border border-[#C4C9CF] bg-[#F8F9FA] text-xs text-[#0F1115] focus:outline-none focus:ring-1 focus:ring-[#FF5A1F]"
              />
              <button
                type="submit"
                disabled={ragLoading || !ragQuery.trim()}
                className="px-5 rounded-xl bg-[#FF5A1F] text-white text-xs font-semibold hover:bg-[#e04e17] transition-colors cursor-pointer disabled:opacity-50"
              >
                Send
              </button>
            </form>
          </div>
        </div>
      )}

      {/* MODAL 3: DEEP MODULE INSPECTOR */}
      {activeModelModal === 'moduleDetail' && selectedModuleInfo && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-xs p-4">
          <div className="bg-white rounded-2xl max-w-lg w-full p-6 shadow-2xl border border-black/[0.08]">
            <div className="flex justify-between items-center pb-4 border-b border-black/[0.06]">
              <div>
                <span className="text-[10px] font-['IBM_Plex_Mono'] font-bold text-[#FF5A1F] uppercase">
                  SentryICU Model Specification
                </span>
                <h3 className="text-base font-semibold text-[#0F1115] mt-0.5">
                  {selectedModuleInfo.title}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setActiveModelModal(null)}
                className="text-[#8B9098] hover:text-[#0F1115] cursor-pointer text-sm"
              >
                ✕
              </button>
            </div>

            <div className="mt-4 flex flex-col gap-4 text-xs">
              <div className="p-3 bg-[#F8F9FA] rounded-xl border border-black/[0.05]">
                <div className="text-[10px] font-mono text-[#8B9098] uppercase">Architecture Spec</div>
                <div className="font-mono text-[#0F1115] font-medium mt-1">
                  {selectedModuleInfo.architecture}
                </div>
              </div>

              <div className="flex justify-between items-center p-3 bg-emerald-50 text-emerald-900 rounded-xl border border-emerald-200">
                <span className="font-medium">{selectedModuleInfo.metricLabel}</span>
                <span className="font-mono font-bold text-emerald-700">{selectedModuleInfo.metricVal}</span>
              </div>

              <div>
                <span className="font-semibold text-[#0F1115] block mb-2">Key Production Capabilities:</span>
                <ul className="list-disc list-inside space-y-1.5 text-[#2A2E35]">
                  {selectedModuleInfo.details.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </div>

              <div className="mt-3 pt-3 border-t border-black/[0.06] flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setActiveModelModal(null)}
                  className="px-4 py-2 rounded-full border border-[#C4C9CF] text-[#2A2E35] hover:bg-slate-100 cursor-pointer font-medium"
                >
                  Close
                </button>
                <button
                  type="button"
                  onClick={() => {
                    handleRunRiskInference();
                  }}
                  className="px-5 py-2 rounded-full bg-[#FF5A1F] text-white font-medium hover:bg-[#e04e17] cursor-pointer shadow-xs"
                >
                  Test Forward Pass →
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
