<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="Mustatil is a GIS-level AI vision workspace for annotation, YOLO training, large-scale detection, satellite-map analysis, GeoPackage export, and visual AI pipeline building." />
  <meta name="keywords" content="Mustatil, YOLO, GIS, AI detection, annotation, satellite detection, GeoPackage, remote sensing, computer vision, object detection, QGIS" />
  <meta name="author" content="Tarek Wasfy" />

  <meta property="og:title" content="Mustatil - GIS-Level AI Vision Workspace" />
  <meta property="og:description" content="Annotation, YOLO training, large-image detection, satellite-map analysis, geospatial export, and graphical AI pipeline building in one desktop application." />
  <meta property="og:type" content="website" />
  <meta property="og:url" content="https://mustatil.de/" />

  <title>Mustatil - GIS-Level AI Vision Workspace</title>

  <style>
    :root {
      --bg: #0b1020;
      --bg-soft: #111831;
      --bg-card: #161f3a;
      --bg-card-2: #10172d;
      --text: #f4f7fb;
      --muted: #b7c0d8;
      --muted-2: #8792b0;
      --accent: #7c5cff;
      --accent-2: #00d0ff;
      --accent-3: #82bea0;
      --danger: #ff4f6d;
      --warning: #ffb84d;
      --ok: #69e19c;
      --border: rgba(255, 255, 255, 0.12);
      --shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
      --radius: 22px;
      --radius-sm: 14px;
      --max: 1180px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    }

    * {
      box-sizing: border-box;
    }

    html {
      scroll-behavior: smooth;
    }

    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(circle at 15% 0%, rgba(124, 92, 255, 0.28), transparent 30%),
        radial-gradient(circle at 85% 10%, rgba(0, 208, 255, 0.18), transparent 26%),
        radial-gradient(circle at 50% 100%, rgba(130, 190, 160, 0.14), transparent 32%),
        var(--bg);
      line-height: 1.6;
    }

    a {
      color: inherit;
      text-decoration: none;
    }

    img {
      max-width: 100%;
      display: block;
    }

    .wrap {
      width: min(var(--max), calc(100% - 40px));
      margin: 0 auto;
    }

    .nav {
      position: sticky;
      top: 0;
      z-index: 20;
      backdrop-filter: blur(20px);
      background: rgba(11, 16, 32, 0.78);
      border-bottom: 1px solid var(--border);
    }

    .nav-inner {
      min-height: 74px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 22px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 800;
      letter-spacing: -0.03em;
      font-size: 1.16rem;
    }

    .brand-mark {
      width: 38px;
      height: 38px;
      border-radius: 12px;
      background:
        linear-gradient(135deg, rgba(124, 92, 255, 1), rgba(0, 208, 255, 0.95));
      box-shadow: 0 12px 40px rgba(124, 92, 255, 0.32);
      position: relative;
    }

    .brand-mark::after {
      content: "";
      position: absolute;
      inset: 9px;
      border: 2px solid rgba(255, 255, 255, 0.9);
      border-radius: 4px;
    }

    .nav-links {
      display: flex;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
      justify-content: flex-end;
      color: var(--muted);
      font-size: 0.94rem;
    }

    .nav-links a {
      padding: 8px 10px;
      border-radius: 10px;
    }

    .nav-links a:hover {
      background: rgba(255, 255, 255, 0.08);
      color: var(--text);
    }

    .hero {
      padding: 76px 0 48px;
    }

    .hero-grid {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 34px;
      align-items: center;
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.06);
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 20px;
    }

    .eyebrow-dot {
      width: 8px;
      height: 8px;
      background: var(--ok);
      border-radius: 999px;
      box-shadow: 0 0 18px rgba(105, 225, 156, 0.7);
    }

    h1 {
      margin: 0;
      font-size: clamp(3rem, 8vw, 6.8rem);
      line-height: 0.92;
      letter-spacing: -0.075em;
    }

    .gradient-text {
      background: linear-gradient(120deg, #ffffff 0%, #d9e2ff 35%, #8fdfff 64%, #b9ffdc 100%);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }

    .lead {
      max-width: 760px;
      margin: 24px 0 0;
      color: var(--muted);
      font-size: clamp(1.08rem, 2vw, 1.35rem);
      line-height: 1.65;
    }

    .hero-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-top: 30px;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      min-height: 48px;
      padding: 13px 18px;
      border-radius: 14px;
      border: 1px solid var(--border);
      font-weight: 750;
      transition: transform 160ms ease, background 160ms ease, border 160ms ease;
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
    }

    .btn:hover {
      transform: translateY(-2px);
    }

    .btn-primary {
      border: 0;
      color: white;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
    }

    .btn-secondary {
      background: rgba(255, 255, 255, 0.08);
      color: var(--text);
    }

    .btn-snap {
      border: 0;
      color: #07120d;
      background: linear-gradient(135deg, #82bea0, #b8ffd8);
    }

    .badges {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 28px;
      align-items: center;
    }

    .badges img {
      height: 20px;
      width: auto;
    }

    .hero-card {
      padding: 18px;
      border-radius: var(--radius);
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.09), rgba(255, 255, 255, 0.04)),
        rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .terminal {
      border-radius: 18px;
      background: #070b17;
      border: 1px solid rgba(255, 255, 255, 0.10);
      overflow: hidden;
    }

    .terminal-bar {
      height: 40px;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 0 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.09);
      background: rgba(255, 255, 255, 0.04);
    }

    .dot {
      width: 11px;
      height: 11px;
      border-radius: 50%;
      background: var(--danger);
    }

    .dot:nth-child(2) {
      background: var(--warning);
    }

    .dot:nth-child(3) {
      background: var(--ok);
    }

    .terminal-body {
      padding: 18px;
      font-family: var(--mono);
      font-size: 0.9rem;
      color: #bde7ff;
      min-height: 330px;
    }

    .terminal-body span {
      color: #8effc1;
    }

    .terminal-body .comment {
      color: #7e8daf;
    }

    .visual-box {
      margin-top: 16px;
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
    }

    .metric {
      padding: 16px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.07);
      border: 1px solid var(--border);
    }

    .metric strong {
      display: block;
      font-size: 1.42rem;
      line-height: 1;
      margin-bottom: 8px;
    }

    .metric span {
      color: var(--muted);
      font-size: 0.9rem;
    }

    section {
      padding: 58px 0;
    }

    .section-head {
      max-width: 820px;
      margin-bottom: 26px;
    }

    .kicker {
      color: var(--accent-2);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 800;
      font-size: 0.82rem;
      margin-bottom: 10px;
    }

    h2 {
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.2rem);
      line-height: 1.06;
      letter-spacing: -0.055em;
    }

    .section-head p {
      color: var(--muted);
      margin: 16px 0 0;
      font-size: 1.06rem;
    }

    .grid {
      display: grid;
      gap: 18px;
    }

    .grid-3 {
      grid-template-columns: repeat(3, 1fr);
    }

    .grid-2 {
      grid-template-columns: repeat(2, 1fr);
    }

    .card {
      border-radius: var(--radius);
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.075), rgba(255, 255, 255, 0.035)),
        var(--bg-card-2);
      border: 1px solid var(--border);
      padding: 24px;
      box-shadow: 0 16px 42px rgba(0, 0, 0, 0.20);
    }

    .card h3 {
      margin: 0 0 10px;
      font-size: 1.25rem;
      letter-spacing: -0.025em;
    }

    .card p {
      margin: 0;
      color: var(--muted);
    }

    .icon {
      width: 42px;
      height: 42px;
      border-radius: 14px;
      display: grid;
      place-items: center;
      margin-bottom: 16px;
      background: rgba(124, 92, 255, 0.16);
      border: 1px solid rgba(124, 92, 255, 0.38);
      font-weight: 900;
      color: #dcd5ff;
    }

    .download-card {
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      min-height: 260px;
    }

    .download-card .platform {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 7px 10px;
      color: var(--muted);
      font-size: 0.88rem;
      margin-bottom: 14px;
    }

    .download-card .download-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }

    .mini-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
      padding: 9px 12px;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--border);
      font-weight: 700;
      color: var(--text);
    }

    .mini-btn:hover {
      background: rgba(255, 255, 255, 0.13);
    }

    .snap-button {
      margin-top: 16px;
      max-width: 220px;
    }

    pre {
      margin: 16px 0 0;
      padding: 16px;
      overflow-x: auto;
      border-radius: 16px;
      background: #070b17;
      border: 1px solid rgba(255, 255, 255, 0.10);
      color: #d7ecff;
      font-family: var(--mono);
      font-size: 0.92rem;
    }

    code {
      font-family: var(--mono);
    }

    .feature-list {
      display: grid;
      gap: 10px;
      margin-top: 16px;
      padding: 0;
      list-style: none;
    }

    .feature-list li {
      display: flex;
      align-items: flex-start;
      gap: 10px;
      color: var(--muted);
    }

    .feature-list li::before {
      content: "✓";
      color: var(--ok);
      font-weight: 900;
    }

    .screenshots {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 18px;
    }

    .screenshot {
      border-radius: 20px;
      overflow: hidden;
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.06);
      box-shadow: 0 16px 42px rgba(0, 0, 0, 0.24);
    }

    .screenshot img {
      width: 100%;
      height: auto;
    }

    .screenshot figcaption {
      padding: 13px 16px;
      color: var(--muted);
      font-size: 0.92rem;
      border-top: 1px solid var(--border);
    }

    .timeline {
      display: grid;
      gap: 14px;
    }

    .step {
      display: grid;
      grid-template-columns: 48px 1fr;
      gap: 16px;
      align-items: start;
    }

    .step-no {
      width: 48px;
      height: 48px;
      display: grid;
      place-items: center;
      border-radius: 16px;
      background: linear-gradient(135deg, rgba(124, 92, 255, 0.85), rgba(0, 208, 255, 0.75));
      font-weight: 900;
    }

    .step-body {
      padding: 20px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.055);
      border: 1px solid var(--border);
    }

    .step-body h3 {
      margin: 0 0 6px;
    }

    .step-body p {
      margin: 0;
      color: var(--muted);
    }

    .note {
      border-left: 4px solid var(--accent-3);
      background: rgba(130, 190, 160, 0.11);
      border-radius: 16px;
      padding: 18px 20px;
      color: var(--muted);
    }

    .models {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
    }

    .model-pill {
      padding: 18px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border);
    }

    .model-pill strong {
      display: block;
      margin-bottom: 8px;
    }

    .model-pill span {
      color: var(--muted);
      font-size: 0.94rem;
    }

    .faq {
      display: grid;
      gap: 14px;
    }

    details {
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.055);
      border: 1px solid var(--border);
      overflow: hidden;
    }

    summary {
      cursor: pointer;
      padding: 18px 20px;
      font-weight: 800;
    }

    details p {
      margin: 0;
      padding: 0 20px 20px;
      color: var(--muted);
    }

    .cta {
      padding: 44px;
      border-radius: 28px;
      background:
        linear-gradient(135deg, rgba(124, 92, 255, 0.55), rgba(0, 208, 255, 0.32)),
        rgba(255, 255, 255, 0.07);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      text-align: center;
    }

    .cta h2 {
      margin-bottom: 14px;
    }

    .cta p {
      color: rgba(255, 255, 255, 0.82);
      max-width: 780px;
      margin: 0 auto 24px;
    }

    footer {
      padding: 36px 0 50px;
      color: var(--muted-2);
      border-top: 1px solid var(--border);
    }

    .footer-grid {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 24px;
      align-items: center;
    }

    .footer-links {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      justify-content: flex-end;
    }

    .footer-links a:hover {
      color: var(--text);
    }

    @media (max-width: 980px) {
      .hero-grid,
      .grid-3,
      .grid-2,
      .models,
      .screenshots {
        grid-template-columns: 1fr;
      }

      .hero {
        padding-top: 46px;
      }

      .nav-inner {
        align-items: flex-start;
        flex-direction: column;
        padding: 16px 0;
      }

      .nav-links {
        justify-content: flex-start;
      }

      .footer-grid {
        grid-template-columns: 1fr;
      }

      .footer-links {
        justify-content: flex-start;
      }
    }

    @media (max-width: 620px) {
      .wrap {
        width: min(100% - 26px, var(--max));
      }

      .hero-actions,
      .download-card .download-actions {
        flex-direction: column;
      }

      .btn,
      .mini-btn {
        width: 100%;
      }

      .visual-box {
        grid-template-columns: 1fr;
      }

      .cta {
        padding: 28px 18px;
      }
    }
  </style>
</head>

<body>
  <header class="nav">
    <div class="wrap nav-inner">
      <a class="brand" href="#top" aria-label="Mustatil home">
        <span class="brand-mark" aria-hidden="true"></span>
        <span>Mustatil</span>
      </a>

      <nav class="nav-links" aria-label="Main navigation">
        <a href="#features">Features</a>
        <a href="#models">AI Models</a>
        <a href="#downloads">Downloads</a>
        <a href="#install">Install</a>
        <a href="#screenshots">Screenshots</a>
        <a href="#faq">FAQ</a>
      </nav>
    </div>
  </header>

  <main id="top">
    <section class="hero">
      <div class="wrap hero-grid">
        <div>
          <div class="eyebrow">
            <span class="eyebrow-dot"></span>
            GIS-level AI vision workspace for desktop workflows
          </div>

          <h1>
            <span class="gradient-text">Mustatil</span><br />
            AI Detection for Large Images and Maps
          </h1>

          <p class="lead">
            Mustatil is an integrated GIS-level AI vision workspace for annotation, YOLO training,
            large-scale detection, satellite-map analysis, GeoPackage export, and visual pipeline building.
            It combines dataset creation, model training, geospatial inference, map-based review, and
            graphical AI workflows in one desktop application.
          </p>

          <div class="hero-actions">
            <a class="btn btn-primary" href="https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-/releases/download/Mustatil-5.6/Mustatil_5.6_Setup.exe">
              Download Windows Installer
            </a>
            <a class="btn btn-snap" href="https://snapcraft.io/mustatil">
              Get it from the Snap Store
            </a>
            <a class="btn btn-secondary" href="https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-">
              GitHub Repository
            </a>
          </div>

          <div class="badges" aria-label="Project badges">
            <a href="https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-">
              <img src="https://img.shields.io/badge/GitHub-Repository-black" alt="GitHub Repository" />
            </a>
            <a href="https://pypi.org/project/mustatil/">
              <img src="https://img.shields.io/pypi/v/mustatil?label=PyPI" alt="PyPI version" />
            </a>
            <a href="https://tarekwasfy01.itch.io/mustatil-qt-workspace">
              <img src="https://img.shields.io/badge/Download-itch.io-red" alt="Download on itch.io" />
            </a>
            <a href="https://snapcraft.io/mustatil">
              <img src="https://img.shields.io/badge/Snap%20Store-mustatil-82BEA0?style=flat&logo=snapcraft" alt="Get Mustatil from the Snap Store" />
            </a>
            <a href="https://doi.org/10.5281/zenodo.20481110">
              <img src="https://img.shields.io/badge/Zenodo-DOI%2010.5281%2Fzenodo.20481110-blue?style=flat&logo=zenodo" alt="Zenodo DOI" />
            </a>
            <a href="https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-/blob/main/LICENSE">
              <img src="https://img.shields.io/badge/License-LGPL%20v3-blue" alt="License LGPL v3" />
            </a>
            <a href="https://www.python.org/downloads/">
              <img src="https://img.shields.io/badge/Python-3.11%2B-brightgreen" alt="Python 3.11+" />
            </a>
            <a href="https://mustatil.de/">
              <img src="https://img.shields.io/badge/Website-mustatil.de-orange" alt="Mustatil Website" />
            </a>
            <a href="https://mustatil-ai.com/">
              <img src="https://img.shields.io/badge/Website-mustatil--ai.com-orange" alt="Mustatil AI Website" />
            </a>
            <a href="#features">
              <img src="https://img.shields.io/badge/Features-YOLO%20Training%20%7C%20GIS%20Detection%20%7C%20AI%20Pipeline-purple" alt="Features" />
            </a>
          </div>
        </div>

        <aside class="hero-card" aria-label="Installation preview">
          <div class="terminal">
            <div class="terminal-bar">
              <span class="dot"></span>
              <span class="dot"></span>
              <span class="dot"></span>
            </div>
            <div class="terminal-body">
              <div><span>$</span> sudo snap install mustatil</div>
              <br />
              <div class="comment"># or install with pip</div>
              <div><span>$</span> py -m pip install mustatil</div>
              <div><span>$</span> mustatil</div>
              <br />
              <div class="comment"># desktop workflow</div>
              <div>1. Annotate objects</div>
              <div>2. Train YOLO models</div>
              <div>3. Detect across large raster areas</div>
              <div>4. Export GeoPackage / GeoJSON</div>
              <div>5. Review results in GIS software</div>
              <br />
              <div class="comment"># designed for large images, satellite maps, and GIS-level AI work</div>
            </div>
          </div>

          <div class="visual-box">
            <div class="metric">
              <strong>YOLO</strong>
              <span>Training and detection workflow</span>
            </div>
            <div class="metric">
              <strong>GIS</strong>
              <span>GeoPackage and geospatial export</span>
            </div>
            <div class="metric">
              <strong>Maps</strong>
              <span>Satellite-map based review</span>
            </div>
            <div class="metric">
              <strong>AI</strong>
              <span>Visual detection pipelines</span>
            </div>
          </div>
        </aside>
      </div>
    </section>

    <section id="about">
      <div class="wrap">
        <div class="section-head">
          <div class="kicker">About</div>
          <h2>One workspace for annotation, training, detection, and map-based review.</h2>
          <p>
            Mustatil was created for workflows where normal computer-vision tools become too small:
            very large rasters, satellite imagery, map areas, archaeological survey regions, remote-sensing
            datasets, and detection outputs that need to be used in GIS environments.
          </p>
        </div>

        <div class="grid grid-2">
          <article class="card">
            <div class="icon">AI</div>
            <h3>AI Vision Workspace</h3>
            <p>
              Build datasets, train YOLO models, run object detection, review detections, and export results.
              Mustatil provides a practical desktop workflow instead of splitting the process across many tools.
            </p>
          </article>

          <article class="card">
            <div class="icon">GIS</div>
            <h3>Geospatial Detection</h3>
            <p>
              Mustatil is designed for GIS-level object detection, including large-image workflows,
              satellite-map analysis, geospatial outputs, and GeoPackage or GeoJSON export for downstream use.
            </p>
          </article>
        </div>
      </div>
    </section>

    <section id="features">
      <div class="wrap">
        <div class="section-head">
          <div class="kicker">Features</div>
          <h2>Core functionality</h2>
          <p>
            Mustatil focuses on the full AI vision lifecycle: annotation, model training, detection,
            map review, pipeline construction, and GIS-compatible export.
          </p>
        </div>

        <div class="grid grid-3">
          <article class="card">
            <div class="icon">01</div>
            <h3>Annotation</h3>
            <p>
              Create training annotations with rectangular boxes and class labels for object-detection datasets.
            </p>
            <ul class="feature-list">
              <li>Positive and negative samples</li>
              <li>Class management</li>
              <li>Dataset creation workflow</li>
            </ul>
          </article>

          <article class="card">
            <div class="icon">02</div>
            <h3>YOLO Training</h3>
            <p>
              Train YOLO models directly from prepared datasets using a desktop interface.
            </p>
            <ul class="feature-list">
              <li>YOLO training setup</li>
              <li>Dataset YAML support</li>
              <li>Training logs and model workflow</li>
            </ul>
          </article>

          <article class="card">
            <div class="icon">03</div>
            <h3>Large-Scale Detection</h3>
            <p>
              Run object detection on large images and geospatial raster workflows.
            </p>
            <ul class="feature-list">
              <li>Large-image detection</li>
              <li>Tiled processing workflow</li>
              <li>Map-based review</li>
            </ul>
          </article>

          <article class="card">
            <div class="icon">04</div>
            <h3>Satellite-Map Analysis</h3>
            <p>
              Use satellite-map based workflows for visual selection, detection, and review.
            </p>
            <ul class="feature-list">
              <li>Map preview workflow</li>
              <li>Area-based processing</li>
              <li>Object detection over map regions</li>
            </ul>
          </article>

          <article class="card">
            <div class="icon">05</div>
            <h3>GIS Export</h3>
            <p>
              Export detections into GIS-friendly vector formats for further analysis in QGIS and similar tools.
            </p>
            <ul class="feature-list">
              <li>GeoPackage export</li>
              <li>GeoJSON export</li>
              <li>QGIS-compatible result workflow</li>
            </ul>
          </article>

          <article class="card">
            <div class="icon">06</div>
            <h3>Visual AI Pipeline</h3>
            <p>
              Build graphical AI pipelines for detection logic, review, and model-based workflows.
            </p>
            <ul class="feature-list">
              <li>Pipeline blocks</li>
              <li>Detection rules</li>
              <li>Exportable results</li>
            </ul>
          </article>
        </div>
      </div>
    </section>

    <section id="models">
      <div class="wrap">
        <div class="section-head">
          <div class="kicker">AI Models</div>
          <h2>Beyond standard YOLO detection</h2>
          <p>
            Mustatil also includes experimental support for additional AI vision models beyond standard YOLO,
            extending it from a YOLO GIS workspace into a broader detection and training environment.
          </p>
        </div>

        <div class="models">
          <div class="model-pill">
            <strong>YOLO</strong>
            <span>Standard object detection, model training, and tiled inference workflows.</span>
          </div>
          <div class="model-pill">
            <strong>OWL-ViT / OWLv2</strong>
            <span>Open-vocabulary object detection from text prompts.</span>
          </div>
          <div class="model-pill">
            <strong>Grounding DINO</strong>
            <span>Text-guided detection for flexible object search.</span>
          </div>
          <div class="model-pill">
            <strong>LAE-DINO</strong>
            <span>DINO-based workflow with project-based dataset creation and training support.</span>
          </div>
        </div>

        <div class="note" style="margin-top: 22px;">
          Some non-YOLO model integrations are experimental and may require additional dependencies,
          model downloads, or hardware-specific configuration.
        </div>
      </div>
    </section>

    <section id="downloads">
      <div class="wrap">
        <div class="section-head">
          <div class="kicker">Downloads</div>
          <h2>Install Mustatil on Windows, macOS, Linux, Snap, PyPI, or Conda.</h2>
          <p>
            The Windows installer downloads Python and all dependencies automatically, then starts the GUI.
            The first start can take some time and is intended for users who are not familiar with Python.
          </p>
        </div>

        <div class="grid grid-3">
          <article class="card download-card">
            <div>
              <div class="platform">Windows</div>
              <h3>Mustatil 5.6 Windows Installer</h3>
              <p>
                Recommended for Windows users who want a regular installer and launcher workflow.
              </p>
            </div>
            <div class="download-actions">
              <a class="mini-btn" href="https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-/releases/download/Mustatil-5.6/Mustatil_5.6_Setup.exe">
                Download .exe
              </a>
            </div>
          </article>

          <article class="card download-card">
            <div>
              <div class="platform">macOS</div>
              <h3>Mustatil 5.6 macOS Installer</h3>
              <p>
                Installer package for Apple macOS workflows.
              </p>
            </div>
            <div class="download-actions">
              <a class="mini-btn" href="https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-/releases/download/Mustatil-5.6/Mustatil_5.6_macOS.pkg">
                Download .pkg
              </a>
            </div>
          </article>

          <article class="card download-card">
            <div>
              <div class="platform">Linux DEB</div>
              <h3>Mustatil 5.6 Linux Installer</h3>
              <p>
                Debian package for compatible Linux distributions.
              </p>
            </div>
            <div class="download-actions">
              <a class="mini-btn" href="https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-/releases/download/Mustatil-5.6/Mustatil_5.6_Linux.deb">
                Download .deb
              </a>
            </div>
          </article>

          <article class="card download-card">
            <div>
              <div class="platform">Snap Store</div>
              <h3>Mustatil on Snapcraft</h3>
              <p>
                Install the Linux Snap package directly from the Snap Store.
              </p>
              <a class="snap-button" href="https://snapcraft.io/mustatil">
                <img alt="Get it from the Snap Store" src="https://snapcraft.io/en/dark/install.svg" />
              </a>
            </div>
            <pre><code>sudo snap install mustatil</code></pre>
          </article>

          <article class="card download-card">
            <div>
              <div class="platform">PyPI</div>
              <h3>Python Package</h3>
              <p>
                Install Mustatil as a Python package on supported systems.
              </p>
            </div>
            <pre><code>py -m pip install mustatil
mustatil</code></pre>
          </article>

          <article class="card download-card">
            <div>
              <div class="platform">Conda</div>
              <h3>Conda Package</h3>
              <p>
                Install Mustatil through the Conda package channel.
              </p>
            </div>
            <pre><code>conda install mustatil::mustatil</code></pre>
          </article>
        </div>
      </div>
    </section>

    <section id="install">
      <div class="wrap">
        <div class="section-head">
          <div class="kicker">Install</div>
          <h2>Installation commands</h2>
          <p>
            Choose the installation path that fits your system. The installer version is best for non-Python users,
            while PyPI and Conda are useful for users who already work with Python environments.
          </p>
        </div>

        <div class="grid grid-2">
          <article class="card">
            <h3>Snap</h3>
            <p>Install from the Snap Store:</p>
            <pre><code>sudo snap install mustatil</code></pre>
          </article>

          <article class="card">
            <h3>PyPI</h3>
            <p>Install and run with Python:</p>
            <pre><code>py -m pip install mustatil
mustatil</code></pre>
          </article>

          <article class="card">
            <h3>Conda</h3>
            <p>Install through Conda:</p>
            <pre><code>conda install mustatil::mustatil</code></pre>
          </article>

          <article class="card">
            <h3>Windows Installer</h3>
            <p>Download and run the setup executable:</p>
            <pre><code>Mustatil_5.6_Setup.exe</code></pre>
          </article>
        </div>
      </div>
    </section>

    <section id="workflow">
      <div class="wrap">
        <div class="section-head">
          <div class="kicker">Workflow</div>
          <h2>Typical Mustatil workflow</h2>
          <p>
            Mustatil is designed around a practical object-detection workflow from data creation to GIS export.
          </p>
        </div>

        <div class="timeline">
          <div class="step">
            <div class="step-no">1</div>
            <div class="step-body">
              <h3>Create or load data</h3>
              <p>Load images, large rasters, satellite-map regions, or project datasets for detection and annotation.</p>
            </div>
          </div>

          <div class="step">
            <div class="step-no">2</div>
            <div class="step-body">
              <h3>Annotate objects</h3>
              <p>Create bounding boxes, define classes, add positive examples, and prepare a dataset for training.</p>
            </div>
          </div>

          <div class="step">
            <div class="step-no">3</div>
            <div class="step-body">
              <h3>Train or load a model</h3>
              <p>Train a YOLO model inside Mustatil or load an existing model for inference.</p>
            </div>
          </div>

          <div class="step">
            <div class="step-no">4</div>
            <div class="step-body">
              <h3>Run detection</h3>
              <p>Detect objects in large images, tiled raster data, or map-based satellite regions.</p>
            </div>
          </div>

          <div class="step">
            <div class="step-no">5</div>
            <div class="step-body">
              <h3>Export GIS results</h3>
              <p>Export detections as GeoPackage or GeoJSON for QGIS and other geospatial workflows.</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section id="screenshots">
      <div class="wrap">
        <div class="section-head">
          <div class="kicker">Screenshots</div>
          <h2>Interface examples</h2>
          <p>
            Screenshots of Mustatil showing detection, training, annotation, satellite analysis, and pipeline workflows.
          </p>
        </div>

        <div class="screenshots">
          <figure class="screenshot">
            <img width="1280" height="720" alt="Mustatil screenshot 2026-05-29 182916" src="https://github.com/user-attachments/assets/e869cd6a-3c36-42e1-b587-65dfc3edcbbb" />
            <figcaption>Mustatil desktop interface</figcaption>
          </figure>

          <figure class="screenshot">
            <img width="1280" height="720" alt="Mustatil screenshot 2026-05-31 005311" src="https://github.com/user-attachments/assets/fe9d1958-78fb-498c-96bd-636aae9ed342" />
            <figcaption>Detection and review workflow</figcaption>
          </figure>

          <figure class="screenshot">
            <img width="1280" height="720" alt="Mustatil screenshot 2026-05-29 183015" src="https://github.com/user-attachments/assets/7e9099d2-20c1-43e7-b5d9-fb0f05e61ff9" />
            <figcaption>AI workflow interface</figcaption>
          </figure>

          <figure class="screenshot">
            <img width="1280" height="720" alt="Mustatil screenshot 2026-05-29 182940" src="https://github.com/user-attachments/assets/03291052-c523-4aee-8007-d5c964e9115b" />
            <figcaption>Model and geospatial workflow</figcaption>
          </figure>
        </div>
      </div>
    </section>

    <section id="links">
      <div class="wrap">
        <div class="section-head">
          <div class="kicker">Links</div>
          <h2>Project links</h2>
          <p>
            Main project pages, downloads, packages, and identifiers.
          </p>
        </div>

        <div class="grid grid-2">
          <article class="card">
            <h3>Websites</h3>
            <ul class="feature-list">
              <li><a href="https://mustatil.de">mustatil.de</a></li>
              <li><a href="https://mustatil-ai.com">mustatil-ai.com</a></li>
            </ul>
          </article>

          <article class="card">
            <h3>Repositories and stores</h3>
            <ul class="feature-list">
              <li><a href="https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-">GitHub Repository</a></li>
              <li><a href="https://tarekwasfy01.itch.io/mustatil-qt-workspace">itch.io Download Page</a></li>
              <li><a href="https://pypi.org/project/mustatil/">PyPI Package</a></li>
              <li><a href="https://snapcraft.io/mustatil">Snap Store</a></li>
            </ul>
          </article>

          <article class="card">
            <h3>DOI</h3>
            <p>
              Mustatil Zenodo DOI:
              <br />
              <a href="https://doi.org/10.5281/zenodo.20481110">https://doi.org/10.5281/zenodo.20481110</a>
            </p>
          </article>

          <article class="card">
            <h3>License</h3>
            <p>
              Mustatil is linked with an LGPL v3 license badge.
              <br />
              <a href="https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-/blob/main/LICENSE">View license on GitHub</a>
            </p>
          </article>
        </div>
      </div>
    </section>

    <section id="qgis">
      <div class="wrap">
        <div class="section-head">
          <div class="kicker">QGIS</div>
          <h2>GeoPackage and QGIS workflow</h2>
          <p>
            Mustatil exports GIS-compatible detection results. Additionally, there is a GeoPackage converter
            for QGIS if there is a problem with exported files. Normally, you can change the EPSG for a layer in QGIS.
          </p>
        </div>

        <div class="card">
          <h3>GIS notes</h3>
          <ul class="feature-list">
            <li>Export detection boxes into geospatial vector formats.</li>
            <li>Open GeoPackage or GeoJSON outputs in QGIS.</li>
            <li>Adjust layer EPSG settings inside QGIS when needed.</li>
            <li>Use exported detections for archaeological survey, mapping, and remote-sensing review.</li>
          </ul>
        </div>
      </div>
    </section>

    <section id="faq">
      <div class="wrap">
        <div class="section-head">
          <div class="kicker">FAQ</div>
          <h2>Frequently asked questions</h2>
        </div>

        <div class="faq">
          <details>
            <summary>What does Mustatil mean?</summary>
            <p>
              Mustatil means rectangle — a reference to both archaeological mustatils and the rectangular
              detection boxes used in AI object detection.
            </p>
          </details>

          <details>
            <summary>Is Mustatil only a YOLO tool?</summary>
            <p>
              No. YOLO is the main training and detection workflow, but Mustatil also includes experimental
              support for additional model families such as OWL-ViT / OWLv2, Grounding DINO, and LAE-DINO.
            </p>
          </details>

          <details>
            <summary>Does the Windows installer require manual Python setup?</summary>
            <p>
              The installer executable downloads Python and all dependencies automatically, then starts the GUI.
              The first start can take some time.
            </p>
          </details>

          <details>
            <summary>Can I use Mustatil with QGIS?</summary>
            <p>
              Yes. Mustatil is designed around GIS-compatible outputs such as GeoPackage and GeoJSON,
              which can be used in QGIS and similar GIS software.
            </p>
          </details>

          <details>
            <summary>Was the program written using AI?</summary>
            <p>
              Yes. The program was written using AI-assisted development.
            </p>
          </details>
        </div>
      </div>
    </section>

    <section>
      <div class="wrap">
        <div class="cta">
          <h2>Download Mustatil</h2>
          <p>
            Use Mustatil for annotation, training, detection, satellite-map analysis, GIS export,
            and graphical AI pipeline workflows.
          </p>

          <div class="hero-actions" style="justify-content: center;">
            <a class="btn btn-primary" href="https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-/releases/download/Mustatil-5.6/Mustatil_5.6_Setup.exe">
              Windows Installer
            </a>
            <a class="btn btn-snap" href="https://snapcraft.io/mustatil">
              Snap Store
            </a>
            <a class="btn btn-secondary" href="https://pypi.org/project/mustatil/">
              PyPI
            </a>
            <a class="btn btn-secondary" href="https://tarekwasfy01.itch.io/mustatil-qt-workspace">
              itch.io
            </a>
          </div>
        </div>
      </div>
    </section>
  </main>

  <footer>
    <div class="wrap footer-grid">
      <div>
        <strong>Mustatil</strong>
        <br />
        GIS-level AI vision workspace for annotation, YOLO training, large-scale detection,
        satellite-map analysis, and visual pipeline building.
      </div>

      <div class="footer-links">
        <a href="https://mustatil.de">mustatil.de</a>
        <a href="https://mustatil-ai.com">mustatil-ai.com</a>
        <a href="https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-">GitHub</a>
        <a href="https://snapcraft.io/mustatil">Snap Store</a>
        <a href="https://doi.org/10.5281/zenodo.20481110">Zenodo DOI</a>
      </div>
    </div>
  </footer>
</body>
</html>
