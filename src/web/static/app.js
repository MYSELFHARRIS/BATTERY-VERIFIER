// Professional industrial control-room UI rendering and polling logic.
// Shared across all pages. Strictly uses textContent and DOM methods (no innerHTML).

const SVG_NS = "http://www.w3.org/2000/svg";

let wasRunning = false;
let currentFilter = "ALL";
let cachedEvents = [];

function createSvgElement(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, v);
  }
  return el;
}

function createCheckIcon(size = 14) {
  const svg = createSvgElement("svg", {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    "stroke-width": "2.5",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
  });
  const poly = createSvgElement("polyline", { points: "20 6 9 17 4 12" });
  svg.appendChild(poly);
  return svg;
}

function createAlertIcon(size = 14) {
  const svg = createSvgElement("svg", {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    "stroke-width": "2.5",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
  });
  const circle = createSvgElement("circle", { cx: "12", cy: "12", r: "10" });
  const l1 = createSvgElement("line", { x1: "12", y1: "8", x2: "12", y2: "12" });
  const l2 = createSvgElement("line", { x1: "12", y1: "16", x2: "12.01", y2: "16" });
  svg.appendChild(circle);
  svg.appendChild(l1);
  svg.appendChild(l2);
  return svg;
}

function createSkipIcon(size = 14) {
  const svg = createSvgElement("svg", {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    "stroke-width": "2.5",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
  });
  const p1 = createSvgElement("polygon", { points: "5 4 15 12 5 20 5 4" });
  const l1 = createSvgElement("line", { x1: "19", y1: "5", x2: "19", y2: "19" });
  svg.appendChild(p1);
  svg.appendChild(l1);
  return svg;
}

function createEmptyIcon(size = 40) {
  const svg = createSvgElement("svg", {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    "stroke-width": "1.5",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
    class: "events-empty-icon",
  });
  const r = createSvgElement("rect", { x: "3", y: "4", width: "18", height: "16", rx: "2" });
  const l = createSvgElement("line", { x1: "3", y1: "10", x2: "21", y2: "10" });
  svg.appendChild(r);
  svg.appendChild(l);
  return svg;
}

function createShieldIcon(size = 24) {
  const svg = createSvgElement("svg", {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    "stroke-width": "2",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
  });
  const p = createSvgElement("path", { d: "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" });
  svg.appendChild(p);
  return svg;
}

function renderEvents(eventsList) {
  const eventsEl = document.getElementById("events");
  if (!eventsEl) return;
  eventsEl.replaceChildren();

  const filtered = currentFilter === "ALL"
    ? eventsList
    : eventsList.filter((ev) => ev.type === currentFilter);

  if (filtered.length === 0) {
    const empty = document.createElement("div");
    empty.className = "events-empty";
    empty.appendChild(createEmptyIcon(40));
    const emptyText = document.createElement("span");
    emptyText.textContent = "No events yet";
    empty.appendChild(emptyText);
    eventsEl.appendChild(empty);
    return;
  }

  filtered.forEach((ev) => {
    const li = document.createElement("li");
    li.className = "event-row";

    const chip = document.createElement("div");
    chip.className = "event-chip";
    if (ev.type === "STEP_VERIFIED") {
      chip.classList.add("chip-verified");
      chip.appendChild(createCheckIcon(14));
    } else if (ev.type === "WRONG_ORIENTATION") {
      chip.classList.add("chip-wrong");
      chip.appendChild(createAlertIcon(14));
    } else if (ev.type === "SKIPPED_STEP") {
      chip.classList.add("chip-skipped");
      chip.appendChild(createSkipIcon(14));
    } else {
      chip.classList.add("chip-verified");
      chip.appendChild(createCheckIcon(14));
    }

    const details = document.createElement("div");
    details.className = "event-details";

    const msg = document.createElement("div");
    msg.className = "event-msg";
    msg.textContent = ev.message;

    const meta = document.createElement("div");
    meta.className = "event-meta";
    if (ev.step !== null && ev.step !== undefined) {
      const stepSpan = document.createElement("span");
      stepSpan.textContent = `Step ${ev.step}`;
      meta.appendChild(stepSpan);
    }
    if (ev.confidence !== undefined && ev.confidence !== null) {
      const confSpan = document.createElement("span");
      confSpan.textContent = `Conf: ${Number(ev.confidence).toFixed(2)}`;
      meta.appendChild(confSpan);
    }

    details.appendChild(msg);
    details.appendChild(meta);

    const timeEl = document.createElement("div");
    timeEl.className = "event-time";
    timeEl.textContent = ev.time;

    li.appendChild(chip);
    li.appendChild(details);
    li.appendChild(timeEl);

    eventsEl.appendChild(li);
  });
}

function updateUI(status) {
  if (!status) return;

  // Reconnect video feed if running switched from false to true
  if (status.running && !wasRunning) {
    const feed = document.getElementById("feed");
    if (feed) {
      feed.src = "/video_feed?t=" + Date.now();
    }
  }
  wasRunning = Boolean(status.running);

  // 1. Status Pill (present in top-bar on all pages)
  const pill = document.getElementById("status-pill");
  const pillText = document.getElementById("status-pill-text");
  if (pill && pillText) {
    if (status.error) {
      pill.className = "status-pill error";
      pillText.textContent = "Camera error";
    } else if (status.running) {
      pill.className = "status-pill live";
      pillText.textContent = "Live";
    } else {
      pill.className = "status-pill idle";
      pillText.textContent = "Idle";
    }
  }

  // 2. Camera Error Banner (on Dashboard page)
  const errorEl = document.getElementById("camera-error");
  if (errorEl) {
    if (status.error) {
      errorEl.textContent = status.error;
      errorEl.style.display = "block";
    } else {
      errorEl.textContent = "";
      errorEl.style.display = "none";
    }
  }

  // 3. Current Step Text
  const stepTextEl = document.getElementById("step-text");
  if (stepTextEl) {
    stepTextEl.textContent = status.step_text || "Ready";
  }

  // 4. Stepper
  const stepsEl = document.getElementById("steps");
  if (stepsEl) {
    stepsEl.replaceChildren();
    const stepsList = status.steps || [];
    stepsList.forEach((s) => {
      const li = document.createElement("li");
      li.className = `step-item state-${s.state}`;

      const badge = document.createElement("span");
      badge.className = "step-badge";
      if (s.state === "done") {
        badge.appendChild(createCheckIcon(12));
      } else {
        badge.textContent = String(s.id);
      }

      const name = document.createElement("span");
      name.className = "step-name";
      name.textContent = s.name;

      li.appendChild(badge);
      li.appendChild(name);
      stepsEl.appendChild(li);
    });
  }

  // 5. Results Timeline
  cachedEvents = status.events || [];
  renderEvents(cachedEvents);

  // 6. Insights & KPIs
  const counts = status.counts || {};
  const vEl = document.getElementById("insight-verified");
  if (vEl) vEl.textContent = String(counts.STEP_VERIFIED ?? 0);

  const wEl = document.getElementById("insight-wrong-orientation");
  if (wEl) wEl.textContent = String(counts.WRONG_ORIENTATION ?? 0);

  const sEl = document.getElementById("insight-skipped");
  if (sEl) sEl.textContent = String(counts.SKIPPED_STEP ?? 0);

  const stepsList = status.steps || [];
  const stepsDone = stepsList.filter((s) => s.state === "done").length;
  const totalSteps = stepsList.length || 3;
  const stepsDoneEl = document.getElementById("insight-steps-done");
  if (stepsDoneEl) stepsDoneEl.textContent = `${stepsDone} / ${totalSteps}`;

  const progressFill = document.getElementById("steps-progress-fill");
  const progressPercent = document.getElementById("progress-percent");
  const pct = Math.round((stepsDone / totalSteps) * 100);
  if (progressFill) progressFill.style.width = `${pct}%`;
  if (progressPercent) progressPercent.textContent = `${pct}%`;

  const msEl = document.getElementById("insight-ms-per-frame");
  if (msEl) msEl.textContent = String(status.ms_per_frame ?? 0);

  const fpsEl = document.getElementById("insight-fps");
  if (fpsEl) fpsEl.textContent = String(status.fps ?? 0);

  // 7. Integrity Card
  const chainEl = document.getElementById("insight-chain");
  const integrityCard = document.getElementById("integrity-card");
  const integrityIconBox = document.getElementById("integrity-icon-box");

  if (chainEl && integrityCard) {
    integrityCard.classList.remove("chain-ok", "chain-broken");
    chainEl.classList.remove("status-ok", "status-broken", "status-pending");

    if (status.chain_ok === true) {
      chainEl.textContent = "CHAIN OK";
      chainEl.classList.add("status-ok");
      integrityCard.classList.add("chain-ok");
    } else if (status.chain_ok === false) {
      chainEl.textContent = "BROKEN";
      chainEl.classList.add("status-broken");
      integrityCard.classList.add("chain-broken");
    } else {
      chainEl.textContent = "not started";
      chainEl.classList.add("status-pending");
    }

    if (integrityIconBox) {
      integrityIconBox.replaceChildren();
      const shield = createShieldIcon(24);
      if (status.chain_ok === true) {
        shield.classList.add("status-ok");
      } else if (status.chain_ok === false) {
        shield.classList.add("status-broken");
      } else {
        shield.classList.add("status-pending");
      }
      integrityIconBox.appendChild(shield);
    }
  }
}

async function sendCommand(url) {
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "X-BV-Request": "1",
      },
    });
    if (res.ok) {
      const data = await res.json();
      if (url === "/api/start") {
        const feed = document.getElementById("feed");
        if (feed) {
          feed.src = "/video_feed?t=" + Date.now();
        }
      }
      updateUI(data);
    }
  } catch (err) {
    console.error("Command error:", err);
  }
}

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    if (res.ok) {
      const data = await res.json();
      updateUI(data);
    }
  } catch (err) {
    console.error("Status poll error:", err);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  // Start verification on Overview page
  const btnStartOverview = document.getElementById("btn-start-overview");
  if (btnStartOverview) {
    btnStartOverview.addEventListener("click", async () => {
      await sendCommand("/api/start");
      window.location.href = "/dashboard";
    });
  }

  // Dashboard buttons
  const btnStart = document.getElementById("btn-start");
  if (btnStart) {
    btnStart.addEventListener("click", () => sendCommand("/api/start"));
  }

  const btnStop = document.getElementById("btn-stop");
  if (btnStop) {
    btnStop.addEventListener("click", () => sendCommand("/api/stop"));
  }

  const btnReset = document.getElementById("btn-reset");
  if (btnReset) {
    btnReset.addEventListener("click", () => sendCommand("/api/reset"));
  }

  // Results filter buttons
  document.querySelectorAll(".filter-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentFilter = btn.getAttribute("data-filter") || "ALL";
      renderEvents(cachedEvents);
    });
  });

  // Poll immediately, then every 1000 ms
  pollStatus();
  setInterval(pollStatus, 1000);
});
