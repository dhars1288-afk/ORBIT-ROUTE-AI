/**
 * ORBITROUTE AI - Frontend Architecture & Interactive Canvas Engine
 * Real-time constellation visualization, ML inference, Q-Learning lab, Chart.js integrations
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- Application State ---
    const state = {
        currentTab: "overview",
        network: null,
        optimalRoute: null,
        dijkstraRoute: null,
        rlRoute: null,
        candidateRoutes: [],
        sourceSat: "SAT-01",
        destSat: "SAT-10",
        simulationStep: 1,
        autoOrbitInterval: null,
        isAutoOrbiting: false,
        selectedNode: null,
        hoveredNode: null,
        packetAnimationProgress: 0,
        charts: {}
    };

    // --- DOM Elements ---
    const canvas = document.getElementById("constellation-canvas");
    const ctx = canvas ? canvas.getContext("2d") : null;

    // --- Tab Switching ---
    const tabs = document.querySelectorAll(".nav-tab");
    const tabContents = document.querySelectorAll(".tab-content");

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const targetTab = tab.getAttribute("data-tab");
            switchTab(targetTab);
        });
    });

    function switchTab(tabId) {
        state.currentTab = tabId;
        tabs.forEach(t => {
            t.classList.toggle("active", t.getAttribute("data-tab") === tabId);
        });
        tabContents.forEach(c => {
            c.classList.toggle("active", c.id === `tab-${tabId}`);
        });

        // Resize canvas if switching to network tab
        if (tabId === "network" && canvas) {
            resizeCanvas();
        }

        // Render charts when opening models or benchmarks or RL tab
        if (tabId === "models") {
            renderFeatureImportanceChart();
        } else if (tabId === "rl") {
            renderRLRewardChart();
        } else if (tabId === "benchmarks") {
            renderBenchmarkCharts();
        } else if (tabId === "dataset") {
            loadDatasetPreview();
        }
    }

    // --- Navigation Quick Action Buttons in Hero ---
    const btnHeroSimulate = document.getElementById("btn-hero-simulate");
    if (btnHeroSimulate) {
        btnHeroSimulate.addEventListener("click", () => {
            switchTab("network");
            advanceSimulation();
        });
    }

    const btnHeroPredict = document.getElementById("btn-hero-predict");
    if (btnHeroPredict) {
        btnHeroPredict.addEventListener("click", () => {
            switchTab("prediction");
        });
    }

    const btnHeroDemo = document.getElementById("btn-hero-demo");
    if (btnHeroDemo) {
        btnHeroDemo.addEventListener("click", () => {
            runVivaDemo();
        });
    }

    // Clear logs button
    const btnClearLogs = document.getElementById("btn-clear-logs");
    if (btnClearLogs) {
        btnClearLogs.addEventListener("click", () => {
            const container = document.getElementById("telemetry-log-entries");
            if (container) container.innerHTML = "";
            appendLog("Telemetry log cleared.", "info");
        });
    }

    // --- Canvas Setup & High-DPI Scaling ---
    function resizeCanvas() {
        if (!canvas) return;
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }
    window.addEventListener("resize", resizeCanvas);

    // --- Dynamic Logging Utility ---
    function appendLog(message, type = "info") {
        const container = document.getElementById("telemetry-log-entries");
        if (!container) return;
        const now = new Date();
        const timeStr = `[${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}]`;
        
        const line = document.createElement("div");
        line.className = `log-line ${type}`;
        line.innerHTML = `<span class="log-time">${timeStr}</span> ${message}`;
        container.prepend(line);

        // Keep maximum 40 log lines
        while (container.children.length > 40) {
            container.removeChild(container.lastChild);
        }
    }

    // --- Canvas Notification Toast ---
    function showCanvasNotification(text) {
        const el = document.getElementById("canvas-notification");
        if (!el) return;
        el.textContent = text;
        el.classList.add("show");
        setTimeout(() => {
            el.classList.remove("show");
        }, 2400);
    }

    // --- API Calls ---

    // 1. Fetch System Status
    async function fetchSystemStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();
            document.getElementById("system-status-text").textContent = `${data.system_name} ${data.status}`;
            document.getElementById("header-epoch").textContent = String(data.simulation_step).padStart(3, '0');
            document.getElementById("kpi-model-acc").textContent = `${(data.model_accuracy * 100).toFixed(1)} %`;
        } catch (err) {
            console.error("Failed to fetch status:", err);
        }
    }

    // 2. Fetch Network Topology
    async function fetchNetworkTopology() {
        try {
            const res = await fetch("/api/network");
            state.network = await res.json();
            updateKPIs(state.network);
            if (!state.optimalRoute) {
                await computeOptimalRoute(state.sourceSat, state.destSat);
            }
        } catch (err) {
            console.error("Failed to load network:", err);
            appendLog("Network connection error.", "alert");
        }
    }

    // 3. Compute Optimal Route
    async function computeOptimalRoute(source, destination) {
        try {
            const res = await fetch("/api/optimal-route", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ source, destination })
            });
            const data = await res.json();

            if (data.error) {
                appendLog(`Routing error: ${data.error}`, "warning");
                return;
            }

            state.optimalRoute = data.optimal_route;
            state.dijkstraRoute = data.dijkstra_route;
            state.rlRoute = data.rl_route;
            state.candidateRoutes = data.all_candidates || [];

            updateRouteDisplays(data);
            appendLog(`Optimal Route: ${data.current_route} (Cost: ${data.optimal_route.routing_score})`, "success");

            if (data.route_changed) {
                appendLog(`Topology reroute detected: Switched from [${data.previous_route}] to [${data.current_route}]`, "alert");
                showCanvasNotification("Reroute Triggered: Channel Condition Shift");
            }
        } catch (err) {
            console.error("Error computing optimal route:", err);
        }
    }

    // 4. Advance Simulation
    async function advanceSimulation() {
        try {
            const res = await fetch("/api/simulate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    source: state.sourceSat,
                    destination: state.destSat
                })
            });
            const data = await res.json();
            state.simulationStep = data.simulation.step;
            state.network = data.network;
            
            document.getElementById("header-epoch").textContent = String(state.simulationStep).padStart(3, '0');
            updateKPIs(state.network);

            if (data.routing && !data.routing.error) {
                state.optimalRoute = data.routing.optimal_route;
                state.dijkstraRoute = data.routing.dijkstra_route;
                state.rlRoute = data.routing.rl_route;
                state.candidateRoutes = data.routing.all_candidates || [];
                updateRouteDisplays(data.routing);

                if (data.routing.route_changed) {
                    appendLog(`Dynamic reroute (Epoch ${state.simulationStep}): New path ${data.routing.current_route}`, "warning");
                    showCanvasNotification("Network Rerouted: Telemetry Shift");
                } else {
                    appendLog(`Epoch ${state.simulationStep}: Orbit advanced. Optimal path stable.`, "info");
                }
            }
        } catch (err) {
            console.error("Simulation advance failed:", err);
        }
    }

    // --- Update KPI Cards ---
    function updateKPIs(net) {
        if (!net) return;
        document.getElementById("kpi-satellites").textContent = `${net.satellites_count} LEO`;
        document.getElementById("kpi-links").textContent = `${net.links_count} Active`;
        document.getElementById("kpi-latency").textContent = `${net.avg_latency_ms} ms`;
        document.getElementById("kpi-loss").textContent = `${net.avg_packet_loss_percent} %`;
        document.getElementById("kpi-energy").textContent = `${net.avg_energy_usage_j} J`;
    }

    // --- Update Route Displays across Dashboard ---
    function updateRouteDisplays(routingData) {
        const opt = routingData.optimal_route;
        if (!opt) return;

        // Overview Quick Route Strip
        const quickStrip = document.getElementById("quick-route-strip");
        if (quickStrip && opt.path) {
            quickStrip.innerHTML = "";
            opt.path.forEach((node, i) => {
                const pill = document.createElement("div");
                pill.className = `node-pill ${i === 0 ? 'source' : (i === opt.path.length - 1 ? 'destination' : 'intermediate')}`;
                pill.textContent = node;
                quickStrip.appendChild(pill);

                if (i < opt.path.length - 1) {
                    const arr = document.createElement("div");
                    arr.className = "route-arrow";
                    arr.textContent = "➔";
                    quickStrip.appendChild(arr);
                }
            });
        }

        // Quick Stats
        const quickDelay = document.getElementById("quick-delay");
        const quickLoss = document.getElementById("quick-loss");
        const quickEnergy = document.getElementById("quick-energy");
        const quickHops = document.getElementById("quick-hops");
        const quickScore = document.getElementById("quick-score");
        const quickRec = document.getElementById("quick-recommendation");

        if (quickDelay) quickDelay.textContent = `${opt.latency_ms} ms`;
        if (quickLoss) quickLoss.textContent = `${opt.packet_loss_percent} %`;
        if (quickEnergy) quickEnergy.textContent = `${opt.energy_consumption} J`;
        if (quickHops) quickHops.textContent = `${opt.number_of_hops}`;
        if (quickScore) quickScore.textContent = `${opt.routing_score}`;
        if (quickRec) quickRec.textContent = `"${opt.recommendation}"`;

        // Canvas bottom banner
        const canvasPath = document.getElementById("canvas-path-val");
        const canvasMetrics = document.getElementById("canvas-path-metrics");
        if (canvasPath) canvasPath.textContent = opt.path_string;
        if (canvasMetrics) {
            canvasMetrics.innerHTML = `
                <span>Delay: <strong>${opt.latency_ms}ms</strong></span>
                <span>Loss: <strong>${opt.packet_loss_percent}%</strong></span>
                <span>Energy: <strong>${opt.energy_consumption}J</strong></span>
                <span>Score: <strong>${opt.routing_score}</strong></span>
            `;
        }

        // Hero gauge percentage
        const heroQuality = document.getElementById("hero-route-quality");
        if (heroQuality) {
            heroQuality.textContent = `${opt.route_quality}%`;
        }

        // Populate Candidate Paths Table
        populateCandidatePathsTable(state.candidateRoutes);
    }

    // --- Candidate Paths Table ---
    function populateCandidatePathsTable(candidates) {
        const tbody = document.getElementById("candidate-routes-tbody");
        if (!tbody) return;
        tbody.innerHTML = "";

        if (!candidates || candidates.length === 0) {
            tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color:var(--text-muted);">No candidate paths calculated.</td></tr>`;
            return;
        }

        candidates.forEach((cand, idx) => {
            const tr = document.createElement("tr");
            const isWinner = idx === 0;
            if (isWinner) {
                tr.style.background = "rgba(0, 240, 255, 0.08)";
            }

            tr.innerHTML = `
                <td><strong>#${idx + 1}</strong> ${isWinner ? '👑' : ''}</td>
                <td><code style="color:${isWinner ? 'var(--neon-cyan)' : '#fff'}; font-family:var(--font-mono);">${cand.path_string}</code></td>
                <td>${cand.number_of_hops}</td>
                <td>${cand.latency_ms} ms</td>
                <td>${cand.packet_loss_percent} %</td>
                <td>${cand.energy_consumption} J</td>
                <td><strong style="color:var(--neon-green);">${cand.route_quality || '-'}</strong></td>
                <td><strong style="color:var(--neon-cyan);">${cand.routing_score}</strong></td>
                <td><span class="badge ${isWinner ? 'badge-success' : 'badge-info'}">${cand.status_label || (isWinner ? 'OPTIMAL' : 'SECONDARY')}</span></td>
            `;
            tbody.appendChild(tr);
        });
    }

    // --- Interactive Canvas Renderer ---
    function renderConstellation() {
        if (!ctx || !canvas) return;
        const width = canvas.parentElement.clientWidth;
        const height = canvas.parentElement.clientHeight;

        ctx.clearRect(0, 0, width, height);

        const centerX = width / 2;
        const centerY = height / 2;

        // 1. Draw Earth Globe in Center
        const earthRadius = 75;
        const earthGrad = ctx.createRadialGradient(centerX - 20, centerY - 20, 15, centerX, centerY, earthRadius);
        earthGrad.addColorStop(0, "#1e3a8a");
        earthGrad.addColorStop(0.65, "#0f172a");
        earthGrad.addColorStop(1, "#020617");

        ctx.beginPath();
        ctx.arc(centerX, centerY, earthRadius, 0, Math.PI * 2);
        ctx.fillStyle = earthGrad;
        ctx.fill();

        // Earth Atmospheric Glow
        ctx.beginPath();
        ctx.arc(centerX, centerY, earthRadius + 6, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(0, 240, 255, 0.35)";
        ctx.lineWidth = 3;
        ctx.stroke();

        // Earth Latitude / Longitude lines
        ctx.strokeStyle = "rgba(56, 189, 248, 0.15)";
        ctx.lineWidth = 1;
        for (let r = 25; r <= 60; r += 20) {
            ctx.beginPath();
            ctx.ellipse(centerX, centerY, earthRadius, r, 0, 0, Math.PI * 2);
            ctx.stroke();
        }

        // 2. Draw Orbital Planes
        const rInner = 175;
        const rOuter = 235;

        // Plane A (Cyan dashed)
        ctx.beginPath();
        ctx.ellipse(centerX, centerY, rInner, rInner * 0.9, Math.PI / 12, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(0, 240, 255, 0.22)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.stroke();

        // Plane B (Purple dashed)
        ctx.beginPath();
        ctx.ellipse(centerX, centerY, rOuter, rOuter * 0.9, -Math.PI / 10, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(139, 92, 246, 0.25)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 5]);
        ctx.stroke();
        ctx.setLineDash([]); // Reset line dash

        if (!state.network || !state.network.nodes) return;

        // Compute coordinate mapping to center canvas
        const nodeMap = {};
        state.network.nodes.forEach(n => {
            const rawX = n.x - 400; // Original center was 400, 300
            const rawY = n.y - 300;
            const mappedX = centerX + rawX * (width / 880);
            const mappedY = centerY + rawY * (height / 640);
            nodeMap[n.id] = { ...n, mappedX, mappedY };
        });

        // 3. Draw Communication Links (Edges)
        const optimalPath = state.optimalRoute ? state.optimalRoute.path : [];

        if (state.network.edges) {
            state.network.edges.forEach(edge => {
                const u = nodeMap[edge.source];
                const v = nodeMap[edge.target];
                if (!u || !v) return;

                // Check if this edge is on the optimal path
                let isOptimalEdge = false;
                for (let i = 0; i < optimalPath.length - 1; i++) {
                    if ((optimalPath[i] === edge.source && optimalPath[i+1] === edge.target) ||
                        (optimalPath[i] === edge.target && optimalPath[i+1] === edge.source)) {
                        isOptimalEdge = true;
                        break;
                    }
                }

                ctx.beginPath();
                ctx.moveTo(u.mappedX, u.mappedY);
                ctx.lineTo(v.mappedX, v.mappedY);

                if (isOptimalEdge) {
                    // Highlighted Optimal Path Link
                    ctx.strokeStyle = "#10b981";
                    ctx.lineWidth = 4;
                    ctx.shadowColor = "#10b981";
                    ctx.shadowBlur = 12;
                    ctx.stroke();
                    ctx.shadowBlur = 0; // reset
                } else if (edge.status === "high_latency") {
                    ctx.strokeStyle = "rgba(244, 63, 94, 0.45)";
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                } else if (edge.status === "weak") {
                    ctx.strokeStyle = "rgba(245, 158, 11, 0.4)";
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                } else {
                    ctx.strokeStyle = "rgba(0, 240, 255, 0.18)";
                    ctx.lineWidth = 1.2;
                    ctx.stroke();
                }
            });
        }

        // 4. Animate Data Packet Pulses along Optimal Path
        if (optimalPath.length >= 2) {
            state.packetAnimationProgress = (state.packetAnimationProgress + 0.012) % 1.0;
            const totalHops = optimalPath.length - 1;
            const hopIndex = Math.min(Math.floor(state.packetAnimationProgress * totalHops), totalHops - 1);
            const hopProgress = (state.packetAnimationProgress * totalHops) - hopIndex;

            const uNode = nodeMap[optimalPath[hopIndex]];
            const vNode = nodeMap[optimalPath[hopIndex + 1]];

            if (uNode && vNode) {
                const packetX = uNode.mappedX + (vNode.mappedX - uNode.mappedX) * hopProgress;
                const packetY = uNode.mappedY + (vNode.mappedY - uNode.mappedY) * hopProgress;

                ctx.beginPath();
                ctx.arc(packetX, packetY, 5, 0, Math.PI * 2);
                ctx.fillStyle = "#ffffff";
                ctx.shadowColor = "#00f0ff";
                ctx.shadowBlur = 14;
                ctx.fill();
                ctx.shadowBlur = 0;
            }
        }

        // 5. Draw Satellites (Nodes)
        Object.values(nodeMap).forEach(node => {
            const isSource = node.id === state.sourceSat;
            const isDest = node.id === state.destSat;
            const isPath = optimalPath.includes(node.id);
            const isHovered = state.hoveredNode === node.id;

            // Halo Aura
            if (isSource || isDest || isPath || isHovered) {
                ctx.beginPath();
                ctx.arc(node.mappedX, node.mappedY, 18, 0, Math.PI * 2);
                ctx.fillStyle = isSource ? "rgba(0, 240, 255, 0.25)" : (isDest ? "rgba(16, 185, 129, 0.25)" : "rgba(139, 92, 246, 0.2)");
                ctx.fill();
            }

            // Node Circle
            ctx.beginPath();
            ctx.arc(node.mappedX, node.mappedY, 10, 0, Math.PI * 2);
            ctx.fillStyle = isSource ? "#00f0ff" : (isDest ? "#10b981" : (node.plane === "A" ? "#38bdf8" : "#a78bfa"));
            ctx.shadowColor = ctx.fillStyle;
            ctx.shadowBlur = 8;
            ctx.fill();
            ctx.shadowBlur = 0;

            // Inner Ring
            ctx.beginPath();
            ctx.arc(node.mappedX, node.mappedY, 4, 0, Math.PI * 2);
            ctx.fillStyle = "#030712";
            ctx.fill();

            // Label
            ctx.font = isHovered ? "bold 12px 'JetBrains Mono', monospace" : "11px 'JetBrains Mono', monospace";
            ctx.fillStyle = isHovered ? "#00f0ff" : "#f1f5f9";
            ctx.fillText(node.id, node.mappedX - 18, node.mappedY - 14);
        });
    }

    // Animation Loop
    function animationLoop() {
        renderConstellation();
        requestAnimationFrame(animationLoop);
    }

    // --- Canvas Mouse Interactivity ---
    if (canvas) {
        canvas.addEventListener("mousemove", (e) => {
            if (!state.network || !state.network.nodes) return;
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            const width = canvas.parentElement.clientWidth;
            const height = canvas.parentElement.clientHeight;
            const centerX = width / 2;
            const centerY = height / 2;

            let hovered = null;
            state.network.nodes.forEach(n => {
                const mappedX = centerX + (n.x - 400) * (width / 880);
                const mappedY = centerY + (n.y - 300) * (height / 640);
                const dist = Math.hypot(mouseX - mappedX, mouseY - mappedY);
                if (dist <= 20) {
                    hovered = n.id;
                    updateNodeInspector(n);
                }
            });

            state.hoveredNode = hovered;
            canvas.style.cursor = hovered ? "pointer" : "default";
        });

        canvas.addEventListener("click", () => {
            if (state.hoveredNode) {
                // If clicked node is not source, set as destination or switch
                if (state.sourceSat !== state.hoveredNode) {
                    state.destSat = state.hoveredNode;
                    document.getElementById("select-dest-sat").value = state.destSat;
                    computeOptimalRoute(state.sourceSat, state.destSat);
                } else {
                    showCanvasNotification(`Node ${state.hoveredNode} is currently Source.`);
                }
            }
        });
    }

    function updateNodeInspector(node) {
        document.getElementById("inspect-sat-id").textContent = node.id;
        document.getElementById("inspect-sat-load").textContent = `${node.cpu_load}%`;
        document.getElementById("inspect-sat-queue").textContent = `${node.queue_length} pkts`;
        document.getElementById("inspect-sat-batt").textContent = `${node.battery_percent}%`;
        document.getElementById("inspect-sat-alt").textContent = `${node.altitude_km} km`;
        document.getElementById("inspect-sat-vel").textContent = `${node.velocity_kms} km/s`;
        document.getElementById("inspect-sat-plane").textContent = `Plane ${node.plane}`;
    }

    // --- Constellation Controls ---
    const selectSourceSat = document.getElementById("select-source-sat");
    const selectDestSat = document.getElementById("select-dest-sat");
    const btnFindOptimalRoute = document.getElementById("btn-find-optimal-route");
    const btnAdvanceSim = document.getElementById("btn-advance-simulation");
    const btnToggleAutoOrbit = document.getElementById("btn-toggle-auto-orbit");

    if (selectSourceSat) {
        selectSourceSat.addEventListener("change", (e) => {
            state.sourceSat = e.target.value;
        });
    }

    if (selectDestSat) {
        selectDestSat.addEventListener("change", (e) => {
            state.destSat = e.target.value;
        });
    }

    if (btnFindOptimalRoute) {
        btnFindOptimalRoute.addEventListener("click", () => {
            computeOptimalRoute(state.sourceSat, state.destSat);
        });
    }

    if (btnAdvanceSim) {
        btnAdvanceSim.addEventListener("click", () => {
            advanceSimulation();
        });
    }

    if (btnToggleAutoOrbit) {
        btnToggleAutoOrbit.addEventListener("click", () => {
            state.isAutoOrbiting = !state.isAutoOrbiting;
            const icon = document.getElementById("auto-orbit-icon");
            const badge = document.getElementById("badge-topology-state");

            if (state.isAutoOrbiting) {
                btnToggleAutoOrbit.classList.add("btn-primary");
                btnToggleAutoOrbit.classList.remove("btn-outline");
                icon.textContent = "⏸";
                badge.textContent = "ORBIT IN MOTION";
                badge.className = "badge badge-success";
                appendLog("Continuous orbital dynamics simulation engaged (1.8s epochs).", "info");

                state.autoOrbitInterval = setInterval(() => {
                    advanceSimulation();
                }, 2200);
            } else {
                btnToggleAutoOrbit.classList.remove("btn-primary");
                btnToggleAutoOrbit.classList.add("btn-outline");
                icon.textContent = "▶";
                badge.textContent = "ORBIT STEADY";
                badge.className = "badge badge-info";
                clearInterval(state.autoOrbitInterval);
                appendLog("Auto-orbit paused.", "info");
            }
        });
    }

    // --- Route Prediction Form ---
    const predForm = document.getElementById("prediction-form");
    const btnFillSample = document.getElementById("btn-fill-sample-inputs");

    if (btnFillSample) {
        btnFillSample.addEventListener("click", () => {
            document.getElementById("pred-distance").value = "2150";
            document.getElementById("pred-bandwidth").value = "850";
            document.getElementById("pred-latency").value = "52.4";
            document.getElementById("pred-loss").value = "0.75";
            document.getElementById("pred-energy").value = "3.1";
            document.getElementById("pred-signal").value = "-54.0";
            document.getElementById("pred-load").value = "42";
            document.getElementById("pred-queue").value = "22";
            document.getElementById("pred-velocity").value = "2.8";
            document.getElementById("pred-stability").value = "0.89";
            document.getElementById("pred-quality").value = "82";
            document.getElementById("pred-hops").value = "3";
            appendLog("Preset link telemetry loaded into prediction form.", "info");
        });
    }

    if (predForm) {
        predForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const btnSubmit = document.getElementById("btn-submit-prediction");
            btnSubmit.disabled = true;
            btnSubmit.innerHTML = `<span class="btn-icon">⏳</span> EVALUATING...`;

            const payload = {
                source_satellite: document.getElementById("pred-source").value,
                destination_satellite: document.getElementById("pred-destination").value,
                distance_km: parseFloat(document.getElementById("pred-distance").value),
                bandwidth_mbps: parseFloat(document.getElementById("pred-bandwidth").value),
                latency_ms: parseFloat(document.getElementById("pred-latency").value),
                packet_loss_percent: parseFloat(document.getElementById("pred-loss").value),
                energy_consumption: parseFloat(document.getElementById("pred-energy").value),
                signal_strength: parseFloat(document.getElementById("pred-signal").value),
                satellite_load: parseFloat(document.getElementById("pred-load").value),
                queue_length: parseInt(document.getElementById("pred-queue").value),
                relative_velocity: parseFloat(document.getElementById("pred-velocity").value),
                connection_stability: parseFloat(document.getElementById("pred-stability").value),
                link_quality: parseFloat(document.getElementById("pred-quality").value),
                number_of_hops: parseInt(document.getElementById("pred-hops").value)
            };

            try {
                const res = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const pred = await res.json();
                updatePredictionResults(pred);
                appendLog(`Custom prediction evaluated: ${pred.status_label} (Optimal Prob: ${pred.optimal_probability}%)`, "success");
            } catch (err) {
                console.error("Prediction error:", err);
                appendLog("Prediction failed: " + err.message, "alert");
            } finally {
                btnSubmit.disabled = false;
                btnSubmit.innerHTML = `<span class="btn-icon">⚡</span> PREDICT ROUTE`;
            }
        });
    }

    function updatePredictionResults(pred) {
        document.getElementById("res-optimal-prob").textContent = `${pred.optimal_probability}%`;
        document.getElementById("res-status-title").textContent = `Route Viability: ${pred.status_label}`;
        document.getElementById("res-recommendation-text").textContent = pred.recommendation;

        const badge = document.getElementById("pred-badge-status");
        badge.textContent = pred.status_label;
        badge.className = `badge badge-${pred.badge_color}`;

        document.getElementById("res-delay").textContent = `${pred.predicted_delay} ms`;
        document.getElementById("res-loss").textContent = `${pred.predicted_packet_loss} %`;
        document.getElementById("res-energy").textContent = `${pred.energy_consumption} J`;
        document.getElementById("res-hops").textContent = pred.number_of_hops;
        document.getElementById("res-score").textContent = pred.routing_score;
        document.getElementById("res-quality").textContent = `${pred.route_quality} / 100`;

        // Update Stacked Cost Bar
        const bd = pred.cost_breakdown;
        const total = (bd.delay_penalty + bd.loss_penalty + bd.energy_penalty + bd.hops_penalty) || 1;
        document.getElementById("bar-delay").style.width = `${(bd.delay_penalty / total) * 100}%`;
        document.getElementById("bar-loss").style.width = `${(bd.loss_penalty / total) * 100}%`;
        document.getElementById("bar-energy").style.width = `${(bd.energy_penalty / total) * 100}%`;
        document.getElementById("bar-hops").style.width = `${(bd.hops_penalty / total) * 100}%`;
    }

    // --- Charts & Model Performance ---
    let performanceData = null;

    async function loadPerformanceData() {
        if (performanceData) return performanceData;
        try {
            const res = await fetch("/api/model-performance");
            performanceData = await res.json();
            updateModelMetricsView(performanceData.metrics);
            updateConfusionMatrix(performanceData.metrics.confusion_matrix);
            populateComparisonCards(performanceData.benchmarks.algorithms);
            return performanceData;
        } catch (err) {
            console.error("Failed to load model performance:", err);
        }
    }

    function updateModelMetricsView(m) {
        if (!m) return;
        document.getElementById("val-accuracy").textContent = `${(m.accuracy * 100).toFixed(2)}%`;
        document.getElementById("val-precision").textContent = `${(m.precision * 100).toFixed(2)}%`;
        document.getElementById("val-recall").textContent = `${(m.recall * 100).toFixed(2)}%`;
        document.getElementById("val-f1").textContent = `${(m.f1_score * 100).toFixed(2)}%`;
    }

    function updateConfusionMatrix(cm) {
        if (!cm || cm.length < 2) return;
        document.getElementById("cm-tn").textContent = `${cm[0][0]} (TN)`;
        document.getElementById("cm-fp").textContent = `${cm[0][1]} (FP)`;
        document.getElementById("cm-fn").textContent = `${cm[1][0]} (FN)`;
        document.getElementById("cm-tp").textContent = `${cm[1][1]} (TP)`;
    }

    async function renderFeatureImportanceChart() {
        const data = await loadPerformanceData();
        if (!data || !data.metrics || !data.metrics.feature_importance) return;

        const fi = data.metrics.feature_importance;
        const labels = Object.keys(fi).map(k => k.replace(/_/g, ' ').toUpperCase());
        const values = Object.values(fi);

        const canvasEl = document.getElementById("featureImportanceChart");
        if (!canvasEl) return;

        if (state.charts.featureChart) {
            state.charts.featureChart.destroy();
        }

        state.charts.featureChart = new Chart(canvasEl, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Relative Importance Score',
                    data: values,
                    backgroundColor: 'rgba(0, 240, 255, 0.65)',
                    borderColor: '#00f0ff',
                    borderWidth: 1.5,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#cbd5e1', font: { family: 'Outfit', size: 11 } }
                    }
                }
            }
        });
    }

    async function renderRLRewardChart() {
        const data = await loadPerformanceData();
        if (!data || !data.rl_history) return;

        const episodes = data.rl_history.map(h => `Ep ${h.episode}`);
        const rewards = data.rl_history.map(h => h.avg_reward);

        const canvasEl = document.getElementById("rlRewardChart");
        if (!canvasEl) return;

        if (state.charts.rlChart) {
            state.charts.rlChart.destroy();
        }

        state.charts.rlChart = new Chart(canvasEl, {
            type: 'line',
            data: {
                labels: episodes,
                datasets: [{
                    label: 'Average Episode Reward',
                    data: rewards,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.15)',
                    fill: true,
                    tension: 0.35,
                    pointRadius: 3,
                    pointBackgroundColor: '#8b5cf6'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#cbd5e1' } }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });
    }

    // --- RL Trace Runner ---
    const btnRunRLTrace = document.getElementById("btn-run-rl-trace");
    if (btnRunRLTrace) {
        btnRunRLTrace.addEventListener("click", async () => {
            const src = document.getElementById("rl-src").value;
            const dst = document.getElementById("rl-dst").value;

            try {
                const res = await fetch("/api/rl/route", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ source: src, destination: dst })
                });
                const trace = await res.json();
                populateRLTransitionTable(trace);
            } catch (err) {
                console.error("RL trace error:", err);
            }
        });
    }

    function populateRLTransitionTable(trace) {
        const tbody = document.getElementById("rl-transitions-tbody");
        const pathSummary = document.getElementById("rl-trace-path-summary");
        if (!tbody) return;
        tbody.innerHTML = "";

        if (!trace || !trace.transitions || trace.transitions.length === 0) {
            tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color:var(--text-muted);">No transitions recorded.</td></tr>`;
            return;
        }

        pathSummary.textContent = `Path: ${trace.path.join(" → ")} (Hops: ${trace.hops}, Delay: ${trace.total_delay_ms}ms)`;

        trace.transitions.forEach(tr => {
            const row = document.createElement("tr");
            const m = tr.link_metrics || {};
            row.innerHTML = `
                <td><strong>Step ${tr.step}</strong></td>
                <td><code>${tr.state}</code></td>
                <td><strong style="color:var(--neon-green);">${tr.action}</strong></td>
                <td style="color:${tr.reward > 0 ? '#86efac' : '#fda4af'}; font-weight:700;">${tr.reward > 0 ? '+' : ''}${tr.reward}</td>
                <td><code>${tr.next_state}</code></td>
                <td><strong style="color:var(--neon-cyan);">${tr.q_value}</strong></td>
                <td>${m.latency_ms || '-'} ms</td>
                <td>${m.packet_loss || '-'} %</td>
                <td>${m.energy_j || '-'} J</td>
            `;
            tbody.appendChild(row);
        });
    }

    // --- Comparison Cards & Benchmark Charts ---
    function populateComparisonCards(algos) {
        const container = document.getElementById("comparison-cards-container");
        if (!container || !algos) return;
        container.innerHTML = "";

        algos.forEach((a, i) => {
            const isWinner = i === 1; // Random Forest
            const card = document.createElement("div");
            card.className = `comparison-card ${isWinner ? 'highlight' : ''}`;
            card.innerHTML = `
                <div class="comp-header">
                    <h3>${a.name}</h3>
                    <span class="comp-type">${a.type}</span>
                </div>
                <div class="comp-stats">
                    <div class="comp-stat-row"><span>Average Latency:</span> <strong>${a.avg_delay_ms} ms</strong></div>
                    <div class="comp-stat-row"><span>Packet Loss:</span> <strong>${a.packet_loss_percent} %</strong></div>
                    <div class="comp-stat-row"><span>Energy Cost:</span> <strong>${a.energy_consumption_j} J</strong></div>
                    <div class="comp-stat-row"><span>Average Hops:</span> <strong>${a.avg_hops}</strong></div>
                    <div class="comp-stat-row"><span>Route Stability:</span> <strong>${a.route_stability_percent} %</strong></div>
                </div>
                <div class="comp-cost-badge">
                    <span>Combined Routing Cost</span>
                    <strong>${a.combined_cost}</strong>
                </div>
            `;
            container.appendChild(card);
        });
    }

    async function renderBenchmarkCharts() {
        const data = await loadPerformanceData();
        if (!data || !data.benchmarks || !data.benchmarks.algorithms) return;

        const algos = data.benchmarks.algorithms;
        const labels = algos.map(a => a.name.split(" ")[0] + " " + a.name.split(" ")[1]);

        // 1. Latency & Energy Chart
        const latCanvas = document.getElementById("comparisonLatencyChart");
        if (latCanvas) {
            if (state.charts.benchLatChart) state.charts.benchLatChart.destroy();
            state.charts.benchLatChart = new Chart(latCanvas, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Avg Delay (ms)',
                            data: algos.map(a => a.avg_delay_ms),
                            backgroundColor: 'rgba(56, 189, 248, 0.7)'
                        },
                        {
                            label: 'Energy (J)',
                            data: algos.map(a => a.energy_consumption_j * 10), // scaled for visualization
                            backgroundColor: 'rgba(245, 158, 11, 0.7)'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
                        x: { grid: { display: false }, ticks: { color: '#cbd5e1' } }
                    }
                }
            });
        }

        // 2. Packet Loss & Combined Cost Chart
        const lossCanvas = document.getElementById("comparisonLossChart");
        if (lossCanvas) {
            if (state.charts.benchLossChart) state.charts.benchLossChart.destroy();
            state.charts.benchLossChart = new Chart(lossCanvas, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Packet Loss (%)',
                            data: algos.map(a => a.packet_loss_percent),
                            backgroundColor: 'rgba(244, 63, 94, 0.7)'
                        },
                        {
                            label: 'Combined Cost (Score × 10)',
                            data: algos.map(a => a.combined_cost * 10),
                            backgroundColor: 'rgba(16, 185, 129, 0.7)'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
                        x: { grid: { display: false }, ticks: { color: '#cbd5e1' } }
                    }
                }
            });
        }
    }

    // --- Dataset Preview Table ---
    let datasetLoaded = false;
    async function loadDatasetPreview() {
        if (datasetLoaded) return;
        try {
            const res = await fetch("/api/dataset");
            const data = await res.json();
            datasetLoaded = true;

            document.getElementById("ds-total-samples").textContent = data.total_samples.toLocaleString();
            document.getElementById("ds-features").textContent = `${data.feature_count} Cols`;
            document.getElementById("ds-train").textContent = data.train_samples.toLocaleString();
            document.getElementById("ds-test").textContent = data.test_samples.toLocaleString();

            const thead = document.getElementById("dataset-preview-thead");
            const tbody = document.getElementById("dataset-preview-tbody");

            if (thead && data.columns) {
                thead.innerHTML = `<tr>${data.columns.map(c => `<th>${c.replace(/_/g, ' ')}</th>`).join('')}</tr>`;
            }

            if (tbody && data.preview) {
                tbody.innerHTML = data.preview.map(row => {
                    return `<tr>${data.columns.map(col => `<td>${row[col] !== undefined ? row[col] : '-'}</td>`).join('')}</tr>`;
                }).join('');
            }
        } catch (err) {
            console.error("Failed to load dataset preview:", err);
        }
    }

    // --- Viva Demo Script ---
    async function runVivaDemo() {
        switchTab("network");
        appendLog("🚀 VIVA DEMO TRIGGERED: Initializing demo sequence...", "alert");
        state.sourceSat = "SAT-01";
        state.destSat = "SAT-10";
        selectSourceSat.value = "SAT-01";
        selectDestSat.value = "SAT-10";

        await computeOptimalRoute("SAT-01", "SAT-10");
        showCanvasNotification("Viva Demo: Computing Multi-Objective Optimal Route");

        // Trigger step-by-step RL trace in background
        const rlBtn = document.getElementById("btn-run-rl-trace");
        if (rlBtn) rlBtn.click();
    }

    // --- Initial System Boot ---
    async function init() {
        resizeCanvas();
        await fetchSystemStatus();
        await fetchNetworkTopology();
        await loadPerformanceData();
        animationLoop();
    }

    init();
});
