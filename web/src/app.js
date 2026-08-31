// OceanEmbed-X Web GIS & 3D Digital Twin Client Engine
// Indian National Centre for Ocean Information Services (INCOIS) / MoES SIH26066

const API_BASE = "http://localhost:8000/api/v1";

// 1. Initialize Leaflet Map centered on North Indian Ocean (5°N–30°N, 45°E–105°E)
const map = L.map('map-container', {
    zoomControl: true,
    minZoom: 4,
    maxZoom: 12
}).setView([15.5, 75.0], 5);

// Basemap Tile Providers
const BASEMAP_TILES = {
    satellite: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: '&copy; Esri & Maxar | OceanEmbed-X MoES',
        maxZoom: 12
    }),
    dark: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
        attribution: '&copy; Esri & HERE | OceanEmbed-X',
        maxZoom: 12
    }),
    ocean: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_OceanBase/MapServer/tile/{z}/{y}/{x}', {
        attribution: '&copy; Esri & GEBCO | OceanEmbed-X',
        maxZoom: 12
    }),
    osm: L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 12
    })
};

let currentBasemap = BASEMAP_TILES.satellite.addTo(map);

const referenceLabels = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 12,
    opacity: 0.85
}).addTo(map);

let currentMarker = null;
let floatLayerGroup = L.layerGroup().addTo(map);
let cycloneLayerGroup = L.layerGroup().addTo(map);
let rasterLayerGroup = L.layerGroup().addTo(map);

let activeCycloneKey = "biparjoy_2023";
let cycloneDatabase = {};

const DEPTH_LEVELS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000];
let currentDepthIndex = 10; // 200m
let currentVariable = "temp";
let currentActiveCoord = { lat: 15.0, lon: 70.0 };

// Color Palette generator for depth maps
function getColormapColor(val, minVal, maxVal, variable) {
    if (val === null || isNaN(val)) return "rgba(0,0,0,0)";
    let norm = (val - minVal) / (maxVal - minVal + 1e-6);
    norm = Math.max(0, Math.min(1, norm));

    if (variable === "temp" || variable === "sst") {
        // Ocean Thermal Palette (Deep Blue -> Cyan -> Yellow -> Dark Red)
        const r = Math.floor(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(norm * 4 - 3))));
        const g = Math.floor(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(norm * 4 - 2))));
        const b = Math.floor(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(norm * 4 - 1))));
        return `rgba(${r}, ${g}, ${b}, 0.80)`;
    } else if (variable === "tchp") {
        // Heat Potential: Gold to Crimson
        const r = Math.floor(255 * norm);
        const g = Math.floor(180 * (1 - norm));
        const b = Math.floor(30);
        return `rgba(${r}, ${g}, ${b}, 0.85)`;
    } else if (variable === "d20") {
        // Thermocline depth: Cyan to Deep Navy
        const r = Math.floor(40);
        const g = Math.floor(220 * (1 - norm * 0.7));
        const b = Math.floor(255 * (1 - norm * 0.4));
        return `rgba(${r}, ${g}, ${b}, 0.80)`;
    } else {
        // Generic Viridis-like
        const r = Math.floor(255 * (0.2 + 0.8 * norm));
        const g = Math.floor(255 * (0.1 + 0.9 * Math.sin(norm * Math.PI)));
        const b = Math.floor(255 * (0.9 - 0.7 * norm));
        return `rgba(${r}, ${g}, ${b}, 0.80)`;
    }
}

// 2. Fetch and Render 2D Raster Grid
async function load2DRasterSlice() {
    try {
        const depthM = DEPTH_LEVELS[currentDepthIndex];
        const res = await fetch(`${API_BASE}/predict/slice?depth_m=${depthM}&variable=${currentVariable}`);
        if (!res.ok) return;
        const data = await res.json();

        rasterLayerGroup.clearLayers();

        const lats = data.lats;
        const lons = data.lons;
        const grid = data.grid;
        const minVal = data.min_val;
        const maxVal = data.max_val;

        // Update Legend
        const unit = (currentVariable === "temp" || currentVariable === "sst") ? "°C" :
                     (currentVariable === "tchp") ? "kJ/cm²" :
                     (currentVariable === "d20") ? "m" : "m/s";
        document.getElementById('legend-min').innerText = `${minVal.toFixed(1)} ${unit}`;
        document.getElementById('legend-max').innerText = `${maxVal.toFixed(1)} ${unit}`;
        document.getElementById('legend-title').innerText = `${data.variable.toUpperCase()} at ${data.depth_m}m Layer`;

        // Render Canvas overlay for performance
        const stepLat = Math.abs(lats[1] - lats[0]);
        const stepLon = Math.abs(lons[1] - lons[0]);

        for (let i = 0; i < lats.length; i += 2) {
            for (let j = 0; j < lons.length; j += 2) {
                const val = grid[i][j];
                if (val !== null) {
                    const color = getColormapColor(val, minVal, maxVal, currentVariable);
                    const bounds = [
                        [lats[i] - stepLat, lons[j] - stepLon],
                        [lats[i] + stepLat, lons[j] + stepLon]
                    ];
                    L.rectangle(bounds, {
                        color: color,
                        fillColor: color,
                        fillOpacity: 0.75,
                        weight: 0
                    }).addTo(rasterLayerGroup);
                }
            }
        }
    } catch (err) {
        console.error("Error loading 2D raster:", err);
    }
}

// 3. Query 3D Temperature Profile & Uncertainty Bounds
async function queryPointProfile(lat, lon) {
    try {
        currentActiveCoord = { lat, lon };
        document.getElementById('cursor-coords').innerText = `${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E (${lon < 78 ? 'Arabian Sea' : 'Bay of Bengal'})`;

        if (currentMarker) map.removeLayer(currentMarker);
        currentMarker = L.circleMarker([lat, lon], {
            radius: 8,
            color: '#38bdf8',
            fillColor: '#0284c7',
            fillOpacity: 0.9,
            weight: 2
        }).addTo(map);

        const res = await fetch(`${API_BASE}/predict/profile?lat=${lat}&lon=${lon}`);
        if (!res.ok) {
            const errData = await res.json();
            console.warn(errData.detail);
            return;
        }
        const data = await res.json();

        // Update Top Mini Metric Cards
        document.getElementById('mld-val').innerText = `${data.mld_m.toFixed(1)} m`;
        document.getElementById('d26-val').innerText = `${data.d26_m.toFixed(1)} m`;
        document.getElementById('tchp-val').innerText = `${data.tchp_kj_cm2.toFixed(1)} kJ/cm²`;

        // Update PFZ & Ecosystem Cards
        if (data.pfz_upwelling) {
            document.getElementById('d20-val').innerText = `${data.pfz_upwelling.d20_isotherm_depth_m} m`;
            document.getElementById('gear-depth-val').innerText = `${data.pfz_upwelling.recommended_gear_depth_m} m`;
            document.getElementById('pfz-badge').innerText = data.pfz_upwelling.pfz_potential_category;
            document.getElementById('pfz-advisory-text').innerText = data.pfz_upwelling.incois_advisory_text;
        }

        // Update Pollution Card
        if (data.oil_and_plastic_risk) {
            const oosa = data.oil_and_plastic_risk.oil_spill_oosa;
            const plastic = data.oil_and_plastic_risk.plastic_debris_analysis;
            document.getElementById('oil-depth-val').innerText = `${oosa.max_droplet_entrainment_depth_m} m`;
            document.getElementById('plastic-zone-val').innerText = plastic.dominant_vertical_zone;
            document.getElementById('pollution-advisory-text').innerText = `${oosa.regime}. ${plastic.marine_debris_advisory}`;
        }

        // Plot 3D Profile Curve with 90% Confidence Envelope
        plotVerticalProfile(data);

        // Update Embedding Inspector for this point
        loadEmbeddingInspection(lat, lon);

    } catch (err) {
        console.error("Error querying profile:", err);
    }
}

// 4. Plot Vertical Profile using Plotly.js
function plotVerticalProfile(data) {
    const depths = data.depths;
    const tMed = data.temperature_median;
    const tLow = data.temperature_lower_10;
    const tUp = data.temperature_upper_90;
    const tClim = data.climatology_baseline;

    const traceEnvelope = {
        x: tLow.concat(tUp.slice().reverse()),
        y: depths.concat(depths.slice().reverse()),
        fill: 'toself',
        fillcolor: 'rgba(56, 189, 248, 0.18)',
        line: { color: 'transparent' },
        name: '90% Conformal Uncertainty',
        showlegend: true,
        type: 'scatter'
    };

    const traceMedian = {
        x: tMed,
        y: depths,
        mode: 'lines+markers',
        name: 'HO-Mamba Predicted T(z)',
        line: { color: '#38bdf8', width: 3.5 },
        marker: { size: 6, color: '#0284c7' },
        type: 'scatter'
    };

    const traceClim = {
        x: tClim,
        y: depths,
        mode: 'lines',
        name: 'Climatology Baseline T_clim(z)',
        line: { color: '#94a3b8', width: 2, dash: 'dot' },
        type: 'scatter'
    };

    const layout = {
        margin: { l: 50, r: 25, t: 25, b: 40 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(15, 23, 42, 0.65)',
        xaxis: {
            title: 'Temperature (°C)',
            color: '#cbd5e1',
            gridcolor: '#334155',
            range: [4, 32]
        },
        yaxis: {
            title: 'Depth (m)',
            color: '#cbd5e1',
            gridcolor: '#334155',
            autorange: 'reversed'
        },
        legend: {
            x: 0.05,
            y: 0.05,
            bgcolor: 'rgba(15, 23, 42, 0.85)',
            bordercolor: '#475569',
            font: { color: '#f8fafc', size: 10 }
        },
        shapes: [
            // MLD Line
            {
                type: 'line',
                x0: 4, x1: 32,
                y0: data.mld_m, y1: data.mld_m,
                line: { color: '#fbbf24', width: 1.5, dash: 'dash' }
            },
            // D26 Line
            {
                type: 'line',
                x0: 4, x1: 32,
                y0: data.d26_m, y1: data.d26_m,
                line: { color: '#f87171', width: 1.5, dash: 'dash' }
            }
        ],
        annotations: [
            {
                x: 28, y: data.mld_m - 5,
                text: `MLD: ${data.mld_m.toFixed(0)}m`,
                showarrow: false,
                font: { color: '#fbbf24', size: 10 }
            },
            {
                x: 28, y: data.d26_m - 5,
                text: `D26: ${data.d26_m.toFixed(0)}m`,
                showarrow: false,
                font: { color: '#f87171', size: 10 }
            }
        ]
    };

    Plotly.newPlot('profile-plot-container', [traceEnvelope, traceClim, traceMedian], layout, { responsive: true, displayModeBar: false });
}

// 5. Latent Satellite Embedding Inspector
async function loadEmbeddingInspection(lat, lon) {
    try {
        const res = await fetch(`${API_BASE}/embeddings/inspect?lat=${lat}&lon=${lon}`);
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById('eddy-energy-val').innerText = `${data.spatial_eddy_energy_index.toFixed(2)}`;
        const m1 = data.dominant_baroclinic_mode_amplitudes[0];
        document.getElementById('mode1-weight-val').innerText = `${m1.toFixed(3)} (Thermocline Mode)`;

        // Plot Channel Group Energy Breakdown
        const groups = data.channel_statistics;
        const xNames = groups.map(g => g.feature_name.split(' ')[0]);
        const yEnergy = groups.map(g => g.energy_norm);

        const trace = {
            x: xNames,
            y: yEnergy,
            type: 'bar',
            marker: {
                color: ['#38bdf8', '#818cf8', '#a78bfa', '#f472b6', '#34d399', '#fbbf24', '#fb923c', '#94a3b8']
            }
        };

        const layout = {
            margin: { l: 40, r: 20, t: 20, b: 60 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(15, 23, 42, 0.65)',
            xaxis: {
                tickangle: -30,
                color: '#cbd5e1',
                gridcolor: '#334155',
                font: { size: 9 }
            },
            yaxis: {
                title: 'Energy Norm',
                color: '#cbd5e1',
                gridcolor: '#334155'
            }
        };

        Plotly.newPlot('embedding-plot-container', [trace], layout, { responsive: true, displayModeBar: false });

    } catch (err) {
        console.error("Error loading embeddings:", err);
    }
}

// 6. Cyclone Track Explorer
async function loadCycloneDatabase() {
    try {
        const res = await fetch(`${API_BASE}/cyclones/tracks`);
        if (!res.ok) return;
        cycloneDatabase = await res.json();
        renderCycloneTrack(activeCycloneKey);
    } catch (err) {
        console.error("Error loading cyclones:", err);
    }
}

function renderCycloneTrack(key) {
    cycloneLayerGroup.clearLayers();
    const cyclone = cycloneDatabase[key];
    if (!cyclone) return;

    const track = cyclone.track;
    const latlngs = track.map(pt => [pt.lat, pt.lon]);

    // Plot Track line on Leaflet map
    L.polyline(latlngs, {
        color: '#f87171',
        weight: 3.5,
        dashArray: '5, 5'
    }).addTo(cycloneLayerGroup);

    track.forEach(pt => {
        L.circleMarker([pt.lat, pt.lon], {
            radius: pt.tchp > 80 ? 9 : 6,
            color: pt.tchp > 80 ? '#ef4444' : '#fbbf24',
            fillColor: pt.tchp > 80 ? '#ef4444' : '#fbbf24',
            fillOpacity: 0.85
        }).bindPopup(`
            <strong>${cyclone.name}</strong><br>
            Date: ${pt.date}<br>
            Stage: ${pt.stage}<br>
            TCHP: ${pt.tchp} kJ/cm²<br>
            RI Risk: <strong>${pt.ri_risk}</strong>
        `).addTo(cycloneLayerGroup);
    });

    // Plot Cyclone TCHP Time Series
    const traceTCHP = {
        x: track.map(p => p.date),
        y: track.map(p => p.tchp),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'TCHP (kJ/cm²)',
        line: { color: '#f87171', width: 3 }
    };

    const layout = {
        margin: { l: 45, r: 20, t: 20, b: 40 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(15, 23, 42, 0.65)',
        xaxis: { color: '#cbd5e1', gridcolor: '#334155' },
        yaxis: { title: 'TCHP (kJ/cm²)', color: '#cbd5e1', gridcolor: '#334155' }
    };

    Plotly.newPlot('cyclone-track-plot', [traceTCHP], layout, { responsive: true, displayModeBar: false });
}

// 7. Benchmark Scorecard Table
async function loadBenchmarkScorecard() {
    try {
        const res = await fetch(`${API_BASE}/evaluation/benchmark`);
        if (!res.ok) return;
        const data = await res.json();

        const tbody = document.getElementById('bench-table-body').querySelector('tbody');
        tbody.innerHTML = '';

        const depths = data.depths_m;
        const rmse = data.rmse_per_depth_degC;
        const mae = data.mae_per_depth_degC;
        const bias = data.bias_per_depth_degC;
        const corr = data.pearson_r_per_depth;

        for (let i = 0; i < depths.length; i++) {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${depths[i]} m</strong></td>
                <td>${rmse[i].toFixed(3)}</td>
                <td>${mae[i].toFixed(3)}</td>
                <td>${bias[i].toFixed(3)}</td>
                <td><span class="badge ${corr[i] > 0.8 ? 'success' : 'highlight'}">${corr[i].toFixed(3)}</span></td>
            `;
            tbody.appendChild(tr);
        }

        // Summary Average Row
        const trAvg = document.createElement('tr');
        trAvg.style.fontWeight = 'bold';
        trAvg.style.backgroundColor = 'rgba(56, 189, 248, 0.12)';
        trAvg.innerHTML = `
            <td>OVERALL MEAN</td>
            <td>${data.mean_overall_rmse.toFixed(3)}</td>
            <td>--</td>
            <td>--</td>
            <td><span class="badge success">${data.mean_overall_correlation.toFixed(3)}</span></td>
        `;
        tbody.appendChild(trAvg);

    } catch (err) {
        console.error("Error loading benchmarks:", err);
    }
}

// 8. 3D Volumetric Plotly Rendering
async function render3DVolumetricView() {
    try {
        const res = await fetch(`${API_BASE}/volume/3d?downsample=4`);
        if (!res.ok) return;
        const data = await res.json();

        const trace3D = {
            x: data.x,
            y: data.y,
            z: data.z,
            mode: 'markers',
            marker: {
                size: 3,
                color: data.temperature,
                colorscale: 'Viridis',
                colorbar: { title: 'Temp (°C)', len: 0.6 },
                opacity: 0.65
            },
            type: 'scatter3d'
        };

        const layout = {
            margin: { l: 0, r: 0, t: 0, b: 0 },
            paper_bgcolor: '#0f172a',
            scene: {
                xaxis: { title: 'Longitude (°E)', color: '#94a3b8', gridcolor: '#334155' },
                yaxis: { title: 'Latitude (°N)', color: '#94a3b8', gridcolor: '#334155' },
                zaxis: { title: 'Depth (m)', autorange: 'reversed', color: '#94a3b8', gridcolor: '#334155' }
            }
        };

        Plotly.newPlot('volume-3d-plot', [trace3D], layout, { responsive: true });

    } catch (err) {
        console.error("Error rendering 3D volume:", err);
    }
}

// 9. Event Listeners & Tab Switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

        const targetTab = e.target.getAttribute('data-tab');
        e.target.classList.add('active');
        document.getElementById(targetTab).classList.add('active');

        if (targetTab === 'bench-tab') loadBenchmarkScorecard();
        if (targetTab === 'cyclone-tab') renderCycloneTrack(activeCycloneKey);
        if (targetTab === 'embedding-tab') loadEmbeddingInspection(currentActiveCoord.lat, currentActiveCoord.lon);
    });
});

// View Switch: 2D vs 3D
document.getElementById('btn-view-2d').addEventListener('click', () => {
    document.getElementById('btn-view-2d').classList.add('active');
    document.getElementById('btn-view-3d').classList.remove('active');
    document.getElementById('map-container').style.display = 'block';
    document.getElementById('volume-view-wrapper').style.display = 'none';
});

document.getElementById('btn-view-3d').addEventListener('click', () => {
    document.getElementById('btn-view-3d').classList.add('active');
    document.getElementById('btn-view-2d').classList.remove('active');
    document.getElementById('map-container').style.display = 'none';
    document.getElementById('volume-view-wrapper').style.display = 'block';
    render3DVolumetricView();
});

// Layer & Depth Selectors
document.getElementById('var-select').addEventListener('change', (e) => {
    currentVariable = e.target.value;
    const depthContainer = document.getElementById('depth-selector-container');
    depthContainer.style.display = (currentVariable === 'temp') ? 'flex' : 'none';
    load2DRasterSlice();
});

document.getElementById('depth-slider').addEventListener('input', (e) => {
    currentDepthIndex = parseInt(e.target.value);
    document.getElementById('depth-val-display').innerText = `${DEPTH_LEVELS[currentDepthIndex]} m`;
    load2DRasterSlice();
});

document.getElementById('basemap-select').addEventListener('change', (e) => {
    map.removeLayer(currentBasemap);
    currentBasemap = BASEMAP_TILES[e.target.value].addTo(map);
});

document.getElementById('cyclone-selector').addEventListener('change', (e) => {
    activeCycloneKey = e.target.value;
    renderCycloneTrack(activeCycloneKey);
});

// Map Click Listener
map.on('click', (e) => {
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;
    if (lat >= 5.0 && lat <= 30.0 && lon >= 45.0 && lon <= 105.0) {
        queryPointProfile(lat, lon);
    }
});

// Modal Virtual Float Deployment
document.getElementById('btn-deploy-mode').addEventListener('click', () => {
    document.getElementById('float-modal').style.display = 'flex';
});

document.getElementById('btn-close-modal').addEventListener('click', () => {
    document.getElementById('float-modal').style.display = 'none';
});

document.getElementById('btn-confirm-deploy').addEventListener('click', async () => {
    const lat = parseFloat(document.getElementById('modal-lat').value);
    const lon = parseFloat(document.getElementById('modal-lon').value);

    try {
        const res = await fetch(`${API_BASE}/floats/inject`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ latitude: lat, longitude: lon })
        });
        const resData = await res.json();
        alert(resData.message);
        document.getElementById('float-modal').style.display = 'none';

        // Add to map
        L.marker([lat, lon], {
            icon: L.divIcon({
                className: 'custom-argo-pin',
                html: '📍',
                iconSize: [24, 24]
            })
        }).bindPopup(`<strong>Virtual Float:</strong> ${resData.float.float_id}`).addTo(floatLayerGroup);

        queryPointProfile(lat, lon);
    } catch (err) {
        console.error("Error deploying float:", err);
    }
});

// Initial Setup
window.addEventListener('DOMContentLoaded', () => {
    load2DRasterSlice();
    queryPointProfile(15.0, 70.0);
    loadCycloneDatabase();
});
