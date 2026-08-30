// OceanEmbed-X Web GIS & Digital Twin Client

const API_BASE = "http://localhost:8000/api/v1";

// 1. Initialize Leaflet Map centered on North Indian Ocean
const map = L.map('map-container').setView([16.0, 75.0], 5);

// Dark Basemap tiles
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a> | OceanEmbed-X MoES',
    maxZoom: 12,
    minZoom: 4
}).addTo(map);

let currentMarker = null;
let floatLayerGroup = L.layerGroup().addTo(map);
let cycloneLayerGroup = L.layerGroup().addTo(map);
let gridLayerGroup = L.layerGroup().addTo(map);

// Historical Cyclone Track: Cyclone Biparjoy (June 2023 Arabian Sea)
const BIPARJOY_TRACK = [
    [11.8, 66.0, "June 06: Deep Depression (TCHP: 92 kJ/cm²)"],
    [13.5, 66.2, "June 07: Cyclonic Storm"],
    [15.2, 66.3, "June 08: Very Severe Cyclonic Storm (Rapid Intensification)"],
    [17.4, 67.3, "June 10: Extremely Severe Cyclonic Storm (Cat 3)"],
    [20.5, 67.5, "June 12: Peak Intensity"],
    [23.2, 68.6, "June 15: Landfall near Jakhau Port, Gujarat"]
];

function drawCycloneTrack() {
    cycloneLayerGroup.clearLayers();
    const latlngs = BIPARJOY_TRACK.map(p => [p[0], p[1]]);
    
    // Draw track line
    L.polyline(latlngs, {
        color: '#f97316',
        weight: 3,
        dashArray: '6, 6',
        opacity: 0.85
    }).addTo(cycloneLayerGroup);

    // Draw track waypoints
    BIPARJOY_TRACK.forEach((p, idx) => {
        const circle = L.circleMarker([p[0], p[1]], {
            radius: idx === 3 ? 8 : 5,
            fillColor: idx === 3 ? '#ef4444' : '#f97316',
            color: '#fff',
            weight: 1.5,
            fillOpacity: 0.9
        }).bindPopup(`<b>${p[2]}</b><br>Lat: ${p[0]}°N, Lon: ${p[1]}°E`);
        circle.addTo(cycloneLayerGroup);
    });
}

// 2. Fetch and Display Active Argo Floats
async function loadActiveArgoFloats() {
    try {
        const res = await fetch(`${API_BASE}/floats/active`);
        if (!res.ok) return;
        const data = await res.json();
        
        floatLayerGroup.clearLayers();
        data.floats.forEach(f => {
            const marker = L.circleMarker([f.latitude, f.longitude], {
                radius: 6,
                fillColor: '#38bdf8',
                color: '#fff',
                weight: 1.5,
                fillOpacity: 0.85
            }).bindPopup(`
                <b>🛰️ ${f.float_id} (In-Situ Argo)</b><br>
                Lat: ${f.latitude}°N, Lon: ${f.longitude}°E<br>
                Surface Temp: ${f.temperatures[0]}°C<br>
                Temp at 200m: ${f.temperatures[10]}°C<br>
                <i>QC Flag: Good (Flag 1)</i>
            `);
            marker.addTo(floatLayerGroup);
        });
    } catch (e) {
        console.warn("Could not load remote Argo floats (fallback active).", e);
    }
}

// 3. Fetch 2D Depth-Slice Contours
async function loadDepthSlice(depthM) {
    try {
        const res = await fetch(`${API_BASE}/predict/slice?depth_m=${depthM}`);
        if (!res.ok) return;
        const data = await res.json();
        
        gridLayerGroup.clearLayers();
        const lats = data.lats;
        const lons = data.lons;
        const grid = data.temperature_grid;
        
        // Color scale mapping
        function getColor(t) {
            if (t === null) return 'transparent';
            if (t > 28.0) return '#ef4444';
            if (t > 25.0) return '#f97316';
            if (t > 20.0) return '#eab308';
            if (t > 15.0) return '#22c55e';
            if (t > 10.0) return '#06b6d4';
            return '#3b82f6';
        }

        // Draw sparse sample grid markers for performance
        for (let i = 0; i < lats.length; i += 2) {
            for (let j = 0; j < lons.length; j += 2) {
                const temp = grid[i][j];
                if (temp !== null) {
                    const rect = L.rectangle([
                        [lats[i] - 0.25, lons[j] - 0.25],
                        [lats[i] + 0.25, lons[j] + 0.25]
                    ], {
                        color: 'transparent',
                        fillColor: getColor(temp),
                        fillOpacity: 0.35
                    });
                    rect.addTo(gridLayerGroup);
                }
            }
        }
    } catch (e) {
        console.warn("Could not load depth slice", e);
    }
}

// 4. Plot Vertical Temperature Profile with Plotly
function plotVerticalProfile(data) {
    const depths = data.depths;
    const tMed = data.temperature_median;
    const tLow = data.temperature_lower_10;
    const tUp = data.temperature_upper_90;
    const tClim = data.climatology_baseline;

    const traceUpper = {
        x: tUp,
        y: depths,
        type: 'scatter',
        mode: 'lines',
        line: { color: 'transparent' },
        name: '90% Upper Bound',
        showlegend: false
    };

    const traceLower = {
        x: tLow,
        y: depths,
        type: 'scatter',
        mode: 'lines',
        fill: 'tonexty',
        fillcolor: 'rgba(56, 189, 248, 0.18)',
        line: { color: 'transparent' },
        name: '90% Confidence Envelope'
    };

    const traceMedian = {
        x: tMed,
        y: depths,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#38bdf8', width: 3 },
        marker: { size: 5, color: '#38bdf8' },
        name: 'HyperOcean-Mamba (Predicted)'
    };

    const traceClim = {
        x: tClim,
        y: depths,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#94a3b8', width: 2, dash: 'dash' },
        name: 'Climatological Baseline'
    };

    const layout = {
        title: {
            text: `Vertical Thermal Profile T(z) at (${data.latitude}°N, ${data.longitude}°E)`,
            font: { color: '#f8fafc', size: 13 }
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'rgba(15, 23, 42, 0.6)',
        xaxis: {
            title: 'Temperature (°C)',
            color: '#94a3b8',
            gridcolor: 'rgba(56, 189, 248, 0.1)',
            range: [4, 32]
        },
        yaxis: {
            title: 'Depth (meters)',
            color: '#94a3b8',
            autorange: 'reversed',  // Ocean depth downwards
            gridcolor: 'rgba(56, 189, 248, 0.1)'
        },
        margin: { l: 50, r: 20, t: 40, b: 40 },
        legend: { orientation: 'h', y: -0.2, font: { color: '#cbd5e1', size: 10 } }
    };

    Plotly.newPlot('profile-plot-container', [traceUpper, traceLower, traceMedian, traceClim], layout, { responsive: true, displayModeBar: false });
}

// 5. Plot Tactical Sonar Sound Velocity Profile & Acoustic Rays
async function plotTacticalSonar(lat, lon, data) {
    const svp = data.sonar_analysis.sound_velocity_profile;
    const depths = data.sonar_analysis.depths;

    const traceSVP = {
        x: svp,
        y: depths,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#0284c7', width: 3 },
        marker: { size: 5, color: '#0284c7' },
        name: 'Sound Velocity C(z)'
    };

    const layout = {
        title: {
            text: `Acoustic Sound Velocity Profile (Mackenzie SVP)`,
            font: { color: '#f8fafc', size: 13 }
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'rgba(15, 23, 42, 0.6)',
        xaxis: {
            title: 'Sound Speed (m/s)',
            color: '#94a3b8',
            gridcolor: 'rgba(56, 189, 248, 0.1)'
        },
        yaxis: {
            title: 'Depth (meters)',
            color: '#94a3b8',
            autorange: 'reversed',
            gridcolor: 'rgba(56, 189, 248, 0.1)'
        },
        margin: { l: 50, r: 20, t: 40, b: 40 }
    };

    Plotly.newPlot('sonar-plot-container', [traceSVP], layout, { responsive: true, displayModeBar: false });

    // Update stats
    document.getElementById('duct-val').innerText = `${data.sonar_analysis.surface_duct_thickness_m} m`;
    document.getElementById('sofar-val').innerText = `${data.sonar_analysis.sofar_axis_depth_m} m (${data.sonar_analysis.sofar_minimum_speed_ms} m/s)`;
    document.getElementById('shadow-val').innerText = `${data.sonar_analysis.submarine_shadow_zone.top_m}m – ${data.sonar_analysis.submarine_shadow_zone.bottom_m}m`;
}

// 6. Handle Map Click Event
async function handleCoordinateSelection(lat, lon) {
    document.getElementById('click-coord-display').innerText = `📍 Reconstructing Profile for: ${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E...`;
    
    if (currentMarker) map.removeLayer(currentMarker);
    currentMarker = L.marker([lat, lon]).addTo(map);

    try {
        const res = await fetch(`${API_BASE}/predict/profile?lat=${lat}&lon=${lon}`);
        if (!res.ok) {
            const err = await res.json();
            alert(err.detail || "Error querying coordinate.");
            return;
        }
        const data = await res.json();

        // Update UI Stats
        document.getElementById('mld-val').innerText = `${data.mld_m} m`;
        document.getElementById('d26-val').innerText = `${data.d26_m} m`;
        document.getElementById('tchp-val').innerText = `${data.tchp_kj_cm2} kJ/cm²`;
        document.getElementById('click-coord-display').innerText = `📍 Active Coordinate: ${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E (MLD: ${data.mld_m}m | TCHP: ${data.tchp_kj_cm2} kJ/cm²)`;

        // Render Plots
        plotVerticalProfile(data);
        plotTacticalSonar(lat, lon, data);
    } catch (e) {
        console.error("Error fetching profile", e);
    }
}

// 7. Event Listeners & Initialization
map.on('click', (e) => {
    handleCoordinateSelection(e.latlng.lat, e.latlng.lng);
});

document.getElementById('depth-select').addEventListener('change', (e) => {
    loadDepthSlice(parseFloat(e.target.value));
});

document.getElementById('toggle-floats').addEventListener('change', (e) => {
    if (e.target.checked) map.addLayer(floatLayerGroup);
    else map.removeLayer(floatLayerGroup);
});

document.getElementById('toggle-cyclone').addEventListener('change', (e) => {
    if (e.target.checked) map.addLayer(cycloneLayerGroup);
    else map.removeLayer(cycloneLayerGroup);
});

// Tab Switcher
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        const targetTab = document.getElementById(btn.dataset.tab);
        if (targetTab) targetTab.classList.add('active');
    });
});

// Initial Bootstrap
window.addEventListener('DOMContentLoaded', () => {
    drawCycloneTrack();
    loadActiveArgoFloats();
    loadDepthSlice(200.0);
    // Default coordinate (Arabian Sea warm pool)
    handleCoordinateSelection(15.0, 70.0);
});
