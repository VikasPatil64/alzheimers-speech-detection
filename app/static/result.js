// static/result.js - Permanent working version

// Load data from sessionStorage
const data = JSON.parse(sessionStorage.getItem('prediction'));

// Debug - check console (F12)
console.log('Data received:', data);

// ============================================================
// 1. PREDICTION TITLE
// ============================================================
const titleElement = document.getElementById('predictionTitle');
if (titleElement) {
    if (data.prediction === 'Dementia') {
        titleElement.textContent = '⚠️ Dementia Risk Detected';
        titleElement.style.color = '#e74c3c';
    } else if (data.prediction === 'Control') {
        titleElement.textContent = '✅ Healthy';
        titleElement.style.color = '#27ae60';

    } else if (data.prediction === 'Uncertain') {
    titleElement.textContent = '⚠️ Inconclusive Result';
    titleElement.style.color = '#f39c12';

    } else if (data.error) {
        titleElement.textContent = '⚠️ Error';
        titleElement.style.color = '#e74c3c';
    } else {
        titleElement.textContent = data.prediction || 'Unknown';
    }
}

// ============================================================
// 2. CONFIDENCE SCORE
// ============================================================
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBar = document.getElementById('confidenceBar');

if (data.confidence !== undefined && data.confidence !== null && !data.error) {
    const percent = (data.confidence * 100).toFixed(1);

    if (confidenceValue) {
        confidenceValue.textContent = `${percent}%`;
    }

    if (confidenceBar) {
        confidenceBar.style.width = `${percent}%`;

        // ✅ UPDATED LOGIC
        if (data.prediction === 'Dementia') {
            confidenceBar.className = 'confidence-bar dementia';   // 🔴 red
        } else if (data.prediction === 'Uncertain') {
            confidenceBar.className = 'confidence-bar uncertain';  // 🟡 yellow
        } else {
            confidenceBar.className = 'confidence-bar';            // 🟢 green
        }
    }

} else {
    if (confidenceValue) confidenceValue.textContent = 'N/A';
    if (confidenceBar) confidenceBar.style.width = '0%';
}

// ============================================================
// 3. TRANSCRIPT
// ============================================================
const transcriptEl = document.getElementById('transcript');
if (transcriptEl) {
    if (data.error) {
        transcriptEl.innerHTML = `<div style="color: #e74c3c;">❌ ${data.error}</div>`;
    } else {
        transcriptEl.innerHTML = data.transcript || 'No transcript available';
    }
}

// ============================================================
// 4. CLINICAL FACTORS
// ============================================================
const clinicalEl = document.getElementById('clinicalFactors');
if (clinicalEl) {
    if (data.error) {
        clinicalEl.innerHTML = '<p style="color: #e74c3c;">Error loading clinical data</p>';
    } else if (data.factors && data.factors.clinical) {
        const c = data.factors.clinical;
        clinicalEl.innerHTML = `
            <p><strong>Age:</strong> ${c.age || 'Not provided'}</p>
            <p><strong>Gender:</strong> ${c.gender || 'Not provided'}</p>
            <p><strong>Education:</strong> ${c.education || 'Not provided'} years</p>
            <p><strong>MMSE:</strong> ${c.mmse || 'Not provided'}</p>
        `;
    } else {
        clinicalEl.innerHTML = '<p>No clinical data available</p>';
    }
}

// ============================================================
// 5. SPEECH MARKERS
// ============================================================
const speechEl = document.getElementById('speechFactors');
if (speechEl) {
    if (data.error) {
        speechEl.innerHTML = '<p style="color: #e74c3c;">Error loading speech markers</p>';
    } else if (data.factors && data.factors.speech) {
        const s = data.factors.speech;
        speechEl.innerHTML = `
            <p><strong>Words per minute:</strong> ${s.words_per_minute || 'N/A'}</p>
            <p><strong>Pause count:</strong> ${s.pause_count || 'N/A'}</p>
            <p><strong>Filler rate:</strong> ${s.filler_rate || 'N/A'}</p>
            <p><strong>Pitch variability:</strong> ${s.pitch_variability || 'N/A'}</p>
        `;
    } else {
        speechEl.innerHTML = '<p>No speech markers available</p>';
    }
}

// ============================================================
// 6. OVERRIDE WARNING (if any)
// ============================================================
if (data.override && data.override_reason) {
    const factorsGrid = document.getElementById('factorsGrid');
    if (factorsGrid) {
        const warningDiv = document.createElement('div');
        warningDiv.style.cssText = 'background: #fff3cd; color: #856404; padding: 12px; border-radius: 8px; margin-top: 15px; font-size: 14px; text-align: center;';
        warningDiv.innerHTML = `⚠️ ${data.override_reason}`;
        factorsGrid.after(warningDiv);
    }
}

// ============================================================
// 7. HANDLE ERRORS (if prediction failed)
// ============================================================
if (data.error) {
    const resultCard = document.getElementById('resultCard');
    if (resultCard) {
        resultCard.style.background = '#f8d7da';
        resultCard.style.border = '1px solid #f5c6cb';
    }
}

console.log('✅ Result page loaded successfully');