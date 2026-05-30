const HISTORY_KEY = 'alzheimersPatientHistory';
const data = JSON.parse(sessionStorage.getItem('prediction') || '{}');

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function readHistory() {
    try {
        return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
    } catch (error) {
        console.warn('Could not read patient history:', error);
        return [];
    }
}

function writeHistory(records) {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(records));
}

function formatDateTime(value) {
    if (!value) return 'Not saved';
    return new Intl.DateTimeFormat(undefined, {
        dateStyle: 'medium',
        timeStyle: 'short'
    }).format(new Date(value));
}

function compactDate(value) {
    if (!value) return 'No date';
    return new Intl.DateTimeFormat(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    }).format(new Date(value));
}

function confidencePercent(record) {
    if (record.confidence === undefined || record.confidence === null || record.error) return null;
    return Number(record.confidence) * 100;
}

function confidenceLabel(record) {
    const percent = confidencePercent(record);
    return percent === null ? 'N/A' : `${percent.toFixed(1)}%`;
}

function isDementiaPrediction(record) {
    return ['dementia', 'alzheimer', "alzheimer's"].some(term =>
        String(record.prediction || '').toLowerCase().includes(term)
    );
}

function isControlPrediction(record) {
    return String(record.prediction || '').toLowerCase().includes('control');
}

function screeningBadge(record) {
    const percent = confidencePercent(record);
    const lowerConfidence = percent !== null && percent < 65;

    if (record.error) return 'Review Needed';

    if (isControlPrediction(record)) {
        return lowerConfidence ? 'Control Pattern (Lower Confidence)' : 'Low Cognitive Risk';
    }

    if (isDementiaPrediction(record)) {
        return lowerConfidence ? "Alzheimer's Pattern (Lower Confidence)" : 'Elevated Cognitive Risk';
    }

    return 'Review Needed';
}

function badgeTone(record) {
    if (isControlPrediction(record)) return 'control';
    if (isDementiaPrediction(record)) return 'elevated';
    return 'review';
}

function getRiskClass(record) {
    const tone = record.badgeTone || badgeTone(record);
    if (tone === 'control') return 'risk-low';
    if (tone === 'elevated') return 'risk-high';
    return 'risk-moderate';
}

function interpretationText(record) {
    const percent = confidencePercent(record);
    const confidencePhrase = percent !== null && percent < 65 ? 'with lower model confidence' : 'with stronger model confidence';

    if (record.error) return 'The assessment could not be completed. Review the error details and try again.';

    if (isDementiaPrediction(record)) {
        return `The model output aligns with an Alzheimer's/dementia speech pattern ${confidencePhrase}. Treat this as a screening signal for clinical review, not a diagnosis.`;
    }

    if (isControlPrediction(record)) {
        return `The model output aligns with a control speech pattern ${confidencePhrase}. This should be interpreted as lower cognitive risk in this screening session, not proof of normal cognition.`;
    }

    return 'The model returned an unclear screening class. Review transcript quality, clinical inputs, and speech markers before interpreting the report.';
}

function normalizedRecord() {
    const patient = data.patient || {};
    const clinical = data.factors?.clinical || {};

    return {
        assessmentId: data.assessmentId || `assessment-${Date.now()}`,
        savedAt: data.savedAt || new Date().toISOString(),
        prediction: data.prediction || 'Unknown',
        confidence: data.confidence,
        confidenceLabel: confidenceLabel(data),
        assessmentBadge: screeningBadge(data),
        badgeTone: badgeTone(data),
        transcript: data.transcript || '',
        patient: {
            name: patient.name || 'Unnamed Patient',
            id: patient.id || 'Not provided',
            age: patient.age || clinical.age || 'Not provided',
            gender: patient.gender || clinical.gender || 'Not provided',
            education: patient.education || clinical.education || 'Not provided',
            mmse: patient.mmse || clinical.mmse || 'Not provided'
        },
        reportData: {
            ...data,
            assessmentId: data.assessmentId || `assessment-${Date.now()}`,
            savedAt: data.savedAt || new Date().toISOString(),
            patient: patient
        }
    };
}

function saveCurrentAssessment() {
    if (!data || (!data.success && !data.prediction && !data.error) || data.error || data.fromHistory) return;

    const record = normalizedRecord();
    const records = readHistory();
    const exists = records.some(item => item.assessmentId === record.assessmentId);

    if (!exists) {
        writeHistory([record, ...records].slice(0, 50));
    }

    data.localHistorySaved = true;
    sessionStorage.setItem('prediction', JSON.stringify(data));
}

function updateStats() {
    const records = readHistory().sort((a, b) => new Date(b.savedAt) - new Date(a.savedAt));
    const countEl = document.getElementById('savedAssessmentsCount');
    const lastEl = document.getElementById('lastAnalysisLabel');
    const sessionEl = document.getElementById('activeSessionLabel');

    if (countEl) countEl.textContent = records.length;
    if (lastEl) lastEl.textContent = records[0] ? compactDate(records[0].savedAt) : 'None';
    if (sessionEl) sessionEl.textContent = data.fromHistory ? 'History report' : 'Report ready';
}

function viewPreviousReport(assessmentId) {
    const record = readHistory().find(item => item.assessmentId === assessmentId);
    if (!record) return;

    sessionStorage.setItem('prediction', JSON.stringify({
        ...record.reportData,
        fromHistory: true,
        localHistorySaved: true
    }));
    window.location.href = '/result';
}

function renderHistory(filter = '') {
    const list = document.getElementById('historyList');
    if (!list) return;

    const query = filter.trim().toLowerCase();
    const records = readHistory()
        .filter(record => (record.patient?.name || '').toLowerCase().includes(query))
        .sort((a, b) => new Date(b.savedAt) - new Date(a.savedAt));

    if (!records.length) {
        list.innerHTML = '<div class="empty-state">No matching assessments found.</div>';
        return;
    }

    list.innerHTML = records.map(record => `
        <article class="history-card">
            <div class="history-card-top">
                <strong>${escapeHtml(record.patient?.name || 'Unnamed Patient')}</strong>
                <span class="mini-risk-badge ${getRiskClass(record)}">${escapeHtml(record.assessmentBadge || screeningBadge(record))}</span>
            </div>
            <span>${escapeHtml(compactDate(record.savedAt))}</span>
            <p>${escapeHtml(record.prediction || 'Prediction unavailable')} - ${escapeHtml(record.confidenceLabel || 'N/A')}</p>
            <button type="button" class="text-button" onclick="viewPreviousReport('${escapeHtml(record.assessmentId)}')">View Previous Report</button>
        </article>
    `).join('');
}

function setupHistorySearch() {
    const search = document.getElementById('historySearch');
    if (!search) return;
    search.addEventListener('input', () => renderHistory(search.value));
    renderHistory();
}

function setupSectionNavigation() {
    const links = [...document.querySelectorAll('[data-section-link]')];
    const sections = links
        .map(link => document.getElementById(link.dataset.sectionLink))
        .filter(Boolean);

    if (!links.length || !sections.length || !('IntersectionObserver' in window)) return;

    const observer = new IntersectionObserver(entries => {
        const visible = entries
            .filter(entry => entry.isIntersecting)
            .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

        if (!visible) return;
        links.forEach(link => {
            link.classList.toggle('active', link.dataset.sectionLink === visible.target.id);
        });
    }, { rootMargin: '-25% 0px -60% 0px', threshold: [0.2, 0.6] });

    sections.forEach(section => observer.observe(section));
}

function renderPatientSummary() {
    const patient = normalizedRecord().patient;
    const name = document.getElementById('reportPatientName');
    const meta = document.getElementById('reportMeta');
    const patientName = document.getElementById('summaryPatientName');
    const patientId = document.getElementById('summaryPatientId');
    const risk = document.getElementById('summaryRiskLevel');
    const mmse = document.getElementById('summaryMmse');
    const summaryConfidence = document.getElementById('summaryConfidence');
    const timestamp = document.getElementById('summaryTimestamp');

    if (name) name.textContent = patient.name || 'Assessment Results';
    if (meta) {
        meta.textContent = `${patient.gender || 'Patient'} | Age ${patient.age || 'N/A'} | MMSE ${patient.mmse || 'N/A'} | ${formatDateTime(data.savedAt)}`;
    }
    if (patientName) patientName.textContent = patient.name || 'Not provided';
    if (patientId) patientId.textContent = patient.id || 'Not provided';
    if (risk) risk.textContent = screeningBadge(data);
    if (mmse) mmse.textContent = patient.mmse || 'Not provided';
    if (summaryConfidence) summaryConfidence.textContent = confidenceLabel(data);
    if (timestamp) timestamp.textContent = formatDateTime(data.savedAt);
}

function renderPrediction() {
    const titleElement = document.getElementById('predictionTitle');
    const riskBadge = document.getElementById('riskBadge');
    const resultCard = document.getElementById('resultCard');
    const interpretation = document.getElementById('aiInterpretation');

    if (!titleElement) return;

    if (data.error) {
        titleElement.textContent = 'Error';
        titleElement.style.color = '#b42318';
        if (riskBadge) {
            riskBadge.textContent = 'Review needed';
            riskBadge.className = 'risk-badge risk-moderate';
        }
        if (resultCard) resultCard.classList.add('result-error');
        if (interpretation) interpretation.textContent = interpretationText(data);
        return;
    }

    if (data.prediction === 'Dementia') {
        titleElement.textContent = 'Dementia Risk Detected';
        titleElement.style.color = '#b42318';
    } else if (data.prediction === 'Control') {
        titleElement.textContent = 'Healthy Control Pattern';
        titleElement.style.color = '#067647';
    } else {
        titleElement.textContent = data.prediction || 'Unknown';
    }

    if (riskBadge) {
        riskBadge.textContent = screeningBadge(data);
        riskBadge.className = `risk-badge ${getRiskClass(data)}`;
    }

    if (interpretation) interpretation.textContent = interpretationText(data);
}

function renderConfidence() {
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceBar = document.getElementById('confidenceBar');
    const riskMeter = document.getElementById('riskMeter');
    const percent = confidencePercent(data);

    if (percent !== null) {
        if (confidenceValue) confidenceValue.textContent = `${percent.toFixed(1)}%`;
        if (riskMeter) {
            const clamped = Math.max(0, Math.min(percent, 100));
            riskMeter.style.setProperty('--meter-value', `${clamped * 3.6}deg`);
            riskMeter.className = `risk-meter ${badgeTone(data) === 'elevated' ? 'meter-high' : badgeTone(data) === 'control' ? 'meter-low' : 'meter-moderate'}`;
        }
        if (confidenceBar) {
            confidenceBar.style.width = `${Math.max(0, Math.min(percent, 100))}%`;
            confidenceBar.className = data.prediction === 'Dementia'
                ? 'confidence-bar dementia'
                : 'confidence-bar';
        }
    } else {
        if (confidenceValue) confidenceValue.textContent = 'N/A';
        if (confidenceBar) confidenceBar.style.width = '0%';
        if (riskMeter) riskMeter.style.setProperty('--meter-value', '0deg');
    }
}

function renderTranscript() {
    const transcriptEl = document.getElementById('transcript');
    if (!transcriptEl) return;

    if (data.error) {
        transcriptEl.innerHTML = `<div class="error-text">${escapeHtml(data.error)}</div>`;
    } else {
        transcriptEl.textContent = data.transcript || 'No transcript available';
    }
}

function renderClinicalFactors() {
    const clinicalEl = document.getElementById('clinicalFactors');
    if (!clinicalEl) return;

    if (data.error) {
        clinicalEl.innerHTML = '<p class="error-text">Error loading clinical data</p>';
        return;
    }

    const patient = normalizedRecord().patient;
    clinicalEl.innerHTML = `
        <p><strong>Age:</strong> ${escapeHtml(patient.age || 'Not provided')}</p>
        <p><strong>Gender:</strong> ${escapeHtml(patient.gender || 'Not provided')}</p>
        <p><strong>Education:</strong> ${escapeHtml(patient.education || 'Not provided')} years</p>
        <p><strong>MMSE:</strong> ${escapeHtml(patient.mmse || 'Not provided')}</p>
    `;
}

function renderSpeechFactors() {
    const speechEl = document.getElementById('speechFactors');
    if (!speechEl) return;

    if (data.error) {
        speechEl.innerHTML = '<p class="error-text">Error loading speech markers</p>';
        return;
    }

    if (data.factors && data.factors.speech) {
        const s = data.factors.speech;
        speechEl.innerHTML = `
            <p><strong>Words per minute:</strong> ${escapeHtml(s.words_per_minute || 'N/A')}</p>
            <p><strong>Pause count:</strong> ${escapeHtml(s.pause_count || 'N/A')}</p>
            <p><strong>Filler rate:</strong> ${escapeHtml(s.filler_rate || 'N/A')}</p>
            <p><strong>Pitch variability:</strong> ${escapeHtml(s.pitch_variability || 'N/A')}</p>
        `;
    } else {
        speechEl.innerHTML = '<p>No speech markers available</p>';
    }
}

function renderOverrideWarning() {
    if (!data.override || !data.override_reason) return;

    const factorsGrid = document.getElementById('factorsGrid');
    if (!factorsGrid) return;

    const warningDiv = document.createElement('div');
    warningDiv.className = 'inline-warning';
    warningDiv.textContent = data.override_reason;
    factorsGrid.after(warningDiv);
}

function reportText() {
    const patient = normalizedRecord().patient;
    return [
        "Alzheimer's Detection System - Assessment Report",
        '',
        `Patient Name: ${patient.name}`,
        `Patient ID: ${patient.id}`,
        `Date & Time: ${formatDateTime(data.savedAt)}`,
        `Age: ${patient.age}`,
        `Gender: ${patient.gender}`,
        `MMSE: ${patient.mmse}`,
        '',
        `Prediction: ${data.prediction || 'Unknown'}`,
        `Screening Badge: ${screeningBadge(data)}`,
        `Confidence: ${confidenceLabel(data)}`,
        '',
        'AI Interpretation:',
        interpretationText(data),
        '',
        'Transcript:',
        data.transcript || 'No transcript available.'
    ].join('\n');
}

function setupDownloadReport() {
    const button = document.getElementById('downloadReportBtn');
    if (!button) return;

    button.addEventListener('click', () => {
        const patient = normalizedRecord().patient;
        const filenameName = (patient.name || 'patient').replace(/[^a-z0-9]+/gi, '-').replace(/^-|-$/g, '').toLowerCase();
        const blob = new Blob([reportText()], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');

        link.href = url;
        link.download = `${filenameName || 'patient'}-assessment-report.txt`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
    });
}

function setupCopyTranscript() {
    const button = document.getElementById('copyTranscriptBtn');
    if (!button) return;

    button.addEventListener('click', async () => {
        const text = data.transcript || '';
        if (!text) return;

        try {
            await navigator.clipboard.writeText(text);
            button.textContent = 'Copied';
        } catch (error) {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            textarea.remove();
            button.textContent = 'Copied';
        }

        setTimeout(() => {
            button.textContent = 'Copy Transcript';
        }, 1600);
    });
}

saveCurrentAssessment();
renderPatientSummary();
renderPrediction();
renderConfidence();
renderTranscript();
renderClinicalFactors();
renderSpeechFactors();
renderOverrideWarning();
setupDownloadReport();
setupCopyTranscript();
setupHistorySearch();
setupSectionNavigation();
updateStats();
