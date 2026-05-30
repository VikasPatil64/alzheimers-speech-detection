const HISTORY_KEY = 'alzheimersPatientHistory';

// ============================================================
// SHARED PATIENT HISTORY HELPERS
// ============================================================
function readHistory() {
    try {
        return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
    } catch (error) {
        console.warn('Could not read patient history:', error);
        return [];
    }
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function writeCurrentPrediction(data, mode, metadata) {
    const payload = {
        ...data,
        assessmentId: metadata.assessmentId,
        assessmentMode: mode,
        savedAt: metadata.savedAt,
        patient: metadata.patient,
        localHistorySaved: false
    };

    sessionStorage.setItem('prediction', JSON.stringify(payload));
}

function fieldValue(id) {
    const field = document.getElementById(id);
    return field ? field.value.trim() : '';
}

function selectText(id) {
    const select = document.getElementById(id);
    if (!select) return 'Not provided';
    return select.options[select.selectedIndex]?.text || 'Not provided';
}

function getPatientMetadata(prefix = '') {
    const key = prefix ? `${prefix}_` : '';
    return {
        assessmentId: `assessment-${Date.now()}-${Math.random().toString(16).slice(2)}`,
        savedAt: new Date().toISOString(),
        patient: {
            name: fieldValue(`${key}patientName`) || 'Unnamed Patient',
            id: fieldValue(`${key}patientId`) || 'Not provided',
            age: fieldValue(`${key}age`),
            gender: selectText(`${key}gender`),
            education: fieldValue(`${key}education`),
            mmse: fieldValue(`${key}mmse`)
        }
    };
}

function formatHistoryDate(value) {
    if (!value) return 'No date';
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

function updateStats() {
    const records = readHistory().sort((a, b) => new Date(b.savedAt) - new Date(a.savedAt));
    const countEl = document.getElementById('savedAssessmentsCount');
    const lastEl = document.getElementById('lastAnalysisLabel');
    const sessionEl = document.getElementById('activeSessionLabel');

    if (countEl) countEl.textContent = records.length;
    if (lastEl) lastEl.textContent = records[0] ? compactDate(records[0].savedAt) : 'None';
    if (sessionEl) sessionEl.textContent = 'Ready';
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
            <p>${escapeHtml(record.prediction || 'Prediction unavailable')} ${record.confidenceLabel ? `- ${escapeHtml(record.confidenceLabel)}` : ''}</p>
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

function updateRecordingIndicator(label, isRecording = false) {
    const indicator = document.getElementById('recordingIndicator');
    const sessionEl = document.getElementById('activeSessionLabel');

    if (indicator) {
        const text = indicator.querySelector('strong');
        indicator.classList.toggle('is-recording', isRecording);
        if (text) text.textContent = label;
    }

    if (sessionEl) sessionEl.textContent = label;
}

// ============================================================
// TAB SWITCHING
// ============================================================
function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById('liveTab').style.display = 'none';
    document.getElementById('uploadTab').style.display = 'none';

    const activeModeLabel = document.getElementById('activeModeLabel');

    if (tab === 'live') {
        document.querySelector('.tab-btn:first-child').classList.add('active');
        document.getElementById('liveTab').style.display = 'block';
        if (activeModeLabel) activeModeLabel.textContent = 'Live microphone';
        updateRecordingIndicator('Ready');
    } else {
        document.querySelector('.tab-btn:last-child').classList.add('active');
        document.getElementById('uploadTab').style.display = 'block';
        if (activeModeLabel) activeModeLabel.textContent = 'File upload';
        updateRecordingIndicator('Upload mode');
    }
}

// ============================================================
// LIVE MICROPHONE RECORDING
// ============================================================
let mediaRecorder;
let audioChunks = [];
let startTime;
let timerInterval;

const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDiv = document.getElementById('status');
const loader = document.getElementById('loader');
const timerContainer = document.getElementById('timerContainer');
const timerDiv = document.getElementById('timer');
const warningDiv = document.getElementById('warning');

recordBtn.onclick = async () => {
    const ageInput = document.getElementById('age');
    const mmseInput = document.getElementById('mmse');

    if (!ageInput || !ageInput.value || ageInput.value.trim() === '') {
        statusDiv.innerText = 'Age is required. Please enter patient age.';
        return;
    }

    if (!mmseInput || !mmseInput.value || mmseInput.value.trim() === '') {
        statusDiv.innerText = 'MMSE score is required. Please enter MMSE (0-30).';
        return;
    }

    const ageNum = parseInt(ageInput.value);
    const mmseNum = parseInt(mmseInput.value);

    if (isNaN(ageNum) || ageNum < 40 || ageNum > 100) {
        statusDiv.innerText = 'Age must be between 40 and 100.';
        return;
    }

    if (isNaN(mmseNum) || mmseNum < 0 || mmseNum > 30) {
        statusDiv.innerText = 'MMSE score must be between 0 and 30.';
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        startTime = Date.now();

        timerContainer.classList.remove('hidden');
        timerDiv.innerText = '0:00 / 1:00';
        warningDiv.classList.add('hidden');

        timerInterval = setInterval(() => {
            const elapsed = (Date.now() - startTime) / 1000;
            const minutes = Math.floor(elapsed / 60);
            const seconds = Math.floor(elapsed % 60);
            const clock = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            timerDiv.innerText = `${clock} / 1:00`;
            updateRecordingIndicator(`Recording - ${clock}`, true);

            if (elapsed < 30) {
                warningDiv.classList.remove('hidden');
            } else {
                warningDiv.classList.add('hidden');
            }
        }, 100);

        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            clearInterval(timerInterval);
            const duration = (Date.now() - startTime) / 1000;

            if (duration < 3) {
                statusDiv.innerText = 'Recording too short. Please speak for at least 3 seconds.';
                resetUI();
                return;
            }

            const clinicalData = {
                age_at_visit: ageNum,
                gender: parseInt(document.getElementById('gender').value),
                education_years: parseInt(document.getElementById('education').value) || null,
                mmse_score: mmseNum
            };

            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const reader = new FileReader();
            reader.readAsDataURL(audioBlob);
            reader.onloadend = async () => {
                statusDiv.innerText = 'Analyzing...';
                loader.classList.remove('hidden');
                updateRecordingIndicator('Analyzing');

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            audio: reader.result.split(',')[1],
                            clinical: clinicalData,
                            duration: duration
                        })
                    });

                    const data = await response.json();

                    if (data.success) {
                        writeCurrentPrediction(data, 'Live microphone', getPatientMetadata());
                        window.location.href = '/result';
                    } else {
                        loader.classList.add('hidden');
                        statusDiv.innerText = `Error: ${data.error}`;
                        resetUI();
                    }
                } catch (error) {
                    loader.classList.add('hidden');
                    statusDiv.innerText = `Network error: ${error.message}`;
                    resetUI();
                }
            };
        };

        mediaRecorder.start(1000);
        recordBtn.disabled = true;
        stopBtn.disabled = false;
        statusDiv.innerText = 'Recording... (max 60 seconds)';
        document.body.classList.add('is-recording');
        updateRecordingIndicator('Recording - 0:00', true);

        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                stream.getTracks().forEach(track => track.stop());
            }
        }, 60000);

    } catch (error) {
        statusDiv.innerText = 'Microphone access denied';
        updateRecordingIndicator('Microphone blocked');
        console.error(error);
    }
};

stopBtn.onclick = () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        if (mediaRecorder.stream) {
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }
    resetUI();
};

function resetUI() {
    recordBtn.disabled = false;
    stopBtn.disabled = true;
    timerContainer.classList.add('hidden');
    document.body.classList.remove('is-recording');
    updateRecordingIndicator('Ready');
}

// ============================================================
// FILE UPLOAD HANDLER
// ============================================================
const uploadBtn = document.getElementById('uploadBtn');
const audioFileInput = document.getElementById('audioFile');
const transcriptFileInput = document.getElementById('transcriptFile');
const uploadStatus = document.getElementById('uploadStatus');
const uploadLoader = document.getElementById('uploadLoader');

uploadBtn.onclick = async () => {
    const audioFile = audioFileInput.files[0];
    if (!audioFile) {
        uploadStatus.innerText = 'Please select an audio file.';
        return;
    }

    const ageInput = document.getElementById('upload_age');
    const mmseInput = document.getElementById('upload_mmse');

    if (!ageInput || !ageInput.value || ageInput.value.trim() === '') {
        uploadStatus.innerText = 'Age is required. Please enter patient age.';
        return;
    }

    if (!mmseInput || !mmseInput.value || mmseInput.value.trim() === '') {
        uploadStatus.innerText = 'MMSE score is required. Please enter MMSE (0-30).';
        return;
    }

    const ageNum = parseInt(ageInput.value);
    const mmseNum = parseInt(mmseInput.value);

    if (isNaN(ageNum) || ageNum < 40 || ageNum > 100) {
        uploadStatus.innerText = 'Age must be between 40 and 100.';
        return;
    }

    if (isNaN(mmseNum) || mmseNum < 0 || mmseNum > 30) {
        uploadStatus.innerText = 'MMSE score must be between 0 and 30.';
        return;
    }

    const clinicalData = {
        age_at_visit: ageNum,
        gender: parseInt(document.getElementById('upload_gender').value),
        education_years: parseInt(document.getElementById('upload_education').value) || null,
        mmse_score: mmseNum
    };

    const formData = new FormData();
    formData.append('audio', audioFile);
    formData.append('clinical', JSON.stringify(clinicalData));

    const transcriptFile = transcriptFileInput.files[0];
    if (transcriptFile) {
        formData.append('transcript', transcriptFile);
        uploadStatus.innerText = 'Using provided transcript...';
        const sessionEl = document.getElementById('activeSessionLabel');
        if (sessionEl) sessionEl.textContent = 'Transcript provided';
    } else {
        uploadStatus.innerText = 'No transcript provided. Auto-transcribing with Whisper...';
        const sessionEl = document.getElementById('activeSessionLabel');
        if (sessionEl) sessionEl.textContent = 'Auto-transcribing';
    }

    uploadLoader.classList.remove('hidden');

    try {
        const response = await fetch('/analyze_upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        uploadLoader.classList.add('hidden');

        if (data.success) {
            writeCurrentPrediction(data, 'File upload', getPatientMetadata('upload'));
            window.location.href = '/result';
        } else {
            uploadStatus.innerText = `Error: ${data.error}`;
        }
    } catch (error) {
        uploadLoader.classList.add('hidden');
        uploadStatus.innerText = `Network error: ${error.message}`;
    }
};

setupHistorySearch();
setupSectionNavigation();
updateStats();
