// ============================================================
// TAB SWITCHING
// ============================================================
function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById('liveTab').style.display = 'none';
    document.getElementById('uploadTab').style.display = 'none';
    
    if (tab === 'live') {
        document.querySelector('.tab-btn:first-child').classList.add('active');
        document.getElementById('liveTab').style.display = 'block';
    } else {
        document.querySelector('.tab-btn:last-child').classList.add('active');
        document.getElementById('uploadTab').style.display = 'block';
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
    // ============================================================
    // VALIDATION: Age and MMSE are MANDATORY
    // ============================================================
    const ageInput = document.getElementById('age');
    const mmseInput = document.getElementById('mmse');
    
    if (!ageInput || !ageInput.value || ageInput.value.trim() === '') {
        statusDiv.innerText = '❌ Age is required. Please enter patient age.';
        return;
    }
    
    if (!mmseInput || !mmseInput.value || mmseInput.value.trim() === '') {
        statusDiv.innerText = '❌ MMSE score is required. Please enter MMSE (0-30).';
        return;
    }
    
    const ageNum = parseInt(ageInput.value);
    const mmseNum = parseInt(mmseInput.value);
    
    if (isNaN(ageNum) || ageNum < 40 || ageNum > 100) {
        statusDiv.innerText = '❌ Age must be between 40 and 100.';
        return;
    }
    
    if (isNaN(mmseNum) || mmseNum < 0 || mmseNum > 30) {
        statusDiv.innerText = '❌ MMSE score must be between 0 and 30.';
        return;
    }
    // ============================================================
    
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
            timerDiv.innerText = `${minutes}:${seconds.toString().padStart(2, '0')} / 1:00`;
            
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
                statusDiv.innerText = '⚠️ Recording too short! Please speak for at least 3 seconds.';
                resetUI();
                return;
            }
            
            // Get clinical data (already validated)
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
                statusDiv.innerText = '🔍 Analyzing...';
                loader.classList.remove('hidden');
                
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
                    sessionStorage.setItem('prediction', JSON.stringify(data));
                    window.location.href = '/result';
                } else {
                    loader.classList.add('hidden');
                    statusDiv.innerText = `❌ Error: ${data.error}`;
                    resetUI();
                }
            };
        };
        
        mediaRecorder.start(1000);
        recordBtn.disabled = true;
        stopBtn.disabled = false;
        statusDiv.innerText = '🔴 Recording... (max 60 seconds)';
        
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                stream.getTracks().forEach(track => track.stop());
            }
        }, 60000);
        
    } catch (error) {
        statusDiv.innerText = '❌ Microphone access denied';
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
        uploadStatus.innerText = '⚠️ Please select an audio file';
        return;
    }
    
    // ============================================================
    // VALIDATION: Age and MMSE are MANDATORY for upload tab
    // ============================================================
    const ageInput = document.getElementById('upload_age');
    const mmseInput = document.getElementById('upload_mmse');
    
    if (!ageInput || !ageInput.value || ageInput.value.trim() === '') {
        uploadStatus.innerText = '❌ Age is required. Please enter patient age.';
        return;
    }
    
    if (!mmseInput || !mmseInput.value || mmseInput.value.trim() === '') {
        uploadStatus.innerText = '❌ MMSE score is required. Please enter MMSE (0-30).';
        return;
    }
    
    const ageNum = parseInt(ageInput.value);
    const mmseNum = parseInt(mmseInput.value);
    
    if (isNaN(ageNum) || ageNum < 40 || ageNum > 100) {
        uploadStatus.innerText = '❌ Age must be between 40 and 100.';
        return;
    }
    
    if (isNaN(mmseNum) || mmseNum < 0 || mmseNum > 30) {
        uploadStatus.innerText = '❌ MMSE score must be between 0 and 30.';
        return;
    }
    // ============================================================
    
    // Get clinical data (using validated values)
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
        uploadStatus.innerText = '📝 Using provided transcript...';
    } else {
        uploadStatus.innerText = '🎙️ No transcript provided. Auto-transcribing with Whisper...';
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
            sessionStorage.setItem('prediction', JSON.stringify(data));
            window.location.href = '/result';
        } else {
            uploadStatus.innerText = `❌ Error: ${data.error}`;
        }
    } catch (error) {
        uploadLoader.classList.add('hidden');
        uploadStatus.innerText = `❌ Network error: ${error.message}`;
    }
};

