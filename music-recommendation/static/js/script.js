const video = document.getElementById('video');
const captureBtn = document.getElementById('capture-btn');
const analyzeTextBtn = document.getElementById('analyze-text-btn');
const analyzeBothBtn = document.getElementById('analyze-both-btn');
const emotionDisplay = document.getElementById('emotion-display');
const playlistsDiv = document.getElementById('playlists');
const loadingOverlay = document.getElementById('loading');
const resultSection = document.getElementById('result-section');
const textInput = document.getElementById('text-input');

// Emoji mapping for emotions
const emotionEmojis = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'fear': '😨',
    'surprise': '😲',
    'neutral': '😐',
    'disgust': '🤢',
    'excited': '🤩',
    'calm': '😌',
    'energetic': '⚡',
    'chill': '😎'
};

// Initialize webcam
let stream = null;
let cameraAvailable = false;

navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
    .then(s => {
        stream = s;
        video.srcObject = stream;
        cameraAvailable = true;
        captureBtn.disabled = false;
    })
    .catch(err => {
        console.error('Error accessing webcam:', err);
        cameraAvailable = false;
        captureBtn.disabled = true;
        captureBtn.innerHTML = '<i class="fas fa-camera-slash"></i> Camera Unavailable';
        captureBtn.title = 'Camera access denied or not available. Please check browser permissions or use text emotion detection instead.';
        
        // Hide video element and show a placeholder
        video.style.display = 'none';
        const videoContainer = document.querySelector('.video-container');
        const placeholder = document.createElement('div');
        placeholder.style.cssText = 'padding: 60px 20px; text-align: center; background: #f5f5f5; border-radius: 15px;';
        placeholder.innerHTML = '<i class="fas fa-camera-slash" style="font-size: 48px; color: #999; margin-bottom: 15px; display: block;"></i><p style="color: #666;">Camera not available</p><p style="font-size: 0.9rem; color: #999;">Please use text emotion detection</p>';
        videoContainer.appendChild(placeholder);
    });

// Capture emotion from webcam
captureBtn.addEventListener('click', async () => {
    if (!cameraAvailable || !stream) {
        alert('⚠️ Camera Not Available\n\nFacial emotion detection requires camera access. Please:\n\n1. Enable camera permissions in your browser\n2. Reload the page\n3. Or use Text Emotion Detection instead');
        return;
    }

    showLoading();
    
    const canvas = document.getElementById('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    const imageData = canvas.toDataURL('image/jpeg');

    try {
        const response = await fetch('/detect_emotion', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData }),
        });

        const data = await response.json();
        hideLoading();

        if (response.ok) {
            displayResult(data);
        } else {
            showError(data.error || 'Failed to detect emotion. Please try again.');
        }
    } catch (error) {
        hideLoading();
        console.error('Error:', error);
        showError('Network error. Please check your connection and try again.');
    }
});

// Analyze text emotion
analyzeTextBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();

    if (!text) {
        showError('Please enter some text to analyze.');
        return;
    }

    showLoading();

    try {
        const response = await fetch('/detect_emotion', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        });

        const data = await response.json();
        hideLoading();

        if (response.ok) {
            displayResult(data);
        } else {
            showError(data.error || 'Failed to analyze text. Please try again.');
        }
    } catch (error) {
        hideLoading();
        console.error('Error:', error);
        showError('Network error. Please check your connection and try again.');
    }
});

// Analyze both text and webcam image
analyzeBothBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();

    if (!cameraAvailable) {
        alert('⚠️ Camera Not Available\n\nFacial emotion detection requires camera access. Please enable camera permissions or use Analyze Text only.');
        return;
    }

    showLoading();

    const canvas = document.getElementById('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    const imageData = canvas.toDataURL('image/jpeg');

    try {
        const response = await fetch('/detect_emotion', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData, text: text })
        });

        const data = await response.json();
        hideLoading();

        if (response.ok) {
            displayResult(data);
        } else {
            showError(data.error || 'Failed to detect emotion. Please try again.');
        }
    } catch (error) {
        hideLoading();
        console.error('Error:', error);
        showError('Network error. Please check your connection and try again.');
    }
});

// Enable button on text input
textInput.addEventListener('input', () => {
    analyzeTextBtn.disabled = textInput.value.trim().length === 0;
});

// Display emotion and playlists
function displayResult(data) {
    resultSection.classList.add('show');
    
    const emotionEmoji = emotionEmojis[data.emotion] || emotionEmojis[data.mood] || '🎵';
    
    emotionDisplay.innerHTML = `
        <div style="font-size: 3rem; margin-bottom: 15px;">${emotionEmoji}</div>
        <div>Detected Emotion: <strong>${capitalizeFirst(data.emotion)}</strong></div>
        <div style="margin-top: 10px;">Mood: <strong>${capitalizeFirst(data.mood)}</strong></div>
    `;

    if (data.playlists && data.playlists.length > 0) {
        playlistsDiv.innerHTML = data.playlists.map(playlist => `
            <div class="playlist">
                <img src="${playlist.image || 'https://via.placeholder.com/200x200?text=No+Image'}" 
                     alt="${escapeHtml(playlist.name)}" 
                     onerror="this.src='https://via.placeholder.com/200x200?text=No+Image'">
                <a href="${playlist.url}" target="_blank" rel="noopener noreferrer">
                    ${escapeHtml(playlist.name)}
                </a>
            </div>
        `).join('');
    } else {
        playlistsDiv.innerHTML = `
            <div style="text-align: center; padding: 40px; color: #999; grid-column: 1/-1;">
                <i class="fas fa-music" style="font-size: 3rem; margin-bottom: 15px;"></i>
                <p>No playlists found. Please check your Spotify API configuration.</p>
            </div>
        `;
    }

    // Smooth scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Show loading overlay
function showLoading() {
    loadingOverlay.classList.add('show');
}

// Hide loading overlay
function hideLoading() {
    loadingOverlay.classList.remove('show');
}

// Show error message
function showError(message) {
    resultSection.classList.add('show');
    emotionDisplay.innerHTML = `
        <div style="font-size: 3rem; margin-bottom: 15px;">⚠️</div>
        <div style="color: #ffebee;">${escapeHtml(message)}</div>
    `;
    playlistsDiv.innerHTML = '';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Helper function to capitalize first letter
function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// Helper function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Clean up webcam stream on page unload
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});
