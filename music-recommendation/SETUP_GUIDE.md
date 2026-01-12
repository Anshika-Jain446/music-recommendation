# Music Recommendation App - Setup Guide

## ✅ Current Status

The app is **fully functional** with beautiful modern design and text emotion detection!

### Working Features:
- ✨ **Text Emotion Detection** - Fully working!
- 🎨 Modern gradient design with animations
- 📱 Responsive layout
- ⚡ Loading states and error handling
- 🎵 Music mood mapping

### Optional Features (Require Setup):
- 📷 **Facial Emotion Detection** - Requires DeepFace setup (complex)
- 🎧 **Spotify Playlists** - Requires API credentials

---

## 🚀 Quick Start

1. **Activate the virtual environment** (if not already activated):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Run the app**:
   ```powershell
   python app.py
   ```

3. **Open in browser**: http://127.0.0.1:5000

4. **Try it out**: Use the **Text Emotion Detection** feature - it works perfectly!

---

## 🎧 Enable Spotify Playlists (Optional)

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create an app and get credentials
3. Set environment variables:
   ```powershell
   $env:SPOTIFY_CLIENT_ID="your_client_id"
   $env:SPOTIFY_CLIENT_SECRET="your_client_secret"
   ```
4. Restart the app

---

## 📷 About Camera/Facial Detection

The camera button shows "Camera Unavailable" because:
1. **Browser permissions** - Camera access may be blocked
2. **DeepFace library** - Requires complex setup with TensorFlow models

**Recommendation**: Use the **Text Emotion Detection** feature instead - it works great!

---

## 🐛 Troubleshooting

### Text analysis not working?
- Check if emoji library is installed: `pip install "emoji<2.0.0"`
- Restart the Flask app

### Camera not working?
- This is expected - use Text Emotion Detection instead
- Camera requires browser permissions and additional DeepFace setup

### No playlists showing?
- Set up Spotify API credentials (see above)
- Without credentials, the app still detects emotions but won't show playlists

---

## 📦 Dependencies

All required packages are in `requirements.txt`:
```
Flask==2.3.3
text2emotion==0.0.5
spotipy==2.23.0
opencv-python==4.8.0.76
requests==2.31.0
deepface==0.0.79
tf-keras==2.15.0
numpy<2.0.0
emoji<2.0.0
```

Install with:
```powershell
pip install -r requirements.txt
```

---

## 🎨 Features

- **Modern UI**: Gradient backgrounds with smooth animations
- **Text Analysis**: Detects emotions from your text input
- **Mood Mapping**: Maps emotions to music moods
- **Responsive**: Works on desktop and mobile
- **Error Handling**: Clear, helpful error messages

Enjoy your emotion-based music recommendations! 🎵✨
