# 🎵 AI-Based Emotion Recognition Playlist Maker

A beautiful web application that detects your emotions through text and recommends music based on your mood. Built with Flask, machine learning, and Spotify API integration.

## ✨ Features

- **Text Emotion Detection** - Analyzes your text to detect emotions
- **Facial Emotion Detection** - Recognizes emotions from your face (optional)
- **Smart Mood Mapping** - Maps emotions to curated music moods
- **Spotify Integration** - Recommends playlists based on detected mood
- **Beautiful UI** - Modern gradient design with smooth animations
- **Responsive Design** - Works perfectly on desktop and mobile
- **Real-time Processing** - Instant emotion analysis and recommendations

## 🛠️ Tech Stack

### Backend
- **Flask 2.3.3** - Python web framework
- **text2emotion 0.0.5** - AI-powered text emotion detection
- **DeepFace 0.0.79** - Facial emotion recognition
- **Spotipy 2.23.0** - Spotify API client
- **OpenCV 4.8.0.76** - Computer vision library
- **TensorFlow/tf-keras** - Deep learning framework

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Modern styling with animations and gradients
- **JavaScript ES6+** - Vanilla JS (no frameworks)
- **Font Awesome 6.4.0** - Icon library
- **Google Fonts (Poppins)** - Typography

### Tools & Libraries
- **NumPy** - Numerical computing
- **NLTK** - Natural language processing
- **Python Virtual Environment** - Dependency isolation

## 📋 Requirements

- Python 3.8+
- pip (Python package manager)
- Spotify account (optional, for real API integration)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/music-recommendation.git
cd music-recommendation
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

### 5. Open in Browser
Navigate to: `http://127.0.0.1:5000`

## 🎧 Features Usage

### Text Emotion Detection (Fully Working)
1. Enter your text in the "Text Emotion Detection" section
2. Click "Analyze Text"
3. The app detects your emotion and mood
4. Browse recommended playlists
5. Click a playlist to open it in Spotify

### Facial Emotion Detection (Optional)
- Requires webcam access and additional setup
- Recommended: Use text emotion detection instead

## 🔌 Spotify Integration (Optional)

To enable real Spotify API integration:

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app and get credentials
3. Set environment variables:
   ```powershell
   $env:SPOTIFY_CLIENT_ID="your_client_id"
   $env:SPOTIFY_CLIENT_SECRET="your_client_secret"
   ```
4. Restart the app

**Note**: Without API setup, the app still works with sample playlists!

## 📊 Emotion Mapping

The app maps detected emotions to music moods:

| Emotion | Mood |
|---------|------|
| Happy | Happy |
| Sad | Sad |
| Angry | Energetic |
| Fear | Calm |
| Surprise | Excited |
| Neutral | Chill |
| Love | Romantic |

## 🏗️ Project Structure

```
music-recommendation/
├── app.py                 # Flask backend
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
├── README.md             # This file
├── SETUP_GUIDE.md        # Detailed setup guide
├── TODO.md               # Todo list
├── templates/
│   └── index.html        # Main HTML page
└── static/
    ├── css/
    │   └── style.css     # Styling
    └── js/
        └── script.js     # Frontend logic
```

## 🔧 Configuration Files

### requirements.txt
Contains all Python dependencies. Install with:
```bash
pip install -r requirements.txt
```

### .env (Optional)
Create a `.env` file for sensitive data:
```
SPOTIFY_CLIENT_ID=your_id
SPOTIFY_CLIENT_SECRET=your_secret
```

## 🎨 UI Features

- **Gradient Background** - Beautiful purple/pink gradient
- **Smooth Animations** - Fade-in, bounce, pulse effects
- **Loading States** - Visual feedback during processing
- **Error Handling** - Clear, helpful error messages
- **Responsive Grid** - Auto-adjusting layouts
- **Icon Integration** - Font Awesome icons throughout

## 📱 Browser Compatibility

- Chrome/Chromium (Latest)
- Firefox (Latest)
- Safari (Latest)
- Edge (Latest)

## 🐛 Troubleshooting

### Issue: "Camera Unavailable"
- **Solution**: Use text emotion detection (recommended)
- Camera requires browser permissions and DeepFace setup

### Issue: "Text analysis failed"
- **Solution**: Make sure emoji library is installed
- Run: `pip install "emoji<2.0.0"`

### Issue: No playlists showing
- **Solution**: Set up Spotify API credentials (see above)
- App still detects emotions without credentials

### Issue: Port 5000 already in use
- **Solution**: Change port in app.py or kill the process using it

## 🚀 Future Enhancements

- [ ] Real-time emotion detection from webcam stream
- [ ] User authentication
- [ ] Save favorite playlists
- [ ] Emotion history tracking
- [ ] Deploy to production
- [ ] Mobile app version
- [ ] Voice emotion detection

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created by Anshika Jain


## 🙏 Acknowledgments

- [Spotify](https://www.spotify.com/) - Music streaming & API
- [text2emotion](https://github.com/tissafary/text2emotion) - Text emotion detection
- [DeepFace](https://github.com/serengalp/deepface) - Facial emotion recognition
- [Font Awesome](https://fontawesome.com/) - Icon library
- [Google Fonts](https://fonts.google.com/) - Typography

## 📞 Support

For issues or questions, please create an issue on GitHub or contact the author.

---

**Made with ❤️ using AI & Music** 🎵✨
