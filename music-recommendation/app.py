from flask import Flask, request, jsonify, render_template
import cv2
import base64
import numpy as np
import json
import re
# DeepFace may import TensorFlow and take long at startup. Do a lazy import when an image is provided.
DEEPFACE_AVAILABLE = False  # Will attempt runtime import when needed
import text2emotion as te
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

app = Flask(__name__)

# Spotify API credentials (set these as environment variables)
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

# Initialize Spotify client
if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET))
else:
    sp = None
    print("Spotify API credentials not set. Playlist functionality will be disabled.")

# Emotion to mood mapping
emotion_to_mood = {
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'energetic',
    'fear': 'calm',
    'surprise': 'excited',
    'neutral': 'chill',
    'disgust': 'dark',
    'joy': 'happy',
    'anticipation': 'excited',
    'trust': 'calm',
    'positive': 'happy',
    'negative': 'sad',
    'worry': 'calm',
    'love': 'romantic'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    # Prefer a robust JSON parse (handles various clients)
    data = request.get_json(force=True, silent=True) or request.json
    print(f"[DEBUG] /detect_emotion called with data: {json.dumps(data) if data is not None else 'None'}")
    emotion = None
    detection_method = None
    # Facial emotion detection (try but don't fail if DeepFace missing)
    facial_emotion = None
    facial_confidence = 0.0
    if 'image' in data:
        # Lazy import DeepFace to avoid heavy imports at startup
        use_deepface = False
        try:
            from deepface import DeepFace
            use_deepface = True
        except Exception as e:
            print('DeepFace not available at runtime; skipping facial analysis:', e)

        if use_deepface:
            try:
                image_data = base64.b64decode(data['image'].split(',')[1])
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    print("Error: Could not decode image")
                else:
                    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                    # result can be a list or dict
                    if isinstance(result, list):
                        res = result[0]
                    else:
                        res = result

                    # DeepFace returns an 'emotion' dict with scores and 'dominant_emotion'
                    emotions_dict = res.get('emotion') if isinstance(res, dict) else None
                    if emotions_dict:
                        facial_emotion = max(emotions_dict, key=emotions_dict.get)
                        facial_confidence = float(emotions_dict.get(facial_emotion, 0.0)) / 100.0 if max(emotions_dict.values()) > 1.0 else float(emotions_dict.get(facial_emotion, 0.0))
                    else:
                        # Fallback to dominant_emotion
                        facial_emotion = res.get('dominant_emotion') if isinstance(res, dict) else None
                        facial_confidence = 0.0
                    detection_method = 'facial'
            except Exception as e:
                print(f"Error in facial emotion detection: {e}")
                facial_emotion = None
                facial_confidence = 0.0
        else:
            # DeepFace not available at runtime — log and continue with text detection if present
            print('Received image but DeepFace not available; skipping facial analysis.')
    print(f"[DEBUG] facial_emotion={facial_emotion}, facial_confidence={facial_confidence}")

    # Text emotion detection
    text_emotion = None
    text_confidence = 0.0
    if 'text' in data and data['text'].strip():
        try:
            text = data['text'].lower().strip()
            
            # Enhanced keyword-based emotion detection (English + Hindi keywords)
            emotion_keywords = {
                'happy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', 'love', 'glad', 'cheerful', 'delighted', 'thrilled', 'excellent', 'good', 'awesome', 'perfect', 'खुश', 'खुशी', 'खुश हूँ', 'मज़ा', 'उत्साहित'],
                'sad': ['sad', 'unhappy', 'depressed', 'lonely', 'miserable', 'down', 'blue', 'disappointed', 'upset', 'hurt', 'heartbroken', 'crying', 'tears', 'awful', 'terrible', 'bad', 'उदास', 'दुख', 'टूट', 'दुखी'],
                'angry': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated', 'rage', 'pissed', 'hate', 'disgusted', 'outraged', 'गुस्सा', 'क्रोधित', 'नाराज़'],
                'fear': ['scared', 'afraid', 'fear', 'anxious', 'worried', 'nervous', 'terrified', 'frightened', 'panic', 'डर', 'घबराहट', 'डरा'],
                'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'wow', 'incredible', 'unbelievable', 'हैरान', 'चकित'],
                'neutral': ['okay', 'fine', 'alright', 'normal', 'meh', 'whatever', 'ठीक', 'अच्छा', 'सामान्य']
            }

            # Count keyword matches
            emotion_scores = {}
            for emotion_key, keywords in emotion_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text)
                if score > 0:
                    emotion_scores[emotion_key] = score

            # Detect if text contains Devanagari characters -> treat as Hindi
            is_hindi = bool(re.search(r'[\u0900-\u097F]', text))
            language = 'hi' if is_hindi else 'en'

            if is_hindi:
                # For Hindi, rely on keyword matching only (text2emotion is English-only)
                if emotion_scores:
                    text_emotion = max(emotion_scores, key=emotion_scores.get)
                    text_confidence = float(min(emotion_scores[text_emotion] / 5.0, 1.0))
                    detection_method = 'text'
                else:
                    return jsonify({'error': 'Could not detect emotion from text. Please provide more descriptive text about your feelings.'}), 400
            else:
                # Try text2emotion first for English text
                text_emotions = te.get_emotion(data['text'])

                # Combine both methods
                if text_emotions and any(text_emotions.values()):
                    te_emotion = max(text_emotions, key=text_emotions.get)
                    te_score = text_emotions[te_emotion]
                    if emotion_scores:
                        keyword_emotion = max(emotion_scores, key=emotion_scores.get)
                        keyword_score = emotion_scores[keyword_emotion]
                        if keyword_score >= 2 or (keyword_score > 0 and te_score < 0.3):
                            text_emotion = keyword_emotion
                            text_confidence = float(min(keyword_score / 5.0, 1.0))
                        else:
                            text_emotion = te_emotion
                            text_confidence = float(te_score)
                    else:
                        text_emotion = te_emotion
                        text_confidence = float(te_score)
                    detection_method = 'text'
                elif emotion_scores:
                    text_emotion = max(emotion_scores, key=emotion_scores.get)
                    text_confidence = float(min(emotion_scores[text_emotion] / 5.0, 1.0))
                    detection_method = 'text'
                else:
                    return jsonify({'error': 'Could not detect emotion from text. Please provide more descriptive text about your feelings.'}), 400
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"Error in text emotion detection: {e}\n{tb}")
            # Return exception details in response for local debugging
            return jsonify({'error': 'Text analysis failed. Please try again.', 'exception': str(e), 'trace': tb}), 400
    else:
        print('[DEBUG] No text provided or text empty')

    print(f"[DEBUG] text_emotion={text_emotion}, text_confidence={text_confidence}")

    # Combine facial and text detections if both available
    final_emotion = None
    final_method = detection_method
    confidence = 0.0

    # Prefer the signal with higher confidence when both present
    if facial_emotion and text_emotion:
        # Compare top confidences
        if facial_confidence >= text_confidence:
            final_emotion = facial_emotion
            confidence = facial_confidence
            final_method = 'facial' if facial_confidence > text_confidence else 'combined'
        else:
            final_emotion = text_emotion
            confidence = text_confidence
            final_method = 'text' if text_confidence > facial_confidence else 'combined'
    elif facial_emotion:
        final_emotion = facial_emotion
        confidence = facial_confidence
        final_method = 'facial'
    elif text_emotion:
        final_emotion = text_emotion
        confidence = text_confidence
        final_method = 'text'

    if final_emotion:
        emotion_lower = final_emotion.lower()
        mood = emotion_to_mood.get(emotion_lower, 'chill')
        playlists = get_spotify_playlists(mood, language)

        # Debug logging
        print(f"Detected emotion (final): {final_emotion} -> Mood: {mood} (method={final_method}, confidence={confidence}, language={language})")

        return jsonify({
            'emotion': emotion_lower,
            'mood': mood,
            'language': language,
            'playlists': playlists,
            'method': final_method,
            'confidence': confidence,
            'facial_emotion': facial_emotion,
            'facial_confidence': facial_confidence,
            'text_emotion': text_emotion,
            'text_confidence': text_confidence
        })
    else:
        print('[DEBUG] final_emotion is None - returning 400')
        return jsonify({'error': 'Could not detect emotion. Please try again with clearer input.'}), 400

def get_spotify_playlists(mood, language='en'):
    if sp is None:
        # Return sample playlists when Spotify API is not configured
        if language == 'hi':
            sample_playlists_hi = {
                'happy': [
                    {'name': 'Bollywood Happy Hits', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX2taZN6Kf4K1', 'image': 'https://via.placeholder.com/300x300/FFD700/000000?text=Bollywood+Happy'},
                    {'name': 'Top Bollywood', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DXcZ6X5YK6xG6', 'image': 'https://via.placeholder.com/300x300/FF6B9D/000000?text=Top+Bollywood'},
                    {'name': 'Bollywood Retro', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX2yvmlOdMYzV', 'image': 'https://via.placeholder.com/300x300/00D4FF/000000?text=Bollywood+Retro'}
                ],
                'sad': [
                    {'name': 'Bollywood Sad', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWSf2RDTDayIx', 'image': 'https://via.placeholder.com/300x300/4169E1/FFFFFF?text=Bollywood+Sad'},
                    {'name': 'Sad Bollywood', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1', 'image': 'https://via.placeholder.com/300x300/708090/FFFFFF?text=Sad+Bollywood'},
                    {'name': 'Melancholic Bollywood', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWX83CujKHHOn', 'image': 'https://via.placeholder.com/300x300/2F4F4F/FFFFFF?text=Melancholy+Bollywood'}
                ],
                'chill': [
                    {'name': 'Bollywood Mellow', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX889U0CL85jj', 'image': 'https://via.placeholder.com/300x300/9370DB/000000?text=Bollywood+Chill'},
                    {'name': 'Indie Hindi Chill', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWWQRwui0ExPn', 'image': 'https://via.placeholder.com/300x300/BA55D3/000000?text=Indie+Hindi'},
                    {'name': 'Romantic Bollywood', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX50QitC6Oqtn', 'image': 'https://via.placeholder.com/300x300/FF1493/FFFFFF?text=Romantic+Bollywood'}
                ]
            }
            return sample_playlists_hi.get(mood, sample_playlists_hi['chill'])
        else:
            sample_playlists = {
                'happy': [
                    {'name': 'Happy Hits', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC', 'image': 'https://via.placeholder.com/300x300/FFD700/000000?text=Happy+Hits'},
                    {'name': 'Feel Good Indie', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX2sUQwD7tbmL', 'image': 'https://via.placeholder.com/300x300/FF6B9D/000000?text=Feel+Good'},
                    {'name': 'Mood Booster', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0', 'image': 'https://via.placeholder.com/300x300/00D4FF/000000?text=Mood+Booster'}
                ],
                'sad': [
                    {'name': 'Life Sucks', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1', 'image': 'https://via.placeholder.com/300x300/4169E1/FFFFFF?text=Sad+Songs'},
                    {'name': 'Sad Indie', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX59NCqCqJtoH', 'image': 'https://via.placeholder.com/300x300/708090/FFFFFF?text=Sad+Indie'},
                    {'name': 'Melancholy', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWX83CujKHHOn', 'image': 'https://via.placeholder.com/300x300/2F4F4F/FFFFFF?text=Melancholy'}
                ],
                'energetic': [
                    {'name': 'Beast Mode', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX76Wlfdnj7AP', 'image': 'https://via.placeholder.com/300x300/FF4500/000000?text=Beast+Mode'},
                    {'name': 'Power Workout', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX70RN3TfWWJh', 'image': 'https://via.placeholder.com/300x300/DC143C/000000?text=Power+Workout'},
                    {'name': 'Adrenaline', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX0pH2SQMRXnC', 'image': 'https://via.placeholder.com/300x300/8B0000/FFFFFF?text=Adrenaline'}
                ],
                'calm': [
                    {'name': 'Peaceful Piano', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO', 'image': 'https://via.placeholder.com/300x300/87CEEB/000000?text=Peaceful+Piano'},
                    {'name': 'Calm Vibes', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWU0ScTcjJBdj', 'image': 'https://via.placeholder.com/300x300/ADD8E6/000000?text=Calm+Vibes'},
                    {'name': 'Relaxing Sounds', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWZd79rJ6a7lp', 'image': 'https://via.placeholder.com/300x300/B0E0E6/000000?text=Relaxing'}
                ],
                'excited': [
                    {'name': 'Party Time', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DXaXB8fQg7xif', 'image': 'https://via.placeholder.com/300x300/FF1493/000000?text=Party+Time'},
                    {'name': 'Dance Party', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX4dyzvuaRJ0n', 'image': 'https://via.placeholder.com/300x300/FF69B4/000000?text=Dance+Party'},
                    {'name': 'Energy Boost', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX3Sp0P28SIer', 'image': 'https://via.placeholder.com/300x300/FFB6C1/000000?text=Energy+Boost'}
                ],
                'chill': [
                    {'name': 'Chill Hits', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX4WYpdgoIcn6', 'image': 'https://via.placeholder.com/300x300/9370DB/000000?text=Chill+Hits'},
                    {'name': 'Lofi Beats', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWWQRwui0ExPn', 'image': 'https://via.placeholder.com/300x300/BA55D3/000000?text=Lofi+Beats'},
                    {'name': 'Chill Vibes', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX889U0CL85jj', 'image': 'https://via.placeholder.com/300x300/DDA0DD/000000?text=Chill+Vibes'}
                ],
                'romantic': [
                    {'name': 'Romantic', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX50QitC6Oqtn', 'image': 'https://via.placeholder.com/300x300/FF1493/FFFFFF?text=Romantic'},
                    {'name': 'Love Songs', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX0UrRvztWcAU', 'image': 'https://via.placeholder.com/300x300/FF69B4/FFFFFF?text=Love+Songs'},
                    {'name': 'Date Night', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX4OzrY981I1W', 'image': 'https://via.placeholder.com/300x300/FFB6C1/000000?text=Date+Night'}
                ],
                'dark': [
                    {'name': 'Dark & Stormy', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX0XUfTFmNBRM', 'image': 'https://via.placeholder.com/300x300/2F4F4F/FFFFFF?text=Dark'},
                    {'name': 'Metal', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWWOaP4H0w5b0', 'image': 'https://via.placeholder.com/300x300/000000/FFFFFF?text=Metal'},
                    {'name': 'Rock Hard', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWXRqgorJj26U', 'image': 'https://via.placeholder.com/300x300/1C1C1C/FFFFFF?text=Rock+Hard'}
                ]
            }
            return sample_playlists.get(mood, sample_playlists['chill'])
    
    try:
        # Enhanced search queries for better results
        search_queries = {
            'happy': ['happy', 'feel good', 'uplifting'],
            'sad': ['sad', 'melancholy', 'emotional'],
            'energetic': ['workout', 'energetic', 'power'],
            'calm': ['calm', 'peaceful', 'relaxing'],
            'excited': ['party', 'dance', 'upbeat'],
            'chill': ['chill', 'lofi', 'relax'],
            'romantic': ['romantic', 'love songs', 'date night'],
            'dark': ['dark', 'intense', 'heavy']
        }
        
        query = search_queries.get(mood, [mood])[0]
        results = sp.search(q=query, type='playlist', limit=5)
        playlists = []
        for item in results['playlists']['items']:
            playlists.append({
                'name': item['name'],
                'url': item['external_urls']['spotify'],
                'image': item['images'][0]['url'] if item['images'] else None
            })
        return playlists
    except Exception as e:
        print(f"Error fetching Spotify playlists: {e}")
        return []

if __name__ == '__main__':
    app.run(debug=True)
