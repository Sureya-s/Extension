import re
import torch
import flask
from flask import Flask, request, jsonify
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi

# Initialize Flask app
app = Flask(__name__)

# Set device to CPU
device = "cpu"

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

# Function to get video transcript
def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([line['text'] for line in transcript])
        return text
    except Exception as e:
        return f"Error: {str(e)}"

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\[.*?\]', '', text)  # Remove timestamps like [00:00]
    text = re.sub(r'\s+', ' ', text)     # Remove extra spaces
    return text.strip()

# Function to summarize long text
def summarize_long_text(text, max_chunk_length=1024):
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    summaries = [summarizer(chunk, max_length=50, min_length=10, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)

# API Endpoint
@app.route('/summarize', methods=['GET'])
def summarize():
    video_id = request.args.get('video_id')
    if not video_id:
        return jsonify({"error": "Missing video_id parameter"}), 400

    transcript = get_video_transcript(video_id)
    if "Error" in transcript:
        return jsonify({"error": transcript}), 400

    cleaned_transcript = preprocess_text(transcript)
    summary = summarize_long_text(cleaned_transcript)
    
    return jsonify({"video_id": video_id, "summary": summary})

# Run the app locally
import os
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))  # Change to 5001
    app.run(host='0.0.0.0', port=port)



