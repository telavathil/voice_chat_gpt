import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment

# Load environment variables
load_dotenv('.envrc')

# initialize openai api
client = OpenAI()

# Initialize Flask app
app = Flask(__name__)

@app.route('/api/voice-to-text', methods=['POST'])
def voice_to_text():
  # Get the audio file from the request
  audio_file = request.files['file']

  # Convert the audio file to wav format
  audio = AudioSegment.from_file(audio_file)
  audio.export('converted_audio.wav', format='wav')

  # Read the audio file
  with open('converted_audio.wav', 'rb') as file:
    transcription = client.audio.transcriptions.create(
      model="whisper-1",
      file=audio_file,
    )

  # Return the transcription
  return jsonify(transcription)

@app.route('/api/text-to-chat', methods=['POST'])
def text_to_chat():
  # Get the text form the request
  data = request.get_json()
  text = data['text']

  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": text}
    ]
  )

  return jsonify({'response': completion.choices[0].message})

if __name__ == '__main__':
  app.run(debug=True)

