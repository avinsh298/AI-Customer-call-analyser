from flask import Flask, render_template, request, jsonify
import csv
import os
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_transcript(transcript):
    """
    Analyze the transcript using Groq API to get summary and sentiment
    """
    try:
        # System prompt for consistent analysis
        system_prompt = """You are a customer service analysis expert. 
        Analyze the customer call transcript and provide:
        1. A concise 2-3 sentence summary
        2. The customer's sentiment (positive/neutral/negative)
        
        Format your response exactly as:
        SUMMARY: [2-3 sentence summary]
        SENTIMENT: [positive/neutral/negative]"""

        # Make API call to Groq
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # You can use other Groq models too
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this customer call transcript:\n\n{transcript}"}
            ],
            temperature=0.3,
            max_tokens=300
        )

        response = completion.choices[0].message.content
        
        # Parse the response
        summary = ""
        sentiment = ""
        
        lines = response.split('\n')
        for line in lines:
            if line.startswith('SUMMARY:'):
                summary = line.replace('SUMMARY:', '').strip()
            elif line.startswith('SENTIMENT:'):
                sentiment = line.replace('SENTIMENT:', '').strip().lower()
        
        return summary, sentiment
        
    except Exception as e:
        return f"Error analyzing transcript: {str(e)}", "error"

def save_to_csv(transcript, summary, sentiment):
    """
    Save the analysis results to a CSV file
    """
    file_exists = os.path.isfile('call_analysis.csv')
    
    with open('call_analysis.csv', 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Timestamp', 'Transcript', 'Summary', 'Sentiment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Transcript': transcript[:500] + '...' if len(transcript) > 500 else transcript,  # Truncate long transcripts
            'Summary': summary,
            'Sentiment': sentiment
        })

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint to analyze transcript"""
    try:
        data = request.get_json()
        transcript = data.get('transcript', '').strip()
        
        if not transcript:
            return jsonify({'error': 'Transcript cannot be empty'}), 400
        
        if len(transcript) < 10:
            return jsonify({'error': 'Transcript is too short'}), 400
        
        # Analyze the transcript
        summary, sentiment = analyze_transcript(transcript)
        
        # Save to CSV
        save_to_csv(transcript, summary, sentiment)
        
        return jsonify({
            'success': True,
            'summary': summary,
            'sentiment': sentiment,
            'transcript_preview': transcript[:100] + '...' if len(transcript) > 100 else transcript
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def get_history():
    """Get analysis history from CSV"""
    try:
        history = []
        if os.path.exists('call_analysis.csv'):
            with open('call_analysis.csv', 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    history.append(row)
        
        return jsonify({'history': history[-10:]})  # Return last 10 entries
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)