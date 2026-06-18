import os
import json
import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Google API Imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Phase 1: Data Ingestion (The Transcript)

# Extraction
url = input("Enter the URL: ")

if 'https://www.youtube.com/watch?v=' not in url:
    print("Invalid YouTube URL. Link format: https://www.youtube.com/watch?v=")
    exit()

video_id = url.replace('https://www.youtube.com/watch?v=', '')

try:
    # The new v1.2+ syntax requires initializing the API first
    yt_api = YouTubeTranscriptApi()
    transcript = yt_api.fetch(video_id)
except Exception as e:
    print(f"Failed to fetch transcript: {e}")
    exit()

# Cleaning
output = ''
for chunk in transcript:
    # The new library version returns objects instead of dictionaries, 
    # so we use a safe fallback to extract the text without crashing!
    sentence = chunk.text if hasattr(chunk, 'text') else chunk.get('text', '')
    output += f" {sentence}\n"

# Phase 2: Synthesis & Question Generation

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Configuration Error: GEMINI_API_KEY not found in .env file.")
    exit()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=api_key
)

# The Summary
summary_template = """Be strictly formal and to the point. 
Do not give your own opinion. 
You have to extract the key takeaways in a professional, 
bulleted format of this text: {output}
"""

summary_prompt = PromptTemplate.from_template(summary_template)


summary_chain = summary_prompt | llm | StrOutputParser()


final_summary = summary_chain.invoke({"output": output})
print("Summary Generated Successfully.")


# The Quiz (structured output)
class Question(BaseModel):
    question_text: str = Field(description="The multiple-choice question tests application, not just recall")
    options: list[str] = Field(description="Exactly 5 multiple choice options")
    correct_answer: str = Field(description="The exact text of the corret option")
    rationale: str = Field(description="Explanation of why this answer is correct")

class Quiz(BaseModel):
    questions: list[Question] = Field(description="Exactly 5 multiple-choice questions")

quiz_template = """
system: You are a senior Educator. Your goal is to test a student's ability to APPLY concepts learned, rather than simply recalling rote facts.
Task: Based on the provided transcript, generate exactly 5 multiple-choice questions.

Transcript:
{output}
"""

quiz_prompt = PromptTemplate.from_template(quiz_template)

# Forcing structured output
structured_llm = llm.with_structured_output(Quiz)

quiz_chain = quiz_prompt | structured_llm
quiz_data = quiz_chain.invoke({"output": output})
print("Quiz Generated Successfully.")


# Phase 3: Pushing to google docs

SCOPES = ['https://www.googleapis.com/auth/documents']

# Standard OAuth2 flow for a local script
creds = None
if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        # Requires credentials.json from Google Cloud Console
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

docs_service = build('docs', 'v1', credentials=creds)

# 1. Create a blank document
doc_title = f"Study Guide: YouTube Video ({video_id})"
document = docs_service.documents().create(body={'title': doc_title}).execute()
document_id = document.get('documentId')

# 2. Prepare the content & formatting requests
requests = []
current_index = 1

# Helper function to append text and update indices
def append_text(text, is_heading=False, is_list=False):
    global current_index
    start_idx = current_index
    
    requests.append({
        'insertText': {
            'location': {'index': current_index},
            'text': text
        }
    })
    current_index += len(text)
    
    if is_heading:
        requests.append({
            'updateParagraphStyle': {
                'range': {'startIndex': start_idx, 'endIndex': current_index},
                'paragraphStyle': {'namedStyleType': 'HEADING_1'},
                'fields': 'namedStyleType'
            }
        })
    elif is_list:
        requests.append({
            'createParagraphBullets': {
                'range': {'startIndex': start_idx, 'endIndex': current_index},
                'bulletPreset': 'BULLET_DISC_CIRCLE_SQUARE'
            }
        })

# Build Summary Section
append_text("Key Takeaways\n", is_heading=True)
append_text(final_summary + "\n\n")

# Build Quiz Section
append_text("Application Quiz\n", is_heading=True)

quiz_text = ""
for i, q in enumerate(quiz_data.questions, 1):
    quiz_text += f"Question {i}: {q.question_text}\n"
    for opt in q.options:
        quiz_text += f"  - {opt}\n"
    quiz_text += f"Correct Answer: {q.correct_answer}\n"
    quiz_text += f"Rationale: {q.rationale}\n\n"

append_text(quiz_text, is_list=True)

# 3. Execute the formatting and text insertion payload
docs_service.documents().batchUpdate(
    documentId=document_id, 
    body={'requests': requests}
).execute()

# Final Loop: Return the URL to the console
doc_url = f"https://docs.google.com/document/d/{document_id}/edit"
print("\nWORKSPACE PERSISTENCE SUCCESSFUL")
print(f"Document URL: {doc_url}")
