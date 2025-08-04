# Automated-Exam-Grading-App
An AI-powered grading system that automates the evaluation of exam answer sheets, supporting both multiple-choice questions (MCQs) and open-ended questions. The app leverages OCR to read scanned answer sheets and uses NLP-based evaluation for grading open-ended responses.
Features
OCR-based Sheet Reading: Extracts answers from scanned exam sheets using Optical Character Recognition.

MCQ Auto-Grading: Compares student responses with the provided answer key to compute scores.

Open-Ended Question Grading: Uses AI models to assess and score descriptive answers based on similarity and key points.

Fast & Scalable: Built with FastAPI for efficient processing and easy integration with other systems.

Tech Stack
Backend: FastAPI

OCR: Tesseract OCR / OpenCV

NLP & AI Models: OpenAI / Transformers (for open-ended grading)

Data Handling: Pandas, JSON

Deployment: Uvicorn
