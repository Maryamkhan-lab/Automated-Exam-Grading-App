import numpy as np
import cv2
from PIL import Image, ImageEnhance
from io import BytesIO
from pdf2image import convert_from_path
import json

from gapi_client import get_genai_client
from utils import extract_json_from_output

# Global GenAI client
CLIENT = None

def init_genai():
    """
    Initialize the global GenAI client with the provided API key.
    """
    global CLIENT
    CLIENT = get_genai_client()


def parse_all_answers(image_input: Image.Image) -> str:
    """
    Extracts answers from a full answer-sheet image using Gemini.
    Returns the raw JSON string from the model.
    """
    output_format = '''
Answer in the following JSON format. Do not write anything else:
{
  "Paper name": {"name": "<paper Alphabet>"},
  "Answers": {
    "1": "<option or text>",
    "2": "<option or text>",
    "3": "<option or text>",
    "4": "<option or text>",
    "5": "<option or text>",
    "6": "<option or text>",
    "7": "<option or text>",
    "8": "<option or text>",
    "9": "<option or text>",
    "10": "<option or text>",
    "11": "<option or text>",
    "12": "<option or text>",
    "13": "<option or text>",
    "14": "<option or text>",
    "15": "<option or text>",
    "16": "<option or text>",
    "17": "<option or text>",
    "18": "<option or text>",
    "19": "<option or text>",
    "20": "<option or text>",
    "21": "<free text answer>",
    "22": "<free text answer>",
    "23": "<free text answer>",
    "24": "<free text answer>",
    "25": "<free text answer>"
  }
}
'''    
    prompt = f"""
You are an assistant that extracts answers from an image.
Write only the Alphabet(A,B,C,D,E,F) of the paper in the \"Paper name\" field.
The image is a screenshot of an answer sheet containing 25 questions.
For questions 1 to 20, the answers are multiple-choice selections.
For questions 21 to 25, the answers are free-text responses.
Extract the answer for each question (1 to 25) and provide the result in JSON using the format below:
{output_format}
"""
    response = CLIENT.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, image_input]
    )
    return response.text


def preprocess_pdf_last_page(image: Image.Image) -> Image.Image:
    """
    Preprocesses the last page PIL image:
      - Convert to OpenCV BGR
      - Mask vertical region
      - Crop to mask
      - Unsharp mask sharpen
      - Enhance with PIL
    """
    # Convert to BGR
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]

    # Mask
    mask = np.zeros((h, w), dtype="uint8")
    top, bottom = int(h * 0.14), int(h * 0.73)
    cv2.rectangle(mask, (0, top), (w, h - bottom), 255, -1)
    masked = cv2.bitwise_and(img_cv, img_cv, mask=mask)

    # Crop
    coords = cv2.findNonZero(mask)
    x, y, cw, ch = cv2.boundingRect(coords)
    cropped = masked[y:y+ch, x:x+cw]

    # Sharpen
    blurred = cv2.GaussianBlur(cropped, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(cropped, 1.5, blurred, -0.5, 0)

    # PIL enhancements
    pil2 = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    pil2 = ImageEnhance.Sharpness(pil2).enhance(1.3)
    pil2 = ImageEnhance.Contrast(pil2).enhance(1.4)
    pil2 = ImageEnhance.Brightness(pil2).enhance(1.1)
    return pil2


def parse_info_with_gemini(pil_img: Image.Image) -> dict:
    """
    Calls Gemini on a header image to extract candidate info fields.
    """
    output_format = '''
Answer in the following JSON format. Do not write anything else:
{
  "Candidate Info": {
    "Paper": "<paper>",
    "Level": "<level>",
    "Candidate Name": "<name>",
    "Candidate Number": "<number>",
    "School": "<school>",
    "Country": "<country>",
    "grade level": "<grade level>",
    "Date": "<date>"
  }
}
'''
    prompt = f"""
You are a helper that accurately reads a sharpened exam header image and extracts exactly these fields:
  • Paper (e.g. \"B\")
  • Level (e.g. \"MIDDLE PRIMARY\")
  • Candidate Name
  • Candidate Number
  • School
  • Country
  • grade level
  • Date (with time)
Return **only** valid JSON in this format:
{output_format}
"""
    response = CLIENT.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, pil_img]
    )
    return extract_json_from_output(response.text)


def extract_candidate_data(image: Image.Image) -> dict:
    """
    Preprocess last page and parse candidate info.
    """
    prepped = preprocess_pdf_last_page(image)
    info = parse_info_with_gemini(prepped)
    return info


def parse_mcq_answers(pil_image: Image.Image) -> str:
    """
    Extracts MCQ answers 1–10 from an image.
    """
    output_format = '''
Answer in the following JSON format. Do not write anything else:
{
  "Answers": {
    "1": "<option>",
    "2": "<option>",
    "3": "<option>",
    "4": "<option>",
    "5": "<option>",
    "6": "<option>",
    "7": "<option>",
    "8": "<option>",
    "9": "<option>",
    "10": "<option>"
  }
}
'''
    prompt = f"""
You are an assistant that extracts MCQ answers from an image.
The image is a screenshot of a 10-question multiple-choice answer sheet.
Extract which option is marked for each question (1–10) and provide the answers in JSON:
{output_format}
"""
    response = CLIENT.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, pil_image]
    )
    return response.text


def get_mcqs1st(pil_image: Image.Image) -> dict:
    """
    Mask, crop, enhance, and parse MCQs 1–10.
    """
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    mask = np.zeros((h, w), dtype="uint8")
    top, bot, right = int(h*0.30), int(h*0.44), int(w*0.35)
    cv2.rectangle(mask, (0, top), (right, h-bot), 255, -1)
    masked = cv2.bitwise_and(img_cv, img_cv, mask=mask)
    coords = cv2.findNonZero(mask)
    x, y, cw, ch = cv2.boundingRect(coords)
    cropped = masked[y:y+ch, x:x+cw]
    blur = cv2.GaussianBlur(cropped, (0,0), sigmaX=3)
    sharp = cv2.addWeighted(cropped, 1.5, blur, -0.5, 0)
    pil_sh = Image.fromarray(cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB))
    pil_sh = ImageEnhance.Sharpness(pil_sh).enhance(1.3)
    pil_sh = ImageEnhance.Contrast(pil_sh).enhance(1.4)
    final = ImageEnhance.Brightness(pil_sh).enhance(1.1)
    raw = parse_mcq_answers(final)
    return extract_json_from_output(raw)


def parse_mcq_answers_11_20(pil_image: Image.Image) -> str:
    """
    Extracts MCQ answers 11–20 from an image.
    """
    output_format = '''
Answer in the following JSON format. Do not write anything else:
{
  "Answers": {
    "11": "<option>",
    "12": "<option>",
    "13": "<option>",
    "14": "<option>",
    "15": "<option>",
    "16": "<option>",
    "17": "<option>",
    "18": "<option>",
    "19": "<option>",
    "20": "<option>"
  }
}
'''
    prompt = f"""
You are an assistant that extracts MCQ answers from an image.
The image is a screenshot of questions 11–20.
Extract the marked option for each and return JSON:
{output_format}
"""
    response = CLIENT.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, pil_image]
    )
    return response.text


def get_mcqs2nd(pil_image: Image.Image) -> dict:
    """
    Mask, crop, enhance, and parse MCQs 11–20.
    """
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    mask = np.zeros((h, w), dtype="uint8")
    top, bottom, right = int(h*0.56), int(h*0.21), int(w*0.35)
    cv2.rectangle(mask, (0, top), (right, h-bottom), 255, -1)
    masked = cv2.bitwise_and(img_cv, img_cv, mask=mask)
    coords = cv2.findNonZero(mask)
    x, y, cw, ch = cv2.boundingRect(coords)
    cropped = masked[y:y+ch, x:x+cw]
    blurred = cv2.GaussianBlur(cropped, (0,0), sigmaX=3)
    sharp = cv2.addWeighted(cropped, 1.5, blurred, -0.5, 0)
    pil_sharp = Image.fromarray(cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB))
    pil_sharp = ImageEnhance.Sharpness(pil_sharp).enhance(1.3)
    pil_sharp = ImageEnhance.Contrast(pil_sharp).enhance(1.4)
    final_pil = ImageEnhance.Brightness(pil_sharp).enhance(1.1)
    raw = parse_mcq_answers_11_20(final_pil)
    return extract_json_from_output(raw)


def parse_text_answers(pil_image: Image.Image) -> str:
    """
    Extracts free-text answers 21–25 from an image.
    """
    output_format = '''
Answer in the following JSON format. Do not write anything else:
{
  "Answers": {
    "21": "<text>",
    "22": "<text>",
    "23": "<text>",
    "24": "<text>",
    "25": "<text>"
  }
}
'''
    prompt = f"""
You are an assistant that extracts free-text answers from an image.
The image shows answers to questions 21–25.
Extract the text for each and return JSON:
{output_format}
"""
    response = CLIENT.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, pil_image]
    )
    return response.text


def get_answer(pil_image: Image.Image) -> dict:
    """
    Mask, crop, enhance, and parse free-text 21–25.
    """
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    mask = np.zeros((h, w), dtype="uint8")
    top, bottom = int(h*0.31), int(h*0.31)
    left, right = int(w*0.35), int(w*0.66)
    cv2.rectangle(mask, (left, top), (right, h-bottom), 255, -1)
    masked = cv2.bitwise_and(img_cv, img_cv, mask=mask)
    coords = cv2.findNonZero(mask)
    x, y, cw, ch = cv2.boundingRect(coords)
    cropped = masked[y:y+ch, x:x+cw]
    blurred = cv2.GaussianBlur(cropped, (0,0), sigmaX=3)
    sharp = cv2.addWeighted(cropped, 1.5, blurred, -0.5, 0)
    pil_sharp = Image.fromarray(cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB))
    pil_sharp = ImageEnhance.Sharpness(pil_sharp).enhance(1.3)
    pil_sharp = ImageEnhance.Contrast(pil_sharp).enhance(1.4)
    final_pil = ImageEnhance.Brightness(pil_sharp).enhance(1.1)
    raw = parse_text_answers(final_pil)
    return extract_json_from_output(raw)


def infer_page(pil_image: Image.Image) -> dict:
    """
    Full pipeline for a single exam page.
    """
    student_info = extract_candidate_data(pil_image)
    mcq1 = get_mcqs1st(pil_image) or {}
    mcq2 = get_mcqs2nd(pil_image) or {}
    free_txt = get_answer(pil_image) or {}
    all_answers = {**mcq1.get("Answers", {}), **mcq2.get("Answers", {}), **free_txt.get("Answers", {})}
    return {"Candidate Info": student_info.get("Candidate Info", {}), "Answers": all_answers}


def infer_all_pages(pdf_path: str) -> dict:
    """
    Processes every page in the PDF and infers student data.
    """
    results = {}
    pages = convert_from_path(pdf_path)
    for idx, page in enumerate(pages, start=1):
        data = infer_page(page)
        info = data.get("Candidate Info", {})
        key = info.get("Candidate Number") or f"Page_{idx}"
        if data.get("Answers"):
            results[key] = data
    return results


def load_answer_key(pdf_path: str) -> dict:
    """
    Parses the official answer-key PDF into a dict of paper->answers.
    """
    images = convert_from_path(pdf_path)
    key_dict = {}
    for page in images:
        raw = parse_all_answers(page)
        parsed = extract_json_from_output(raw)
        name = parsed.get("Paper name", {}).get("name")
        key_dict[name] = parsed.get("Answers", {})
    return key_dict


def grade_page(student_page_data: dict, answer_key_dict: dict) -> dict:
    """
    Grades a single student page against the loaded key.
    """
    paper = student_page_data.get("Candidate Info", {}).get("Paper")
    correct = answer_key_dict.get(paper, {})
    student_ans = student_page_data.get("Answers", {})
    total_q = len(correct)
    correct_count = 0
    detailed = {}
    for q, key_ans in correct.items():
        stud_ans = student_ans.get(q, "")
        is_corr = str(stud_ans).strip().upper() == str(key_ans).strip().upper()
        if is_corr:
            correct_count += 1
        detailed[q] = {"Correct Answer": key_ans, "Student Answer": stud_ans, "Is Correct": is_corr}
    percentage = round(correct_count/total_q*100, 2) if total_q else 0.0
    return {"Candidate Info": student_page_data.get("Candidate Info", {}), "Total Marks": correct_count, "Total Questions": total_q, "Percentage": percentage, "Detailed Results": detailed}


def grade_all_students(answer_key_pdf: str, student_pdf: str, out_json: str = "results.json") -> dict:
    """
    Loads key, infers all students, grades them, and writes JSON.
    """
    key_dict = load_answer_key(answer_key_pdf)
    students = infer_all_pages(student_pdf)
    results = {}
    for cand, data in students.items():
        results[cand] = grade_page(data, key_dict)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    return results
