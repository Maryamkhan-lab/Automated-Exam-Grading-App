from fastapi import APIRouter, FastAPI, UploadFile, File, HTTPException
import tempfile

from services import init_genai, grade_all_students

router = APIRouter()

@router.post('/grade')
async def grade(
    answer_key: UploadFile = File(...),
    student_pdf: UploadFile = File(...)
):
    """
    Single endpoint: initializes GenAI, grades students against the answer key,
    and returns the result cards.
    """
    try:
        # Initialize GenAI client with provided API key
        init_genai()

        # Save uploaded PDFs to temp files
        with tempfile.NamedTemporaryFile(suffix=answer_key.filename, delete=False) as akf:
            akf.write(await answer_key.read())
            ak_path = akf.name
        with tempfile.NamedTemporaryFile(suffix=student_pdf.filename, delete=False) as spf:
            spf.write(await student_pdf.read())
            sp_path = spf.name

        # Grade all students and return results
        results = grade_all_students(ak_path, sp_path)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))