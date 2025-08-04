from fastapi import FastAPI
from routes import router

app = FastAPI(title='Exam Processing API')
app.include_router(router, prefix='/api')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app.main:app', host='0.0.0.0', port=8000, reload=True)