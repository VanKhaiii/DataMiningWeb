from fastapi import FastAPI
from BPRMF_pipeline import BPRMFPipeline
from src.utils.extract_course import extract_course
from src.utils.extract_course_info import extract_course_info
from src.parser.parser_bprmf import *
import uvicorn

app = FastAPI()

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

@app.post("/predict/")
async def predict():
    args = parse_bprmf_args()
    
    args.data_dir = "src\\datasets"
    args.data_name = "mooccube"
    args.use_pretrain = 2
    args.Ks = '[1, 5, 10]'
    args.pretrain_model_path = 'src\\pretrained_model\\model_BPRMF.pth'

    pipeline = BPRMFPipeline()
    
    top_courses = extract_course(pipeline(args))

    return  extract_course_info(top_courses)

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
