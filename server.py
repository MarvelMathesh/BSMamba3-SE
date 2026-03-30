import os
import time
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from inference import load_model, enhance_file

app = FastAPI(title="BSMamba3-SE Speech Enhancement API")

# Enable CORS for Next.js app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold model
global_model = None
global_config = None
global_device = None

@app.on_event("startup")
def startup_event():
    global global_model, global_config, global_device
    print("[Server] Loading BSMamba3-SE model into GPU memory...")
    checkpoint_path = "./checkpoints/bsmamba3_run1/checkpoint_epoch75.pt"
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found at {checkpoint_path}")
        return
        
    global_model, global_config, global_device = load_model(checkpoint_path, "cuda")
    print(f"[Server] Model loaded successfully: {global_model.count_parameters()/1e6:.2f}M params")

def cleanup_files(*filepaths):
    """Background task to remove temp files after they've been sent."""
    time.sleep(1)  # small buffer so FileResponse completes
    for fp in filepaths:
        try:
            if os.path.exists(fp):
                os.remove(fp)
        except Exception as e:
            print(f"Error cleaning up {fp}: {e}")

@app.post("/api/enhance")
async def enhance_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Receives an audio file, runs inference, and returns enhanced audio."""
    if not global_model:
        return {"error": "Model not loaded properly."}
        
    try:
        # Save uploaded file to temp directory
        fd_input, input_path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd_input, 'wb') as f:
            f.write(await file.read())
            
        fd_output, output_path = tempfile.mkstemp(suffix="_enhanced.wav")
        os.close(fd_output)
        
        # Run BSMamba3-SE inference
        print(f"[API] Processing {file.filename}...")
        result = enhance_file(global_model, input_path, output_path, global_device, sr=16000)
        
        # Cleanup files dynamically after they are returned to client!
        background_tasks.add_task(cleanup_files, input_path, output_path)
        
        return FileResponse(
            path=output_path, 
            media_type="audio/wav", 
            filename=f"enhanced_{file.filename}"
        )
        
    except Exception as e:
        print(f"[API] Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("[Server] Starting uvicorn...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
