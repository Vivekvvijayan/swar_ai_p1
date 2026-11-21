import os
import argparse
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from utils.YParams import YParams
from inference import Evaluator
from tune_generator import generate_tunes_for_raga, cleanup_raga_audio_files

TUNES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_tunes")
os.makedirs(TUNES_DIR, exist_ok=True)

# 1. Initialize FastAPI Server with CORS
server = FastAPI(
    title="Carnatic Raga AI Server",
    description="Backend API powering Carnatic Raga Recognition & Sanchara Melody Generation"
)

server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. REST Endpoints for Mobile App Integration
@server.get("/api/tunes/{raga_name}")
async def get_raga_tunes(raga_name: str):
    try:
        result = generate_tunes_for_raga(raga_name, TUNES_DIR)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@server.delete("/api/tunes/{raga_name}")
async def delete_raga_tunes(raga_name: str):
    try:
        deleted = cleanup_raga_audio_files(raga_name, TUNES_DIR)
        return JSONResponse(content={"status": "cleaned", "raga": raga_name, "files_deleted": deleted})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@server.post("/api/tunes/cleanup/{raga_name}")
async def cleanup_tunes_post(raga_name: str):
    try:
        deleted = cleanup_raga_audio_files(raga_name, TUNES_DIR)
        return JSONResponse(content={"status": "cleaned", "raga": raga_name, "files_deleted": deleted})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@server.get("/api/audio/{filename}")
async def get_audio_file(filename: str):
    file_path = os.path.join(TUNES_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    return JSONResponse(status_code=404, content={"error": "Audio file not found"})


def build_gradio_demo(params):
    evaluator = Evaluator(params)

    # Classifier Interface
    classifier_interface = gr.Interface(
        title="Carnatic Raga AI Classifier",
        description="End-to-End AI-based Melodic Classifier for 150 Carnatic Ragas (ResNet Deep Audio Architecture).",
        inputs=[
            gr.Slider(
                minimum=1,
                maximum=150,
                value=5,
                step=1,
                label="Top K Ragas",
                info="Number of ranked raga candidates to return"
            ),
            gr.Audio(
                label="Audio Sample (Live Mic or WAV/MP3 Upload)"
            )
        ],
        fn=evaluator.inference,
        outputs=gr.Label(label="Predicted Ragas & Confidence"),
        api_name="inference",
        flagging_mode="never"
    )

    # Web UI Tune Generator Interface
    def web_generate_tunes(raga_name):
        res = generate_tunes_for_raga(raga_name or "Mohanam", TUNES_DIR)
        tunes = res["tunes"]
        details = f"### Generated Melodic Tunes for Raga: **{raga_name}**\n\n"
        for t in tunes:
            details += f"- **Tune {t['id']} ({t['title']}):** `{t['swaras']}`\n"
            details += f"  *{t['description']}*\n\n"
        first_audio = os.path.join(TUNES_DIR, tunes[0]["filename"])
        return details, first_audio

    tune_interface = gr.Interface(
        title="AI Carnatic Raga Tune Generator",
        description="Generate authentic melodic phrases (sancharas) and scale cadences in any Carnatic raga with Tanpura accompaniment.",
        inputs=[
            gr.Textbox(label="Carnatic Raga Name", value="Mohanam", placeholder="e.g. Mohanam, Kalyani, Hamsadhwani, Shankarabharanam")
        ],
        outputs=[
            gr.Markdown(label="Swara Notation & Melodic Phrasing"),
            gr.Audio(label="Synthesized Audio Preview (Tanpura + Flute/Veena)")
        ],
        fn=web_generate_tunes,
        flagging_mode="never"
    )

    with gr.Blocks(title="Carnatic Raga AI Studio") as demo:
        with gr.Tab("Raga Classifier"):
            classifier_interface.render()
        with gr.Tab("Raga Tune Generator"):
            tune_interface.render()

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carnatic Raga Recognition & Tune Generation Server")
    parser.add_argument("--yaml_config", default="config.yaml", type=str, help="Path to config yaml")
    parser.add_argument("--config", default="resnet_0.7", type=str, help="Configuration profile name")
    parser.add_argument("--host", default="0.0.0.0", type=str, help="Server bind host")
    parser.add_argument("--port", default=7860, type=int, help="Server bind port")
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # Device selection
    if torch.cuda.is_available():
        params.device = torch.device(torch.cuda.current_device())
    else:
        params.device = torch.device("cpu")

    exp_dir = os.path.join("ckpts", "resnet_0.7", "150classes_alldata_cliplength30", "training_checkpoints")
    params.best_checkpoint_path = os.path.join(exp_dir, "best_ckpt.tar")

    # Build Gradio Demo and mount onto FastAPI server
    demo = build_gradio_demo(params)
    app = gr.mount_gradio_app(server, demo, path="/")

    print(f"Starting Carnatic Raga Server on {args.host}:{args.port} using device: {params.device}")
    uvicorn.run(app, host=args.host, port=args.port)
