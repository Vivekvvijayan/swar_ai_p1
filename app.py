import os
import argparse
import torch
import gradio as gr
from fastapi.responses import JSONResponse, FileResponse
from utils.YParams import YParams
from inference import Evaluator
from tune_generator import generate_tunes_for_raga

TUNES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_tunes")
os.makedirs(TUNES_DIR, exist_ok=True)


def build_app(params):
    evaluator = Evaluator(params)

    # 1. Classifier Interface
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

    # 2. Web UI Tune Generator Function
    def web_generate_tunes(raga_name):
        res = generate_tunes_for_raga(raga_name or "Mohanam", TUNES_DIR)
        tunes = res["tunes"]
        details = f"### Generated Melodic Tunes for Raga: **{raga_name}**\n\n"
        for t in tunes:
            details += f"- **Tune {t['id']} ({t['title']}):** `{t['swaras']}`\n"
            details += f"  *{t['description']}*\n\n"
        # Return first tune path for the audio player
        first_audio = os.path.join(TUNES_DIR, tunes[0]["filename"])
        return details, first_audio

    tune_interface = gr.Interface(
        title="AI Carnatic Raga Tune Generator",
        description="Generate random authentic melodic phrases (sancharas) and scale cadences in any Carnatic raga with Tanpura drone.",
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

    # Mount REST API Endpoints on the FastAPI app instance for Mobile Integration
    @demo.app.get("/api/tunes/{raga_name}")
    async def get_raga_tunes(raga_name: str):
        try:
            result = generate_tunes_for_raga(raga_name, TUNES_DIR)
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @demo.app.get("/api/audio/{filename}")
    async def get_audio_file(filename: str):
        file_path = os.path.join(TUNES_DIR, filename)
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type="audio/wav")
        return JSONResponse(status_code=404, content={"error": "Audio file not found"})

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

    print(f"Starting Carnatic Raga Server on {args.host}:{args.port} using device: {params.device}")
    app = build_app(params)
    app.launch(server_name=args.host, server_port=args.port)
