import os
import argparse
import torch
import gradio as gr
from utils.YParams import YParams
from inference import Evaluator


def build_app(params):
    evaluator = Evaluator(params)

    demo = gr.Interface(
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

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carnatic Raga Recognition AI Model Server")
    parser.add_argument("--yaml_config", default="config.yaml", type=str, help="Path to config yaml")
    parser.add_argument("--config", default="resnet_0.7", type=str, help="Configuration profile name")
    parser.add_argument("--host", default="0.0.0.0", type=str, help="Server bind host")
    parser.add_argument("--port", default=7860, type=int, help="Server bind port")
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # Device selection (GPU if available, otherwise CPU)
    if torch.cuda.is_available():
        params.device = torch.device(torch.cuda.current_device())
    else:
        params.device = torch.device("cpu")

    # Set checkpoint path
    exp_dir = os.path.join("ckpts", "resnet_0.7", "150classes_alldata_cliplength30", "training_checkpoints")
    params.best_checkpoint_path = os.path.join(exp_dir, "best_ckpt.tar")

    print(f"Starting Carnatic Raga Server on {args.host}:{args.port} using device: {params.device}")
    app = build_app(params)
    app.launch(server_name=args.host, server_port=args.port)
