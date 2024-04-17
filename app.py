import os
from inference import Evaluator
import argparse
from utils.YParams import YParams
import torch
import gradio as gr

def read_markdown_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()


if __name__ == '__main__':

    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='config.yaml', type=str)
    parser.add_argument("--config", default='resnet_0.7', type=str)

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    #GPU stuff
    try:
        params.device = torch.device(torch.cuda.current_device())
    except:
        params.device = "cpu"
    
    #checkpoint stuff
    expDir = "ckpts/resnet_0.7/150classes_alldata_cliplength30"
    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')

    evaluator = Evaluator(params)

    with gr.Blocks() as demo:
        with gr.Tab("Classifier"):
            gr.Interface(
            title="Carnatic Raga Classifier",
            description="**Welcome!** This is a deep-learning based raga classifier. Upload or record an audio clip to test it out. Provide at least 30 seconds of audio for best results. Wait for the audio waves to appear (and stay) before clicking Submit! \n",
            article = "**Get in Touch:** Feel free to reach out to [me](https://sanjeevraja.com/) via email (sanjeevr AT berkeley DOT edu) with any questions or feedback! ",
            fn=evaluator.inference,
            inputs=[
                gr.Slider(minimum = 1, maximum = 150, value = 5, label = "Number of displayed ragas", info = "Choose number of top predictions to display"), 
                gr.Audio()
                ],
            outputs="label",
            allow_flagging = False
            )
            
        with gr.Tab("About"):
            gr.Markdown(read_markdown_file('about.md'))
            gr.Image('site/tsne.jpeg', height = 800, width=800)

    demo.launch()


