import gradio as gr
import numpy as np
import torch.multiprocessing as mp
from schedulers import schedulers_names
import json

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    from main import inference, MP as model_parallel, MODEL_ID, devices, load_pipeline
    def change_model(choice: str):
        if choice == "Base Model":
            load_pipeline(MODEL_ID, devices)
            return gr.Image.update(interactive=False)
        elif choice == "Inpainting":
            load_pipeline("stabilityai/stable-diffusion-2-inpainting", devices)
            return gr.Image.update(interactive=True)

    prompts = []
    def dream(
        prompt: str,
        *args
    ):
        # return [(np.random.randn(512, 512, 3)).astype(np.uint8)], [["test"]]
        if not len(prompt.strip()):
            return [], prompts
        images = inference(prompt, *args)
        if not len(prompts) or [prompt] != prompts[-1]:
            prompts.append([prompt])

        return images, prompts

    # v2 model was trained on sfw data only, hence no safety checker
    enable_nsfw_toggle=not model_parallel and MODEL_ID!="stabilityai/stable-diffusion-2-base"

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Tabs():
                    inputs = []
                    with gr.TabItem("Text2Img"):
                        with gr.Column():
                            # FIXME crashes with weird error if no input
                            inputs.append(gr.Textbox(placeholder="Place your input prompt here and start dreaming!", label="Input Prompt")),
                            inputs.append(gr.Slider(1, max(24, len(devices)*2), 1, step=1, label="Number of Images")),
                            inputs.append(gr.Slider(1, 200, 50, step=1, label="Steps")),
                            inputs.append(gr.Slider(256, 1024, 512, step=64, label="Height")),
                            inputs.append(gr.Slider(256, 1024, 512, step=64, label="Width")),
                            inputs.append(gr.Slider(0, 20, 7.5, step=0.5, label="Guidance Scale")),
                            inputs.append(gr.Number(label="Seed", precision=0)),
                            # inputs.append(# gr.Checkbox(True, label="FP16")),
                            inputs.append(gr.Checkbox(False, label="NSFW Filter", interactive=enable_nsfw_toggle)),
                            inputs.append(gr.Checkbox(False, label="Low VRAM mode")),
                            inputs.append(gr.Dropdown(schedulers_names, value="PNDM", label="Noise Scheduler")),
                    with gr.TabItem("Img2Img"):
                        with gr.Column():
                            gr.Markdown("Image and prompt guided generation. Use one of the box below to provide an input image: if two are provided, `Sketch2Img` has priority. Remember to clear the input by pressing `clear`.")
                            inputs.append(gr.Slider(0, 1, 0.25, step=0.05, label="Img2Img input fidelity")),
                            inputs.append(gr.Image(type="pil", tool=None, label="Image Conditioning")),
                            # FIXME state is not resetting when clicking x! resets when clear is pressed
                            inputs.append(gr.Image(type="pil", source='canvas', tool='color-sketch', label="Sketch2Img"))
                    with gr.TabItem("Image Inpainting"):
                        with gr.Column():
                            gr.Markdown("NOTE: Using image inpainting requires loading a different model from disk!")
                            inpainting = gr.Image(type="pil", tool='sketch', label="Image Inpaint", interactive=False)
                            inputs.append(inpainting),
                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    button = gr.Button("Generate Image!", variant="primary")
            with gr.Column(variant="box"):
                outputs=[gr.Gallery(show_label=False).style(grid=2, container=True)]           
                load_radio = gr.Radio(["Base Model", "Inpainting"], value="Base Model",label="Model to load:")
                outputs.append(gr.Dataframe(col_count=(1, "fixed"),headers=["Prompt History"], interactive=True))
        # sample prompt from https://strikingloo.github.io/DALL-E-2-prompt-guide
        # NOTE prompt MUST be first input, since it is passed here
        gr.Examples(["A digital illustration of a medieval town, 4k, detailed, trending in artstation, fantasy"], inputs=inputs[:1])
        button.click(dream, inputs=inputs, outputs=outputs)
        # clear inputs and outputs
        clear_btn.click(
            None,
            [],
            (
                inputs
                + outputs
            ),
            _js=f"""() => {json.dumps(
                [component.cleared_value if hasattr(component, "cleared_value") else None
                    for component in inputs+outputs]
                )
            }
            """,
        )
        load_radio.change(change_model, inputs=load_radio,outputs=inpainting)
        demo.launch(share=False)
