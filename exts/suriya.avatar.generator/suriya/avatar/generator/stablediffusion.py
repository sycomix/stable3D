
import carb
import torch
import asyncio
from diffusers import StableDiffusionPipeline

async def generateImage(progress_widget, image_prompt: str, image_widget):
    carb.log_info("Stable Diffusion Stage")

    if (len(image_prompt) != 0):
        run_loop = asyncio.get_event_loop()
        progress_widget.show_bar(True)
        task = run_loop.create_task(progress_widget.play_anim_forever())

        print("creating image with prompt: "+image_prompt)
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
        pipe.to("cuda")
        #prompt = "a photograph of an astronaut riding a horse"
        prompt = image_prompt
        image = pipe(prompt).images[0]
        # you can save the image with
        image_url = "D:\\CG_Source\\Omniverse\\Extensions\\3DAvatarExtensionPOC\\stable3D\\"+prompt.replace(" ", "")+".png"
        image.save(image_url)
        print("image created")

        task.cancel()
        await asyncio.sleep(1)
        # todo bug fix: reload image if same prompt
        image_widget.source_url = image_url
        progress_widget.show_bar(False)