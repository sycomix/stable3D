import carb
import torch
import asyncio
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionDepth2ImgPipeline
from PIL import Image

async def generateTextToImage(progress_widget, outputImage_widget, image_prompt: str):
    carb.log_info("Stable Diffusion Stage: Text to Image")

    if (len(image_prompt) != 0):
        run_loop = asyncio.get_event_loop()
        progress_widget.show_bar(True)
        task = run_loop.create_task(progress_widget.play_anim_forever())

        print("creating image with prompt: "+image_prompt)
        model_id  = "runwayml/stable-diffusion-v1-5"
        #model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
        pipe.to("cuda")
        #prompt = "a photograph of an astronaut riding a horse"
        image = pipe(image_prompt).images[0]
        # you can save the image with
        image_url = "D:\\CG_Source\\Omniverse\\Extensions\\3DAvatarExtensionPOC\\stable3D\\"+image_prompt.replace(" ", "")+".png"
        image.save(image_url)
        print("image created")

        task.cancel()
        await asyncio.sleep(1)
        # todo bug fix: reload image if same prompt
        outputImage_widget.source_url = image_url
        progress_widget.show_bar(False)

async def generateImageToImage(progress_widget, outputImage_widget, image_prompt: str, inputImageUrl):
    carb.log_info("Stable Diffusion Stage: Image to Image")

    if (len(image_prompt) != 0):
        run_loop = asyncio.get_event_loop()
        progress_widget.show_bar(True)
        task = run_loop.create_task(progress_widget.play_anim_forever())

        print("creating image with prompt+image: "+image_prompt)
        model_id  = "runwayml/stable-diffusion-v1-5"
        #model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
        #optimizations
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing(1)
        #pipe.to("cuda")
        #pipe.enable_vae_tiling()
        #pipe.enable_xformers_memory_efficient_attention()

        init_image = Image.open(inputImageUrl).convert("RGB")
        init_image = init_image.resize((768, 512))
        image = pipe(prompt=image_prompt, image=init_image, strength=0.3, guidance_scale=7.5, num_inference_steps=20).images[0]

        image_url = "D:\\CG_Source\\Omniverse\\Extensions\\3DAvatarExtensionPOC\\stable3D\\"+image_prompt.replace(" ", "")+".png"
        image.save(image_url)
        print("image created")

        task.cancel()
        await asyncio.sleep(1)
        # todo bug fix: reload image if same prompt
        outputImage_widget.source_url = image_url
        progress_widget.show_bar(False)

def generateDepthToImage(progress_widget, outputImage_widget, image_prompt: str, inputImageUrl):
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth",
    torch_dtype=torch.float16,
    ).to("cuda")

    init_image = Image.open(inputImageUrl).convert("RGB")
    init_image = init_image.resize((480, 720))
    n_propmt = "bad, deformed, ugly, bad anotomy"
    image = pipe(prompt=image_prompt, image=init_image, negative_prompt=n_propmt, strength=0.7).images[0]
    
    image_url = "D:\\CG_Source\\Omniverse\\Extensions\\3DAvatarExtensionPOC\\stable3D\\"+image_prompt.replace(" ", "")+".png"
    image.save(image_url)
    print("image created")
    outputImage_widget.source_url = image_url
