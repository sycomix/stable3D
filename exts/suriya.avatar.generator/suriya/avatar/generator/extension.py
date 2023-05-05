import omni.ext
import omni.ui as ui
import omni.kit.commands
import carb
import torch
from diffusers import StableDiffusionPipeline

# Functions and vars are available to other extension as usual in python: `example.python_ext.some_public_function(x)`
def some_public_function(x: int):
    print("[suriya.avatar.generator] some_public_function was called with x: ", x)
    return x ** x


# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class SuriyaSpanCubeExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("[suriya.avatar.generator] suriya avatar generator startup")

        self.prompt = ui.SimpleStringModel()

        self._window = ui.Window("Stable3D", width=300, height=300)
        with self._window.frame:
            with ui.VStack():
                #label = ui.Label("")
                ui.StringField(model=self.prompt)

                def generateImage():
                    carb.log_info("Stable Diffusion Stage")
                    print("creating image with prompt: "+self.prompt.as_string)
                    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
                    pipe.to("cuda")
                    #prompt = "a photograph of an astronaut riding a horse"
                    prompt = self.prompt.as_string
                    image = pipe(prompt).images[0]
                    # you can save the image with
                    image.save("D:\\CG_Source\\Omniverse\\Extensions\\3DAvatarExtensionPOC\\stable3D\\stableDiffusionImage.png")
                    print("image created")
                    #shutil.copy("D:\\CG_Source\\Omniverse\\Extensions\\3DAvatarExtensionPOC\\stable3D\\stableDiffusionImage.png",
                    #            "D:\\CG_Source\\Omniverse\\Extensions\\3DAvatarExtensionPOC\\stable3D\\stableDiffusionImageDefault.png")

                with ui.HStack():
                    ui.Button("Generate", clicked_fn=generateImage)

    def on_shutdown(self):
        print("[suriya.avatar.generator] suriya avatar generator shutdown")
