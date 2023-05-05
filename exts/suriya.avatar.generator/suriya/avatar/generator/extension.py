import omni.ext
import omni.ui as ui
import torch
from diffusers import StableDiffusionPipeline

# Functions and vars are available to other extension as usual in python: `example.python_ext.some_public_function(x)`
def some_public_function(x: int):
    print("[suriya.avatar.generator] some_public_function was called with x: ", x)
    return x ** x


# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class SuriyaAvatarGeneratorExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("[suriya.avatar.generator] suriya avatar generator startup")

        self._count = 0
        self.textPrompt = ui.SimpleStringModel()

        self._window = ui.Window("My Window", width=300, height=300)
        with self._window.frame:
            #with ui.VStack():
                #label = ui.Label("")


                def on_click():
                    self._count += 1
                    label.text = f"count: {self._count}"

                def on_reset():
                    self._count = 0
                    label.text = "empty"

                def genereate_image(prompt):
                    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
                    pipe.to("cuda")
                    #prompt = "a photograph of an astronaut riding a horse"
                    print(prompt)
                    image = pipe(prompt).images[0]
                    # you can save the image with
                    image.save("D:\\CG_Source\\Omniverse\\Extensions\\3DAvatarExtensionPOC\\stable3D\\stableDiffusionImage.png")
                    print("image created")
                    loadImage()

                with ui.ScrollingFrame():
                    with ui.VStack():
                        #ui.Button("Add", clicked_fn=on_click)
                        ui.StringField(self.textPrompt)
                        
                        def loadImage():
                            print("loading image")
                            with ui.HStack(height=ui.Percent(50)):
                                ui.Image("D:\\CG_Source\\Omniverse\\Extensions\\3DAvatarExtensionPOC\\stable3D\\stableDiffusionImage.png")
                        
                        with ui.HStack(height=30):    
                            ui.Button("Generate", clicked_fn=genereate_image(str(self.textPrompt)))
                            ui.Button("Reload", clicked_fn=loadImage())

    def on_shutdown(self):
        print("[suriya.avatar.generator] suriya avatar generator shutdown")
