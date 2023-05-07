#import omni.ext
import omni.ui as ui
import asyncio
from .stablediffusion import generateImage
from .widgets import ProgressBar

class AvatarWindow(ui.Window):

    def __init__(self, title: str, **kwargs) -> None:
        super().__init__(title, **kwargs)
        # Models
        self.prompt = ui.SimpleStringModel()
        self.frame.set_build_fn(self._build_fn)
        
    def _generate(self):
        run_loop = asyncio.get_event_loop()
        run_loop.create_task(generateImage(self.progress, self.prompt.as_string, self.stableImage))

    def _build_fn(self):
        with self.frame:
            with ui.VStack():
                #label = ui.Label("")
                with ui.HStack(height=0):
                    ui.StringField(model=self.prompt)

                with ui.HStack(height=0):
                    ui.Button("Generate", clicked_fn=self._generate)
                self.progress = ProgressBar()
                with ui.HStack():
                    self.stableImage = ui.Image()