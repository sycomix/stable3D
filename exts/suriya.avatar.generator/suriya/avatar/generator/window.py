#import omni.ext
import omni.ui as ui
import asyncio
from .stablediffusion import generateTextToImage, generateImageToImage
from .widgets import ProgressBar

class AvatarWindow(ui.Window):

    def __init__(self, title: str, **kwargs) -> None:
        super().__init__(title, **kwargs)
        # Models
        self.prompt = ui.SimpleStringModel()
        self.frame.set_build_fn(self._build_fn)
        
    def _generate(self):
        run_loop = asyncio.get_event_loop()
        run_loop.create_task(generateTextToImage(self.progress, self.stableImage, self.prompt.as_string))
    
    def _generateI(self):
        run_loop = asyncio.get_event_loop()
        run_loop.create_task(generateImageToImage2(self.progress, self.stableImage, self.prompt.as_string, self.inputImage.source_url))

    #-------------------------Drag and Drop Functions----------------------------
    def drop_accept(self, url):
        print("drop accept")
        # accepts drop of specific extension only
        return True

    def drop(self, widget, event):
        print("drop")
        # called when dropping the data
        widget.source_url = event.mime_data

    def drop_area(self):
        print("drop area")
        # a drop area that shows image when drop
        stack = ui.ZStack()
        with stack:
            ui.Rectangle()
            #ui.Label(f"Accepts {ext}")
            self.inputImage = ui.Image()
        
        self.inputImage.set_accept_drop_fn(lambda d: self.drop_accept(d))
        self.inputImage.set_drop_fn(lambda a, w=self.inputImage: self.drop(w, a))

    def _build_fn(self):
        with self.frame:
            with ui.VStack():
                #label = ui.Label("")
                with ui.HStack(height=0):
                    ui.StringField(model=self.prompt)

                with ui.HStack():
                    self.drop_area()
                    #self.inputImage

                with ui.HStack(height=0):
                    ui.Button("Generate", clicked_fn=self._generate)
                    ui.Button("Generate+I", clicked_fn=self._generateI)
                self.progress = ProgressBar()
                with ui.HStack():
                    self.stableImage = ui.Image()