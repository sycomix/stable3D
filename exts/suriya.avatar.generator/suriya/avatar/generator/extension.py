import omni.kit.commands
from .window import AvatarWindow

# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class SuriyaAvatarGeneratorExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        print("[suriya.avatar.generator] suriya avatar generator startup")
        self._window = AvatarWindow("Stable3D", width=300, height=300)
        
    def on_shutdown(self):
        print("[suriya.avatar.generator] suriya avatar generator shutdown")
        self._window.destroy()
        self._window = None
