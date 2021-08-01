from easydict import EasyDict

_C = EasyDict()
APP_CONFIG = _C

_C.app = EasyDict()

_C.app.console = EasyDict()
_C.app.console.font_name = 'Consolas'
_C.app.console.text_size = 10

_C.app.window = EasyDict()
_C.app.window.preferred_width = 1280
_C.app.window.preferred_height = 720

_C.app.core = EasyDict()

_C.app.core.face_detection = EasyDict()

_C.app.core.face_detection.algorithms = EasyDict()

_C.app.core.face_detection.algorithms.s3fd = EasyDict()
_C.app.core.face_detection.algorithms.s3fd.weight_path = "C:\\Users\\tonkec\\Documents\\deepfake\\data\\weights\\s3fd\\s3fd.pth"


_C.app.gui = EasyDict()

_C.app.gui.video_widget = EasyDict()
_C.app.gui.video_widget.video_aspect_ratio = 16. / 9
