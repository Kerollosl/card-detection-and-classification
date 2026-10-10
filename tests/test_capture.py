import sys,types,unittest
from unittest.mock import Mock,patch
sys.modules['cv2']=types.SimpleNamespace()
sys.modules['ultralytics']=types.SimpleNamespace(YOLO=Mock())
import web_app
class CaptureTests(unittest.TestCase):
 def setUp(self):
  web_app.model=Mock(return_value=[types.SimpleNamespace(plot=lambda:object())])
  self.camera=Mock();self.camera.isOpened.return_value=True
  web_app.cv2.VideoCapture=Mock(return_value=self.camera)
  web_app.cv2.imencode=Mock(return_value=(True,types.SimpleNamespace(tobytes=lambda:b'jpeg')))
 def test_missing_camera_releases_lock(self):
  self.camera.isOpened.return_value=False
  self.assertEqual(web_app.app.test_client().get('/video_feed').status_code,503)
  self.camera.release.assert_called_once();self.assertFalse(web_app.camera_lock.locked())
 def test_disconnect_releases_camera(self):
  self.camera.read.return_value=(True,object())
  response=web_app.app.test_client().get('/video_feed',buffered=False)
  self.assertIn(b'jpeg',next(iter(response.response)))
  self.assertEqual(web_app.app.test_client().get('/video_feed').status_code,409)
  response.close();self.camera.release.assert_called_once()
  self.assertFalse(web_app.camera_lock.locked())
 def test_missing_model(self):
  web_app.model=None
  self.assertEqual(web_app.app.test_client().get('/video_feed').status_code,503)
