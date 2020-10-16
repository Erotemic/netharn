## Known Bugs

* The background matplotlib dumps of logged tensorboard scalars sometimes outputs garbage images. The fonts are in the wrong place, and parts of the image are cut off. This may be due to some race condition. Perhaps multiple dumps were spawned at once and were clobbering the encoding of one another?


* The per-batch iteration metrics seem to jump on the x-axis in the tensorboard logs. Not sure why this is. Perhaps there is a scheduler bug? 


* If PyQt5 is installed and there is a problem with the matplotlib qt backend
  then you may just get an error that crashes the system:

  ```
    QObject::moveToThread: Current thread (0x5636e99e0690) is not the object's thread (0x5636ea1e26b0).
    Cannot move to target thread (0x5636e99e0690)

    qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "$HOME/.local/conda/envs/py38/lib/python3.8/site-packages/cv2/qt/plugins" even though it was found.
    This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

    Available platform plugins are: xcb, eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl.

```

The workaround is to uninstall PyQt5, but that's not great. Need to detect that
this will happen before it does so we can warn and avoid it.
