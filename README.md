# Detect-Facial-Features in real time
Using dlib, OpenCV, and Python, to detect in real time how to open/closed eyes,mouth, position of the head, and the emotion


This detection is based on Dlib which help to extract the cordinates for facial features like eyes, nose, mouth and jaw using 68 facial landmark indexes.

68 Facial landmark indexes
The facial landmark detector implemented inside dlib produces 68 (x, y)-coordinates that map to specific facial structures. These 68 point mappings were obtained by training a shape predictor on the labeled iBUG 300-W dataset.

Below we can visualize what each of these 68 coordinates map to:

![Image description](68747470733a2f2f7777772e7079696d6167657365617263682e636f6d2f77702d636f6e74656e742f75706c6f6164732f323031372f30342f66616369616c5f6c616e646d61726b735f36386d61726b75702e6a7067.jpg)

N|Solid

Examining the image, we can see that facial regions can be accessed via simple Python indexing (assuming zero-indexing with Python since the image above is one-indexed):

⋅⋅*The mouth can be accessed through points [48, 68].
⋅⋅*The right eyebrow through points [17, 22].
⋅⋅*The left eyebrow through points [22, 27].
⋅⋅*The right eye using [36, 42].
⋅⋅*The left eye with [42, 48].
⋅⋅*The nose using [27, 35].
⋅⋅*And the jaw via [0, 17].

These mappings are encoded inside the FACIAL_LANDMARKS_IDXS dictionary inside face_utils of the imutils library.
