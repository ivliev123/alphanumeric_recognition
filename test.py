import cv2
import pytesseract
try:
    import Image
except ImportError:
    from PIL import Image

img = cv2.imread ( 'info/image_for_tesseract.png' )
print (pytesseract.image_to_string(img))

print(pytesseract.image_to_string(Image.fromarray(img)))
