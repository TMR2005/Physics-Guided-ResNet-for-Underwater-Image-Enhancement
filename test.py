import cv2
from utils.dcp import dcp_restore

img = cv2.imread("test.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

J, t, A = dcp_restore(img)
print("A:", A)
print("t min/max:", t.min(), t.max())
cv2.imwrite("output.jpg", cv2.cvtColor(J, cv2.COLOR_RGB2BGR))