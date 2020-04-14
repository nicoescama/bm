"""Departure Warning System with a Monocular Camera"""

__author__ = "Junsheng Fu"
__email__ = "junsheng.fu@yahoo.com"
__date__ = "March 2017"


from lane import *
from imutils.video import VideoStream
import time
import imutils


if __name__ == "__main__":

        print("[INFO] starting video stream...")
        #vs = VideoStream(src=0).start()
        vs = VideoStream(usePiCamera=True).start()
        time.sleep(2.0)
        total = 0

        while True:
            frame = vs.read()
            orig = frame.copy()
            frame = imutils.resize(frame, width=500)
            #frame = process_frame(frame)
            imagepath = 'examples/test3.jpg'
            img = cv2.imread(imagepath)
            img_aug = process_frame(img, True)

            # detect faces in the grayscale frame
            #cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("g"):
                total += 1

            # if the `q` key was pressed, break from the loop
            elif key == ord("q"):
                break

        # do a bit of cleanup
        print("[INFO] {} face images stored".format(total))
        print("[INFO] cleaning up...")
        cv2.destroyAllWindows()
        vs.stop()
        

