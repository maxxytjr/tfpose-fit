import argparse
import logging
import time

import cv2
import numpy as np
import math
import imutils

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tqdm import tqdm

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def auto_canny(image, sigma=0.33):
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma)*v))
    upper = int(min(255, (1.0 + sigma)*v))
    edged = cv2.Canny(image, lower, upper)

    return edged



def find_bodypart_xy(pose, p):
    for body in pose:
        try:
            body_part = body.body_parts[p]
            return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))

        except:
            print("Error finding human parts.")
            return 0, 0

    print("Couldn't find human.")
    return 0, 0


def get_euclidean_dist(pt1, pt2):
    return (math.sqrt(np.power(pt1[0] - pt2[0], 2) + np.power(pt1[1] - pt2[1], 2)))


def get_angle(pt0, pt1, pt2):
    """
    :param pt0: reference pt from which we measure the angle  between pt1 and pt2
    :param pt1: point 1 of interest
    :param pt2: point 2 of interest
    :return: angle between pt0pt1, and pt0pt2
    """
    try:
        # cosine rule
        a2 = np.power(pt2[0] - pt1[0], 2) + np.power(pt2[1] - pt1[1], 2)
        b2 = np.power(pt2[0] - pt0[0], 2) + np.power(pt2[1] - pt0[1], 2)
        c2 = np.power(pt1[0] - pt0[0], 2) + np.power(pt1[1] - pt0[1], 2)

        # return angle in degrees
        return(math.acos((b2 + c2 - a2)/math.sqrt(4 * b2 * c2)) * 180/math.pi)

    except:
        return 0



def plank(lh_angle, rh_angle, ll_angle, rl_angle, la_dist, ra_dist):
    """

    :param lh_angle: angle between left hand and left elbow/shoulder
    :param rh_angle: angle between right hand and torso
    :param ll_angle: angle between left leg and torso
    :param rl_angle: angle between right leg and torso
    :param la_dist: distance from left ankle to head
    :param ra_dist: distance from right ankle to head
    :return: boolean
    """

    if (lh_angle in range(60, 110) or rh_angle in range(60, 110))\
            and (ll_angle in range(140, 180) or rl_angle in range(140, 180))\
            and (la_dist in range(50, 250) or ra_dist in range(50, 250)):
        return True
    else:
        return False


def draw_str(image, text_coords, s, color, scale):
    (x, y) = text_coords
    if (color[0] + color[1] + color[2] == 255 * 3):
        cv2.putText(image, s, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness=4, lineType=10)
    else:
        cv2.putText(image, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness=4, lineType=10)

    cv2.putText(image, s, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, (255, 255, 255), lineType=11)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation video')
    parser.add_argument('--video', type=str, default=0)

    parser.add_argument('--resize', type=str, default='656x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    
    parser.add_argument('--video_out', type=str, default="processed_video.mp4",
                        help='name of directory of processed video file')

    args = parser.parse_args()

    print("mode 0: Only Pose Estimation \nmode 1: Fall Detection \nmode 2: Plank Detection \nmode 3: Squat Analysis(Frontal)")
    mode = int(input("Enter a mode : "))


    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)


    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')

    cam = cv2.VideoCapture(args.video)
    # get frame count, frame width and height of input video
    nb_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))

    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))


    # output video after processing with tensor pose
    video_out = args.video_out
    video_writer = cv2.VideoWriter(filename=video_out,
                                   fourcc=cv2.VideoWriter_fourcc(*'MP4V'),
                                   fps=cam.get(cv2.CAP_PROP_FPS),
                                   frameSize=(frame_w, frame_h))

    y1 = [0,0]
    global height, width
    orange_color = (0, 140, 255)

    for i in tqdm(range(int(nb_frames))):
        ret, image = cam.read()

        if not ret:
            print("Video Ended. Exiting...")
            break


        logger.debug('image process+')

        humans = e.inference(image, resize_to_default=(frame_w > 0 and frame_h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        height, width = image.shape[0], image.shape[1]

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        # human counter mode
        if mode == 1:
            for human in humans:
                # select one person
                try:
                    a = human.body_parts[0]  # head point
                    x = a.x * image.shape[1]  # x coord of head relative to image
                    y = a.y * image.shape[0]  # y coord of head relative to image
                    y1.append(y)
                except:
                    pass
                # comparing distance between current y value and that 2 iterations ago, and see if it is more than
                # the threshold (y value increases down the frame)
                y_thresh = 25
                if ((y - y1[-2]) > y_thresh):
                    cv2.putText(image, "Fall detected!", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 2.5, (0, 0, 255), 2, 11)

        elif mode == 2:
            if len(humans) > 0:
                # distance calculations
                head_to_lh_dist = int(get_euclidean_dist(find_bodypart_xy(humans, 0), find_bodypart_xy(humans, 7)))
                head_to_rh_distance = int(get_euclidean_dist(find_bodypart_xy(humans, 0), find_bodypart_xy(humans, 4)))

                # angle calculations
                angle_lh_lelbow_lshoulder = get_angle(find_bodypart_xy(humans, 7), find_bodypart_xy(humans, 6), find_bodypart_xy(humans, 5))
                angle_rh_relbow_rshoulder = get_angle(find_bodypart_xy(humans, 4), find_bodypart_xy(humans, 3), find_bodypart_xy(humans, 2))

                angle_lhip_lknee_lankle = get_angle(find_bodypart_xy(humans, 11), find_bodypart_xy(humans, 12), find_bodypart_xy(humans, 13))
                angle_rhip_rknee_rankle = get_angle(find_bodypart_xy(humans, 8), find_bodypart_xy(humans, 9), find_bodypart_xy(humans, 10))

                angle_lshoulder_lhip_lfeet = get_angle(find_bodypart_xy(humans, 11), find_bodypart_xy(humans, 5), find_bodypart_xy(humans, 13))
                angle_rshoulder_rhip_rfeet = get_angle(find_bodypart_xy(humans, 8), find_bodypart_xy(humans, 2), find_bodypart_xy(humans, 10))

                if plank(angle_lh_lelbow_lshoulder, angle_rh_relbow_rshoulder, angle_lhip_lknee_lankle, angle_rhip_rknee_rankle, head_to_rh_distance, head_to_lh_dist):
                    action = "Planking"
                    draw_str(image, (20, 50), "Planking", orange_color, 2)
                    logger.debug("***Plank***")

                if angle_lshoulder_lhip_lfeet in range(50, 100) or angle_rshoulder_rhip_rfeet in range(50, 100):
                    action = "Downward Dog"
                    cv2.putText(image, action, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                    
        elif mode == 3:

            squat_text = 'SQUAT'
            caving_text = 'KNEES CAVING IN !!!'

            squat_text_size = cv2.getTextSize(squat_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
            squat_text_x = int((image.shape[1] - squat_text_size[0])/2)

            caving_text_size = cv2.getTextSize(caving_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            caving_text_x = int((image.shape[1] - caving_text_size[0])/2)

            rknee_xy = find_bodypart_xy(humans, 9)
            rfeet_xy = find_bodypart_xy(humans, 10)
            rhip_xy = find_bodypart_xy(humans, 8)

            lknee_xy = find_bodypart_xy(humans, 12)
            lfeet_xy = find_bodypart_xy(humans, 13)
            lhip_xy = find_bodypart_xy(humans, 11)

            head_xy = find_bodypart_xy(humans, 0)

            rhip_rfeet_dist = int(get_euclidean_dist(rhip_xy, rfeet_xy))
            lhip_lfeet_dist = int(get_euclidean_dist(lhip_xy, lfeet_xy))

            average_hip_to_feet_dist = int((rhip_rfeet_dist + lhip_lfeet_dist)/2)

            # knees cave in
            if ((rknee_xy[0] > rfeet_xy[0]) or (lknee_xy[0] < lfeet_xy[0])) and head_xy[1] in range(300, int(height * 0.75)):
                cv2.putText(image, "KNEES CAVING IN!!!",
                        (caving_text_x, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 0, 255), 3)

            if (head_xy[1] in range(300, int(height * 0.75)) and ((rknee_xy[0] <= rfeet_xy[0]) or (lknee_xy[0] >= lfeet_xy[0]))):
                cv2.putText(image, "SQUAT",
                            (squat_text_x, 110), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 255, 0), 2)

            cv2.putText(image, "r_knee: {}".format(rknee_xy[0]),
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 140, 255), 2)

            cv2.putText(image, "r_ankle: {}".format(rfeet_xy[0]),
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 140, 255), 2)

            cv2.putText(image, "l_knee: {}".format(lknee_xy[0]),
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 140, 255), 2)

            cv2.putText(image, "l_ankle: {}".format(lfeet_xy[0]),
                        (10,190), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 140, 255), 2)

        elif mode == 0:
            pass

        fps_time = time.time()

        print("writing to video")

        video_writer.write(np.uint8(image))

        logger.debug('finished+')

    video_writer.release()
    cam.release()
    cv2.destroyAllWindows()
