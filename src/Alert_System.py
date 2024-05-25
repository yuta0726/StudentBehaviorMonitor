import cv2
import mediapipe as mp
import statistics

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.re = 0
        self.end_list = []
        self.nose_list_x = []
        self.left_eye_list_x = []
        self.right_eye_list_x = []
        self.left_shoulder_list = []
        self.right_shoulder_list = []
        self.warning_count = 0

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

    def process_frame(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        mp_pose = mp.solutions.pose

        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh, \
                mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

            while self.video.isOpened():
                success, image = self.video.read()

                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results_face = face_mesh.process(image)
                results_pose = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results_face.multi_face_landmarks:
                    for face_landmarks in results_face.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                        left_eye_landmarks = face_landmarks.landmark[362:386]
                        right_eye_landmarks = face_landmarks.landmark[386:410]

                        for landmark in left_eye_landmarks:
                            left_eye_x = int(landmark.x * image.shape[1])
                            self.left_eye_list_x.append(left_eye_x)

                        for landmark in right_eye_landmarks:
                            right_eye_x = int(landmark.x * image.shape[1])
                            self.right_eye_list_x.append(right_eye_x)

                        nose_landmark = face_landmarks.landmark[4]
                        nose_x = int(nose_landmark.x * image.shape[1])
                        self.nose_list_x.append(nose_x)

                    mean_left_eye_x = statistics.mean(self.left_eye_list_x)
                    mean_right_eye_x = statistics.mean(self.right_eye_list_x)
                    mean_nose_x = statistics.mean(self.nose_list_x)
                    mean_left_x = mean_left_eye_x - mean_nose_x
                    mean_right_x = mean_right_eye_x - mean_nose_x
                    left_x = left_eye_x - nose_x
                    right_x = right_eye_x - nose_x

                    if not mean_left_x - 20 <= left_x <= mean_left_x + 20 and mean_right_x - 30 <= right_x <= mean_right_x + 30:
                        self.end_list.append(right_x)
                        print("end_list", len(self.end_list))

                if results_pose.pose_landmarks:

                    left_shoulder = results_pose.pose_landmarks.landmark[11]
                    right_shoulder = results_pose.pose_landmarks.landmark[12]

                    left_shoulder_x = int(left_shoulder.x * image.shape[1])
                    left_shoulder_y = int(left_shoulder.y * image.shape[0])
                    right_shoulder_x = int(right_shoulder.x * image.shape[1])
                    right_shoulder_y = int(right_shoulder.y * image.shape[0])

                    self.left_shoulder_list.append((left_shoulder_x, left_shoulder_y))
                    self.right_shoulder_list.append((right_shoulder_x, right_shoulder_y))

                    mean_left_shoulder_x = statistics.mean([coord[0] for coord in self.left_shoulder_list])
                    mean_left_shoulder_y = statistics.mean([coord[1] for coord in self.left_shoulder_list])
                    mean_right_shoulder_x = statistics.mean([coord[0] for coord in self.right_shoulder_list])
                    mean_right_shoulder_y = statistics.mean([coord[1] for coord in self.right_shoulder_list])

                    if not mean_left_shoulder_x - 20 <= left_shoulder_x <= mean_left_shoulder_x + 20 or \
                       not mean_left_shoulder_y - 20 <= left_shoulder_y <= mean_left_shoulder_y + 20:
                        self.end_list.append((left_shoulder_x, left_shoulder_y))
                        print("end_list", len(self.end_list))

                    if not mean_right_shoulder_x - 20 <= right_shoulder_x <= mean_right_shoulder_x + 20 or \
                       not mean_right_shoulder_y - 20 <= right_shoulder_y <= mean_right_shoulder_y + 20:
                        self.end_list.append((right_shoulder_x, right_shoulder_y))
                        print("end_list", len(self.end_list))

                if len(self.end_list) >= 15:
                    self.warning_count += 1
                    if self.warning_count == 1:
                        image[:] = (0, 255, 255)  # 画像全体を黄色くする
                        self._put_centered_text(image, "Warning 1", (0, 0, 0))
                        print("Warning 1")
                    elif self.warning_count == 2:
                        image[:] = (0, 165, 255)  # 画像全体をオレンジ色にする
                        self._put_centered_text(image, "Warning 2", (0, 0, 0))
                        print("Warning 2")
                    elif self.warning_count == 3:
                        self.re = 1
                        image[:] = (0, 0, 255)  # 画像全体を赤くする
                        self._put_centered_text(image, "Qualification revoked", (255, 255, 255))
                        print("Qualification revoked")
                        cv2.imshow('Video Feed', image)
                        cv2.waitKey(0)  # 無限に待機してキー入力を待つ
                        break

                    cv2.imshow('Video Feed', image)
                    cv2.waitKey(1500)  # 1.5秒間表示
                    self.end_list.clear()  # end_listをリセット

                cv2.imshow('Video Feed', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.__del__()

    def _put_centered_text(self, image, text, color):
        font_scale = 2
        thickness = 3
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = (image.shape[0] + text_size[1]) // 2
        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)

if __name__ == "__main__":
    video_camera = VideoCamera()
    video_camera.process_frame()
