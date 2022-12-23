from flask import Flask, request, flash, url_for, redirect, render_template, session, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

from io import BytesIO
import os
import cv2
import utils_pose as utils
from movenet import Movenet
from data import BodyPart
# from list_function import *
import tensorflow as tf
import threading
import numpy as np
import cv2
from tensorflow import keras
import time

import base64

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database//database.sqlite3'
# app = Flask (__name__)
# app.config ['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.sqlite3'
app.config['SECRET_KEY'] = "random string"

db = SQLAlchemy(app)

class_names = ['Pose_1', 'Pose_10', 'Pose_11', 'Pose_12', 'Pose_13', 'Pose_14',
                     'Pose_15', 'Pose_16', 'Pose_2', 'Pose_3', 'Pose_4', 'Pose_5',
                     'Pose_6', 'Pose_7', 'Pose_8', 'Pose_9']

description_class_name = [
"Quay sang trái 90°, đưa chân trái ra ngoài thành tư thế đi bên trái thực hiện động tác chặn tay trái thấp.",
"Bước sang phải 90° bằng chân phải thành tư thế thẳng về phía trước và thực hiện động tác cản phá thấp bằng cánh tay phải sau đó giữ nguyên tư thế và thực hiện cú đấm vào phần giữa bằng nắm tay trái",
"Bước sang trái 90° sang tư thế đi bên trái và thực hiện động tác cản phá bằng tay trái.",
"Giữ nguyên chân trái, thực hiện cú đá trước khu trung tuyến bằng chân phải. Bước xuống bằng chân phải vào tư thế đi bên phải và thực hiện cú đấm khu trung tuyến bằng nắm tay phải.",
"Bước sang phải 180° sang tư thế đi bộ bên phải và thực hiện động tác đỡ đòn bằng tay phải.",
"Giữ nguyên chân phải, thực hiện quả đá trước khu trung tuyến bằng chân trái sau đó bước xuống bằng chân trái sang tư thế đi bên trái và thực hiện cú đấm khu trung tuyến bằng nắm tay trái.",
"Bước sang phải 90° bằng chân trái và thực hiện động tác cản phá tầm thấp bằng cánh tay trái.",
"Bước chân phải xuống tư thế thẳng về phía trước và thực hiện cú đấm khu trung tuyến bằng nắm tay phải.",
"Bước chân phải về phía trước thành tư thế đi bên phải và đấm bằng nắm tay phải vào phần giữa.",
"Quay sang phải 180°, bước bằng chân phải sang tư thế đi bên phải thực hiện động tác hạ thấp cánh tay phải.",
"bước lên trước bằng chân trái thành tư thế đi bên trái và đấm bằng nắm tay trái vào phần giữa.",
"Bước sang trái 90° bằng chân trái sang tư thế trái, thực hiện động tác chặn tay trái thấp và thực hiện một cú đấm vào phần giữa bằng nắm tay phải.",
"Bước sang phải 90° bằng chân phải sang tư thế đi bộ sang và thực hiện một cú chặn giữa vào bên trong bằng cánh tay trái.",
"Bước lên trước bằng chân trái sang tư thế đi bên trái và dùng tay phải đấm vào phần giữa.",
"Bước sang trái 180° bằng chân trái sang tư thế đi bên trái và thực hiện cú truy cản từ giữa vào trong bằng cánh tay phải.",
"Bước lên trước bằng chân phải sang tư thế đi bên phải và thực hiện cú đấm khu trung tuyến bằng nắm tay trái."
]


model = tf.keras.models.load_model("./models/model_pose_taekwondo_5fps.h5")
movenet = Movenet('movenet_thunder')

label = "waiting"
Tmp_result = ''
# session["video_path"] = path
Tmp_path = ''

predicted_pose = ""
unpredicted_pose = ""

class Users(db.Model):
   id = db.Column(db.Integer, primary_key=True)
   username = db.Column(db.String(100))
   email = db.Column(db.String(50),  unique=True,)
   password = db.Column(db.String(200))

   def __init__(self, username, email,  password):
      self.username = username
      self.email = email
      self.password = password

class Videos(db.Model):
   id = db.Column(db.Integer, primary_key=True)
   location = db.Column(db.String(1000))
   result = db.Column(db.Float, nullable=False)
   predicted = db.Column(db.String(10000))
   unpredicted = db.Column(db.String(10000))
   user_id = db.Column(db.Integer)

   def __init__(self, location,result, predicted, unpredicted, user_id):
      self.location = location
      self.result = result
      self.predicted = predicted
      self.unpredicted = unpredicted
      self.user_id = user_id


def render_picture(data):
    render_pic = base64.b64encode(data).decode('ascii')
    return render_pic

def decode_base64(data):
#     <img src="data:;base64,{{ image }}"/>
    return b64encode(data).decode("utf-8")

# detect pose on video

def get_center_point(landmarks, left_bodypart, right_bodypart):
  """Calculates the center point of the two given landmarks."""

  left = tf.gather(landmarks, left_bodypart.value, axis=1)
  right = tf.gather(landmarks, right_bodypart.value, axis=1)
  center = left * 0.5 + right * 0.5
  return center

def write_video(file_path, frames, fps):


    height, width, channels = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
# write_video('output.avi', np.array(t), 30)

def get_pose_size(landmarks, torso_size_multiplier=2.5):
  """Calculates pose size.

  It is the maximum of two values:
    * Torso size multiplied by `torso_size_multiplier`
    * Maximum distance from pose center to any pose landmark
  """
  # Hips center
  hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                 BodyPart.RIGHT_HIP)

  # Shoulders center
  shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER,
                                      BodyPart.RIGHT_SHOULDER)

  # Torso size as the minimum body size
  torso_size = tf.linalg.norm(shoulders_center - hips_center)

  # Pose center
  pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                     BodyPart.RIGHT_HIP)
  pose_center_new = tf.expand_dims(pose_center_new, axis=1)
  # Broadcast the pose center to the same size as the landmark vector to
  # perform substraction
  pose_center_new = tf.broadcast_to(pose_center_new,
                                    [tf.size(landmarks) // (17*2), 17, 2])

  # Dist to pose center
  d = tf.gather(landmarks - pose_center_new, 0, axis=0,
                name="dist_to_pose_center")
  # Max dist to pose center
  max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

  # Normalize scale
  pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)

  return pose_size


def normalize_pose_landmarks(landmarks):
  """Normalizes the landmarks translation by moving the pose center to (0,0) and
  scaling it to a constant pose size.
  """
  # Move landmarks so that the pose center becomes (0,0)
  pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                 BodyPart.RIGHT_HIP)
  pose_center = tf.expand_dims(pose_center, axis=1)
  # Broadcast the pose center to the same size as the landmark vector to perform
  # substraction
  pose_center = tf.broadcast_to(pose_center,
                                [tf.size(landmarks) // (17*2), 17, 2])
  landmarks = landmarks - pose_center

  # Scale the landmarks to a constant pose size
  pose_size = get_pose_size(landmarks)
  landmarks /= pose_size

  return landmarks


def landmarks_to_embedding(landmarks_and_scores):
  """Converts the input landmarks into a pose embedding."""
  # Reshape the flat input into a matrix with shape=(17, 3)
  reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)

  # Normalize landmarks 2D
  landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])

  # Flatten the normalized landmark coordinates into a vector
  embedding = keras.layers.Flatten()(landmarks)

  return embedding

def draw_prediction_on_image(
    image, person, crop_region=None, close_figure=True,
    keep_input_size=False):
  # Draw the detection result on top of the image.
  image_np = utils.visualize(image, [person])

  # Plot the image with detection results.
#   height, width, channel = image.shape
#   aspect_ratio = float(width) / height
#   fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
#   im = ax.imshow(image_np)

  if close_figure:
    plt.close(fig)

  if not keep_input_size:
    image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))

  return image_np

def draw_prediction_on_image_danger(
    image, person, crop_region=None, close_figure=True,
    keep_input_size=False):
  # Draw the detection result on top of the image.
  image_np = utils.visualize(image, [person],(0,0,255))

  # Plot the image with detection results.
#   height, width, channel = image.shape
#   aspect_ratio = float(width) / height
#   fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
#   im = ax.imshow(image_np)

  if close_figure:
    plt.close(fig)

  if not keep_input_size:
    image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))

  return image_np

def get_keypoint_landmarks(person):
    pose_landmarks = np.array(
      [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
        for keypoint in person.keypoints],
      dtype=np.float32)
    return pose_landmarks

def predict_pose(model, lm_list):
    global label
#     lm_list = np.array(lm_list)
#     lm = lm.reshape(lm,(1,5,34))
#     lm_list = np.expand_dims(lm_list, axis=0)
#     print(lm_list.shape)
    results = model.predict(lm_list)
    print(np.argmax(results))
    if np.max(results)>=0.6 and np.argmax(results) <= 15:
        label = class_names[np.argmax(results)]
    return label

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (43, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def draw_class_on_image_top(label, img, bottomLeftCornerOfText_p):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = bottomLeftCornerOfText_p
    fontScale = 0.5
    fontColor = (43, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def draw_class_on_image_bot(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 60)
    fontScale = 1
    fontColor = (10, 255, 255)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(input_tensor, inference_count=3):
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)

    for _ in range(inference_count - 1):
        detection = movenet.detect(input_tensor.numpy(),
                                reset_crop_region=False)

    return detection

def is_available_dou(list , var1, var2):
    for i in range(len(list)-1):
        if list[i] == var1 and list[i+1] ==  var2:
            return True
    return False

def is_available(list , var):
    for i in list:
        if i == var:
            return True
    return False

def standard_pose16(list):
    print(len(list)-1)
    for i in range(len(list)-2):
        if list[i] == 'Pose_16':
            list.pop(i)
    return list

def standardize(list):
    xyz = list
    j = 0
    for i in range(0,len(xyz)):
        if xyz[i] == 'Pose_16':
            j = i

    new_list_label = []
    for i in range(0,j+1):
        new_list_label.append(xyz[i])
#     print(new_list_label)
    return new_list_label

def normalsize(list):
    i = 0
    while i < len(list)-1:
        if list[i] == list[i+1]:
            list.pop(i)
#             print(i)
            i -= 1
#             print(i)
        i += 1
    return list

def list_except(list1 , list2):
    list = []
    for i in list2:
        if not is_available(list1 ,i):
            list.append(i)
    return list

def colab_list(list1, list2):
    list = []
    seri_label = ['Pose_1','Pose_2','Pose_3','Pose_4','Pose_5','Pose_6','Pose_7','Pose_8',
                      'Pose_9','Pose_10','Pose_11','Pose_12','Pose_13','Pose_14','Pose_15','Pose_16','Pose_16']

    for i in seri_label:
        for l1 in list1:
            if i == l1:
                list.append(i)
        for l2 in list2:
            if i == l2:
                list.append(i)
    return list



def caculator_acc(list_label):

    seri_label = ['Pose_1','Pose_2','Pose_3','Pose_4','Pose_5','Pose_6','Pose_7','Pose_8',
                  'Pose_9','Pose_10','Pose_11','Pose_12','Pose_13','Pose_14','Pose_15','Pose_16','Pose_16']

    list_after_standardized = []
    tmp = 0

    list_label = standardize(list_label)
    print(list_label)
    list_label = standard_pose16(list_label)
    print(list_label)
    list_label.append('Pose_16')
    while not is_available(list_after_standardized,'Pose_16'):
        for i in range(0,len(list_label)-1):
            if is_available_dou(seri_label,list_label[i],list_label[i+1]):
                list_after_standardized.append(list_label[i])
                list_after_standardized.append(list_label[i+1])
                i = i + 1


    list_after_standardized = normalsize(list_after_standardized)

    list_exc = list_except(list_after_standardized,list_label)

    final_list = colab_list(list_exc,list_after_standardized)
    final_list = normalsize(final_list)
    print(list_exc)

    global predicted, unpredicted
    string = ''
    for i in final_list:
        if( i == final_list[len(final_list)-1]):
            string = string + i
        else:
            string = string + i + ", "
#     print(string)
    predicted = string

    unpredicted_list = list_except(final_list,['Pose_1','Pose_2','Pose_3','Pose_4','Pose_5','Pose_6','Pose_7','Pose_8',
                                                                        'Pose_9','Pose_10','Pose_11','Pose_12','Pose_13','Pose_14','Pose_15','Pose_16'])
    string = ''
    for i in unpredicted_list:
        if( i == unpredicted_list[len(unpredicted_list)-1]):
            string = string + i
        else:
            string = string + i + ", "

    unpredicted = string

    print(list_after_standardized)
    print(final_list)
    result = len(final_list)*100/15
    print(str(result)+"%")

    if result >= 100:
        result = 100
    return result

def gen_video(path):
    model.summary()
    cap = cv2.VideoCapture(path)
    lm = []
    time_step = 5

    warmup_frames = 60
    i = 0
    list = []
    list_label = []
    list_label.append('waiting')
    while cap.isOpened():
        ret ,frame = cap.read()
        if ret == True:
            #Reshape Image
            img = frame.copy()
            img = cv2.resize(img , (980,540))
            img = tf.convert_to_tensor(img, dtype=tf.uint8)

            i = i + 1
        #     print(img)

            if( i > warmup_frames):

                print("start detect:")
                person = detect(img)
                landmarks = get_keypoint_landmarks(person)
                lm_pose = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(landmarks), (1, 51)))
        #         print(lm_pose)

                lm.append(lm_pose)
                img = np.array(img)
                img = draw_prediction_on_image(img, person, crop_region=None,
                                           close_figure=False, keep_input_size=True)

                if(len(lm) == time_step):
                    lm = tf.reshape(lm ,(1,5,34))
                    t1 = threading.Thread(target=predict_pose, args=(model, lm,))
                    t1.start()
                    lm = []

                    if label != list_label[len(list_label)-1] and len(label) >  1:
                        list_label.append(label)

            img = np.array(img)
            img = draw_class_on_image(label,img)

            list.append(img)

        else:
            break
    cap.release()
    write_video(path, np.array(list), 40)
    list_label.remove('waiting')
    return caculator_acc(list_label)

def stream_video(path):
    camera = cv2.VideoCapture(path)
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def stream_webcam_html():
    cap = cv2.VideoCapture(0)
    lm = []
    time_step = 5
    seri_label = ['Pose_1','Pose_2','Pose_3','Pose_4','Pose_5','Pose_6','Pose_7','Pose_8',
                  'Pose_9','Pose_10','Pose_11','Pose_12','Pose_13','Pose_14','Pose_15','Pose_16']
    warmup_frames = 60
    i = 0
    count = 0
    count_1 = 0
    tmp = 0
    tmp_1 = 0
    tmp_2 = 0
    global label
    label = "waiting"
    while cap.isOpened():
        ret ,frame = cap.read()
        if ret == True:
            #Reshape Image
            img = frame.copy()
            img = cv2.resize(img , (980,540))
            img = tf.convert_to_tensor(img, dtype=tf.uint8)

            i = i + 1

            if( i > warmup_frames):

                print("start detect:")
                person = detect(img)
                landmarks = get_keypoint_landmarks(person)
                lm_pose = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(landmarks), (1, 51)))
        #         print(lm_pose)

                lm.append(lm_pose)
                img = np.array(img)
                if tmp == 0:
                    img = draw_prediction_on_image(img, person, crop_region=None,
                                               close_figure=False, keep_input_size=True)
                else :
                    img = draw_prediction_on_image_danger(img, person, crop_region=None,
                                                                   close_figure=False, keep_input_size=True)

                if(len(lm) == time_step):
                    lm = tf.reshape(lm ,(1,5,34))
                    t1 = threading.Thread(target=predict_pose, args=(model, lm,))
                    t1.start()

                    lm = []



                    if label == seri_label[count]:
#                         is_available(list , var)
                        tmp_1 = tmp_1 + 1
                        if tmp_1 == 1:
                            count = count + 1
                            tmp_1 = 0
                        tmp = 0
                    else :
                        tmp = tmp + 1

            img = np.array(img)

            label_al_pr = "alrealdy predicted label: "
            for i_1 in range(0,count):
                label_al_pr = label_al_pr+", "+seri_label[i_1]
            img = draw_class_on_image_top(label_al_pr,img,(10,90))

            if(count == 15):
                string_label = "complete taekwondo lesson 1"
            else :
                string_label = "label: "+label

            img = draw_class_on_image(string_label,img)
            img = draw_class_on_image_bot("waiting predicted label: "+seri_label[count],img)

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/stream_webcam')
def stream_webcam():
    return Response(stream_webcam_html(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/learningTaekwondo')
def learningTaekwondo():
    return render_template('stream_web.html')


@app.route('/play', methods=["GET",'POST'])
def play():
    global Tmp_result, Tmp_path, predicted, unpredicted
    if session.get('user_id') == None:
        flash('Vui lòng đăng nhập tài khoản', 'error')
        return redirect(url_for('login'))
    if request.method == 'POST':

        video_id = request.form.get('video_id')
        Video = Videos.query.filter_by(id=video_id).first()
        Tmp_result = Video.result
        Tmp_path = os.path.join('./static/Videos/',Video.location)
        return render_template('show_result.html',Tmp_result = Tmp_result, predicted = Video.predicted, unpredicted = Video.unpredicted)

#     return render_template('show_result.html',Tmp_result = Tmp_result)
    return render_template('show_result.html',Tmp_result = Tmp_result, predicted = predicted, unpredicted = unpredicted )

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/about')
def about():
   return render_template('about.html')

@app.route('/stream')
def stream():
    global Tmp_path
    return Response(stream_video(Tmp_path),mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/upload', methods=['GET','POST'])
def upload():
    global Tmp_result
    global Tmp_path
    if session.get('user_id') == None:
        flash('Vui lòng đăng nhập tài khoản', 'error')
        return redirect(url_for('login'))
    if Tmp_result != '':
#         session.pop("result", None)
#         session.pop("video_path", None)
        Tmp_result = ''
        Tmp_path = ''

    if request.method == 'POST':


        if not request.files['file']:
            flash('Vui lòng chọn file', 'error')
            return redirect(url_for('upload'))
        else :
            file = request.files['file']
            file_name = file.filename
            f_name, _ =  file_name.split(".")

            global predicted
            global unpredicted

            name = str(time.time())+f_name+".mp4"
            path = os.path.join('./static/Videos/',name)
            file.save(path)
            result = gen_video(path)
            Video = Videos(name , result ,predicted,unpredicted,session['user_id'])
#             predicted = ''
#             unpredicted = ''
            Tmp_result = result
#             session["video_path"] = path
            Tmp_path = path
            db.session.add(Video)
            db.session.commit()
#             flash('upload was successfully added', 'error')

            return redirect(url_for('play'))
    else :
        return render_template('upload.html')

from sqlalchemy import desc
@app.route('/video')
def video():

    global Tmp_result
    global Tmp_path
    if session.get('user_id') == None:
        flash('Vui lòng đăng nhập tài khoản', 'error')
        return redirect(url_for('login'))
#     if session["result"] != None :
    if Tmp_result != '':
#         session.pop("result", None)
#         session.pop("video_path", None)
        Tmp_result = ''
        Tmp_path = ''
    Video = Videos.query.filter_by(user_id=session['user_id']).all()
    user = Users.query.filter_by(id = session['user_id']).first()
    return render_template('video.html', List_Video = Video,user = user)


@app.route('/login', methods = ['GET', 'POST'])
def login():
    if request.method == 'POST':
        if not request.form['email'] or not request.form['password']:
            flash('Vui lòng điền đầy đủ thông tin', 'error')
        else :
            email = request.form.get('email')
            password = request.form.get('password')
            remember = True if request.form.get('remember') else False

            user = Users.query.filter_by(email = email).first()
#             return user.password
            if user:
                if check_password_hash(user.password, password):
#                     login_user( user, remember = request.form.get('remember'))
#                     flash('login successfully!')
                    session["user_id"] = user.id
                    session["user_name"] = user.username
                    return redirect(url_for('index'))
                else :
                    flash('Nhập sai tài khoản hoặc mật khẩu', 'error')
                    return render_template('login.html')

    return render_template('login.html' )


@app.route('/register', methods = ['GET', 'POST'])
def register():
    if request.method == 'POST':
        if not request.form['username'] or not request.form['email'] or not request.form['password']:
            flash('Vui lòng điền đầy đủ thông tin', 'error')
        else :
            email = request.form.get('email')
            exists = Users.query.filter_by(email = email).first()
            if exists:
                flash('your email is available', 'error')
                return redirect(url_for('login'))
            else :
#                 file = request.files['avatar']
#                 data = file.read()
#                 render_file = render_picture(data)
#                 return render_file

                hashed_password = generate_password_hash(request.form['password'], method='sha256')
                User = Users(request.form['username'], request.form['email'],hashed_password)

                db.session.add(User)
                db.session.commit()
                return redirect(url_for('index'))

    return render_template('register.html')


@app.route('/logout', methods=['GET'])
def logout():
    session.pop("result", None)
    session.pop("video_path", None)
    session.pop("user_id", None)
    session.pop("user_name", None)
    return redirect(url_for('index'))


if __name__ == '__main__':
   db.create_all()
   app.run(debug = True)
