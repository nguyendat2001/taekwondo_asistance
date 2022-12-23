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

import base64

#   function to process program

movenet = Movenet('movenet_thunder')

label = "waiting"
class_names = ['pose_1', 'pose_10', 'pose_11', 'pose_12', 'pose_13', 'pose_14',
           'pose_15', 'pose_16', 'pose_2', 'pose_3', 'pose_4', 'pose_5',
           'pose_6', 'pose_7', 'pose_8', 'pose_9']

def write_video(file_path, frames, fps):
    height, width, channels = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

def get_center_point(landmarks, left_bodypart, right_bodypart):
  """Calculates the center point of the two given landmarks."""

  left = tf.gather(landmarks, left_bodypart.value, axis=1)
  right = tf.gather(landmarks, right_bodypart.value, axis=1)
  center = left * 0.5 + right * 0.5
  return center


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
    fontColor = (0, 0, 0)
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

seri_label = ['pose_1','pose_2','pose_3','pose_4','pose_5','pose_6','pose_7','pose_8',
              'pose_9','pose_10','pose_11','pose_12','pose_13','pose_14','pose_15','pose_16','pose_16']


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
        if list[i] == 'pose_16':
            list.pop(i)
    return list

def standardize(list):
    xyz = list
    j = 0
    for i in range(0,len(xyz)):
        if xyz[i] == 'pose_16':
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
    seri_label = ['pose_1','pose_2','pose_3','pose_4','pose_5','pose_6','pose_7','pose_8',
              'pose_9','pose_10','pose_11','pose_12','pose_13','pose_14','pose_15','pose_16']
    list = []
    for i in seri_label:
        for l1 in list1:
            if i == l1:
                list.append(i)
        for l2 in list2:
            if i == l2:
                list.append(i)
    return list

#  end function

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database//database.sqlite3'
app.config['SECRET_KEY'] = "random string"

db = SQLAlchemy(app)

class Users(db.Model):
   id = db.Column(db.Integer, primary_key=True)
   username = db.Column(db.String(100))
   email = db.Column(db.String(50),  unique=True,)
   avatar = db.Column(db.String(100000), nullable=False)
   password = db.Column(db.String(200))

   def __init__(self, username, email, avatar, password):
      self.username = username
      self.email = email
      self.avatar = avatar
      self.password = password

class Videos(db.Model):
   id = db.Column(db.Integer, primary_key=True)
#    name = db.Column(db.String(100))
#    data = db.Column()
   location = db.Column(db.String(1000))
   result = db.Column(db.Float, nullable=False)
   user_id = db.Column(db.Integer)

   def __init__(self, location,result , user_id):
#       self.name = name
      self.location = location
      self.result = result
      self.user_id = user_id




def render_picture(data):
    render_pic = base64.b64encode(data).decode('ascii')
    return render_pic

def decode_base64(data):
#     <img src="data:;base64,{{ image }}"/>
    return b64encode(data).decode("utf-8")

# detect pose on video

def caculator_acc(list_label):

    list_label.remove('waiting')

    seri_label = ['pose_1','pose_2','pose_3','pose_4','pose_5','pose_6','pose_7','pose_8',
                  'pose_9','pose_10','pose_11','pose_12','pose_13','pose_14','pose_15','pose_16','pose_16']

    list_after_standardized = []
    tmp = 0
    list_label = standardize(list_label)
    # print(list_label)
    list_label = standard_pose16(list_label)
    list_label.append('pose_16')
    while not is_available(list_after_standardized,'pose_16'):
        for i in range(0,len(list_label)-1):
            if is_available_dou(seri_label,list_label[i],list_label[i+1]):
                list_after_standardized.append(list_label[i])
                list_after_standardized.append(list_label[i+1])
                i = i + 1

    list_after_standardized = normalsize(list_after_standardized)
    list_exc = list_except(list_after_standardized,list_label)
    final_list = colab_list(list_exc,list_after_standardized)

    print(final_list)
    result = len(final_list)*100/13

    if result >= 100:
        result = 100
#     print(str(result)+"%")
    return result

def gen_video(path):
    model = tf.keras.models.load_model("./models/model_pose_taekwondo.h5")
    movenet = Movenet('movenet_thunder')

    cap = cv2.VideoCapture(path)
    lm = []
    time_step = 5
    label = "waiting"
    warmup_frames = 60
    i = 0
    list = []
    list_label = []
    list_label.append('waiting')

#     while False:
    while cap.isOpened():
        ret ,frame=cap.read()

        #Reshape Image
        if ret == True:

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
        #         _ = draw_prediction_on_image(img, person, crop_region=None,
        #                                    close_figure=False, keep_input_size=True)

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
        else :
            break

    write_video(path, np.array(list), 40)
    list_label.remove('waiting')
    print(list_label)
    return caculator_acc(list_label)


@app.route('/')
def index():
   return render_template('index.html')

@app.route('/stream/<upload_id>')
def download(upload_id):
    video = Videos.query.filter_by(id=upload_id).first()
    return Response(gen(video.location),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video1')
def video1():
    return Response(gen(video),mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        if not request.files['file'] :
                flash('Please upload your file', 'error')
        else :
            file = request.files['file']
            path = os.path.join('./static/Videos',file.filename)
            file.save(path)

            Video = Videos(path ,gen_video(path) ,session['user_id'])

            db.session.add(Video)
            db.session.commit()
            flash('upload was successfully added', 'error')
            return redirect(url_for('upload'))

    return render_template('upload.html')

@app.route('/video')
def video():
    Video = Videos.query.filter_by(user_id=session['user_id']).all()
    return render_template('video.html', List_Video = Video)


@app.route('/login', methods = ['GET', 'POST'])
def login():
    if request.method == 'POST':
        if not request.form['email'] or not request.form['password']:
            flash('Please enter all the fields', 'error')
        else :
            email = request.form.get('email')
            password = request.form.get('password')
            remember = True if request.form.get('remember') else False

            user = Users.query.filter_by(email = email).first()
#             return user.password
            if user:
                if check_password_hash(user.password, password):
#                     login_user( user, remember = request.form.get('remember'))
                    flash('login successfully!')
                    session["user_id"] = user.id
                    session["user_name"] = user.username
                    return redirect(url_for('index'))
                else :
                    flash('login unsucessfully!')
                    return render_template('login.html' )

    return render_template('login.html' )


@app.route('/register', methods = ['GET', 'POST'])
def register():
    if request.method == 'POST':
        if not request.form['username'] or not request.form['email'] or not request.form['password'] or not request.files['avatar']:
            flash('Please enter all the fields', 'error')
        else:
            exists = db.session.query(db.exists().where(Users.email == request.form['email'])).scalar()
            if exists:
                flash('your email is available', 'error')
                return redirect(url_for('signup'))
            else :
                file = request.files['avatar']
                data = file.read()
                render_file = render_picture(data)
#                 return render_file

                hashed_password = generate_password_hash(request.form['password'], method='sha256')
                User = Users(request.form['username'], request.form['email'], render_file ,hashed_password)

                db.session.add(User)
                db.session.commit()
                flash('Record was successfully added')
                return redirect(url_for('index'))

    return render_template('register.html')


@app.route('/logout', methods=['GET'])
def logout():
    session.pop("user_id")
    session.pop("user_name")
    return redirect(url_for('index'))


if __name__ == '__main__':
   db.create_all()
   app.run(debug = True)
