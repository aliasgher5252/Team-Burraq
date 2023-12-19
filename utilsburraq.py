from pymavlink import mavutil
import numpy as np
import cv2
import time
import math
import Jetson.GPIO as GPIO

def arm(drone):
  drone.mav.command_long_send(drone.target_system,drone.target_component,mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,0,1,0,0,0,0,0,0) 
  msg = request_message(drone,type='COMMAND_ACK')
  print(msg)

def takeoff(drone,altitude):
  drone.mav.command_long_send(drone.target_system,drone.target_component,mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,0,0,0,0,0,0,0,altitude)
  msg = request_message(drone,type="GLOBAL_POSITION_INT")
  print(msg.relative_alt)
  while(msg.relative_alt<=(altitude*1000-100)):
    msg = request_message(drone,type="GLOBAL_POSITION_INT")
    print(msg.relative_alt)
  print(f"Reached Height of {altitude} m")

def initialize_drone(connection_string="/dev/ttyACM0"):
  the_connection=mavutil.mavlink_connection('/dev/ttyACM0')
  the_connection.wait_heartbeat()
  print("Heartbeat from system (system %u component %u)" %
    (the_connection.target_system, the_connection.target_component))
  return the_connection

def JetsonGPIO():
  output_pin = 13  # BOARD pin 13
  GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi
    # set pin as an output pin with optional initial state of HIGH
  GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)

def trackPerson(drone, w, pError, pError_y, pError_z, y, boxes, out, cap):
  pid=[0.003,0.0005]
  xmin=boxes[0][0]
  ymin=boxes[0][1]
  xmax=boxes[0][2]
  ymax=boxes[0][3]

  cx=xmin+(xmax-xmin)//2
  cy=ymin+(ymax-ymin)//2
  area=(xmax-xmin)*(ymax-ymin)
  print("The center in x is {}".format(cx))
  print("The center in y is {}".format(cy))
  error_z=area-10000
  error_y=cy-y
  error = cx-w
  print(f"Error in X: {error} | Error in Y: {error_y} | Error in Z: {error_z}")
  if ((cx!=0) and (abs(error)>80 or abs(error_y)>80)):

    speed_y=pid[0]*error_y + pid[1]*(error_y-pError_y)
    speed_y=np.clip(speed_y,-5,5)

    speed=0.0008*error + 0.00025*(error-pError)
    speed=np.clip(speed,-5,5)

    speed_z=0.00001*error_z+0.00005*(error_z-pError_z)
    speed_z=np.clip(speed_z, -0.5, 0.5)

    print(f"Speed in x: {speed_y} | Speed in Z: {speed_z} | Yaw Rate: {speed}")

    drone.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(10,drone.target_system,drone.target_component,
                  mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,int(0b010111000111),0,0,0,-speed_y,0,0,0,0,0,0,speed))
  else:
    print("Navigation Completed")
    speed=0
    speed_y=0
    drone.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(10,drone.target_system,drone.target_component,
                  mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,int(0b110111000111),0,0,0,0,0,0,0,0,0,0,0))
    while True:
      frame=preprocessor(cap)
      out.write(frame)
      drone.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(10,drone.target_system,drone.target_component,
                  mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,int(0b110111111000),0,0,0,0,0,0,0,0,0,0,0))
      msg = request_message(drone,type="GLOBAL_POSITION_INT")
      print(msg.relative_alt)
      #while(msg.relative_alt>=2300):
      #  frame=preprocessor(cap)
      #  out.write(frame)
      #  msg = request_message(drone,type="GLOBAL_POSITION_INT")
      #  print(msg.relative_alt)
      #print("Reached height of 2m")
      GPIO.output(13, GPIO.LOW)
      drone.mav.param_set_send(drone.target_system, drone.target_component,
                       b'WPNAV_SPEED', 500, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
      mode_id = drone.mode_mapping()['RTL']
      drone.mav.set_mode_send(
      drone.target_system,
      mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
      mode_id)
      while True:
        frame=preprocessor(cap)
        out.write(frame)
        print("RTL Mode Set. Returning to Home.")

  return pError, pError_y, pError_z, speed_y, speed

def InferenceTensorFlow(interpreter,image):

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  floating_model = False
  if input_details[0]['dtype'] == np.float32:
    floating_model = True

  initial_h, initial_w, channels = image.shape

  picture = cv2.resize(image, (width, height))

  input_data = np.expand_dims(picture, axis=0)
  if floating_model:
    input_data = (np.float32(input_data) - 127.5) / 127.5

  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  detected_boxes = interpreter.get_tensor(output_details[0]['index'])
  detected_classes = interpreter.get_tensor(output_details[1]['index'])
  detected_scores = interpreter.get_tensor(output_details[2]['index'])
  num_boxes = interpreter.get_tensor(output_details[3]['index'])

  scores=[]
  rectangles = []
  for i in range(int(num_boxes)):
    top, left, bottom, right = detected_boxes[0][i]
    classId = int(detected_classes[0][i])
    score = detected_scores[0][i]
    if score > 0.5 and classId==27:
      xmin = left * initial_w
      ymin = bottom * initial_h
      xmax = right * initial_w
      ymax = top * initial_h
      box = [int(xmin), int(ymin), int(xmax), int(ymax)]
      rectangles.append(box)
      scores.append(score)
  return rectangles,scores

def request_message(vehicle,type='NAV_CONTROLLER_OUTPUT'):
  if type=='NAV_CONTROLLER_OUTPUT':
    vehicle.mav.command_long_send(vehicle.target_system,vehicle.target_component,mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,0,62,0,0,0,0,0,0)
    msg=vehicle.recv_match(type=type, blocking=True)
  elif type=='LOCAL_POSITION_NED':
    vehicle.mav.command_long_send(vehicle.target_system,vehicle.target_component,mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,0,32,0,0,0,0,0,0)
    msg=vehicle.recv_match(type=type, blocking=True)
  elif type=='COMMAND_ACK':
    vehicle.mav.command_long_send(vehicle.target_system,vehicle.target_component,mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,0,77,0,0,0,0,0,0)
    msg=vehicle.recv_match(type=type, blocking=True)
  elif type=='HEARTBEAT':
    vehicle.mav.command_long_send(vehicle.target_system,vehicle.target_component,mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,0,0,0,0,0,0,0,0)
    msg=vehicle.recv_match(type=type, blocking=True)
  elif type=='ALTITUDE':
    vehicle.mav.command_long_send(vehicle.target_system,vehicle.target_component,mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,0,141,0,0,0,0,0,0)
    msg=vehicle.recv_match(type=type,blocking=True)
  elif type=='GLOBAL_POSITION_INT':
    vehicle.mav.command_long_send(vehicle.target_system, vehicle.target_component, mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,0,33,0,0,0,0,0,0)
    msg=vehicle.recv_match(type=type, blocking=True)
  else:
    print("Invalid Type")
    msg=None
  return msg

def haversine_distance(coord1, coord2):
  R = 6371.0  # Radius of the Earth in kilometers

  lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
  lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

  dlat = lat2 - lat1
  dlon = lon2 - lon1

  a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

  distance = R * c
  return distance*1000

def gps_midpoint(coord_list):
  latitudes = [coord[0] for coord in coord_list]
  longitudes = [coord[1] for coord in coord_list]

  avg_latitude = sum(latitudes) / len(coord_list)
  avg_longitude = sum(longitudes) / len(coord_list)

  return [avg_latitude, avg_longitude]

def preprocessor(cap):
  _,frame=cap.read()
  resized_frame = cv2.resize(frame,(640,640))
  return resized_frame
  

#----------Function to change----------- (Done Testing Required)
def wait_wp(drone,yolo,cap,out,detection=True):
  message = request_message(drone)
  print(message)
  print("Waiting to reach Waypoint!!")
  if detection:
    while(message.wp_dist>=0.5):
      message = request_message(drone)
      start_time=time.time()
      frame=preprocessor(cap)
      _,boxes=yolo.get_bbox(frame,0.4,["X"])
      print(boxes)
      end_time=time.time()
      print(f"FPS:{1/(end_time-start_time)}")
      
      if(len(boxes)!=0):
        print("X Detected. Converging......")
        pError=0
        pError_y=0
        pError_z=0
        while True:
          if (len(boxes)!=0):
            cv2.rectangle(frame, (boxes[0][0],boxes[0][1]), (boxes[0][2],boxes[0][3]), (0,255,0),2)
            out.write(frame)
            pError, pError_y, pError_z, speed_y, speed=trackPerson(drone, 320, pError, pError_y, pError_z, 320,boxes,out,cap)
          else:
            out.write(frame)
            print("X Out of Frame")
            drone.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(10,drone.target_system,drone.target_component,
              mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,int(0b010111000111),0,0,0,-speed_y,0,0,0,0,0,0,speed))
          frame=preprocessor(cap)
          _,boxes=yolo.get_bbox(frame,0.4,["X"])
      out.write(frame)
  else:
    while(message.wp_dist>=0.5):
      message = request_message(drone)
      frame=preprocessor(cap)
      out.write(frame)
  print("Waypoint reached!!")   

def wait4start(vehicle):
  # Wait a heartbeat before sending commands
  print("Waiting for heartbeat")
  vehicle.wait_heartbeat()
  print("Heartbeat Checked")
  while True:
    msg =request_message(vehicle,type='HEARTBEAT')
    print("The message is {}".format(msg))
    if msg:
      mode = mavutil.mode_string_v10(msg)
      print("Waiting for Guided")
      time.sleep(1)
      if mode=='GUIDED':
        print("Guided Mode")
        break
  return


def movement(drone,yolo,cap,out,x=0,y=0,gps=[0,0],speed_x=0,speed_y=0,angle=0,angle_rate=0,type="gps",alt=None):

  lat=gps[0]
  lon=gps[1]
  if type=='gps':
    drone.mav.send(mavutil.mavlink.MAVLink_set_position_target_global_int_message(10, drone.target_system,
                    drone.target_component, 6, int(0b010111111000), int(lat* 10 ** 7), int(lon * 10 ** 7), alt, 0, 0, 0, 0, 0, 0, 0, 0))
    wait_wp(drone,yolo,cap,out,detection=False)

  else:
    drone.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(10,drone.target_system,drone.target_component,
                            9,int(0b010111111000),x,y,0,0,0,0,0,0,0,0,0))

    wait_wp(drone,yolo,cap,out)
  drone.mav.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(10,drone.target_system,drone.target_component,
                              9,int(0b010111111000),0,0,0,0,0,0,0,0,0,0,0))

def start_mission(drone,coord_list,sa_list,yolo,cap,out,altitude,alpha,beta,land=0):
  for i in range(len(coord_list)):
    movement(drone,yolo,cap,out,gps=coord_list[i],alt=altitude)
  center_gps=gps_midpoint(sa_list)
  print(f"The center GPS is: {center_gps}")
  radius=haversine_distance(center_gps, sa_list[0])
  print(f"The radius is: {radius}")
  print("Descending to 4 metres") 
  spiral_algo(drone,yolo,cap,out,4,alpha,beta,center_gps,radius,land=0)

def spiral_algo(drone,yolo,cap,out,altitude,alpha,beta,center_gps,radius,land=0):

  diameter = 2*radius

  cell_w=2*altitude*math.tan(alpha/2.0)
  cell_l=2*altitude*math.tan(beta/2.0)
  print(f'Width: {cell_w}, Length = {cell_l}')

  no_of_rows = math.ceil(diameter / cell_w)
  no_of_columns = math.ceil(diameter / cell_l)

  # making grid rows and columns equal
  if (no_of_rows > no_of_columns):
    no_of_columns = no_of_rows
  else:
    no_of_rows = no_of_columns

  # making rows and columns odd
  if (no_of_columns % 2 == 0):
    no_of_rows += 1
    no_of_columns += 1

  print(f"Columns = {no_of_columns}")
  print("Changing speed to 100 cm/s")
  drone.mav.param_set_send(drone.target_system, drone.target_component,
                       b'WPNAV_SPEED', 50,mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
  movement(drone,yolo,cap,out,gps=center_gps,alt=altitude)

  for i in range(no_of_columns):
    if not land:
      if i%2==0:
        movement(drone,yolo,cap,out,(i+1)*cell_w,0,type='movement')
        movement(drone,yolo,cap,out,0,(i+1)*cell_l,type='movement')
      else:
        movement(drone,yolo,cap,out,-(i+1)*cell_w,0,type='movement')
        movement(drone,yolo,cap,out,0,-(i+1)*cell_l,type='movement')
    else:
        break

