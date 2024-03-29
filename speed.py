import cv2
import cx_Oracle
import pandas as pd
import numpy as np
import sys
import time
from math import dist
from pyzbar.pyzbar import decode
from playsound import playsound
import tkinter as tk
from tkinter import messagebox
from ultralytics import YOLO
from tracker import* 
from PIL import Image, ImageTk , ImageFont, ImageDraw # 한글처리
import configparser
from flask import Flask, Response, render_template , jsonify
import threading

#####################################################
# Flask 애플리케이션 설정
# http://localhost:8000/video_feed  브라우져 주소
#####################################################
app = Flask(__name__)
# 전역 변수로 frame 저장
outputFrame = None
lock = threading.Lock()
#####################################################
# Flask 라우트 설정
#####################################################
@app.route("/")
def index():
    # 홈페이지 반환
    return render_template("index.html")

@app.route("/startService", methods=['GET'])
def video_start():
    print("서비스 시작")
    return jsonify({"message": "서비스가 시작되었습니다."}), 200

@app.route("/video_feed")
def video_feed():
    # 비디오 스트림 라우트
    return Response(generate(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")

def generate():
    # 비디오 스트림을 loop하는 함수
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')


############################################################
#
############################################################
# 설정 파일 읽기
config = configparser.ConfigParser()
config.read('config.ini' , encoding="utf-8")

# 설정 값 할당
video_source = config.get('DEFAULT', 'video_source', fallback='0')  # 기본값은 '0' (웹캠)
model_path = config.get('DEFAULT', 'model_path', fallback='last.pt')
machine_name= config.get('DEFAULT', 'machine_name', fallback='None')
image_size = config.get('DEFAULT', 'image_size', fallback='256')

max_retries = config.get('DEFAULT', 'max_retries', fallback=10) # 10 최대 재시도 횟수
retry_delay = config.get('DEFAULT', 'retry_delay', fallback=5) # 5  재시도 간 지연 시간 (초)
noimage = 0 # 영상취득 실패 횟수 카운트
############################################################
# 마우스 이벤트 처리를 위한 변수들
############################################################
dragging = False  # 드래그 상태 여부
line_selected = None  # 선택된 라인 (None, 'cy1', 'cy2')
line_hover = None  # 마우스가 가까이 있는 라인 ('cy1', 'cy2')
line_threshold = 10  # 라인 선택 가능한 최대 거리

last_down_time = None
elapsed_time = .000
tracbar_visible = False

scale_factor = 1  # 조정할 스케일 팩터

# 트랙바 위치 업데이트 함수에도 스케일 팩터 적용
def update_cy1(val):
    global cy1
    cy1 = int(val * scale_factor)

def update_cy2(val):
    global cy2
    cy2 = int(val * scale_factor)
    
############################################################
# 마우스 이벤트 콜백 함수
############################################################
def mouse_event(event, x, y, flags, param):
    global cy1, cy2, dragging, line_selected, line_hover

    # 마우스가 라인 근처에 있는지 확인
    if event == cv2.EVENT_MOUSEMOVE:
        if abs(y - cy1) < line_threshold:
            line_hover = 'cy1'
        elif abs(y - cy2) < line_threshold:
            line_hover = 'cy2'
        else:
            line_hover = None

    # 마우스 클릭으로 라인 선택
    if event == cv2.EVENT_LBUTTONDOWN:
        if line_hover:
            dragging = True
            line_selected = line_hover

    # 마우스 드래그로 라인 이동
    if event == cv2.EVENT_MOUSEMOVE:
        if dragging and line_selected:
            if line_selected == 'cy1':
                cy1 = y
            elif line_selected == 'cy2':
                cy2 = y

    # 마우스 버튼을 놓으면 드래그 종료
    if event == cv2.EVENT_LBUTTONUP:
        dragging = False
        line_selected = None
##################################################
# Oracle Database 연결 설정
##################################################
dsn = cx_Oracle.makedsn('107.105.20.170', 1521, service_name='SMVNPDB')
# 데이터베이스에 데이터 삽입하는 함수 정의
def insert_data(barcode_data , object_name, object_id, current_time):
    sql = "INSERT INTO ICOM_OBJECT_DETECT (barcode , object_name, object_id, detect_datetime) VALUES (:barcode_data, :object_name, :object_id, :detection_time)"
    cursor.execute(sql, [barcode_data, object_name, object_id, current_time])
    connection.commit() 
    
def puttext_kr(frame, text, position, font_size, font_color):
    """
    OpenCV 이미지에 한글 텍스트를 추가합니다.
    Parameters:
    - frame: OpenCV 이미지 (numpy 배열)
    - text: 이미지에 추가할 한글 텍스트 (str)
    - position: 텍스트를 추가할 위치 (tuple, 예: (x, y))
    - font_path: 사용할 폰트의 경로 (str)
    - font_size: 폰트 크기 (int)
    - font_color: 폰트 색상 (tuple, 예: (B, G, R))

    Returns:
    - 한글 텍스트가 추가된 OpenCV 이미지 (numpy 배열)
    """
    
    # OpenCV 이미지를 PIL 이미지로 변환
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    # 폰트 설정
    font = ImageFont.truetype("./TTF/MaruBuri-Bold.ttf", font_size)
    # PIL을 사용하여 이미지에 한글 텍스트 추가
    draw.text(position, text, font=font, fill=font_color)
    # PIL 이미지를 다시 OpenCV 이미지로 변환
    frame_with_text = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    return frame_with_text    



# 비디오 소스를 열기 위한 함수
def open_video_source(source, max_retries, retry_delay):
    for attempt in range(int(max_retries)):
        if source.isdigit():
            cap = cv2.VideoCapture(int(0))  # 숫자면 웹캠 ID로 간주
        else:
            cap = cv2.VideoCapture(0)  # 그렇지 않으면 파일 경로로 간주
        
        if cap.isOpened():
            print("비디오 소스 연결 성공")
            return cap
        else:
            print(f"비디오 소스 연결 실패, {attempt + 1}/{max_retries} 시도...")
            time.sleep(int(retry_delay))
    
    print("비디오 소스 연결을 실패했습니다. 프로그램을 종료합니다.")
    return None

#################################################
# 학습된 모델 명 설정 파일 위치 정학하게 세팅해줄것
# cap=cv2.imread('qrbarcode.jpg')
#################################################
model=YOLO(model_path)
names = model.names
#################################################
try:
    with open("pcb.txt", "r", encoding="utf-8") as f:
        class_list = f.read().split("\n")
except FileNotFoundError:
    print("pcb.txt 파일을 찾을 수 없습니다.")
    
#################################################
# 영상윈도명 / 마우스이벤트 설정 / 영상 입력 싸이즈 추출 
#################################################
cv2.namedWindow(machine_name) 
cv2.setMouseCallback(machine_name, mouse_event)

# cap=cv2.VideoCapture(0) 
if video_source.isdigit():
    # cap = cv2.VideoCapture(int(video_source))  # 숫자면 웹캠 ID로 간주   
    # 비디오 소스 열기 시도
    cap = open_video_source(int(video_source), max_retries, retry_delay)
    # cap이 None이면 비디오 소스 연결에 실패한 것이므로 프로그램 종료
    if cap is None:
        sys.exit(1)
else:
    cap = open_video_source(video_source, max_retries, retry_delay)
    # cap이 None이면 비디오 소스 연결에 실패한 것이므로 프로그램 종료
    if cap is None:
        sys.exit(1)  
    
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

#################################################
# Video writer
# video_writer = cv2.VideoWriter("object_counting_output.avi",
#                        cv2.VideoWriter_fourcc(*'mp4v'),
#                        fps,
#                        (w, h))
#################################################
# 추적 클레스 생성
#################################################
tracker=Tracker()

#################################################
# 설정값 정의
#################################################
blur_ratio = 50 # Blur ratio
barcode_data = "None"
cy1=  int( h - (h * 0.3 ) )  # 1 차라인의 y 값
cy2=  int( h - (h * 0.2 ) ) # 2 차라인의 y 값
offset= 20 # 기준라인과 객체의 거리차이 오프셑값 20 픽셀 좌우도 포함
count=0
use_db_yn = 'N'

#################################################
# 
#################################################
vh_down={} # 딕셔너리 
counter=[] # 어레이

vh_up={}
counter1=[]
#################################################
#
#################################################
if use_db_yn =='Y':
    try:
        # 데이터베이스에 연결 시도
        connection = cx_Oracle.connect(user='INFINITY21_JSMES', password='INFINITY21_JSMES', dsn=dsn)
        cursor = connection.cursor()
        # 여기에 데이터베이스 작업을 수행하는 코드를 추가할 수 있습니다.
        print("데이터베이스에 성공적으로 연결되었습니다.")
        
    except cx_Oracle.DatabaseError as e:
        error, = e.args
        print(f"데이터베이스 연결 실패: {error.code} - {error.message}")
        
        response = messagebox.askyesno("데이터베이스 연결 실패", "연결에 실패했습니다. 계속 진행하시겠습니까?")
        if response:  # Yes를 클릭한 경우
            print("사용자가 계속 진행하기로 선택했습니다.")
            use_db_yn = 'N'
        else:  # No를 클릭한 경우
            print("사용자가 작업을 중단하기로 선택했습니다.")    
            sys.exit()  # 프로그램 종료
            
##################################################
# 기존 코드에서 Flask 스트리밍을 위한 코드 추가
##################################################
def main():
    
    global cap, outputFrame, lock , tracbar_visible , count , cy1 , cy2 , elapsed_time , barcode_data , last_down_time , noimage

    # Flask 애플리케이션을 별도의 스레드에서 실행
    t = threading.Thread(target=lambda: app.run(host="0.0.0.0", port="8000", debug=True, use_reloader=False))
    t.daemon = True
    t.start()
    
    #################################################
    # 처리 시작 루핑...
    #################################################
    while True:   
        ret,frame = cap.read()
        
        if not ret:
            print("No Signal (최대 10회 시도후 종료)."+str(noimage)+" 회 시도중...")
            noimage += 1
            
            if noimage == 10:
                break
            continue
            
        count += 1
        ########################################
        # 3 프레임 마다 한번씩만 실행
        ########################################
        if count % 3 != 0:
            continue
        ########################################
        # 영상 싸이즈 변경
        ########################################  
        frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    #   frame=cv2.resize(frame,(1024,600))
    
        
        if tracbar_visible == False :
            # 조정된 영상의 높이
            resized_height = int(h * scale_factor)
            cv2.createTrackbar('In Pos ', machine_name, cy1, resized_height, update_cy1)
            cv2.createTrackbar('Out Pos', machine_name, cy2, resized_height, update_cy2)  
            tracbar_visible = True
            # cy1, cy2 위치 조정
            cy1 = int(cy1 * scale_factor)
            cy2 = int(cy2 * scale_factor)
        
        ###############################################
        # QR 바코드 인식 
        ###############################################   
        try:
            decoded = decode(frame)
            if not decoded:
                pass
                # print("No Detect Barcode")
            else:
                pass
                # print("detected: {len(decoded)}")
                
        except Exception as e:
            pass
            # print(f"{e}")    
        else:    
            for bd in decoded:
                bx , by , bw , bh  = bd.rect
                barcode_data = bd.data.decode("utf-8")
                barcode_type = bd.type
                cv2.rectangle(frame ,( bx , by ),(bx + bw , by+bh),(255,0,0) , 2   )
                # print ( "Barcde="+barcode_data , "Type="+barcode_type)   
        
        ###############################################
        # predict 객체인식 실행 추론실행 파라메터
        ###############################################
        # source	str	'ultralytics/assets'	추론할 데이터 소스를 지정합니다. 이미지 경로, 동영상 파일, 디렉토리, URL 또는 실시간 피드용 디바이스 ID가 될 수 있습니다. 다양한 형식과 소스를 지원하므로 다양한 유형의 입력에 유연하게 적용할 수 있습니다.
        # conf	float	0.25	탐지에 대한 최소 신뢰도 임계값을 설정합니다. 이 임계값보다 낮은 신뢰도로 탐지된 개체는 무시됩니다. 이 값을 조정하면 오탐지를 줄이는 데 도움이 될 수 있습니다.
        # iou	float	0.7	비최대 억제(NMS)에 대한 교차점 초과(IoU) 임계값입니다. 값이 높을수록 중복되는 상자를 제거하여 탐지 횟수가 줄어들어 중복을 줄이는 데 유용합니다.
        # imgsz	int or tuple	640	추론할 이미지 크기를 정의합니다. 단일 정수일 수 있습니다. 640 를 사용하여 정사각형 크기 조정 또는 (높이, 너비) 튜플을 사용합니다. 적절한 크기 조정은 감지 정확도와 처리 속도를 향상시킬 수 있습니다.
        # half	bool	False	정확도에 미치는 영향을 최소화하면서 지원되는 GPU에서 모델 추론 속도를 높일 수 있는 반정확도(FP16) 추론을 활성화합니다.
        # device	str	None	추론할 장치를 지정합니다(예, cpu, cuda:0 또는 0). 사용자가 모델 실행을 위해 CPU, 특정 GPU 또는 기타 컴퓨팅 장치 중에서 선택할 수 있습니다.
        # max_det	int	300	이미지당 허용되는 최대 감지 횟수. 모델이 한 번의 추론에서 감지할 수 있는 총 오브젝트 수를 제한하여 밀집된 장면에서 과도한 출력을 방지합니다.
        # vid_stride	int	1	비디오 입력의 프레임 보폭. 동영상에서 프레임을 건너뛰어 시간 해상도를 희생하면서 처리 속도를 높일 수 있습니다. 값이 1이면 모든 프레임을 처리하고, 값이 클수록 프레임을 건너뜁니다.
        # stream_buffer	bool	False	비디오 스트림을 처리할 때 모든 프레임을 버퍼링할지 여부를 결정합니다(True), 또는 모델이 가장 최근 프레임을 반환해야 하는지(False). 실시간 애플리케이션에 유용합니다.
        # visualize	bool	False	추론 중에 모델 기능의 시각화를 활성화하여 모델이 '보고 있는 것'에 대한 인사이트를 제공합니다. 디버깅 및 모델 해석에 유용합니다.
        # augment	bool	False	예측을 위한 테스트 시간 증강(TTA)을 지원하여 추론 속도를 희생하더라도 탐지 견고성을 향상시킬 수 있습니다.
        # agnostic_nms	bool	False	서로 다른 클래스의 겹치는 상자를 병합하는 클래스 무관 NMS(Non-Maximum Suppression)를 활성화합니다. 클래스 중복이 일반적인 다중 클래스 탐지 시나리오에 유용합니다.
        # classes	list[int]	None	클래스 ID 집합으로 예측을 필터링합니다. 지정된 클래스에 속하는 탐지만 반환됩니다. 다중 클래스 탐지 작업에서 관련 개체에 집중하는 데 유용합니다.
        # retina_masks	bool	False	모델에서 사용 가능한 경우 고해상도 세분화 마스크를 사용합니다. 이렇게 하면 세분화 작업의 마스크 품질을 향상시켜 더 세밀한 디테일을 제공할 수 있습니다.
        # embed	list[int]	None	특징 벡터 또는 임베딩을 추출할 레이어를 지정합니다. 클러스터링이나 유사도 검색과 같은 다운스트림 작업에 유용합니다.
        ###############################################
        #
        ###############################################
        
        results=model.predict(frame ,imgsz=int(image_size) , conf= 0.8 , max_det = 1 , classes= [0,1,2])
        # print(results)
        #     속성	유형	설명
        # orig_img	numpy.ndarray	원본 이미지가 널빤지 배열로 표시됩니다.
        # orig_shape	tuple	(높이, 너비) 형식의 원본 이미지 모양입니다.
        # boxes	Boxes, optional	감지 경계 상자가 포함된 Boxes 개체입니다.
        # masks	Masks, optional	감지 마스크가 포함된 마스크 객체입니다.
        # probs	Probs, optional	분류 작업에 대한 각 클래스의 확률을 포함하는 Probs 객체입니다.
        # keypoints	Keypoints, optional	각 개체에 대해 감지된 키포인트를 포함하는 키포인트 개체입니다.
        # obb	OBB, optional	방향이 지정된 바운딩 박스를 포함하는 OBB 객체입니다.
        # speed	dict	이미지당 밀리초 단위의 전처리, 추론, 후처리 속도 사전입니다.
        # names	dict	클래스 이름 사전입니다.
        # path	str	이미지 파일의 경로입니다.
        ###############################################
        # 블러링 전처리 로직
        ###############################################
        # boxes = results[0].boxes.xyxy.cpu().tolist()
        # clss = results[0].boxes.cls.cpu().tolist()
        # annotator = Annotator(frame, line_width=2, example=names) 
        
        ###############################################
        # 리스트 데이터 잘라내기
        ###############################################
        a=results[0].boxes.data
        # print(a)  #tensor([[  1.0071, 530.1204, 268.5498, 719.8320,   0.9678,   0.0000]])
        px=pd.DataFrame(a).astype("float")
        # print(px)
        list=[]
                
        for index,row in px.iterrows():
            # print(row)
            x1=int(row[0]) # 물체좌표
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            confd=float(row[4]) # 감지된 물제 정확도
            class_id=int(row[5])       # 물체 id 
            
            try:
                class_name=class_list[class_id] # 물제 id 를 통한 물체명 리턴
            except IndexError:
                class_name=class_list[0]
                # print (class_name)
                
            ####################################
            # PCB 만 선별해서 처리 나머지 스킵
            ####################################
            # if 'pcb1' in class_name:
            list.append([x1,y1,x2,y2])
                
        ####################################
        # 박스처리된  리스트    
        ####################################         
        bbox_id=tracker.update(list)
        # print("Boxid="+str(bbox_id))
        for bbox in bbox_id:
            
            x3,y3,x4,y4,id=bbox
            cx=int(x3+x4)//2  # 목을 구해서 2 로 나누어 x 좌표 박스의 중앙값을 구함 
            cy=int(y3+y4)//2  

            ####################################################
            # 사각형 박스 를 그려줌 빨간색 
            # 중앙에 파란점 찍기
            ####################################################
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
            cv2.circle(frame,(cx,cy),4,(255,0,0),-1) # Blue dot

            ####################################################
            # 객체이름을 중앙에 표시
            # 동일객체인데 감지된 ID 가 매번 바뀌는 현상 있음 나중에 개선해야함
            # tracker.py 에서 설정값 조정 겹치는 범위 설정
            ####################################################

            # cv2.putText(frame, str(round(confd,3)) +' '+str(class_id)+':'+class_name+' ID['+str(id)+']' , (cx-10, cy-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,255), 1)
            frame = puttext_kr(frame , str(round(confd,2))+' '+str(class_id)+' '+class_name +' '+str(id), (cx-10, cy-10) , 18 , (0,255,255))   
            cv2.imshow(machine_name, frame)
            
            #####위에서 아래도 통과##### ###########################    
            # 객체의 중앙값에 오프셋을 더한 값이  1 번라인을 통과 할경우  
            ######################################################
                    
            if cy1<(cy+offset) and cy1 > (cy-offset): 
                vh_down[id]=time.time() # 감지된 객체 아이디 인텍스에 시간 저장  
                cv2.circle(frame,(cx,cy),4,(0,255,255),-1) # 여기서 감지되면 노란색으로 중앙에 표시
            
            if id in vh_down:  
                if cy2<(cy+offset) and cy2 > (cy-offset): # 1 번을 통과한 객체가 2 번라인도 통과한 경우 
                # 경과된 시간계산 
                #  elapsed_time=time.time() - vh_down[id] 
                #  print ("경과시간="+str(elapsed_time)+'  '+str(time.localtime()))
                
                 if counter.count(id)==0:
                    counter.append(id)
                    # distance = 10 # meters
                    # a_speed_ms = distance / elapsed_time
                    # a_speed_kh = a_speed_ms * 3.6
                    
                    ######################################################
                    # 선에 닿으면 비프음 발생시킴 ( 나중에 제거)
                    ######################################################
                    # beepsound()
                    playsound("detect.mp3" , False)
                
                    if last_down_time != None:
                        elapsed_time = time.time() - last_down_time
                        
                        print("경과"+str(elapsed_time))
                        
                    last_down_time = time.time()  
                    count = 0  
                    ######################################################
                    if use_db_yn =='Y':
                        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        insert_data(barcode_data , class_name, id, current_time)
                    ######################################################
                    # 블러링 후 처리
                    ######################################################
                    # for box, cls in zip(boxes, clss):
                    #     annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])
                    #     obj = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    #     blur_obj = cv2.blur(obj, (blur_ratio, blur_ratio))
                    #     frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = blur_obj            
                    ######################################################
                    # 선을 통과 하면 붉은색으로 
                    # cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    # 박스는 로란색으로
                    # cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255),2)   
                
                    # 계산된 속도 표시 
                    # cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,255,255),1)
                    # cv2.putText(frame,str(int(a_speed_kh))+'Km/h',(x4,y4 ),cv2.FONT_HERSHEY_COMPLEX,0.4,(0,255,255),2)

            #####아래서 위도 통과############################################################
            #
            ################################################################################    
            # if cy2<(cy+offset) and cy2 > (cy-offset):
            #    vh_up[id]=time.time()
            # #    print( 'ucv2' , id , cy1 , cy2 , cy+offset , cy-offset )
            
            # if id in vh_up:

            #    if cy1<(cy+offset) and cy1 > (cy-offset):
            #      elapsed1_time=time.time() - vh_up[id]
            #     #  print( 'ucv1' , id , cy1 , cy2 , cy+offset , cy-offset )
            #     #  cv2.circle(frame,(cx,cy),4,(255,255,255),-1)
            #      if counter1.count(id)==0:
            #         counter1.append(id)      
            #         distance1 = 10 # meters
            #         a_speed_ms1 = distance1 / elapsed1_time
            #         a_speed_kh1 = a_speed_ms1 * 3.6
                    
            #         # 라인 2 에 닿으면 빨간 선을 그림 
            #         cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
            #         cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
            #         cv2.putText(frame,str(int(a_speed_kh1))+'Km/h',(x4,y4),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                    
        ########################################################
        # 화면에 통과선 그리기 2 개
        # 영상 해상도에 따라 그리도록 소스 수정해야 함.
        ########################################################

        cv2.line(frame,(10,cy1),(w - 10,cy1),(255,255,255),3)
        cv2.putText(frame,('In Y Pos-'+ str(cy1)),(10,cy1-20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
        
        cv2.line(frame,(10,cy2),(w - 10,cy2),(255,255,255),3) 
        cv2.putText(frame,('Out Y Pos-'+ str(cy2)),(10,cy2-20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,255),1)
        
        ########################################################    
        # 화면에 통과 수량 표시
        ########################################################    
        d=(len(counter))  # 아래로 내려오는 경우 수 
        u=(len(counter1)) # 위로 올라가는 경우 수 
        
        # cv2.putText(frame,('goingdown:-')+str(d),(10,20),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,255),1)
        frame = puttext_kr(frame , "생산실적수량-"+str(d) , (10,20) , 48 , (0,255,255))   
        cv2.putText(frame, "Elapsed="+str(f"{elapsed_time:.2f}")+"/Sec" , (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0, 255, 255), 2)
    
        
        
        # cv2.putText(frame,('goingup:-')+str(u),(10,40),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,255),1)
        # cv2.putText(frame,('w/h/fps:-')+str(w)+':'+str(h)+':'+str(fps),(10,60),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,255),1)
        ########################################
        # 추출된 바코드 표시 파란색
        ########################################
        cv2.putText(frame,('Barcode:-')+barcode_data,(10,180),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),1)  
        cv2.putText(frame,('Database Use:-')+use_db_yn,(10,220),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),1)  
        cv2.putText(frame,('Count :-')+str(count),(10,260),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),1)  
        cv2.putText(frame,('W+h :-')+str(w)+"*"+str(h),(10,300),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),1)  
        cv2.imshow(machine_name, frame) # 한글처리 때문에 영상출력 한번 해준다.
        ##########################
        # 영상표시
        ##########################
        # cv2.imshow(machine_name, frame)
        with lock:
            outputFrame = frame.copy()
        ###########################
        # 감지된 영상 저장 
        ###########################
        #video_writer.write(frame)
    

        if cv2.waitKey(1)&0xFF==27:
            break
########################################################
# 여기까지 while 루프        
########################################################    
    if use_db_yn == 'Y':    
        cursor.close()
        connection.close()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()