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
from flask import Flask, Response, render_template , jsonify , request , redirect , url_for
import threading
#####################################################
# Flask 애플리케이션 설정
# http://localhost:8000/showVideo  브라우져 주소
#####################################################
app = Flask(__name__)
# 전역 변수로 frame 저장
outputFrame = None
lock = threading.Lock()
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>쓰레드락설정")
service_active = False
main_thread = None
#####################################################
# Flask 라우트 설정
#####################################################
@app.route("/")
def index():
    # 홈페이지 반환
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>인텍스페이지회신")
    return render_template("index.html")

@app.route("/startService", methods=['GET'])
def start_service():
    global service_active , main_thread
    print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>서비스시작요청")
    if not service_active:
        service_active = True
        # 별도의 스레드에서 main 함수 실행
        main_thread = threading.Thread(target=main).start()
        return jsonify({"message": "서비스가 시작되었습니다."}), 200
    else:
        return jsonify({"message": "서비스가 이미 실행 중입니다."}), 200

@app.route("/stopService", methods=['GET'])
def stop_service():
    global service_active, main_thread
    print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>서비스종료요청")
    if service_active:
        service_active = False  # 스트리밍 상태를 False로 설정하여 스트리밍 종료
        if main_thread is not None:
            main_thread.join()  # main 쓰레드가 종료될 때까지 기다립니다.
            main_thread = None  # main 쓰레드 참조를 제거합니다.
        print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>서비스가 종료되었습니다")    
        return jsonify({"message": "서비스가 종료되었습니다."}), 200
    else:
        print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>실행 중인 서비스가 없습니다")
        return jsonify({"message": "실행 중인 서비스가 없습니다."}), 200

# @app.route("/checkVideoStreamStatus", methods=['GET'])
# def checkVideoStreamStatus():
#     global service_active
#     print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>스트림상태체크요청")    
#     if service_active == False:
#       return jsonify({"message": "스트림이 종료되었습니다."}), 200
#     else:
#       return jsonify({"message": "스트림이 진행중입니다."}), 200
    
@app.route("/showVideo")
def showVideo():
    print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>서비스상태정상 영상전송시작")
    global service_active
    if service_active:
        print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>서비스상태정상 영상전송시작")
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
    else:
        print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>영상전송 서비스가 실행중이 아닙니다")
        return jsonify({"message": "영상전송 서비스가 실행중이 아닙니다."}), 200

def generate():
    # global outputFrame, lock
    print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>영상디코딩시작")
    while service_active:
        # with lock:  #락이 잡혀있는 동안 실행 main 함안에서 영상처리 시 with lock 으로 잡았다 풀었다 함
        if outputFrame is None:
            # print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>영상프레임 없음.  서비스상태:"+str(service_active))
            continue
        
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        if not flag:
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        
@app.route("/editConfig", methods=["GET", "POST"])
def editConfig():
    config_file = "config.ini"
    if request.method == "POST":
        content = request.form["content"]
        with open(config_file, "w" , encoding="utf-8") as file:
            file.write(content)
        return redirect(url_for("index"))
    else:
        try:
            with open(config_file, "r" , encoding="utf-8") as file:
                content = file.read()
        except FileNotFoundError:
            content = ""
        except UnicodeDecodeError as e:
            print("유니코드문자처리오류" , e)
                
        return render_template("editConfig.html", content=content)
    
@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    global outputFrame, lock, service_active
    if service_active:
        # Convert string of image data to uint8
        nparr = np.fromstring(request.data, np.uint8)
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Optional: Any processing on the image
        # Example: Display the image
        cv2.imshow("Received Frame", img)
        cv2.waitKey(1)

        # Update the global frame to be the newly received frame
        with lock:
            outputFrame = img

        return Response(status=200)
    else:
        return jsonify({"error": "Service not active"}), 403
    
    

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
cursor = None
connection = None
############################################################
# 트랙바 위치 업데이트 함수에도 스케일 팩터 적용
############################################################
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
    global cursor , connection
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
        if source == 0:
            print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>웹캠으로연결")
            cap = cv2.VideoCapture(int(source))  # 숫자면 웹캠 ID로 간주
        else:
            print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>파일로연결")
            cap = cv2.VideoCapture(source)  # 그렇지 않으면 파일 경로로 간주
        
        if cap.isOpened():
            print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>비디오소스연결성공")
            return cap
        else:
            print(f"비디오 소스 연결 실패, {attempt + 1}/{max_retries} 시도...")
            time.sleep(int(retry_delay))
    
    print("비디오 소스 연결을 실패했습니다. 프로그램을 종료합니다.")
    return None
    
##################################################
# 기존 코드에서 Flask 스트리밍을 위한 코드 추가
##################################################
def main():
    print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>메인쓰레드시작")   
    global service_active , cap, outputFrame, lock , tracbar_visible , count , cy1 , cy2 , elapsed_time , barcode_data  , barcode_type , last_down_time , noimage

    ############################################################
    # 설정 파일 읽기
    ############################################################
    config = configparser.ConfigParser()
    config.read('config.ini' , encoding="utf-8")
    print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>설정파일로딩")   
    # 설정 값 할당
    video_source = config.get('DEFAULT', 'video_source', fallback='0')  # 기본값은 '0' (웹캠)
    model_path = config.get('DEFAULT', 'model_path', fallback='last.pt')
    machine_name= config.get('DEFAULT', 'machine_name', fallback='None')
    image_size = config.get('DEFAULT', 'image_size', fallback='256')

    max_retries = config.get('DEFAULT', 'max_retries', fallback=10) # 10 최대 재시도 횟수
    retry_delay = config.get('DEFAULT', 'retry_delay', fallback=5) # 5  재시도 간 지연 시간 (초)
    noimage = 0 # 영상취득 실패 횟수 카운트
    
    #################################################
    # 학습된 모델 명 설정 파일 위치 정학하게 세팅해줄것
    # cap=cv2.imread('qrbarcode.jpg')
    #################################################
    model=YOLO(model_path)
    names = model.names
    print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>AI 모델로딩")  
    
    #################################################
    # 클레스 이름 매핑파일 열기
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
    print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>영상연결")   
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
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>추적기로딩")
    #################################################
    # 설정값 정의
    #################################################
    blur_ratio = 50 # Blur ratio
    barcode_data = "None"
    barcode_type = None
    
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
                
                
        # # Flask 애플리케이션을 별도의 스레드에서 실행
        # t = threading.Thread(target=lambda: app.run(host="0.0.0.0", port="8000", debug=True, use_reloader=False))
        # t.daemon = True
        # t.start()
    
    #################################################
    # 처리 시작 루핑...
    #################################################
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>루프진입")
    while True:   
        ret,frame = cap.read()
        
        if not ret:
            print("영싱처리 No Signal (최대 10회 시도후 종료)."+str(noimage+1)+" 회 시도")
            noimage += 1
            
            if noimage == 10 or service_active ==False:
                service_active==False # 쓰레드 작업 빠져나감.
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>서비스 종료상태로 변경 (service_active=False)")
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
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>리싸이즈시작")
        frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    #   frame=cv2.resize(frame,(1024,600))
    
        
        
        if tracbar_visible == False :
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>트렉바그리기")
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
        # 바코드가 계속 바뀐다는 가정하에 그냥 계속감지함
        ###############################################   
        
        try:
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>바코드인식")
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
        # 객체추론 시작
        # 
        ###############################################
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>객체추론로딩시작")
        results=model.predict(frame ,imgsz=int(image_size) , conf= 0.8 , max_det = 1 , classes= [0,1,2]) 
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>객체추론로딩종료")
        ###############################################
        # 리스트 데이터 잘라내기
        ###############################################
        a=results[0].boxes.data
        # print(a)  #tensor([[  1.0071, 530.1204, 268.5498, 719.8320,   0.9678,   0.0000]])
        px=pd.DataFrame(a).astype("float")
        # print(px)
        list=[]
        
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>물체인식루프시작")        
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
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>박스처리시작")      
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
                        
                        # print("경과"+str(elapsed_time))
                        
                    last_down_time = time.time()  
                    count = 0  
                    ######################################################
                    if use_db_yn =='Y':
                        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        insert_data(barcode_data , class_name, id, current_time)      
                    ######################################################
                
        ########################################################    
        # 화면에 통과 수량 표시
        ########################################################    
        d=(len(counter))  # 아래로 내려오는 경우 수 
        u=(len(counter1)) # 위로 올라가는 경우 수 
        cv2.line(frame,(10,cy1),(w - 10,cy1),(255,255,255),3)
        cv2.putText(frame,('In Y Pos-'+ str(cy1)),(10,cy1-20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
        
        cv2.line(frame,(10,cy2),(w - 10,cy2),(255,255,255),3) 
        cv2.putText(frame,('Out Y Pos-'+ str(cy2)),(10,cy2-20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,255),1)
        
        frame = puttext_kr(frame , "생산실적수량-"+str(d) , (10,20) , 48 , (0,255,255))   
        cv2.putText(frame, "Elapsed="+str(f"{elapsed_time:.2f}")+"/Sec" , (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0, 255, 255), 2)
        ########################################
        # 추출된 바코드 표시 파란색
        ########################################
        cv2.putText(frame,('Barcode:-')+barcode_data,(10,180),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),1)  
        cv2.putText(frame,('Database Use:-')+use_db_yn,(10,220),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),1)  
        cv2.putText(frame,('Count :-')+str(count),(10,260),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),1)  
        cv2.putText(frame,('W+h :-')+str(w)+"*"+str(h),(10,300),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),1)  
     
        ##########################
        # OPEN CV 영상표시창 보이기
        ##########################
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>이미지보여주기")
        cv2.imshow(machine_name, frame)
        ##########################
        # 영상을 스트림으로 복사
        ##########################
        # with lock:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>이미지복사")
        outputFrame = frame.copy()
        # print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>영상복사 Lock 설정"+str(lock))
        ###########################
        # 감지된 영상 저장 
        ###########################
        #video_writer.write(frame)
    
        # ESC 키가 눌리거나 서비스 종료요청이 들어오면 
        
        if cv2.waitKey(1)&0xFF==27 or service_active==False:
            print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>종료키눌림")
            service_active==False # 서비스 종료로 처리 
            break
########################################################
# 여기까지 Main 루프        
########################################################    
    if use_db_yn == 'Y':    
        cursor.close()
        connection.close()

    cap.release()
    print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>영상연결해제")
    cv2.destroyAllWindows()
    print( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>영상처리화면종료")
    service_active = False    
    
if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+" >>스트림서버시작")
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=True)