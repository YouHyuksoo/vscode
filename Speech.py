import speech_recognition as sr
import pyttsx3
import pyautogui
import pyperclip

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen_and_act():
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

        while True:
            speak("말씀하세요. 프로그램을 종료하려면 '꺼져'라고 말해주세요.")
            audio = recognizer.listen(source)
            
            try:
                text = recognizer.recognize_google(audio, language='ko-KR')
                print("인식된 내용: " + text)  # Recognized text

                if text.strip() == "꺼져":
                    speak("프로그램을 종료합니다.")
                    break
                else:
                    # Copy text to clipboard
                    pyperclip.copy(text)
                    # Simulate a "paste" command
                    pyautogui.hotkey('ctrl', 'v', interval=0.15)  # Use 'command' instead of 'ctrl' on macOS
                    pyautogui.press('enter')
            except sr.UnknownValueError:
                speak("죄송합니다. 못 알아들었어요. 다시 말씀해주세요.")
            except sr.RequestError as e:
                speak("음성 인식 서비스를 요청할 수 없습니다.")
            except Exception as e:
                speak(f"예상치 못한 오류가 발생했습니다: {e}")

listen_and_act()