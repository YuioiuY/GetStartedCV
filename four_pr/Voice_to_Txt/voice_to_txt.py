import speech_recognition as sr

def recording():
    rec = sr.Recognizer()

    with sr.Microphone(device_index = 2) as device:
        print("Recording...")
        audio = rec.listen(device)

    result = rec.recognize_google(audio, language="ru-RU")
    print("Recording this -> " + result)


def main():

    while True:

        rec = sr.Recognizer()

        with sr.Microphone(device_index = 2) as device:

            audio = rec.listen(device)
            result = rec.recognize_google(audio, language="ru-RU")
            if result == "Геннадий":
                recording()


main()