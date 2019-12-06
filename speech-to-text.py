import speech_recognition as sr

r = sr.Recognizer()

file = input ('Enter audio file name :')

harvard = sr.AudioFile(file)
with harvard as source:
    audio = r.record(source)

txt = r.recognize_google(audio)

print(txt)

