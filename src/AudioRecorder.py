import pyaudio
import os
import wave

def record_audio(train):
    Name = (input("Entre com seu nome: "))
    quant = int(input("Quantidade de audios de 2s que serao gravados: "))
    for count in range(quant):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512
        RECORD_SECONDS = 2
        device_index = 2
        audio = pyaudio.PyAudio()
        if count == 0:
            print("----------------------lista de dispositivos para gravacao---------------------")
            info = audio.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            for i in range(0, numdevices):
                if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                    print("Input Device id ", i, " - ",
                          audio.get_device_info_by_host_api_device_index(0, i).get('name')) 
            print("-------------------------------------------------------------")
            index = int(input())
            print("Gravando via index "+str(index))
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, input_device_index=index,
                            frames_per_buffer=CHUNK)
        print("Gravacao {0} comecou.".format(count+1))
        Recordframes = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            Recordframes.append(data)
        print("Gravacao {0} terminou".format(count+1))
        stream.stop_stream()
        stream.close()
        audio.terminate()
        OUTPUT_FILENAME = str(count)+"_"+Name.lower()+"-sample"+".wav"
        if train:
          WAVE_OUTPUT_FILENAME = os.path.join("../data/recordings/moreSpeakersTrain", OUTPUT_FILENAME)
        else:
          WAVE_OUTPUT_FILENAME = os.path.join("../data/recordings/moreSpeakersTest", OUTPUT_FILENAME)
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()

print("Gravacao de audio")
while True:
  opc = input("[0] - Sair\n[1] - Gravar audio para treinamento\n[2] - Gravar audio para teste\n")
  if opc == "1":
    record_audio(True)
  elif opc == "2":
    record_audio(False)
  else:
    break
