import pyaudio
import numpy as np
import wave
import os
import sys
from kivy.uix.progressbar import ProgressBar
from scipy import signal
from scipy.signal import butter, lfilter

## Record from n microphones
def record_N_mic(num, RECORD_SECONDS):
    
    # Set recording parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 8192
    INDEXES = [2,3]

    mics = []

    ##intializing microphones
    for i in range(num):
        mics.append(pyaudio.PyAudio())
    stream = []
    
    for i in range(num):
        stream.append(mics[i].open(format=FORMAT, channels=CHANNELS,
                              rate=RATE, input=True, output =False,
                              input_device_index=INDEXES[i],
                              frames_per_buffer=CHUNK))
    frames = [[]] * num

    print("recording")                  
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        for j in range(num):
            data = stream[j].read(CHUNK)
            frames[j].append(data)

    print("recording stopped")
    for i in range(num):
        stream[i].stop_stream()
        stream[i].close()
        mics[i].terminate()

        
    return frames

#-----------------------------------------------------------------------------

# write to a .wav file
def write_wav(frames, file_name):
    
    waveFile1 = wave.open( file_name, 'wb')
    waveFile1.setnchannels(1)
    waveFile1.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
    waveFile1.setframerate(44100)
    waveFile1.writeframes(b''.join(frames))
    waveFile1.close()

#-----------------------------------------------------------------------------
def get_mics():
    p = pyaudio.PyAudio()
    
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
          if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
               print ("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

    p.terminate()
#----------------------------------------------------------------------------
def time_delay(nphones, sound_speed, spacing, noise_pwr, signal_pwr, sampling_rate, samples, signal_hz, signal_dir):
    data = np.sqrt(noise_pwr)*np.random.randn(samples, nphones)
    time = np.linspace(0, samples/sampling_rate, samples, endpoint = False)
##    signal = np.sqrt(signal_pwr)*np.random.randn(samples)
    time_delays = spacing/sound_speed
    fft_freqs = np.matrix(np.linspace( 0, sampling_rate, samples, endpoint = False))
    time_delays = np.matrix(np.cos(signal_dir)*time_delays)
    spacial_filt = np.exp(2j*np.pi*fft_freqs.transpose()*time_delays)
    spacial_filt = np.array(spacial_filt).transpose()
    replicas = np.fft.irfft( np.array(np.fft.fft(signal))*spacial_filt, samples, 1)
    
    data = data + replicas.transpose()
    return data, time
#----------------------------------------------------------------------------
def signal_process(nphones, sound_speed, spacing, look_dirs, samples, data):
    sampling_rate = 250
    bf_data = np.zeros((samples, len(look_dirs)))
    time_delays = np.matrix ( (spacing/sound_speed))
    fft_freqs = np.matrix(np.linspace( 0, sampling_rate, samples, endpoint=False)).transpose()

    for ind, direction in enumerate(look_dirs):
        spacial_filt = 1.0/nphones*np.exp(-2j*np.pi*fft_freqs*time_delays*np.cos(direction))
        bf_data[:, ind] = np.sum(np.fft.irfft(np.fft.fft(data, samples, 0)*np.array(spacial_filt), samples, 0), 1)
        return bf_data
#----------------------------------------------------------------------------
def set_mic_file():
    # Ensures alsamixer runs
    file = open("/home/pi/.asoundrc", 'w')
    data = """pcm.!default {
                    type hw
                    card 1
                    card 2
            }

            ctl.!default {
            type hw
            card 1
            card 2
            }"""
    
    file.write(data)
    file.close()
#----------------------------------------------------------------------------
def start_Bluetooth():
    p = os.popen('pulseaudio --start',"r")
    os.popen('blutetoothctl',"r")
    os.popen('power on',"r")
    os.popen('agent on',"r")
    os.popen('default-agent',"r")

#----------------------------------------------------------------------------
def bessel_bandpass_filter(inputfile, outputfile):
    wr = wave.open(inputfile, 'r')
    par = list(wr.getparams()) # Get the parameters from the input.
    # This file is stereo, 2 bytes/sample, 44.1 kHz.
    par[3] = 0 # The number of samples will be set by writeframes.

    # Open the output file
    ww = wave.open(outputfile, 'w')
    ww.setparams(tuple(par)) # Use the same parameters as the input file.

    lowpass = 200 # Remove lower frequencies.
    highpass = 1500 # Remove higher frequencies.

    sz = wr.getframerate() # Read and process 1 second at a time.
    c = int(wr.getnframes()/sz) # whole file
    for num in range(c):
        print('Processing {}/{} s'.format(num+1, c))
        da = np.fromstring(wr.readframes(sz), dtype=np.int16)
        left, right = da[0::2], da[1::2] # left and right channel
        lf, rf = np.fft.rfft(left), np.fft.rfft(right)
        lf[:lowpass], rf[:lowpass] = 0, 0 # low pass filter
        lf[55:66], rf[55:66] = 0, 0 # line noise
        lf[highpass:], rf[highpass:] = 0,0 # high pass filter
        nl, nr = np.fft.irfft(lf), np.fft.irfft(rf)
        ns = np.column_stack((nl,nr)).ravel().astype(np.int16)
        ww.writeframes(ns.tostring())
    # Close the files.
    wr.close()
    ww.close()
#----------------------------------------------------------------------------
def record_2_mic(RECORD_SECONDS):
    # Connect to Bluetooth headphones
    p = os.popen('pulseaudio --start',"r")

    # Ensures alsamixer runs
    file = open("/home/pi/.asoundrc", 'w')
    data = """pcm.!default {
                    type hw
                    card 1
                    card 2
            }

            ctl.!default {
            type hw
            card 1
            card 2
            }"""
    
    file.write(data)
    file.close()
            
    p = pyaudio.PyAudio()
    audio2 = pyaudio.PyAudio()

    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
          
           if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
               print ("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


    # Set recording parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 8192
    
    INDEXES = [2,3]
    
    audio1 = p

    # start Recording
    stream1 = audio1.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, output =False,
                        input_device_index=INDEXES[0],
                        frames_per_buffer=CHUNK)
    # start Recording
    stream2 = audio2.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, output =False,
                        input_device_index=INDEXES[1],
                        frames_per_buffer=CHUNK)


    print ("recording...")
    frames1 = []
    frames2 = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data1 = stream1.read(CHUNK)
        data2 = stream2.read(CHUNK)
        frames1.append(data1)
        frames2.append(data2)
    print ("finished recording")

    # stop Recording
    stream1.stop_stream()
    stream1.close()
    stream2.stop_stream()
    stream2.close()
    audio1.terminate()
    audio2.terminate()


    waveFile1 = wave.open("file1.wav", 'wb')
    waveFile1.setnchannels(CHANNELS)
    waveFile1.setsampwidth(audio1.get_sample_size(FORMAT))
    waveFile1.setframerate(RATE)
    waveFile1.writeframes(b''.join(frames1))
    waveFile1.close()

    waveFile2 = wave.open("file2.wav", 'wb')
    waveFile2.setnchannels(CHANNELS)
    waveFile2.setsampwidth(audio2.get_sample_size(FORMAT))
    waveFile2.setframerate(RATE)
    waveFile2.writeframes(b''.join(frames2))
    waveFile2.close()  

#-----------------------------------------------------------------------------
def record_2_mic_progress(RECORD_SECONDS, progress):
    # Connect to Bluetooth headphones
    p = os.popen('pulseaudio --start',"r")

    # Ensures alsamixer runs
    file = open("/home/pi/.asoundrc", 'w')
    data = """pcm.!default {
                    type hw
                    card 1
                    card 2
            }

            ctl.!default {
            type hw
            card 1
            card 2
            }"""
    
    file.write(data)
    file.close()
            
    p = pyaudio.PyAudio()
    audio2 = pyaudio.PyAudio()

    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
          
           if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
               print ("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


    # Set recording parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 8192
    
    INDEXES = [2,3]
    
    audio1 = p

    # start Recording
    stream1 = audio1.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, output =False,
                        input_device_index=INDEXES[0],
                        frames_per_buffer=CHUNK)
    # start Recording
    stream2 = audio2.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, output =False,
                        input_device_index=INDEXES[1],
                        frames_per_buffer=CHUNK)


    print ("recording...")
    frames1 = []
    frames2 = []
    progress.value_normalized = 0.1
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data1 = stream1.read(CHUNK)
        data2 = stream2.read(CHUNK)
        frames1.append(data1)
        frames2.append(data2)
        progress.value_normalized += 0.8/RECORD_SECONDS
    print ("finished recording")

    # stop Recording
    stream1.stop_stream()
    stream1.close()
    stream2.stop_stream()
    stream2.close()
    audio1.terminate()
    audio2.terminate()


    waveFile1 = wave.open("file1.wav", 'wb')
    waveFile1.setnchannels(CHANNELS)
    waveFile1.setsampwidth(audio1.get_sample_size(FORMAT))
    waveFile1.setframerate(RATE)
    waveFile1.writeframes(b''.join(frames1))
    waveFile1.close()

    waveFile2 = wave.open("file2.wav", 'wb')
    waveFile2.setnchannels(CHANNELS)
    waveFile2.setsampwidth(audio2.get_sample_size(FORMAT))
    waveFile2.setframerate(RATE)
    waveFile2.writeframes(b''.join(frames2))
    waveFile2.close()

#----------------------------------------------------------------------------
def record_3_mic(RECORD_SECONDS):
    # Connect to Bluetooth headphones
    p = os.popen('pulseaudio --start',"r")

    # Ensures alsamixer runs
    file = open("/home/pi/.asoundrc", 'w')
    data = """pcm.!default {
                    type hw
                    card 1
                    card 2
                    card 3
                    
            }

            ctl.!default {
            type hw
            card 1
            card 2
            card 3
            
            }"""
    
    file.write(data)
    file.close()
            
    p = pyaudio.PyAudio()
    audio2 = pyaudio.PyAudio()
    audio3 = pyaudio.PyAudio()

    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
          
           if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
               print ("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


    # Set recording parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 8192
    
    INDEXES = [2,3,4]
    
    audio1 = p

    # start Recording
    stream1 = audio1.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, output =False,
                        input_device_index=INDEXES[0],
                        frames_per_buffer=CHUNK)
    # start Recording
    stream2 = audio2.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, output =False,
                        input_device_index=INDEXES[1],
                        frames_per_buffer=CHUNK)
    
    stream3 = audio3.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, output =False,
                        input_device_index=INDEXES[2],
                        frames_per_buffer=CHUNK)


    print ("recording...")
    frames1 = []
    frames2 = []
    frames3 = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data1 = stream1.read(CHUNK)
        data2 = stream2.read(CHUNK)
        data3 = stream3.read(CHUNK)
        frames1.append(data1)
        frames2.append(data2)
        frames3.append(data3)
    print ("finished recording")

    # stop Recording
    stream1.stop_stream()
    stream1.close()
    stream2.stop_stream()
    stream2.close()
    stream3.stop_stream()
    stream3.close()
    audio1.terminate()
    audio2.terminate()
    audio3.terminate()


    waveFile1 = wave.open("file1.wav", 'wb')
    waveFile1.setnchannels(CHANNELS)
    waveFile1.setsampwidth(audio1.get_sample_size(FORMAT))
    waveFile1.setframerate(RATE)
    waveFile1.writeframes(b''.join(frames1))
    waveFile1.close()

    waveFile2 = wave.open("file2.wav", 'wb')
    waveFile2.setnchannels(CHANNELS)
    waveFile2.setsampwidth(audio2.get_sample_size(FORMAT))
    waveFile2.setframerate(RATE)
    waveFile2.writeframes(b''.join(frames2))
    waveFile2.close()

    waveFile3 = wave.open("file3.wav", 'wb')
    waveFile3.setnchannels(CHANNELS)
    waveFile3.setsampwidth(audio3.get_sample_size(FORMAT))
    waveFile3.setframerate(RATE)
    waveFile3.writeframes(b''.join(frames3))
    waveFile3.close()

#-----------------------------------------------------------------------------
def record_4_mic(RECORD_SECONDS):
    # Connect to Bluetooth headphones
    p = os.popen('pulseaudio --start',"r")

    # Ensures alsamixer runs
    file = open("/home/pi/.asoundrc", 'w')
    data = """pcm.!default {
                    type hw
                    card 1
                    card 2
                    card 3
                    card 4
        
            }

            ctl.!default {
            type hw
            card 1
            card 2
            card 3
            card 4
            
            }"""
    
    file.write(data)
    file.close()
            
    p = pyaudio.PyAudio()
    audio2 = pyaudio.PyAudio()
    audio3 = pyaudio.PyAudio()
    audio4 = pyaudio.PyAudio()

    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
          
           if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
               print ("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


    # Set recording parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 8192
    
    INDEXES = [2,3,4,5]
    
    audio1 = p

    # start Recording
    stream1 = audio1.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, output =False,
                        input_device_index=INDEXES[0],
                        frames_per_buffer=CHUNK)
    # start Recording
    stream2 = audio2.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, output =False,
                        input_device_index=INDEXES[1],
                        frames_per_buffer=CHUNK)
    
    stream3 = audio3.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, output =False,
                        input_device_index=INDEXES[2],
                        frames_per_buffer=CHUNK)
    
    stream4 = audio4.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, output =False,
                        input_device_index=INDEXES[3],
                        frames_per_buffer=CHUNK)


    print ("recording...")
    frames1 = []
    frames2 = []
    frames3 = []
    frames4 = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data1 = stream1.read(CHUNK)
        data2 = stream2.read(CHUNK)
        data3 = stream3.read(CHUNK)
        data4 = stream4.read(CHUNK)
        frames1.append(data1)
        frames2.append(data2)
        frames3.append(data3)
        frames4.append(data4)
    print ("finished recording")

    # stop Recording
    stream1.stop_stream()
    stream1.close()
    stream2.stop_stream()
    stream2.close()
    stream3.stop_stream()
    stream3.close()
    stream4.stop_stream()
    stream4.close()
    audio1.terminate()
    audio2.terminate()
    audio3.terminate()
    audio4.terminate()


    waveFile1 = wave.open("./file1.wav", 'wb')
    waveFile1.setnchannels(CHANNELS)
    waveFile1.setsampwidth(audio1.get_sample_size(FORMAT))
    waveFile1.setframerate(RATE)
    waveFile1.writeframes(b''.join(frames1))
    waveFile1.close()

    waveFile2 = wave.open("./file2.wav", 'wb')
    waveFile2.setnchannels(CHANNELS)
    waveFile2.setsampwidth(audio2.get_sample_size(FORMAT))
    waveFile2.setframerate(RATE)
    waveFile2.writeframes(b''.join(frames2))
    waveFile2.close()

    waveFile3 = wave.open("./file3.wav", 'wb')
    waveFile3.setnchannels(CHANNELS)
    waveFile3.setsampwidth(audio3.get_sample_size(FORMAT))
    waveFile3.setframerate(RATE)
    waveFile3.writeframes(b''.join(frames3))
    waveFile3.close()

    waveFile4 = wave.open("./file4.wav", 'wb')
    waveFile4.setnchannels(CHANNELS)
    waveFile4.setsampwidth(audio4.get_sample_size(FORMAT))
    waveFile4.setframerate(RATE)
    waveFile4.writeframes(b''.join(frames4))
    waveFile4.close()

 
#-----------------------------------------------------------------------------
def record_4_mic_realtime(RECORD_SECONDS):
    # Connect to Bluetooth headphones
    p = os.popen('pulseaudio --start',"r")

    # Ensures alsamixer runs
    file = open("/home/pi/.asoundrc", 'w')
    data = """pcm.!default {
                    type hw
                    card 1
                    card 2
                    card 3
                    card 4
        
            }

            ctl.!default {
            type hw
            card 1
            card 2
            card 3
            card 4
            
            }"""
    
    file.write(data)
    file.close()
            
    p = pyaudio.PyAudio()
    audio2 = pyaudio.PyAudio()
    audio3 = pyaudio.PyAudio()
    audio4 = pyaudio.PyAudio()

    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
          
           if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
               print ("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))


    # Set recording parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 8192
    
    INDEXES = [2,3,4,5]
    
    audio1 = p

    # start Recording
    stream1 = audio1.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, output =False,
                        input_device_index=INDEXES[0],
                        frames_per_buffer=CHUNK)
    # start Recording
    stream2 = audio2.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, output =False,
                        input_device_index=INDEXES[1],
                        frames_per_buffer=CHUNK)
    
    stream3 = audio3.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, output =False,
                        input_device_index=INDEXES[2],
                        frames_per_buffer=CHUNK)
    
    stream4 = audio4.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, output =False,
                        input_device_index=INDEXES[3],
                        frames_per_buffer=CHUNK)


    print ("recording...")
    frames1 = []
    frames2 = []
    frames3 = []
    frames4 = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data1 = stream1.read(CHUNK)
        data2 = stream2.read(CHUNK)
        data3 = stream3.read(CHUNK)
        data4 = stream4.read(CHUNK)
        frames1.append(data1)
        frames2.append(data2)
        frames3.append(data3)
        frames4.append(data4)
    print ("finished recording")

    # stop Recording
    stream1.stop_stream()
    stream1.close()
    stream2.stop_stream()
    stream2.close()
    stream3.stop_stream()
    stream3.close()
    stream4.stop_stream()
    stream4.close()
    audio1.terminate()
    audio2.terminate()
    audio3.terminate()
    audio4.terminate()


    waveFile1 = wave.open("file1.wav", 'wb')
    waveFile1.setnchannels(CHANNELS)
    waveFile1.setsampwidth(audio1.get_sample_size(FORMAT))
    waveFile1.setframerate(RATE)
    waveFile1.writeframes(b''.join(frames1))
    waveFile1.close()

    waveFile2 = wave.open("file2.wav", 'wb')
    waveFile2.setnchannels(CHANNELS)
    waveFile2.setsampwidth(audio2.get_sample_size(FORMAT))
    waveFile2.setframerate(RATE)
    waveFile2.writeframes(b''.join(frames2))
    waveFile2.close()

    waveFile3 = wave.open("file3.wav", 'wb')
    waveFile3.setnchannels(CHANNELS)
    waveFile3.setsampwidth(audio3.get_sample_size(FORMAT))
    waveFile3.setframerate(RATE)
    waveFile3.writeframes(b''.join(frames3))
    waveFile3.close()

    waveFile4 = wave.open("file4.wav", 'wb')
    waveFile4.setnchannels(CHANNELS)
    waveFile4.setsampwidth(audio4.get_sample_size(FORMAT))
    waveFile4.setframerate(RATE)
    waveFile4.writeframes(b''.join(frames4))
    waveFile4.close()
#-----------------------------------------------------------------------------

       
        

