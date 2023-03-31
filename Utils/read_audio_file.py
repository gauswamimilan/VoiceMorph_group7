import pyaudio
import wave
import numpy as np

def play_audio_file(audio_file, volume=1.0):
    # Open the audio file using wave library
    wf = wave.open(audio_file, 'rb')

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream and start playing audio
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
                    
    data = wf.readframes(1024)

    while data:
        samples = np.frombuffer(data, dtype=np.int16)
        samples = volume * samples
        stream.write(samples.astype(np.int16).tobytes())
        # stream.write(data)
        data = wf.readframes(1024)

    # Stop stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
