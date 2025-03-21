import urllib.request

url = "https://www2.cs.uic.edu/~i101/SoundFiles/taunt.wav"
urllib.request.urlretrieve(url, "sample_audio.wav")

print("Sample audio downloaded successfully!")
