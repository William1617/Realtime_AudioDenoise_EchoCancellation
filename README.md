# Realtime_AudioDenoise_EchoCancellation

This project is c++ port of DTLN denoise.<br/>
For doise and AEC purpose, I've tried speex, webrtc, rnnoise and etc, but didn't find optimal solution (realtime & performance).<br/>
Lately, I found DTLN solution by [**breizhn**](https://github.com/breizhn).<br/>
DTLN not only performs denoising, but also has deecho effect.<br/>

It's VC++ project, but use pure c++, so you can adopt to any platforms.<br/>
I've included all the dependencies for wav, tflite, fft, resampling.<br/>
Also I've included pretrained tflite models from DTLN project.<br/>
So you can test project using raw pcm or wav files in varous format.<br/>

## TIPS
I've tested project on armv7 1GHz environment and got realtime performance.<br/>
If performance not match your need, you can use quantized model in models folder and tried to input 8Khz directly. (Not recommended)<br/>

#### Personally thanks to breizhn for his great work
And looking forward to [DTLN-aec](https://github.com/breizhn/DTLN-aec) repo completion.

## STAR if project is helpful
And welcome pull requests and issue.

## REFERENCES
[DTLN](https://github.com/breizhn/DTLN)  (https://github.com/breizhn/DTLN)  (Great thanks to breizhn)<br/>
[PFFT](https://gitlab.mpcdf.mpg.de/mtr/pocketfft/-/tree/cpp)  (https://gitlab.mpcdf.mpg.de/mtr/pocketfft/-/tree/cpp)<br/>
[Resample](https://github.com/cpuimage/resampler)  (https://github.com/cpuimage/resampler)<br/>

## LICENSE
Anti 996 License from (https://github.com/kattgu7/Anti-996-License)
