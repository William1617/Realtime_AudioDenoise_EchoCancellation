# Realtime_AudioDenoise_EchoCancellation

This project is c++ port of DTLN denoise.<br/>
For doise and AEC purpose, I've tried speex, webrtc, rnnoise and etc, but didn't find optimal solution (realtime & performance).<br/>
Lately, I found DTLN solution by [**breizhn**](https://github.com/breizhn).<br/>
DTLN not only performs denoising, but also has deecho effect.<br/>

It's VC++ project, but use pure c++, so you can adopt to any platforms.<br/>
I've included all the dependencies for wav, tflite, fft, resampling.<br/>
Also I've included pretrained tflite models from DTLN project.<br/>
So you can test project using raw pcm or wav files in varous format.<br/>

## USAGE
You can create multiple instances.<br/>
Please refer to project for detailed usage.<br/>
Input format of denoise function is 16KHz, fp32 pcm.<br/>

    void tflite_create(trg_engine* engine)
    void trg_denoise(trg_engine* engine, float* samples, float* out, int sampleCount)
    void tflite_destroy(trg_engine* engine)

## TIPS
I've tested project on armv7 1GHz environment and got realtime performance.<br/>
If performance not match your need, you can use quantized model in models folder and tried to input 8Khz directly instead of 16Khz pcm**.<br/>
** I've tested 8Khz pcm directly without resampling and got x2 speed up, good quality on test samples. (Not recommended)


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
