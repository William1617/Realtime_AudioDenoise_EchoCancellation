#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "timing.h"
#include "fft.h"

#ifdef _WIN32

#include <Windows.h>

#else
#include <unistd.h>
#endif

//采用https://github.com/mackron/dr_libs/blob/master/dr_wav.h 解码
#define DR_WAV_IMPLEMENTATION

#include "dr_wav.h"

#ifndef nullptr
#define nullptr 0
#endif

#ifndef MIN
#define MIN(A, B)        ((A) < (B) ? (A) : (B))
#endif

//写wav文件
void wavWrite_scalar(char *filename, float *buffer, size_t sampleRate, size_t totalSampleCount) {
    drwav_data_format format ;
    format.container = drwav_container_riff;     // <-- drwav_container_riff = normal WAV files, drwav_container_w64 = Sony Wave64.
    format.channels = 1;
    format.sampleRate = (drwav_uint32) sampleRate;
    format.bitsPerSample = sizeof(float) * 8;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;

    drwav *pWav = drwav_open_file_write(filename, &format);
    if (pWav) {
        drwav_uint64 samplesWritten = drwav_write(pWav, totalSampleCount, buffer);
        drwav_uninit(pWav);
        if (samplesWritten != totalSampleCount) {
            fprintf(stderr, "ERROR\n");
            exit(1);
        }
    }
}

//读取wav文件
float *wavRead_scalar(char *filename, uint32_t *sampleRate, uint64_t *totalSampleCount) {
    unsigned int channels;
    float *buffer = drwav_open_and_read_file_f32(filename, &channels, sampleRate,
                                                           totalSampleCount);
    if (buffer == nullptr) {
        printf("读取wav文件失败.");
    }
    //仅仅处理单通道音频
    if (channels != 1) {
        drwav_free(buffer);
        buffer = nullptr;
        *sampleRate = 0;
        *totalSampleCount = 0;
    }
    return buffer;
}

//分割路径函数
void splitpath(const char *path, char *drv, char *dir, char *name, char *ext) {
    const char *end;
    const char *p;
    const char *s;
    if (path[0] && path[1] == ':') {
        if (drv) {
            *drv++ = *path++;
            *drv++ = *path++;
            *drv = '\0';
        }
    } else if (drv)
        *drv = '\0';
    for (end = path; *end && *end != ':';)
        end++;
    for (p = end; p > path && *--p != '\\' && *p != '/';)
        if (*p == '.') {
            end = p;
            break;
        }
    if (ext)
        for (s = end; (*ext = *s++);)
            ext++;
    for (p = end; p > path;)
        if (*--p == '\\' || *p == '/') {
            p++;
            break;
        }
    if (name) {
        for (s = p; s < end;)
            *name++ = *s++;
        *name = '\0';
    }
    if (dir) {
        for (s = path; s < p;)
            *dir++ = *s++;
        *dir = '\0';
    }
}

void DTLN(float* samples, int sampleCount);

void audio_deNoise(char *in_file, char *out_file) {
    uint32_t sampleRate = 0; 
    uint64_t inSampleCount = 0;
    float *inBuffer = wavRead_scalar(in_file, &sampleRate, &inSampleCount);
	
#if 1
	DTLN(inBuffer, inSampleCount);
#else
    if (inBuffer != nullptr) {
        int32_t time_win = 50;
        float sigma_noise =0.047f;
        double startTime = now();
        DenoiseProc(inBuffer, sampleRate, inSampleCount, time_win, sigma_noise);
        double time_interval = calcElapsed(startTime, now());
        printf("time interval: %d ms\n ", (int) (time_interval * 1000));
        wavWrite_scalar(out_file, inBuffer, sampleRate, inSampleCount);
        free(inBuffer);
    }
#endif
}

int main(int argc, char *argv[]) {
    printf("Audio Denoise by Time-Frequency Block Thresholding\n");
    printf("blog:http://cpuimage.cnblogs.com/\n");
//    if (argc < 2)
//        return -1;
    char *in_file = "E:/micin_16k_s16_mono.wav";
    char drive[3];
    char dir[256];
    char fname[256];
    char ext[256];
    char out_file[1024];
    splitpath(in_file, drive, dir, fname, ext);
    sprintf(out_file, "%s%s%s_out2%s", drive, dir, fname, ext);
    audio_deNoise(in_file, out_file);

    printf("press any key to exit.\n");
    getchar();
    return 0;
}

void calc_mag_phase(fft_complex* fft_res, float* in_mag, float* in_phase, int count)
{
	for (int i = 0; i < count; i++)
	{
		in_mag[i] = sqrtf(fft_res[i].real * fft_res[i].real + fft_res[i].imag * fft_res[i].imag);
		in_phase[i] = atan2f(fft_res[i].imag, fft_res[i].real);
	}
}

void DTLN(float* samples, int sampleCount)
{
#define block_len		512
#define block_shift		128
	float in_buffer[block_len] = { 0 };
	float out_buffer[block_len] = { 0 };
	float in_mag[block_len / 2 + 1] = { 0 };
	float in_phase[block_len / 2 + 1] = { 0 };

	int num_blocks = (sampleCount - (block_len - block_shift)) / block_shift;

	for (int idx = 0; idx < num_blocks; idx++)
	{
		memmove(in_buffer, in_buffer + block_shift, (block_len - block_shift) * sizeof(float));
		memcpy(in_buffer + (block_len - block_shift), samples + idx * block_shift, block_shift * sizeof(float));

		fft_complex* fft_res = (fft_complex *)malloc(sizeof(fft_complex) * (block_len / 2 + 1));

		fft_plan forward_plan = fft_plan_dft_r2c_1d(block_len, in_buffer, fft_res, 0);
		fft_execute(forward_plan);
		fft_destroy_plan(forward_plan);
		
		calc_mag_phase(fft_res, in_mag, in_phase, block_len / 2 + 1);
	}
}
/*
void tflite_infer()
{
	std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("model/model_1.tflite");

	if (!model) {
		printf("Failed to mmap model\n");
		exit(0);
	}

	tflite::ops::builtin::BuiltinOpResolver resolver;
	std::unique_ptr<tflite::Interpreter> interpreter;
	tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

	// Resize input tensors, if desired.
	interpreter->AllocateTensors();

	float* input = interpreter->typed_input_tensor<float>(0);
	// Dummy input for testing
	*input = 2.0;

	interpreter->Invoke();

	float* output = interpreter->typed_output_tensor<float>(0);

	printf("Result is: %f\n", *output);
}*/