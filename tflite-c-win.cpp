#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef _WIN32

#include <Windows.h>

#else
#include <unistd.h>
#endif


//采用https://github.com/mackron/dr_libs/blob/master/dr_wav.h 解码
#define DR_WAV_IMPLEMENTATION

#include "dr_wav.h"
#include <tensorflow\lite\c\c_api.h>

#ifndef MIN
#define MIN(A, B)        ((A) < (B) ? (A) : (B))
#endif

//写wav文件
void wavWrite_scalar(char* filename, float* buffer, size_t sampleRate, size_t totalSampleCount) {
    drwav_data_format format;
    format.container = drwav_container_riff;     // <-- drwav_container_riff = normal WAV files, drwav_container_w64 = Sony Wave64.
    format.channels = 1;
    format.sampleRate = (drwav_uint32)sampleRate;
    format.bitsPerSample = sizeof(float) * 8;
    format.format = 0x3;

    drwav* pWav = drwav_open_file_write(filename, &format);
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
float* wavRead_scalar(char* filename, uint32_t* sampleRate, uint64_t* totalSampleCount) {
    unsigned int channels;
    float* buffer = drwav_open_and_read_file_f32(filename, &channels, sampleRate,
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
void splitpath(const char* path, char* drv, char* dir, char* name, char* ext) {
    const char* end;
    const char* p;
    const char* s;
    if (path[0] && path[1] == ':') {
        if (drv) {
            *drv++ = *path++;
            *drv++ = *path++;
            *drv = '\0';
        }
    }
    else if (drv)
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

void audio_denoise(char* in_file, char* out_file);
#define S16_INPUT_RAW

int main(int argc, char* argv[]) {
    printf("Audio Denoise Using DTLN by Turing311\n");
    char* in_file = (char*)"E:\\Ref\\AudioDenoise\\noisyPcm\\noise.raw";
    char drive[3];
    char dir[256];
    char fname[256];
    char ext[256];
    char out_file[1024];
    splitpath(in_file, drive, dir, fname, ext);
    sprintf(out_file, "%s%s%s_out2%s", drive, dir, fname, ext);
    audio_denoise(in_file, out_file);

    printf("press any key to exit.\n");
    getchar();
    return 0;
}


#include "pocketfft_hdronly.h"

#define block_len		512
#define block_shift		128
#define fft_out_size    (block_len / 2 + 1)
#define BLOCK_SIZE 256
#define STATE_SIZE 512

using namespace pocketfft;
using namespace std;
typedef complex<double> cpx_type;

struct trg_engine {
    float in_buffer[block_len] = { 0 };
    float out_buffer[block_len] = { 0 };
    float states_1[STATE_SIZE] = { 0 };
    float states_2[STATE_SIZE] = { 0 };

    TfLiteTensor* input_details_1[2], * input_details_2[2];
    const TfLiteTensor* output_details_1[2], * output_details_2[2];
    TfLiteInterpreter* interpreter_1, * interpreter_2;
    TfLiteModel* model1, * model2;
};


uint64_t resample_s16(const int16_t* input, int16_t* output, int inSampleRate, int outSampleRate, uint64_t inputSize, uint32_t channels) {
    if (input == NULL)
        return 0;
    uint64_t outputSize = (uint64_t)(inputSize * (double)outSampleRate / (double)inSampleRate);
    outputSize -= outputSize % channels;
    if (output == NULL)
        return outputSize;
    double stepDist = ((double)inSampleRate / (double)outSampleRate);
    const uint64_t fixedFraction = (1LL << 32);
    const double normFixed = (1.0 / (1LL << 32));
    uint64_t step = ((uint64_t)(stepDist * fixedFraction + 0.5));
    uint64_t curOffset = 0;
    for (uint32_t i = 0; i < outputSize; i += 1) {
        for (uint32_t c = 0; c < channels; c += 1) {
            *output++ = (int16_t)(input[c] + (input[c + channels] - input[c]) * (
                (double)(curOffset >> 32) + ((curOffset & (fixedFraction - 1)) * normFixed)
                )
                );
        }
        curOffset += step;
        input += (curOffset >> 32) * channels;
        curOffset &= (fixedFraction - 1);
    }
    if (inSampleRate < outSampleRate)
        *(output - 1) = *(output - 2);
    return outputSize;
}

void s16_8khz_to_f32_16khz(short* in, float* out, int count)
{
    short s16_sample[BLOCK_SIZE * 2];
    for (int i = 0; i < count / BLOCK_SIZE; i++)
    {
        resample_s16(in, s16_sample, 8000, 16000, BLOCK_SIZE, 1);
        for (int j = 0; j < BLOCK_SIZE * 2; j++)
            out[j] = s16_sample[j] / 32767.f;
        in += BLOCK_SIZE;
        out += BLOCK_SIZE * 2;
    }
}

void f32_16khz_to_s16_8khz(float* in, short* out, int count)
{
    short s16_sample[BLOCK_SIZE];
    for (int i = 0; i < count / BLOCK_SIZE; i++)
    {
        for (int j = 0; j < BLOCK_SIZE; j++)
            s16_sample[j] = in[j] * 32767.f;

        resample_s16(s16_sample, out, 16000, 8000, BLOCK_SIZE, 1);
        in += BLOCK_SIZE;
        out += BLOCK_SIZE / 2;
    }
}

void calc_mag_phase(vector<cpx_type> fft_res, float* in_mag, float* in_phase, int count)
{
    for (int i = 0; i < count; i++)
    {
        in_mag[i] = sqrtf(fft_res[i].real() * fft_res[i].real() + fft_res[i].imag() * fft_res[i].imag());
        in_phase[i] = atan2f(fft_res[i].imag(), fft_res[i].real());
    }
}

void tflite_create(trg_engine* engine)
{
    //---------------------------------------------
    engine->model1 = TfLiteModelCreateFromFile("model/model_quant_1.tflite");

    // Build the interpreter
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);

    // Create the interpreter.
    engine->interpreter_1 = TfLiteInterpreterCreate(engine->model1, options);
    if (engine->interpreter_1 == nullptr) {
        printf("Failed to create interpreter");
        return;
    }

    // Allocate tensor buffers.
    if (TfLiteInterpreterAllocateTensors(engine->interpreter_1) != kTfLiteOk) {
        printf("Failed to allocate tensors!");
        return;
    }

    //---------------------------------------------
    engine->model2 = TfLiteModelCreateFromFile("model/model_quant_2.tflite");
    // Create the interpreter.
    engine->interpreter_2 = TfLiteInterpreterCreate(engine->model2, options);
    if (engine->interpreter_2 == nullptr) {
        printf("Failed to create interpreter");
        return;
    }

    // Allocate tensor buffers.
    if (TfLiteInterpreterAllocateTensors(engine->interpreter_2) != kTfLiteOk) {
        printf("Failed to allocate tensors!");
        return;
    }

    ////////////////////////////////////

    engine->input_details_1[0] = TfLiteInterpreterGetInputTensor(engine->interpreter_1, 0);
    engine->input_details_1[1] = TfLiteInterpreterGetInputTensor(engine->interpreter_1, 1);
    engine->output_details_1[0] = TfLiteInterpreterGetOutputTensor(engine->interpreter_1, 0);
    engine->output_details_1[1] = TfLiteInterpreterGetOutputTensor(engine->interpreter_1, 1);

    engine->input_details_2[0] = TfLiteInterpreterGetInputTensor(engine->interpreter_2, 0);
    engine->input_details_2[1] = TfLiteInterpreterGetInputTensor(engine->interpreter_2, 1);
    engine->output_details_2[0] = TfLiteInterpreterGetOutputTensor(engine->interpreter_2, 0);
    engine->output_details_2[1] = TfLiteInterpreterGetOutputTensor(engine->interpreter_2, 1);
}

void tflite_destroy(trg_engine* engine)
{
    TfLiteModelDelete(engine->model1);
    TfLiteModelDelete(engine->model2);
}

void tflite_infer(trg_engine* engine)
{
    float in_mag[block_len / 2 + 1] = { 0 };
    float in_phase[block_len / 2 + 1] = { 0 };
    float estimated_block[block_len];

    double fft_in[block_len];
    vector<cpx_type> fft_res(block_len);

    shape_t shape;
    shape.push_back(block_len);
    shape_t axes;
    axes.push_back(0);
    stride_t stridel, strideo;
    strideo.push_back(sizeof(cpx_type));
    stridel.push_back(sizeof(double));

    for (int i = 0; i < block_len; i++)
        fft_in[i] = engine->in_buffer[i];

    r2c(shape, stridel, strideo, axes, FORWARD, fft_in, fft_res.data(), 1.0);

    calc_mag_phase(fft_res, in_mag, in_phase, fft_out_size);

    memcpy(engine->input_details_1[0]->data.f, in_mag, fft_out_size * sizeof(float));
    memcpy(engine->input_details_1[1]->data.f, engine->states_1, STATE_SIZE * sizeof(float));

    if (TfLiteInterpreterInvoke(engine->interpreter_1) != kTfLiteOk) {
        printf("Error invoking detection model");
    }

    float* out_mask = engine->output_details_1[0]->data.f;
    memcpy(engine->states_1, engine->output_details_1[1]->data.f, STATE_SIZE * sizeof(float));

    for (int i = 0; i < fft_out_size; i++)
        fft_res[i] = cpx_type(in_mag[i] * out_mask[i] * cosf(in_phase[i]), in_mag[i] * out_mask[i] * sinf(in_phase[i]));

    c2r(shape, strideo, stridel, axes, BACKWARD, fft_res.data(), fft_in, 1.0);
    for (int i = 0; i < block_len; i++)
        estimated_block[i] = fft_in[i] / block_len;

    memcpy(engine->input_details_2[0]->data.f, estimated_block, block_len * sizeof(float));
    memcpy(engine->input_details_2[1]->data.f, engine->states_2, STATE_SIZE * sizeof(float));

    if (TfLiteInterpreterInvoke(engine->interpreter_2) != kTfLiteOk) {
        printf("Error invoking detection model");
    }

    float* out_block = engine->output_details_2[0]->data.f;
    memcpy(engine->states_2, engine->output_details_2[1]->data.f, STATE_SIZE * sizeof(float));

    memmove(engine->out_buffer, engine->out_buffer + block_shift, (block_len - block_shift) * sizeof(float));
    memset(engine->out_buffer + (block_len - block_shift), 0, block_shift * sizeof(float));
    for (int i = 0; i < block_len; i++)
        engine->out_buffer[i] += out_block[i];
}

void trg_denoise(trg_engine* engine, float* samples, float* out, int sampleCount)
{
    int num_blocks = sampleCount / block_shift;

    for (int idx = 0; idx < num_blocks; idx++)
    {
        memmove(engine->in_buffer, engine->in_buffer + block_shift, (block_len - block_shift) * sizeof(float));
        memcpy(engine->in_buffer + (block_len - block_shift), samples, block_shift * sizeof(float));
        tflite_infer(engine);
        memcpy(out, engine->out_buffer, block_shift * sizeof(float));
        samples += block_shift;
        out += block_shift;
    }
}

void s16_8khz_to_f32_8khz(short* in, float* out, int count)
{
    for (int j = 0; j < count; j++)
        out[j] = in[j] / 32767.f;
}

void audio_denoise(char* in_file, char* out_file) {
    uint32_t sampleRate = 16000;
    uint64_t inSampleCount = 0;

#ifdef S16_INPUT_RAW
    inSampleCount = 80 * 1024;
    short* inBuffer_s16_8k = (short*)malloc(inSampleCount * sizeof(short));
    FILE* fp = fopen(in_file, "rb");
    if (!fp)
    {
        printf("Please change input file path.\n");
        return;
    }
    fread(inBuffer_s16_8k, inSampleCount, 2, fp);
    fclose(fp);
#else
    float* inBuffer = wavRead_scalar(in_file, &sampleRate, &inSampleCount);
#endif
    FILE* fpf32 = fopen(out_file, "wb");

    trg_engine eng1;
    tflite_create(&eng1);

    int blockCount = inSampleCount / BLOCK_SIZE;
    float f32_sample[BLOCK_SIZE * 2];
    float outBuffer_f32_16khz[BLOCK_SIZE * 2];

    short out_s16_8khz[BLOCK_SIZE];

    for (int i = 0; i < blockCount; i++)
    {
//        s16_8khz_to_f32_8khz(inBuffer_s16_8k, f32_sample, BLOCK_SIZE * 2);    //  Using 8KHz directly (when you need more fps)
        s16_8khz_to_f32_16khz(inBuffer_s16_8k, f32_sample, BLOCK_SIZE);         

        trg_denoise(&eng1, f32_sample, outBuffer_f32_16khz, BLOCK_SIZE * 2);
        inBuffer_s16_8k += BLOCK_SIZE;

        fwrite(outBuffer_f32_16khz, BLOCK_SIZE * 2, 4, fpf32);
    }

    tflite_destroy(&eng1);
    fclose(fpf32);
}
