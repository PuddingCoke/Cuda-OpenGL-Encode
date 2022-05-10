#include<iostream>
#include<cmath>
#include<vector>
#include<fstream>

#include<NvEnc/nvEncodeAPI.h>
#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cudaGL.h>

using std::cout;

constexpr int scrWidth = 1920;
constexpr int scrHeight = 1080;

int main()
{
	if (!glfwInit())
		return 0;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	GLFWwindow* window = glfwCreateWindow(scrWidth, scrHeight, u8"Cuda Encode Test", nullptr, nullptr);

	if (!window)
	{
		glfwTerminate();
		return 0;
	}

	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "failed to initialize GLAD\n";
		return 0;
	}

	glfwSwapInterval(0);
	glfwHideWindow(window);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	unsigned int VAO;

	glGenVertexArrays(1, &VAO);

	cout << "cuda ini status " << cuInit(0) << "\n";

	int deviceCount;

	cudaGetDeviceCount(&deviceCount);

	cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, 0);

	cout << prop.name << "\n";

	CUdevice cuDevice;

	cout << "get device status " << cuDeviceGet(&cuDevice, 0) << "\n";

	CUcontext cuCtx;

	cout << "context create status " << cuCtxCreate(&cuCtx, CU_CTX_SCHED_AUTO, cuDevice) << "\n";

	NV_ENCODE_API_FUNCTION_LIST nvencAPI = { NV_ENCODE_API_FUNCTION_LIST_VER };

	cout << "instance create status " << NvEncodeAPICreateInstance(&nvencAPI) << "\n";

	NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS params = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
	params.device = cuCtx;
	params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
	params.apiVersion = NVENCAPI_VERSION;

	void* encoder;

	cout << "open encode status " << nvencAPI.nvEncOpenEncodeSessionEx(&params, &encoder) << "\n";

	NV_ENC_PRESET_CONFIG presetConfig = { NV_ENC_PRESET_CONFIG_VER,{NV_ENC_CONFIG_VER} };

	cout << "get preset config status " << nvencAPI.nvEncGetEncodePresetConfigEx(encoder, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P3_GUID, NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY, &presetConfig) << "\n";

	NV_ENC_CONFIG config;
	memcpy(&config, &presetConfig.presetCfg, sizeof(NV_ENC_CONFIG));
	config.version = NV_ENC_CONFIG_VER;
	config.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;
	config.rcParams.averageBitRate = 50000000U;

	NV_ENC_INITIALIZE_PARAMS encoderParams = { NV_ENC_INITIALIZE_PARAMS_VER };
	encoderParams.encodeConfig = &config;
	encoderParams.encodeGUID = NV_ENC_CODEC_H264_GUID;
	encoderParams.presetGUID = NV_ENC_PRESET_P3_GUID;
	encoderParams.tuningInfo = NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
	encoderParams.encodeWidth = scrWidth;
	encoderParams.encodeHeight = scrHeight;
	encoderParams.darWidth = scrWidth;
	encoderParams.darHeight = scrHeight;
	encoderParams.frameRateNum = 60;
	encoderParams.frameRateDen = 1;
	encoderParams.enablePTD = 1;
	encoderParams.maxEncodeWidth = scrWidth;
	encoderParams.maxEncodeHeight = scrHeight;
	encoderParams.enableEncodeAsync = 0;

	cout << "initialize encoder status " << nvencAPI.nvEncInitializeEncoder(encoder, &encoderParams) << "\n";

	int idx = 0;

	int frameRecorded = 0;

	constexpr int pboNum = 8;

	unsigned int pbos[pboNum];

	const int byteNum = scrWidth * scrHeight * 4;

	const int totalFrame = 1000;

	glGenBuffers(pboNum, pbos);
	for (size_t i = 0; i < pboNum; i++)
	{
		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[i]);
		glBufferData(GL_PIXEL_PACK_BUFFER, byteNum, nullptr, GL_STREAM_READ);
	}
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

	CUgraphicsResource resource[pboNum];

	for (size_t i = 0; i < pboNum; i++)
	{
		cout << "cuda register pbo status " << cuGraphicsGLRegisterBuffer(&resource[i], pbos[i], cudaGraphicsRegisterFlagsReadOnly) << "\n";
	}

	NV_ENC_CREATE_BITSTREAM_BUFFER bitstream = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER };

	cout << "create bitstream status " << nvencAPI.nvEncCreateBitstreamBuffer(encoder, &bitstream) << "\n";

	FILE* stream = _popen("ffmpeg -f h264 -i pipe: -c copy output.mp4", "wb");

	bool finished = false;

	float theta = 0;

	while (!finished)
	{
		glClearColor(cosf(theta), sinf(theta), 1.f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT);

		theta += 0.01f;

		if (frameRecorded < pboNum)
		{
			glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[idx]);
			glReadPixels(0, 0, scrWidth, scrHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		}
		else
		{
			cuGraphicsMapResources(1, &resource[idx], (CUstream)0);

			CUdeviceptr devicePtr;

			size_t pSize;

			cuGraphicsResourceGetMappedPointer(&devicePtr, &pSize, resource[idx]);

			NV_ENC_REGISTER_RESOURCE registerResource = { NV_ENC_REGISTER_RESOURCE_VER };

			registerResource.bufferFormat = NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ABGR;

			registerResource.bufferUsage = NV_ENC_BUFFER_USAGE::NV_ENC_INPUT_IMAGE;

			registerResource.resourceToRegister = (void*)devicePtr;

			registerResource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;

			registerResource.width = scrWidth;

			registerResource.height = scrHeight;

			registerResource.pitch = scrWidth * 4;

			registerResource.pInputFencePoint = nullptr;

			registerResource.pOutputFencePoint = nullptr;

			nvencAPI.nvEncRegisterResource(encoder, &registerResource);

			NV_ENC_MAP_INPUT_RESOURCE mapResource = { NV_ENC_MAP_INPUT_RESOURCE_VER };

			mapResource.registeredResource = registerResource.registeredResource;

			nvencAPI.nvEncMapInputResource(encoder, &mapResource);

			NV_ENC_PIC_PARAMS picParams = { NV_ENC_PIC_PARAMS_VER };

			if (frameRecorded == totalFrame + pboNum)
			{
				picParams.encodePicFlags = NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_EOS;

				nvencAPI.nvEncEncodePicture(encoder, &picParams);

				finished = true;
			}
			else
			{
				picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;

				picParams.inputBuffer = mapResource.mappedResource;

				picParams.bufferFmt = NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ABGR;

				picParams.inputWidth = scrWidth;

				picParams.inputHeight = scrHeight;

				picParams.outputBitstream = bitstream.bitstreamBuffer;

				nvencAPI.nvEncEncodePicture(encoder, &picParams);

				NV_ENC_LOCK_BITSTREAM lockBitsream = { NV_ENC_LOCK_BITSTREAM_VER };

				lockBitsream.outputBitstream = bitstream.bitstreamBuffer;

				lockBitsream.doNotWait = 0;

				nvencAPI.nvEncLockBitstream(encoder, &lockBitsream);

				fwrite(lockBitsream.bitstreamBufferPtr, lockBitsream.bitstreamSizeInBytes, 1, stream);

				nvencAPI.nvEncUnlockBitstream(encoder, lockBitsream.outputBitstream);
			}

			nvencAPI.nvEncUnregisterResource(encoder, registerResource.registeredResource);

			nvencAPI.nvEncUnmapInputResource(encoder, mapResource.mappedResource);

			cuGraphicsUnmapResources(1, &resource[idx], (CUstream)0);

			glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[idx]);
			glReadPixels(0, 0, scrWidth, scrHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		}

		frameRecorded++;
		idx++;
		idx = idx % pboNum;

		glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

		glfwPollEvents();
	}

	for (int i = 0; i < pboNum; i++)
	{
		cuGraphicsUnregisterResource(resource[i]);
	}

	_pclose(stream);

	glDeleteBuffers(pboNum, pbos);

	nvencAPI.nvEncDestroyBitstreamBuffer(encoder, bitstream.bitstreamBuffer);
	nvencAPI.nvEncDestroyEncoder(encoder);
	cuCtxDestroy(cuCtx);

	glDeleteVertexArrays(1, &VAO);
	glfwTerminate();

	return 0;
}