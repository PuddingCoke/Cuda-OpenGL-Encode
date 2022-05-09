#include<iostream>
#include<cmath>
#include<vector>


#include"Shader.hpp"
#include"FBO.hpp"

#include<NvEnc/nvEncodeAPI.h>
#include<glad/glad.h>
#include<GLFW/glfw3.h>

#include<cuda.h>
#include<cuda_runtime.h>
#include<driver_types.h>
#include<cudaGL.h>

using std::cout;

constexpr int scrWidth = 1280;
constexpr int scrHeight = 720;

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

	std::cout << glGetString(GL_VERSION) << "\n";
	std::cout << glGetString(GL_RENDERER) << "\n";

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	Shader shader("vertShader.vert", "fragShader.frag");

	unsigned int VAO;

	glGenVertexArrays(1, &VAO);

	FBO frameBuffer(scrWidth, scrHeight);

	cout << "cuda ini status " << cuInit(0) << "\n";

	int deviceCount;

	cudaGetDeviceCount(&deviceCount);

	cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, 0);

	CUdevice cuDevice;

	cout << "get device status " << cuDeviceGet(&cuDevice, 0) << "\n";

	CUcontext cuCtx;

	cout << "context create status " << cuCtxCreate(&cuCtx, CU_CTX_SCHED_AUTO, cuDevice) << "\n";

	NV_ENCODE_API_FUNCTION_LIST encodeAPI = { NV_ENCODE_API_FUNCTION_LIST_VER };

	cout << "instance create status " << NvEncodeAPICreateInstance(&encodeAPI) << "\n";

	void* encoder;

	NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS params = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
	params.device = cuCtx;
	params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
	params.apiVersion = NVENCAPI_VERSION;

	cout << "open encode status " << encodeAPI.nvEncOpenEncodeSessionEx(&params, &encoder) << "\n";

	NV_ENC_INITIALIZE_PARAMS initializeParams;

	memset(&initializeParams, 0, sizeof(NV_ENC_INITIALIZE_PARAMS));

	initializeParams.version = NV_ENC_INITIALIZE_PARAMS_VER;
	initializeParams.encodeGUID = NV_ENC_CODEC_H264_GUID;
	initializeParams.presetGUID = NV_ENC_PRESET_P5_GUID;
	initializeParams.encodeWidth = scrWidth;
	initializeParams.encodeHeight = scrHeight;
	initializeParams.darWidth = scrWidth;
	initializeParams.darHeight = scrHeight;
	initializeParams.frameRateNum = 60;
	initializeParams.frameRateDen = 1;
	initializeParams.bufferFormat = NV_ENC_BUFFER_FORMAT_ABGR;
	initializeParams.enableEncodeAsync = 0;
	initializeParams.enablePTD = 1;
	initializeParams.maxEncodeWidth = scrWidth;
	initializeParams.maxEncodeHeight = scrHeight;
	initializeParams.tuningInfo = NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;

	NV_ENC_PRESET_CONFIG presetConfig = { NV_ENC_PRESET_CONFIG_VER,{NV_ENC_CONFIG_VER} };

	cout << "get encode config status " << encodeAPI.nvEncGetEncodePresetConfigEx(encoder, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P5_GUID, NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY, &presetConfig) << "\n";
	
	NV_ENC_CONFIG encodeCfg;

	memcpy(&encodeCfg, &presetConfig.presetCfg, sizeof(NV_ENC_CONFIG));

	encodeCfg.version = NV_ENC_CONFIG_VER;
	encodeCfg.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;
	encodeCfg.frameIntervalP = 1;
	encodeCfg.gopLength = NVENC_INFINITE_GOPLENGTH;

	initializeParams.encodeConfig = &encodeCfg;

	cout << "initialize encoder status " << encodeAPI.nvEncInitializeEncoder(encoder, &initializeParams) << "\n";

	CUgraphicsResource resource;
	
	cout << "cuda register image status " << cuGraphicsGLRegisterImage(&resource, frameBuffer.textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly) << "\n";

	float theta = 0;
	
	NV_ENC_CREATE_BITSTREAM_BUFFER bitstream = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER };

	cout << "bitstream create status " << encodeAPI.nvEncCreateBitstreamBuffer(encoder, &bitstream) << "\n";

	while (!glfwWindowShouldClose(window))
	{
		frameBuffer.bind();
		glClearColor(cosf(theta), sinf(theta), 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		frameBuffer.unbind();
		
		cuGraphicsMapResources(1, &resource, (CUstream)0);

		CUarray cuArray;

		cuGraphicsSubResourceGetMappedArray(&cuArray, resource, 0, 0);

		NV_ENC_REGISTER_RESOURCE nvencResource = { NV_ENC_REGISTER_RESOURCE_VER };

		nvencResource.bufferFormat = NV_ENC_BUFFER_FORMAT_ABGR;

		nvencResource.bufferUsage = NV_ENC_BUFFER_USAGE::NV_ENC_INPUT_IMAGE;

		nvencResource.width = scrWidth;

		nvencResource.height = scrHeight;

		nvencResource.pitch = scrWidth;

		nvencResource.resourceToRegister = cuArray;

		nvencResource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY;

		nvencResource.pInputFencePoint = nullptr;

		nvencResource.pOutputFencePoint = nullptr;

		cout << encodeAPI.nvEncRegisterResource(encoder, &nvencResource) << "\n";
		
		NV_ENC_MAP_INPUT_RESOURCE mapResource = { NV_ENC_MAP_INPUT_RESOURCE_VER };

		mapResource.registeredResource = nvencResource.registeredResource;

		cout << encodeAPI.nvEncMapInputResource(encoder, &mapResource) << "\n";

		NV_ENC_PIC_PARAMS picParams = { 0 };

		picParams.version = NV_ENC_PIC_PARAMS_VER;

		picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;

		picParams.inputBuffer = mapResource.mappedResource;

		picParams.bufferFmt = mapResource.mappedBufferFmt;

		picParams.inputWidth = scrWidth;

		picParams.inputHeight = scrHeight;

		picParams.outputBitstream = bitstream.bitstreamBuffer;

		picParams.inputPitch = scrWidth;

		cout << encodeAPI.nvEncEncodePicture(encoder, &picParams) << "\n";

		cuGraphicsUnmapResources(1, &resource, (CUstream)0);

		glClearColor(0.f, 0.f, 0.f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT);

		shader.bind();
		frameBuffer.attach();
		glBindVertexArray(VAO);

		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

		shader.unbind();
		frameBuffer.dettach();
		glBindVertexArray(0);

		theta += 0.01f;

		glfwSwapBuffers(window);

		glfwPollEvents();
	}

	glDeleteVertexArrays(1, &VAO);
	glfwTerminate();

	encodeAPI.nvEncDestroyEncoder(encoder);
	cuCtxDestroy(cuCtx);

	return 0;
}