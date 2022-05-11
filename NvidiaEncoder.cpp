#include "NvidiaEncoder.hpp"

NvidiaEncoder::NvidiaEncoder(const int& frameToEncode, const int frameRate, const uint32_t& encodeWidth, const uint32_t& encodeHeight) :
	frameToEncode(frameToEncode), idx(0), encodeWidth(encodeWidth), encodeHeight(encodeHeight), frameEncoded(0), encoding(true)
{
	std::cout << "cuda ini status " << cuInit(0) << "\n";

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	std::cout << deviceProp.name << "\n";

	CUdevice cuDevice;
	std::cout << "get device status" << cuDeviceGet(&cuDevice, 0) << "\n";
	std::cout << "cuda context create status " << cuCtxCreate(&cuCtx, CU_CTX_SCHED_AUTO, cuDevice) << "\n";

	nvencAPI = { NV_ENCODE_API_FUNCTION_LIST_VER };
	std::cout << "api instance create status " << NvEncodeAPICreateInstance(&nvencAPI) << "\n";

	NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sessionParams = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
	sessionParams.device = cuCtx;
	sessionParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
	sessionParams.apiVersion = NVENCAPI_VERSION;

	std::cout << "open encode session status " << nvencAPI.nvEncOpenEncodeSessionEx(&sessionParams, &encoder) << "\n";

	NV_ENC_PRESET_CONFIG presetConfig = { NV_ENC_PRESET_CONFIG_VER,{NV_ENC_CONFIG_VER} };

	std::cout << "get preset config status " << nvencAPI.nvEncGetEncodePresetConfigEx(encoder, codec, preset, tunningInfo, &presetConfig) << "\n";

	NV_ENC_CONFIG config;
	memcpy(&config, &presetConfig.presetCfg, sizeof(NV_ENC_CONFIG));
	config.version = NV_ENC_CONFIG_VER;
	config.profileGUID = profile;
	config.rcParams.averageBitRate = 50000000U;
	config.rcParams.maxBitRate = 60000000U;

	NV_ENC_INITIALIZE_PARAMS encoderParams = { NV_ENC_INITIALIZE_PARAMS_VER };
	encoderParams.encodeConfig = &config;
	encoderParams.encodeGUID = codec;
	encoderParams.presetGUID = preset;
	encoderParams.tuningInfo = tunningInfo;
	encoderParams.encodeWidth = encodeWidth;
	encoderParams.encodeHeight = encodeHeight;
	encoderParams.darWidth = encodeWidth;
	encoderParams.darHeight = encodeHeight;
	encoderParams.maxEncodeWidth = encodeWidth;
	encoderParams.maxEncodeHeight = encodeHeight;
	encoderParams.frameRateNum = frameRate;
	encoderParams.frameRateDen = 1;
	encoderParams.enablePTD = 1;
	encoderParams.enableEncodeAsync = 0;

	std::cout << "ini encoder status " << nvencAPI.nvEncInitializeEncoder(encoder, &encoderParams) << "\n";

	const int byteNum = encodeWidth * encodeHeight * 4;

	glGenBuffers(pboNum, pbos);
	for (int i = 0; i < pboNum; i++)
	{
		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[i]);
		glBufferData(GL_PIXEL_PACK_BUFFER, byteNum, nullptr, GL_STREAM_READ);
	}
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

	for (size_t i = 0; i < pboNum; i++)
	{
		std::cout << "cuda register pbo status " << cuGraphicsGLRegisterBuffer(&resources[i], pbos[i], cudaGraphicsRegisterFlagsReadOnly) << "\n";
	}

	bitstream = { NV_ENC_CREATE_BITSTREAM_BUFFER_VER };

	std::cout << "create bitstream status " << nvencAPI.nvEncCreateBitstreamBuffer(encoder, &bitstream) << "\n";

	std::cout << "render at " << encodeWidth << " x " << encodeHeight << "\n";
	std::cout << "frameRate " << frameRate << "\n";
	std::cout << "frameToEncode " << frameToEncode << "\n";

	stream = _popen("ffmpeg -y -f h264 -i pipe: -c copy output.mp4", "wb");
}

NvidiaEncoder::~NvidiaEncoder()
{
	glDeleteBuffers(pboNum, pbos);
	std::cout << "destroy bitstream status " << nvencAPI.nvEncDestroyBitstreamBuffer(encoder, bitstream.bitstreamBuffer) << "\n";
	std::cout << "destroy encoder status " << nvencAPI.nvEncDestroyEncoder(encoder) << "\n";
	for (int i = 0; i < pboNum; i++)
	{
		std::cout << "unregister resource status " << cuGraphicsUnregisterResource(resources[i]) << "\n";
	}
	cuCtxDestroy(cuCtx);
}

bool NvidiaEncoder::encode()
{
	if (frameEncoded < pboNum)
	{
		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[idx]);
		glReadPixels(0, 0, encodeWidth, encodeHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	}
	else
	{
		std::cout << cuGraphicsMapResources(1, &resources[idx], (CUstream)0) << "\n";

		CUdeviceptr devicePtr;

		size_t pSize;

		std::cout << cuGraphicsResourceGetMappedPointer(&devicePtr, &pSize, resources[idx]) << "\n";

		NV_ENC_REGISTER_RESOURCE registerResource = { NV_ENC_REGISTER_RESOURCE_VER };

		registerResource.bufferFormat = NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ABGR;

		registerResource.bufferUsage = NV_ENC_BUFFER_USAGE::NV_ENC_INPUT_IMAGE;

		registerResource.resourceToRegister = (void*)devicePtr;

		registerResource.resourceType = NV_ENC_INPUT_RESOURCE_TYPE::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;

		registerResource.width = encodeWidth;

		registerResource.height = encodeHeight;

		registerResource.pitch = encodeWidth * 4;

		registerResource.pInputFencePoint = nullptr;

		registerResource.pOutputFencePoint = nullptr;

		std::cout << nvencAPI.nvEncRegisterResource(encoder, &registerResource) << "\n";

		NV_ENC_MAP_INPUT_RESOURCE mapResource = { NV_ENC_MAP_INPUT_RESOURCE_VER };

		mapResource.registeredResource = registerResource.registeredResource;

		std::cout << nvencAPI.nvEncMapInputResource(encoder, &mapResource) << "\n";

		NV_ENC_PIC_PARAMS picParams = { NV_ENC_PIC_PARAMS_VER };

		if (frameEncoded == frameToEncode + pboNum)
		{
			picParams.encodePicFlags = NV_ENC_PIC_FLAGS::NV_ENC_PIC_FLAG_EOS;

			nvencAPI.nvEncEncodePicture(encoder, &picParams);

			encoding = false;

			_pclose(stream);
		}
		else
		{
			picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;

			picParams.inputBuffer = mapResource.mappedResource;

			picParams.bufferFmt = NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ABGR;

			picParams.inputWidth = encodeWidth;

			picParams.inputHeight = encodeHeight;

			picParams.outputBitstream = bitstream.bitstreamBuffer;

			nvencAPI.nvEncEncodePicture(encoder, &picParams);

			NV_ENC_LOCK_BITSTREAM lockBitsream = { NV_ENC_LOCK_BITSTREAM_VER };

			lockBitsream.outputBitstream = bitstream.bitstreamBuffer;

			lockBitsream.doNotWait = 0;

			nvencAPI.nvEncLockBitstream(encoder, &lockBitsream);

			fwrite(lockBitsream.bitstreamBufferPtr, lockBitsream.bitstreamSizeInBytes, 1, stream);

			nvencAPI.nvEncUnlockBitstream(encoder, lockBitsream.outputBitstream);
		}

		std::cout << nvencAPI.nvEncUnmapInputResource(encoder, mapResource.mappedResource) << "\n";

		std::cout << nvencAPI.nvEncUnregisterResource(encoder, registerResource.registeredResource) << "\n";

		std::cout << cuGraphicsUnmapResources(1, &resources[idx], (CUstream)0) << "\n";

		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[idx]);
		glReadPixels(0, 0, encodeWidth, encodeHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	}

	frameEncoded++;
	idx++;
	idx = idx % pboNum;

	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

	return encoding;
}
