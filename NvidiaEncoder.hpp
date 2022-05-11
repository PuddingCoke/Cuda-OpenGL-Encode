#pragma once

#ifndef _NVIDIA_ENCODER_HPP_
#define _NVIDIA_ENCODER_HPP_

#include<iostream>
#include<fstream>

#include<NvEnc/nvEncodeAPI.h>

#include<glad/glad.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cudaGL.h>

class NvidiaEncoder
{
public:

	NvidiaEncoder() = delete;

	NvidiaEncoder(const NvidiaEncoder&) = delete;

	NvidiaEncoder(const int& frameToEncode, const int frameRate,const uint32_t& encodeWidth,const uint32_t& encodeHeight);

	~NvidiaEncoder();

	bool encode();

private:

	int idx;

	int frameEncoded;

	const int frameToEncode;

	const uint32_t encodeWidth;

	const uint32_t encodeHeight;

	bool encoding;

	static constexpr int pboNum = 8;

	unsigned int pbos[pboNum];

	void* encoder;

	CUgraphicsResource resources[pboNum];

	CUcontext cuCtx;

	NV_ENCODE_API_FUNCTION_LIST nvencAPI;

	NV_ENC_CREATE_BITSTREAM_BUFFER bitstream;

	FILE* stream;

	const GUID codec = NV_ENC_CODEC_H264_GUID;

	const GUID preset = NV_ENC_PRESET_P3_GUID;

	const GUID profile = NV_ENC_H264_PROFILE_HIGH_GUID;

	const NV_ENC_TUNING_INFO tunningInfo = NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;

};

#endif // !_NVIDIA_ENCODER_HPP_