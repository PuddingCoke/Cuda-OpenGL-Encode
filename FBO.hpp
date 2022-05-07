#pragma once

#include<glad/glad.h>

class FBO
{
public:

	FBO(const int& width,const int& height)
	{
		glGenFramebuffers(1, &fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);

		glGenTextures(1, &textureID);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID, 0);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	~FBO()
	{
		glDeleteFramebuffers(1, &fbo);
		glDeleteTextures(1, &textureID);
	}

	void bind() const
	{
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	}

	void unbind() const
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void attach() const
	{
		glBindTexture(GL_TEXTURE_2D, textureID);
	}

	void dettach() const
	{
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	unsigned int fbo;

	unsigned int textureID;

};