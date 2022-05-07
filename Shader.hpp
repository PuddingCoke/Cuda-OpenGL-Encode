#pragma once

#include<string>
#include<iostream>
#include<fstream>
#include<sstream>

#include<glad/glad.h>

using std::cout;

unsigned int compileShader(const std::string source, const unsigned& type)
{
	unsigned int id = glCreateShader(type);
	const char* src = source.c_str();
	glShaderSource(id, 1, &src, nullptr);
	glCompileShader(id);

	int result = GL_FALSE;

	glGetShaderiv(id, GL_COMPILE_STATUS, &result);

	if (result == GL_FALSE)
	{
		int length;
		glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
		char* message = new char[length];
		glGetShaderInfoLog(id, length, &length, message);
		cout << message << "\n";
		delete[] message;
		glDeleteShader(id);
		return 0;
	}

	return id;
}

std::string readAllText(const std::string& filePath)
{
	std::ifstream stream(filePath);
	std::string str;
	if (stream.is_open())
	{
		std::ostringstream ss;
		ss << stream.rdbuf();
		str = ss.str();
		stream.close();
	}
	return str;
}

class Shader
{
public:

	Shader(const std::string vertPath, const std::string fragPath)
	{
		const std::string vertSource = readAllText(vertPath);
		const std::string fragSource = readAllText(fragPath);

		const unsigned int vertID = compileShader(vertSource, GL_VERTEX_SHADER);
		const unsigned int fragID = compileShader(fragSource, GL_FRAGMENT_SHADER);

		programID = glCreateProgram();

		glAttachShader(programID, vertID);
		glDeleteShader(vertID);

		glAttachShader(programID, fragID);
		glDeleteShader(fragID);

		glLinkProgram(programID);
		glValidateProgram(programID);

	}

	~Shader()
	{
		glDeleteProgram(programID);
	}

	void bind()
	{
		glUseProgram(programID);
	}

	void unbind()
	{
		glUseProgram(0);
	}

	unsigned programID;

};