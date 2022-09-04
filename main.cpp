#include<iostream>
#include<cmath>

#include"NvidiaEncoder.hpp"
#include<GLFW/glfw3.h>

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

	NvidiaEncoder* encoder = new NvidiaEncoder(1000, 60, scrWidth, scrHeight);

	float theta = 0;

	do
	{
		glClearColor(cosf(theta), sinf(theta), 1.f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT);
		theta += 0.01f;
	} while (encoder->encode());

	delete encoder;

	glfwTerminate();

	return 0;
}