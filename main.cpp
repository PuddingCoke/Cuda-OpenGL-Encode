#include<iostream>

#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include<cmath>

#include"Shader.hpp"
#include"FBO.hpp"

using std::cout;

int main()
{
	if (!glfwInit())
		return 0;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	GLFWwindow* window = glfwCreateWindow(1920, 1080, u8"Cuda Encode Test", nullptr, nullptr);

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

	float theta = 0;

	unsigned int VAO;

	glGenVertexArrays(1, &VAO);

	FBO frameBuffer(1920, 1080);

	while (!glfwWindowShouldClose(window))
	{
		frameBuffer.bind();
		glClearColor(cosf(theta), sinf(theta), 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		frameBuffer.unbind();

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

	return 0;
}