#version 430 core

out vec4 color;

in vec2 texCoord;

uniform sampler2D u_Texture;

void main()
{
	color = texture(u_Texture, texCoord);
};