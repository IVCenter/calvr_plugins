#version 460

uniform float Center = .7;
uniform float Width = 0.5;

out vec4 FragColor;

in vs_out {
	vec2 uv;
} i;



void main() {
	float alpha = 1.0;

	//Width is scalar for triangle
	float xPos = (i.uv.x - .5) * Width;
	alpha = 1.0 - step(1.0, xPos + Center);//Right Bounds
	if(alpha == 1.0)
		alpha = step(0.0, Center + xPos);//Left Bounds
	vec3 col = vec3(0.8, 0.9, 1.0);
	FragColor = vec4(col, alpha);
}