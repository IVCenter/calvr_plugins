#version 460

uniform vec3 Color;

out vec4 FragColor;



in vs_out {
	vec2 uv;
	vec3 ver;
} i;



void main() {
	

	
	FragColor = vec4(Color,1.0);
}