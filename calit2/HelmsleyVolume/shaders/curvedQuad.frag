#version 460

uniform vec4 Color;

out vec4 FragColor;



in vs_out {
	vec2 uv;
} i;



void main() {



	 
		
	FragColor = vec4(Color);
}