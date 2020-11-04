#version 460

uniform uint histMax;

layout(std430, binding = 6) buffer histBuffer{ uint hist[]; };

out vec4 FragColor;

in vs_out {
	vec2 uv;
} i;



void main() {
	//normalize u and v
	uint index = uint(i.uv.x * 254);
	uint histBinVal = hist[index];
	float histValNorm = float(histBinVal)/(float(histMax));

	if(i.uv.y < histValNorm){
	//if(0 < histValNorm){
		FragColor = vec4(1.0,1.0,1.0,1.0);
	}
	else{
		FragColor = vec4(0.0,0.0,0.0,1.0);
	}
}