#version 460

uniform vec3 Color;

out vec4 FragColor;



in vs_out {
	vec2 uv;
} i;



void main() {
	float alpha = 1.0;
	float borderColor = 0.0f;
	vec3 col;


	if(i.uv.y < 0.05)
		;//col = vec3(borderColor, borderColor, borderColor);
	else if(i.uv.y > (i.uv.x*2)-0.05 && i.uv.y < (i.uv.x*2)+0.05)
		;//col =  vec3(0.0, 1.0, borderColor);
	else if(i.uv.y > 2*(-i.uv.x + 1)-0.05  && i.uv.y < 2*((-i.uv.x + 1)+0.05))	
		;//col =  vec3(1.0, borderColor, borderColor);
	else if(i.uv.x < .05)
		;//col =  vec3(0.0, 0.0, 0.0);
	else if(true)
		col = Color;
		
	FragColor = vec4(col, alpha);
}