#version 460

uniform float Center;
uniform float Width;
uniform vec3 Color;
uniform bool Selected;

out vec4 FragColor;



in vs_out {
	vec2 uv;
} i;



void main() {
	float alpha = 1.0;
	float borderColor = 0.0f;
	if(Selected){
		borderColor = 1.0f;
	}
	//Width is scalar for triangle
	float xPos = (i.uv.x - .5) * Width;
	alpha = 1.0 - step(1.0, xPos + Center);//Right Bounds
	if(alpha == 1.0)
		alpha = step(0.0, Center + xPos);//Left Bounds
	vec3 col = vec3(0.8, 0.9, 1.0);
//	if(i.uv.y < 0.05)
//		col = vec3(borderColor, borderColor, borderColor);
//	else if(i.uv.y > (i.uv.x*2)-0.05 && i.uv.y < (i.uv.x*2)+0.05)
//		col =  vec3(borderColor, borderColor, borderColor);
//	else if(i.uv.y > 2*(-i.uv.x + 1)-0.05  && i.uv.y < 2*((-i.uv.x + 1)+0.05))	
//		col =  vec3(borderColor, borderColor, borderColor);
	//else
		col = Color;
		
	if(alpha > 0.0)
		alpha = .7f;
	FragColor = vec4(col, alpha);
}